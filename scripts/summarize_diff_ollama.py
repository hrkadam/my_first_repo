#!/usr/bin/env python3
"""
Summarize a unified diff using Ollama (local/remote), LLM-only — FINAL PATCHED VERSION.

- Builds MINIMAL patch text with prefixes preserved ('+', '-', ' ') so additions are detectable.
- Extracts ADDED_ONLY directly from the minimal text (not from unidiff flags).
- Sanitizes ADDED_ONLY and wraps it in fenced Python code blocks so general models (e.g., llama3.2:latest) reliably treat it as code.
- Updated prompt explicitly instructs the model to ANALYZE the fenced code and to treat ALL lines in ADDED_ONLY as meaningful additions (including simple statements).
- Summarizes structural (functions/classes) and non-structural additions (print/log/assignments) in ≤10 bullets.

CLI:
  python scripts/summarize_diff_ollama_final.py --patch diff.patch --repo <owner/name> \
      --branch <branch> --sha <sha> --out summary.txt
"""
import os
import sys
import argparse
import textwrap
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Tuple

import requests
from unidiff import PatchSet

# ---------------- config ----------------
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_API_TOKEN = os.getenv("OLLAMA_API_TOKEN", "")

SKIP_DIRS = {"node_modules", "vendor", "dist", "build", ".venv", ".git"}
MAX_LINES_PER_FILE = 4000
MAX_HUNKS_PER_FILE = 200
TIMEOUT_PER_FILE = int(os.getenv("OLLAMA_TIMEOUT_PER_FILE", "180"))
TIMEOUT_OVERALL = int(os.getenv("OLLAMA_TIMEOUT_OVERALL", "600"))

# ---------------- logging ----------------
def _setup_logging() -> logging.Logger:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "summarize_patch.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = _setup_logging()

# ---------------- http helpers ----------------
def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if OLLAMA_API_TOKEN:
        h["Authorization"] = f"Bearer {OLLAMA_API_TOKEN}"
    return h

# ---------------- diff helpers ----------------
DIFF_NOISE_PREFIXES = (
    "diff --git",
    "index ",
    "--- ",
    "+++ ",
    "@@",
)
ADDED_NOISE_EXACT = {
    "+ No newline at end of file",
    "+\\ No newline at end of file",
}
BACKSLASH_NOISE = {"\\ No newline at end of file"}


def trim_patch_text(text: str, max_lines: int = 2000) -> str:
    """Keep only +/- lines, @@ headers, and minimal metadata; cap length."""
    out: List[str] = []
    for ln in text.splitlines():
        if ln.startswith(('+', '-', '@@')):
            out.append(ln)
        else:
            for pref in DIFF_NOISE_PREFIXES:
                if ln.startswith(pref):
                    out.append(ln)
                    break
    if len(out) > max_lines:
        out = out[:max_lines]
    return "\n".join(out)


def _looks_like_sig_or_decorator(line: str) -> bool:
    s = line.lstrip('+ ').strip()
    return s.startswith(('def ', 'class ', '@'))

# ---------------- MINIMAL (WITH PREFIXES) ----------------
def build_minimal_text_for_file(pf) -> str:
    """Compose a minimal unified text for a PatchFile: hunk headers + lines WITH diff prefixes."""
    def _prefix(l):
        if l.is_added:
            return '+'
        if l.is_removed:
            return '-'
        return ' '  # context line

    text_chunks: List[str] = []
    hcount = 0
    for h in pf:
        hcount += 1
        if hcount > MAX_HUNKS_PER_FILE:
            logger.warning("Hunk cap reached for %s (%d > %d)", pf.path, hcount, MAX_HUNKS_PER_FILE)
            break
        hdr = f"@@ -{h.source_start},{h.source_length} +{h.target_start},{h.target_length} @@"
        changes = "\n".join(_prefix(l) + l.value.rstrip("\n") for l in h)
        text_chunks.append(hdr + "\n" + changes)
    return "\n".join(text_chunks)

# ---------------- ADDED ONLY (FROM MINIMAL) ----------------
def build_added_only_from_minimal(minimal_patch: str, max_lines: int = 3000) -> str:
    """Robustly extract added lines directly from the minimal patch text.
    - Treat any line that starts with '+' (and is not metadata) as an addition.
    - Also capture a preceding context line if it is a signature/decorator (`def`, `class`, `@`).
    - Ignore '+ No newline at end of file' and header-like lines starting with '+++', '+@@', etc.
    - Tail-artifact heuristic: if an added line is immediately followed by a
      backslash 'No newline at end of file' line AND the previous context line equals the
      added line (ignoring leading/trailing whitespace), skip it.
    """
    out: List[str] = []
    lines = minimal_patch.splitlines()

    def is_added_code(ln: str) -> bool:
        if not ln.startswith('+'):
            return False
        if ln in ADDED_NOISE_EXACT:
            return False
        if ln.startswith(('+++', '+@@', '+index', '+diff --git', '+--')):
            return False
        if ln.strip() == '+':  # blank '+' line
            return False
        return True

    for i, ln in enumerate(lines):
        if is_added_code(ln):
            # --- Tail-artifact filter (newline metadata + duplicate context) ---
            next_is_backslash_meta = (i + 1 < len(lines) and lines[i+1] in BACKSLASH_NOISE)
            prev_is_same_context = False
            if i > 0:
                prev = lines[i-1]
                if prev.startswith(' '):  # unified diff context
                    prev_is_same_context = (prev.strip() == ln[1:].strip())
            if next_is_backslash_meta and prev_is_same_context:
                logger.info("Filtered tail artifact: %r", ln[1:].strip())
                continue

            # include previous context signature/decorator if present
            if i > 0:
                prev = lines[i-1]
                if (not prev.startswith(('+', '-'))) and _looks_like_sig_or_decorator(prev):
                    out.append(prev.strip())
            out.append(ln[1:].rstrip())
        if len(out) >= max_lines:
            break

    return "\n".join(out)

# ---------------- ollama ----------------
def ollama_chat(messages: List[dict], timeout: int) -> str:
    url = f"{OLLAMA_ENDPOINT}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192,
            "top_p": 0.9,
            "top_k": 40,
        },
    }
    logger.info("Calling Ollama: model=%s timeout=%s", OLLAMA_MODEL, timeout)
    r = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}) or {}).get("content", "").strip()

# ---------------- language hints ----------------
def _file_language_hint(fname: str) -> str:
    ext = os.path.splitext(fname)[1].lower()
    if ext == ".py":
        return "python"
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        return "javascript/typescript"
    elif ext in (".java",):
        return "java"
    elif ext in (".go",):
        return "go"
    elif ext in (".cs",):
        return "csharp"
    else:
        return "generic"

# ---------------- sanitization ----------------
def sanitize_added_only(added_view: str) -> str:
    safe: List[str] = []
    for ln in added_view.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("\\ No newline"):
            continue
        if s in {"No newline at end of file"}:
            continue
        if s.startswith('+'):
            s = s[1:].strip()
        s = s.replace('\r', '')
        safe.append(s)
    return "\n".join(safe)

# ---------------- per-file summarization ----------------
def summarize_file(fname: str, pf) -> str:
    lang = _file_language_hint(fname)

    minimal = build_minimal_text_for_file(pf)
    minimal = "\n".join(minimal.splitlines()[:MAX_LINES_PER_FILE])

    added_view_raw = build_added_only_from_minimal(minimal)
    added_view = sanitize_added_only(added_view_raw)
    trimmed_view = trim_patch_text(minimal)

    logger.info("ADDED_ONLY_SANITIZED:\n%s", added_view)

    # ✅ UPDATED PROMPT — force the model to analyze the fenced code and treat all lines as additions
    system = textwrap.dedent(
        """
        You are a senior code reviewer. Summarize ONLY the actual additions made in this file.

        Rules:
        - ADDED_ONLY contains pure code inside a fenced ```python``` block. You MUST analyze the code in this block.
        - If ADDED_ONLY is not empty, you MUST treat every line inside it as a real code addition.
        - Summarize ALL added code, including:
          • Functions/classes (name + 1–2 line purpose)
          • Simple statements (e.g., print(), logging, assignments, list/var initialization, returns, expressions, imports)
        - Do NOT ignore simple statements. They ARE meaningful additions.
        - Reflect only what is visible in ADDED_ONLY; no speculation.
        - Keep output ≤ 10 lines in short, scannable bullets.
        - Only if ADDED_ONLY is truly empty, return exactly: "No code additions detected."
        """
    ).strip()

    language_tip = {
        "python": "Focus on: def name(params):, class Name:, and added statements like print(...), assignments, list/var init, logging.*",
        "javascript/typescript": "Focus on: function name(...), const name = (...) =>, class Name, console.log(...)",
        "java": "Focus on: returnType name(...), class Name, System.out.println(...)",
        "go": "Focus on: func name(...), type Name struct, fmt.Println(...)",
        "csharp": "Focus on: returnType Name(...), class Name, Console.WriteLine(...)",
        "generic": "If nothing recognizable exists, return the 'No code additions detected.' message.",
    }[lang]

    user = (
        f"FILE: {fname}\n"
        f"LANGUAGE_HINT: {lang} — {language_tip}\n\n"
        "ADDED_ONLY (code):\n```python\n" + added_view + "\n```\n\n" +
        "TRIMMED_DIFF (context only):\n```diff\n" + trimmed_view + "\n```"
    )

    logger.info("Summarizing file: %s (added-only len=%d)", fname, len(added_view.splitlines()))
    return ollama_chat(
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        timeout=TIMEOUT_PER_FILE,
    )

# ---------------- overall synthesis ----------------
def synthesize_overall(repo: str, branch: str, sha: str, per_file_sections: List[Tuple[str, str]]) -> str:
    system = textwrap.dedent(
        """
        You are an engineering lead. Given multiple per-file summaries produced from actual diffs,
        synthesize an overall commit summary that aggregates the facts WITHOUT adding new assumptions.
        - Start with one line on overall intent if clearly implied by the file summaries.
        - Then bullets: file name → key additions (function/class names) and brief overviews.
        - Keep it scannable, ≤ 12 lines. No extra speculation.
        """
    ).strip()

    user = (
        f"Repository: {repo}\n"
        f"Branch: {branch}\n"
        f"Commit: {sha}\n\n" +
        "\n\n".join([f"FILE: {f}\nSUMMARY:\n{s}" for f, s in per_file_sections])
    )

    return ollama_chat(
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        timeout=TIMEOUT_OVERALL,
    )

# ---------------- main ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", default="summary.txt")
    args = ap.parse_args()

    logger.info(
        "Starting script with repo=%s branch=%s sha=%s out=%s",
        args.repo, args.branch, args.sha, args.out,
    )

    # Load patch
    try:
        with open(args.patch, "r", encoding="utf-8", errors="ignore") as f:
            patch = PatchSet(f)
        logger.info("Loaded patch file: %s (files in patch: %d)", args.patch, len(patch))
    except Exception:
        logger.exception("Failed to load/parse patch: %s", args.patch)
        sys.exit(2)

    per_file: List[Tuple[str, str]] = []
    for pf in patch:
        fname = pf.path or pf.target_file or pf.source_file or "<unknown>"
        # Skip noise dirs
        if fname:
            parts = fname.split("/")
            if any(d in SKIP_DIRS for d in parts):
                logger.info("Skipping file in ignored directory: %s", fname)
                continue

        # Summarize via LLM
        try:
            s = summarize_file(fname, pf)
            first_line = (s or "").splitlines()[0] if s else ""
            logger.info("Summary for file %s: %s", fname, first_line.replace("\n", " ")[:150])
        except Exception as e:
            logger.exception("Error summarizing file: %s", fname)
            s = f"(Error summarizing {fname}: {e})"

        per_file.append((fname, s))
        time.sleep(0.2)  # small pacing

    # Overall synthesis
    try:
        overall = synthesize_overall(args.repo, args.branch, args.sha, per_file)
        logger.info("Overall summary (head): %s", (overall[:200] or "").replace("\n", " "))
    except Exception as e:
        logger.exception("Error synthesizing overall summary")
        overall = f"(Error synthesizing overall summary: {e})"

    # Assemble final report
    report: List[str] = [
        f"Repository: {args.repo}",
        f"Branch: {args.branch}",
        f"Commit: {args.sha}",
        "",
        "Overall",
        overall or "",
        "",
        "Changes",
    ]
    for f, s in per_file:
        first_line = (s or "").splitlines()[0] if s else ""
        report.append(f"- {f}: {first_line}")

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(report).strip() + "\n")
        logger.info("Wrote summary to %s", args.out)
    except Exception:
        logger.exception("Failed writing output file: %s", args.out)
        sys.exit(3)

    logger.info("Completed successfully")


if __name__ == "__main__":
    logger.info("Entrypoint reached")
    main()
