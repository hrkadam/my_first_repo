#!/usr/bin/env python3
"""
Summarize a unified diff using Ollama (local/remote).

Fully rewritten, clean version that:
- Parses unified diff with unidiff.
- Builds an ADDED_ONLY view (robust): collects true added lines from hunks,
  filters diff noise, and includes preceding signature/decorator context when needed.
- Prompts the LLM to enumerate *actual* additions: new functions/classes and other code changes
  (e.g., added print statements), with 1–2 line overviews.
- Synthesizes an overall commit summary.
- Writes summary.txt.

CLI:
  python summarize_diff_ollama_clean.py --patch diff.patch --repo <owner/name> \
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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
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
    lines: List[str] = []
    for ln in text.splitlines():
        if ln.startswith(("+", "-", "@@")) or ln.startswith(DIFF_NOISE_PREFIXES):
            lines.append(ln)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return "\n".join(lines)


def build_added_only_from_patch_file(pf) -> str:
    """Construct an 'added-only' text view for a PatchFile.
    - Include true added lines (l.is_added).
    - Exclude diff metadata noise lines.
    - If an added block is preceded by a *context* signature/decorator line, include it too.
    """
    out: List[str] = []
    hcount = 0
    for h in pf:
        hcount += 1
        if hcount > MAX_HUNKS_PER_FILE:
            logger.warning("Hunk cap reached for %s (%d > %d)", pf.path, hcount, MAX_HUNKS_PER_FILE)
            break
        lines = list(h)
        for i, line in enumerate(lines):
            val = line.value.rstrip("\n")
            # Skip pure metadata noise lines regardless of flag
            if val in BACKSLASH_NOISE:
                continue
            # Capture added code lines
            if line.is_added:
                if val in ADDED_NOISE_EXACT:
                    continue
                # If previous is context and looks like signature/decorator, include it
                if i > 0:
                    prev = lines[i-1]
                    pval = prev.value.rstrip("\n")
                    if (prev.is_context) and looks_like_sig_or_decorator(pval):
                        out.append(pval.lstrip("+ ").strip())
                # Add the code line (strip leading '+')
                out.append(val[1:].rstrip() if val.startswith("+") else val)
    return "\n".join(out)


def looks_like_sig_or_decorator(text_line: str) -> bool:
    s = text_line.lstrip("+ ").strip()
    return (
        s.startswith("def ") or s.startswith("class ") or s.startswith("@")
    )

# ---------------- ollama ----------------
def ollama_chat(messages: List[dict], timeout: int) -> str:
    """Call Ollama /api/chat with low temperature for deterministic output."""
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

# ---------------- per-file summarization ----------------
def summarize_file(fname: str, minimal_patch: str, pf) -> str:
    """Prompt the LLM to produce an additions-focused summary per file."""
    lang = _file_language_hint(fname)
    added_view = build_added_only_from_patch_file(pf)
    trimmed_view = trim_patch_text(minimal_patch)

    
    system = textwrap.dedent("""
    You are a senior code reviewer. Summarize ONLY the actual additions made in this file.

    Rules:
    - If ADDED_ONLY is empty, return exactly: "No code additions detected."
    - Otherwise, summarize ALL added code:
    • New functions/classes (name + 1–2 line purpose)
    • Added statements such as print(), logging, returns, expressions, conditionals, variable updates, etc.
    - Each bullet must reflect only what is visible in ADDED_ONLY.
    - Do NOT invent behavior that is not present in the diff.
    - Keep it ≤ 10 lines, use short bullets.
    """).strip()


    language_tip = {
        "python": "Look for: def name(params):, class Name:, and added statements like print(...) or logging.*",
        "javascript/typescript": "Look for: function name(...), const name = (...) =>, class Name, and console.log(...)",
        "java": "Look for: returnType name(...), class Name, and System.out.println(...)",
        "go": "Look for: func name(...), type Name struct, and fmt.Println(...)",
        "csharp": "Look for: returnType Name(...), class Name, and Console.WriteLine(...)",
        "generic": "If nothing recognizable exists, return the 'No code additions detected.' message.",
    }[lang]

    user = (
        f"FILE: {fname}\n"
        f"LANGUAGE_HINT: {lang} — {language_tip}\n\n"
        "ADDED_ONLY:\n" + added_view + "\n\n" +
        "TRIMMED_DIFF (context only):\n" + trimmed_view
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

    logger.info("Synthesizing overall summary for %d file(s)", len(per_file_sections))
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

        # Build minimal unified text per file
        text_chunks: List[str] = []
        hcount = 0
        for h in pf:
            logger.info("Processing hunk: %s", h)
            hcount += 1
            if hcount > MAX_HUNKS_PER_FILE:
                logger.warning("Hunk cap reached for %s (%d > %d)", fname, hcount, MAX_HUNKS_PER_FILE)
                break
            hdr = f"@@ -{h.source_start},{h.source_length} +{h.target_start},{h.target_length} @@"
            changes = "\n".join(l.value.rstrip("\n") for l in h)
            logger.info("Hunk changes length: %d chars", len(changes))
            text_chunks.append(hdr + "\n" + changes)

        if not text_chunks:
            logger.info("No hunks to summarize for file: %s", fname)
            continue

        minimal = "\n".join(text_chunks)
        logger.info("Total minimal patch length for %s: %d chars", fname, len(minimal))
        minimal = "\n".join(minimal.splitlines()[:MAX_LINES_PER_FILE])
        logger.info("Trimmed minimal patch length for %s: %d chars", fname, len(minimal))

        # Summarize via LLM
        try:
            s = summarize_file(fname, minimal, pf)
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
