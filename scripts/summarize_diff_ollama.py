#!/usr/bin/env python3
"""
Summarize a unified diff using Ollama (local/remote).
Fixes: additions-only view includes preceding signature context lines so new functions/classes
are detected even when the 'def/class' line appears as context.
"""
import os, sys, argparse, textwrap, time
from typing import List, Tuple
import requests
from unidiff import PatchSet
import logging
from logging.handlers import RotatingFileHandler

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_API_TOKEN = os.getenv("OLLAMA_API_TOKEN", "")
SKIP_DIRS = {"node_modules", "vendor", "dist", "build", ".venv", ".git"}
MAX_LINES_PER_FILE = 2000
MAX_HUNKS_PER_FILE = 50
TIMEOUT_PER_FILE = int(os.getenv("OLLAMA_TIMEOUT_PER_FILE", "120"))
TIMEOUT_OVERALL = int(os.getenv("OLLAMA_TIMEOUT_OVERALL", "600"))


def _setup_logging():
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


def _headers():
    h = {"Content-Type": "application/json"}
    if OLLAMA_API_TOKEN:
        h["Authorization"] = f"Bearer {OLLAMA_API_TOKEN}"
    return h


def trim_patch_text(text: str, max_lines=1200):
    lines = []
    for ln in text.splitlines():
        if ln.startswith(("+", "-", "@@", "diff --git", "index", "---", "+++")):
            lines.append(ln)
    return "
".join(lines[:max_lines])


def ollama_chat(messages: List[dict], timeout: int) -> str:
    url = f"{OLLAMA_ENDPOINT}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192, "top_p": 0.9, "top_k": 40},
    }
    logger.info("Calling Ollama: model=%s timeout=%s", OLLAMA_MODEL, timeout)
    r = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "").strip()


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

ADDED_PREFIXES_EXCLUDE = (
    "+++", "+--", "+@@", "+index", "+diff --git", "+ No newline at end of file",
)

def added_only(minimal_patch: str, max_lines: int = 3000) -> str:
    """Return '+' code lines; include preceding signature context if present."""
    out = []
    lines = minimal_patch.splitlines()
    def is_noise(ln: str) -> bool:
        return ln.startswith(ADDED_PREFIXES_EXCLUDE)
    def looks_like_sig(ln: str) -> bool:
        l = ln.lstrip('+ ').strip()
        return l.startswith('def ') or l.startswith('class ')
    for i, ln in enumerate(lines):
        if ln.startswith('+') and not is_noise(ln):
            if i > 0:
                prev = lines[i-1]
                if (not prev.startswith(('+','-'))) and looks_like_sig(prev):
                    out.append(prev.strip())
            code = ln[1:].rstrip()
            if code:
                out.append(code)
            if len(out) >= max_lines:
                break
    return "
".join(out)


def summarize_file(fname: str, minimal_patch: str) -> str:
    lang = _file_language_hint(fname)
    system = textwrap.dedent(
        """
        You are a senior code reviewer. Summarize ONLY the actual additions made in this file.
        STRICT RULES:
        - Use the ADDED_ONLY block to locate newly ADDED functions/classes/methods.
        - List each new function/class by exact name/signature. For each, add a 1–2 line overview
          (prefer adjacent docstrings/comments if present).
        - If ADDED_ONLY is empty or has no recognizable code, return exactly: "No code additions detected."
        - Do NOT invent content or infer behavior that is not visible in ADDED_ONLY.
        - Keep it ≤ 10 lines, scannable bullets.
        """
    ).strip()

    language_tip = {
        "python": "Look for: def name(params): and class Name:",
        "javascript/typescript": "Look for: function name(...), const name = (...) =>, class Name",
        "java": "Look for: returnType name(...), class Name",
        "go": "Look for: func name(...), type Name struct",
        "csharp": "Look for: returnType Name(...), class Name",
        "generic": "If nothing recognizable exists, return the 'No code additions detected.' message.",
    }[lang]

    added_view = added_only(minimal_patch)
    trimmed_view = trim_patch_text(minimal_patch)

    user = (
        f"FILE: {fname}
"
        f"LANGUAGE_HINT: {lang} — {language_tip}

"
        "ADDED_ONLY:
" + added_view + "

" +
        "TRIMMED_DIFF (context only):
" + trimmed_view
    )

    logger.info("Summarizing file: %s (added-only len=%d)", fname, len(added_view))
    return ollama_chat([
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ], timeout=TIMEOUT_PER_FILE)


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
        f"Repository: {repo}
Branch: {branch}
Commit: {sha}

" +
        "

".join([f"FILE: {f}
SUMMARY:
{s}" for f, s in per_file_sections])
    )

    logger.info("Synthesizing overall summary for %d file(s)", len(per_file_sections))
    return ollama_chat([
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ], timeout=TIMEOUT_OVERALL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", default="summary.txt")
    args = ap.parse_args()

    logger.info("Starting script with repo=%s branch=%s sha=%s out=%s", args.repo, args.branch, args.sha, args.out)

    try:
        with open(args.patch, "r", encoding="utf-8", errors="ignore") as f:
            patch = PatchSet(f)
        logger.info("Loaded patch file: %s (files in patch: %d)", args.patch, len(patch))
    except Exception:
        logger.exception("Failed to load/parse patch: %s", args.patch)
        sys.exit(2)

    per_file = []
    for pf in patch:
        fname = pf.path or pf.target_file
        if fname:
            parts = fname.split("/")
            if any(d in SKIP_DIRS for d in parts):
                logger.info("Skipping file in ignored directory: %s", fname)
                continue

        text_chunks = []
        hcount = 0
        for h in pf:
            logger.info("Processing hunk: %s", h)
            hcount += 1
            if hcount > MAX_HUNKS_PER_FILE:
                logger.warning("Hunk cap reached for %s (%d > %d)", fname, hcount, MAX_HUNKS_PER_FILE)
                break
            hdr = f"@@ -{h.source_start},{h.source_length} +{h.target_start},{h.target_length} @@"
            changes = "
".join(l.value.rstrip("
") for l in h)
            logger.info("Hunk changes length: %d chars", len(changes))
            text_chunks.append(hdr + "
" + changes)

        if not text_chunks:
            logger.info("No hunks to summarize for file: %s", fname)
            continue

        minimal = "
".join(text_chunks)
        logger.info("Total minimal patch length for %s: %d chars", fname, len(minimal))
        minimal = "
".join(minimal.splitlines()[:MAX_LINES_PER_FILE])
        logger.info("Trimmed minimal patch length for %s: %d chars", fname, len(minimal))

        try:
            s = summarize_file(fname or "<unknown>", minimal)
            first_line = (s or "").splitlines()[0] if s else ""
            logger.info("Summary for file %s: %s", fname, first_line.replace("
", " ")[:150])
        except Exception as e:
            logger.exception("Error summarizing file: %s", fname)
            s = f"(Error summarizing {fname}: {e})"

        per_file.append((fname or "<unknown>", s))
        time.sleep(0.2)

    try:
        overall = synthesize_overall(args.repo, args.branch, args.sha, per_file)
        logger.info("Overall summary (head): %s", (overall[:200] or "").replace("
", " "))
    except Exception as e:
        logger.exception("Error synthesizing overall summary")
        overall = f"(Error synthesizing overall summary: {e})"

    report = [
        f"Repository: {args.repo}",
        f"Branch: {args.branch}",
        f"Commit: {args.sha}",
        "",
        "Overall",
        overall or "",
        "",
        "Changes"
    ]
    for f, s in per_file:
        first_line = (s or "").splitlines()[0] if s else ""
        report.append(f"- {f}: {first_line}")

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("
".join(report).strip() + "
")
        logger.info("Wrote summary to %s", args.out)
    except Exception:
        logger.exception("Failed writing output file: %s", args.out)
        sys.exit(3)

    logger.info("Completed successfully")


if __name__ == "__main__":
    logger.info("Entrypoint reached")
    main()
