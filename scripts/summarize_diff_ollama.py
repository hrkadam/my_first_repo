
#!/usr/bin/env python3
"""
Summarize a unified diff using Ollama (remote or local).
- Parses patch with unidiff
- Chunks per file/hunk (skip noise)
- Calls Ollama /api/chat to generate per-file and overall summary
- Writes summary.txt
"""

import os, sys, argparse, textwrap, time
from typing import List, Tuple
import requests
from unidiff import PatchSet

# ADDED: logging imports and logger
import logging  # ADDED
from logging.handlers import RotatingFileHandler  # ADDED

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_API_TOKEN = os.getenv("OLLAMA_API_TOKEN", "")

SKIP_DIRS = {"node_modules", "vendor", "dist", "build", ".venv", ".git"}
MAX_LINES_PER_FILE = 1200      # safety cap per file
MAX_HUNKS_PER_FILE = 50        # safety cap
#TIMEOUT = 120                  # seconds
TIMEOUT_PER_FILE = int(os.getenv("OLLAMA_TIMEOUT_PER_FILE", "120"))
TIMEOUT_OVERALL  = int(os.getenv("OLLAMA_TIMEOUT_OVERALL",  "600"))

# ADDED: simple logger setup — writes to file in script directory
def _setup_logging():  # ADDED
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "summarize_patch.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if _setup_logging is called twice
    if not logger.handlers:
        # Rotate at ~5 MB, keep 3 backups
        handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Optional: also log warnings+ to stderr; comment out if you want file-only
        # console = logging.StreamHandler()
        # console.setLevel(logging.WARNING)
        # console.setFormatter(formatter)
        # logger.addHandler(console)

    return logger  # ADDED

# ADDED: initialize module-level logger
logger = _setup_logging()  # ADDED


def _headers():
    h = {"Content-Type": "application/json"}
    if OLLAMA_API_TOKEN:
        h["Authorization"] = f"Bearer {OLLAMA_API_TOKEN}"
    return h

def trim_patch_text(text: str, max_lines=800):
    """Keep +/- lines and hunk headers; cap length to control tokens."""
    lines = []
    for ln in text.splitlines():
        if ln.startswith(("+", "-", "@@", "diff --git", "index", "---", "+++")):
            lines.append(ln)
        # avoid unchanged context lines
    return "\n".join(lines[:max_lines])

def ollama_chat(messages: List[dict], timeout: int) -> str:  # CHANGED signature arrow fix
    url = f"{OLLAMA_ENDPOINT}/api/chat"
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    logger.info("Calling Ollama: model=%s timeout=%s", OLLAMA_MODEL, timeout)  # ADDED
    try:  # ADDED
        r = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "").strip()
        logger.info("Ollama response length: %d chars", len(content))  # ADDED
        return content
    except Exception:
        logger.exception("Ollama chat request failed")  # ADDED
        raise

def summarize_file(fname: str, minimal_patch: str) -> str:  # CHANGED signature arrow fix
    system = textwrap.dedent("""
    You are a senior code reviewer. Given a code diff for ONE file, produce:
    - Intent & high-level change
    - file name
    - code changes    
    Keep it concise (<= 12 lines). Avoid reprinting the patch.
    """)
    user = f"FILE: {fname}\nPATCH:\n{trim_patch_text(minimal_patch)}"
    logger.info("Summarizing file: %s", fname)  # ADDED
    return ollama_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ], timeout=TIMEOUT_PER_FILE)

def synthesize_overall(repo: str, branch: str, sha: str, per_file_sections: List[Tuple[str, str]]) -> str:  # CHANGED
    system = textwrap.dedent("""
    You are an engineering lead. Given multiple per-file summaries, produce an overall commit summary:
    - Overall intent & scope
    - file name
    - code changes
    Keep it scannable; use bullets.
    """)
    user = f"Repository: {repo}\nBranch: {branch}\nCommit: {sha}\n\n" + \
           "\n\n".join([f"FILE: {f}\nSUMMARY:\n{s}" for f, s in per_file_sections])
    logger.info("Synthesizing overall summary for %d file(s)", len(per_file_sections))  # ADDED
    return ollama_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ], timeout=TIMEOUT_OVERALL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", default="summary.txt")
    args = ap.parse_args()

    logger.info("Starting script with repo=%s branch=%s sha=%s out=%s", args.repo, args.branch, args.sha, args.out)  # ADDED

    # Load patch
    try:  # ADDED
        with open(args.patch, "r", encoding="utf-8", errors="ignore") as f:
            patch = PatchSet(f)
        logger.info("Loaded patch file: %s (files in patch: %d)", args.patch, len(patch))  # ADDED
    except Exception:
        logger.exception("Failed to load/parse patch: %s", args.patch)  # ADDED
        sys.exit(2)

    per_file = []
    for pf in patch:
        fname = pf.path or pf.target_file
        # Skip noise dirs
        if fname:
            parts = fname.split("/")
            if any(d in SKIP_DIRS for d in parts):
                logger.info("Skipping file in ignored directory: %s", fname)  # ADDED
                continue
        # Minimal unified text per file
        text_chunks = []
        hcount = 0
        for h in pf:
            hcount += 1
            if hcount > MAX_HUNKS_PER_FILE:
                logger.warning("Hunk cap reached for %s (%d > %d)", fname, hcount, MAX_HUNKS_PER_FILE)  # ADDED
                break
            hdr = f"@@ -{h.source_start},{h.source_length} +{h.target_start},{h.target_length} @@"
            changes = "\n".join(l.value.rstrip("\n") for l in h)
            text_chunks.append(hdr + "\n" + changes)
        if not text_chunks:
            logger.info("No hunks to summarize for file: %s", fname)  # ADDED
            continue
        minimal = "\n".join(text_chunks)
        minimal = "\n".join(minimal.splitlines()[:MAX_LINES_PER_FILE])
        # Summarize per file via Ollama
        try:
            s = summarize_file(fname, minimal)
        except Exception as e:
            logger.exception("Error summarizing file: %s", fname)  # ADDED
            s = f"(Error summarizing {fname}: {e})"
        per_file.append((fname, s))
        time.sleep(0.2)  # small pacing

    # Overall synthesis
    try:
        overall = synthesize_overall(args.repo, args.branch, args.sha, per_file)
    except Exception as e:
        logger.exception("Error synthesizing overall summary")  # ADDED
        overall = f"(Error synthesizing overall summary: {e})"

    report = []
    report.append(f"Repository: {args.repo}")
    report.append(f"Branch:     {args.branch}")
    report.append(f"Commit:     {args.sha}")
    report.append("")
    report.append("Overall")
    report.append(overall)
    report.append("")
    report.append("Changes")
    for f, s in per_file:
        # Use only the first line of the model output to keep it simple
        first_line = (s or "").splitlines()[0] if s else ""
        report.append(f"- {f}: {first_line}")

    try:  # ADDED
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(report).strip() + "\n")
        logger.info("Wrote summary to %s", args.out)  # ADDED
    except Exception:
        logger.exception("Failed writing output file: %s", args.out)  # ADDED
        sys.exit(3)

    logger.info("Completed successfully")  # ADDED

if __name__ == "__main__":
    # print("Start of script....1")  # REMOVED
    logger.info("Entrypoint reached")  # ADDED
    main()
