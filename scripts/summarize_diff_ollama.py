#!/usr/bin/env python3
"""
Summarize a unified diff using Ollama (local/remote).
- Parses patch with unidiff
- Builds per-file "minimal" diff (only +/- and @@ headers)
- LLM generates per-file summary focused on ACTUAL additions
- LLM synthesizes overall summary
- Writes summary.txt
"""
import os, sys, argparse, textwrap, time
from typing import List, Tuple
import requests
from unidiff import PatchSet

# logging
import logging
from logging.handlers import RotatingFileHandler

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_API_TOKEN = os.getenv("OLLAMA_API_TOKEN", "")
SKIP_DIRS = {"node_modules", "vendor", "dist", "build", ".venv", ".git"}
MAX_LINES_PER_FILE = 1200
MAX_HUNKS_PER_FILE = 50
TIMEOUT_PER_FILE = int(os.getenv("OLLAMA_TIMEOUT_PER_FILE", "120"))
TIMEOUT_OVERALL = int(os.getenv("OLLAMA_TIMEOUT_OVERALL", "600"))


# --- Add near the top of the file (under imports/helpers) ---
# Exclude typical diff metadata that can also start with '+'
ADDED_PREFIXES_EXCLUDE = (
    "+++",      # file header
    "+--",      # uncommon header
    "+@@",      # hunk header rendered with '+'
    "+index",   # index line
    "+diff --git",
    "+ No newline at end of file",  # noise line seen in your log
)

def added_only(minimal_patch: str, max_lines: int = 2000) -> str:
    """
    Return only '+' code lines from the minimal patch, excluding headers and noise.
    Leading '+' is removed; lines are kept as-is otherwise.
    """
    out = []
    for ln in minimal_patch.splitlines():
        if not ln.startswith("+"):
            continue
        # exclude metadata/headers/noise that also start with '+'
        if ln.startswith(ADDED_PREFIXES_EXCLUDE):
            continue
        code = ln[1:].rstrip()
        if code:
            out.append(code)
        if len(out) >= max_lines:
            break
    return "\n".join(out)


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

def trim_patch_text(text: str, max_lines=800):
    """Keep +/- lines and hunk headers; cap length."""
    lines = []
    for ln in text.splitlines():
        if ln.startswith(("+", "-", "@@", "diff --git", "index", "---", "+++")):
            lines.append(ln)
    return "\n".join(lines[:max_lines])

def ollama_chat(messages: List[dict], timeout: int) -> str:
    """
    Call Ollama /api/chat with low-temperature settings for deterministic output.
    """
    url = f"{OLLAMA_ENDPOINT}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        # Make decoding stricter/deterministic
        "options": {
            "temperature": 0.0,
            "num_ctx": 12288,        # larger context for bigger diffs (adjust if needed)
            "top_p": 0.9,
            "top_k": 40,
        },
    }
    logger.info("Calling Ollama: model=%s timeout=%s", OLLAMA_MODEL, timeout)
    try:
        r = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        content = data.get("message", {}).get("content", "").strip()
        logger.info("Ollama response length: %d chars", len(content))
        return content
    except Exception:
        logger.exception("Ollama chat request failed")
        raise

def _file_language_hint(fname: str) -> str:
    ext = os.path.splitext(fname)[1].lower()
    if ext == ".py":
        return "python"
    # elif ext in (".js", ".jsx", ".ts", ".tsx"):
    #     return "javascript/typescript"
    # elif ext in (".java",):
    #     return "java"
    # elif ext in (".go",):
    #     return "go"
    # elif ext in (".cs",):
    #     return "csharp"
    else:
        return "generic"



# --- Replace your summarize_file(...) with this version ---
def summarize_file(fname: str, minimal_patch: str) -> str:
    """
    Prompt the LLM to produce an additions-focused summary per file.
    """
    lang = _file_language_hint(fname)
    system = textwrap.dedent(f"""
    You are a senior code reviewer. Summarize ONLY the actual additions made in this file.

    STRICT RULES:
    - Use the ADDED_ONLY block to locate newly ADDED functions/classes/methods.
    - List each new function/class by exact name/signature. For each, add a 1–2 line overview
      (prefer adjacent docstrings/comments if present).
    - If ADDED_ONLY is empty or has no recognizable code, return exactly: "No code additions detected."
    - Do NOT invent content or infer behavior that is not visible in ADDED_ONLY.
    - Keep it ≤ 10 lines, scannable bullets.
    """).strip()

    language_tip = {
        "python": "Look for: def name(params): and class Name:",
        "javascript/typescript": "Look for: function name(...), const name = (...) =>, class Name",
        "java": "Look for: returnType name(...), class Name",
        "go": "Look for: func name(...), type Name struct",
        "csharp": "Look for: returnType Name(...), class Name",
        "generic": "If nothing recognizable exists, return the 'No code additions detected.' message.",
    }[lang]

    # Build views
    added_view = added_only(minimal_patch)
    trimmed_view = trim_patch_text(minimal_patch)

    user = (
        f"FILE: {fname}\n"
        f"LANGUAGE_HINT: {lang} — {language_tip}\n\n"
        "ADDED_ONLY:\n"
        f"{added_view}\n\n"
        "TRIMMED_DIFF (context only):\n"
        f"{trimmed_view}"
    )

    logger.info("Summarizing file: %s (added-only len=%d)", fname, len(added_view))
    return ollama_chat(
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        timeout=TIMEOUT_PER_FILE
    )

# def summarize_file(fname: str, minimal_patch: str) -> str:
#     """
#     Prompt the LLM to produce an additions-focused summary per file.
#     """
#     lang = _file_language_hint(fname)
#     system = textwrap.dedent(f"""
#     You are a senior code reviewer. Given a unified diff for ONE file, produce a concise,
#     fact-based summary focusing ONLY on ACTUAL additions (lines starting with '+').
#     Rules:
#     - Identify newly ADDED functions/classes by exact name/signature if present in '+' lines.
#     - Prefer overview from adjacent docstrings/comments; summarize in 1–2 lines per item.
#     - If there are no true additions, say: "No code additions detected."
#     - Do NOT invent names, behavior, tickets, or intent not present in the diff.
#     - Ignore unchanged context, metadata (diff headers), and deletions unless they explain the additions.
#     - Keep it ≤ 10 lines, scannable bullets.
#     """).strip()

#     # Nudge by language
#     language_tip = {
#         "python": "Focus on `def name(params):` and `class Name:` lines in '+' additions.",
#         "javascript/typescript": "Focus on `function name(...)` / `const name = (...) =>` / `class Name` in '+' additions.",
#         "java": "Focus on method signatures `returnType name(...)` and `class Name` in '+' additions.",
#         "go": "Focus on `func name(...)` and `type Name struct` in '+' additions.",
#         "csharp": "Focus on `type name(...)` methods and `class Name` in '+' additions.",
#         "generic": "Report additions at a high level; if no recognizable functions/classes, give counts.",
#     }[lang]

#     user = (
#         f"FILE: {fname}\n"
#         f"LANGUAGE_HINT: {lang} — {language_tip}\n"
#         "PATCH (trimmed to +/- and @@ lines only):\n"
#         f"{trim_patch_text(minimal_patch)}"
#     )

#     logger.info("Summarizing file: %s", fname)
#     return ollama_chat(
#         [
#             {"role": "system", "content": system},
#             {"role": "user",   "content": user},
#         ],
#         timeout=TIMEOUT_PER_FILE
#     )

def synthesize_overall(repo: str, branch: str, sha: str, per_file_sections: List[Tuple[str, str]]) -> str:
    system = textwrap.dedent("""
    You are an engineering lead. Given multiple per-file summaries produced from actual diffs,
    synthesize an overall commit summary that aggregates the facts WITHOUT adding new assumptions.
    - Start with one line on overall intent if clearly implied by the file summaries.
    - Then bullets: file name → key additions (function/class names) and brief overviews.
    - Keep it scannable, ≤ 12 lines. No extra speculation.
    """).strip()

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
        timeout=TIMEOUT_OVERALL
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", default="summary.txt")
    args = ap.parse_args()

    logger.info("Starting script with repo=%s branch=%s sha=%s out=%s", args.repo, args.branch, args.sha, args.out)

    # Load patch
    try:
        with open(args.patch, "r", encoding="utf-8", errors="ignore") as f:
            patch = PatchSet(f)
        logger.info("Loaded patch file: %s (files in patch: %d)", args.patch, len(patch))
    except Exception:
        logger.exception("Failed to load/parse patch: %s", args.patch)
        sys.exit(2)

    per_file = []
    for pf in patch:
        logger.info("Processing file patch: %s", pf)  # ADDED
        fname = pf.path or pf.target_file
        # Skip noise dirs
        if fname:
            parts = fname.split("/")
            if any(d in SKIP_DIRS for d in parts):
                logger.info("Skipping file in ignored directory: %s", fname)
                continue

        # Build minimal unified text per file
        text_chunks = []
        hcount = 0
        for h in pf:
            logger.info("Processing hunk: %s", h)  # ADDED
            hcount += 1
            if hcount > MAX_HUNKS_PER_FILE:
                logger.warning("Hunk cap reached for %s (%d > %d)", fname, hcount, MAX_HUNKS_PER_FILE)
                break
            hdr = f"@@ -{h.source_start},{h.source_length} +{h.target_start},{h.target_length} @@"
            changes = "\n".join(l.value.rstrip("\n") for l in h)
            logger.info("Hunk changes length: %d chars", len(changes))  # ADDED
            text_chunks.append(hdr + "\n" + changes)
            logger.info("Added hunk to text_chunks for file: %s", fname)  # ADDED
        if not text_chunks:
            logger.info("No hunks to summarize for file: %s", fname)
            continue

        minimal = "\n".join(text_chunks)
        logger.info("Total minimal patch length for %s: %d chars", fname, len(minimal))  # ADDED
        minimal = "\n".join(minimal.splitlines()[:MAX_LINES_PER_FILE])
        logger.info("Trimmed minimal patch length for %s: %d chars", fname, len(minimal))  # ADDED
        # Summarize per file via Ollama
        try:
            s = summarize_file(fname or "<unknown>", minimal)
            # Use first line for the 'Changes' quick list
            first_line = (s or "").splitlines()[0] if s else ""
            logger.info("Summary for file %s: %s", fname, first_line.replace("\n", " ")[:150])
        except Exception as e:
            logger.exception("Error summarizing file: %s", fname)
            s = f"(Error summarizing {fname}: {e})"

        per_file.append((fname or "<unknown>", s))
        time.sleep(0.2)  # small pacing

    # Overall synthesis
    try:
        overall = synthesize_overall(args.repo, args.branch, args.sha, per_file)
        logger.info("Overall summary (head): %s", (overall[:200] or "").replace("\n", " "))
    except Exception as e:
        logger.exception("Error synthesizing overall summary")
        overall = f"(Error synthesizing overall summary: {e})"

    # Assemble final report
    report = []
    report.append(f"Repository: {args.repo}")
    report.append(f"Branch: {args.branch}")
    report.append(f"Commit: {args.sha}")
    report.append("")
    report.append("Overall")
    report.append(overall or "")
    report.append("")
    report.append("Changes")
    for f, s in per_file:
        # Use only the first line of the model output to keep it simple
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
