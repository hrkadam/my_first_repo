
#!/usr/bin/env python3
"""
Summarize a unified diff using Ollama (remote or local), with comprehensive logging and audit artifacts.

What this script does:
- Reads a unified diff patch file (robust to UTF-8/UTF-16/BOM encodings)
- Parses with unidiff to enumerate changed files and hunks
- Creates per-file minimal patch snapshots under .changes/patches/<sha>/
- Calls Ollama /api/chat to generate per-file summaries and an overall summary
- Emits a human-readable summary.txt
- Emits a structured audit JSON (.changes/audit_<sha>.json) capturing exactly what changed,
  how it was processed, and the generated summaries (good for progress tracking & debugging)

Environment variables (optional):
- OLLAMA_ENDPOINT (default: http://localhost:11434)
- OLLAMA_MODEL    (default: qwen2.5-coder:7b)
- OLLAMA_API_TOKEN (optional: Bearer token if your endpoint enforces auth)
- OLLAMA_TIMEOUT_PER_FILE (default: 120 seconds)
- OLLAMA_TIMEOUT_OVERALL  (default: 600 seconds)
- SUMMARY_DIR (default: .changes)  # directory to store audit & patch snapshots
- LOG_LEVEL (default: INFO)        # INFO|DEBUG|WARNING|ERROR

CLI arguments:
--patch  : path to diff.patch
--repo   : repository "owner/name"
--branch : branch name
--sha    : commit sha
--out    : summary output file path (default: summary.txt)
"""

import os
import sys
import json
import time
import argparse
import textwrap
import logging
from typing import List, Tuple
from datetime import datetime
from logging.handlers import RotatingFileHandler

import requests
from unidiff import PatchSet

# ---------------- Configuration ----------------
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_API_TOKEN = os.getenv("OLLAMA_API_TOKEN", "")

SKIP_DIRS = {"node_modules", "vendor", "dist", "build", ".venv", ".git"}
MAX_LINES_PER_FILE = 1200
MAX_HUNKS_PER_FILE = 50

TIMEOUT_PER_FILE = int(os.getenv("OLLAMA_TIMEOUT_PER_FILE", "120"))
TIMEOUT_OVERALL = int(os.getenv("OLLAMA_TIMEOUT_OVERALL", "600"))
SUMMARY_DIR = os.getenv("SUMMARY_DIR", ".changes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# ---------------- Logging Setup ----------------
def _setup_logging() -> logging.Logger:
    """Create a rotating file logger and a console logger (warnings+)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "summarize_patch.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if not logger.handlers:
        handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


logger = _setup_logging()


# ---------------- Utility Functions ----------------
def ensure_dirs(sha: str) -> Tuple[str, str]:
    """Ensure base .changes directory and per-sha patch directory exist."""
    base = os.path.abspath(SUMMARY_DIR)
    patches_dir = os.path.join(base, "patches", sha)
    os.makedirs(base, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)
    return base, patches_dir


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if OLLAMA_API_TOKEN:
        h["Authorization"] = f"Bearer {OLLAMA_API_TOKEN}"
    return h


def trim_patch_text(text: str, max_lines: int = 800) -> str:
    """
    Keep only +/- lines and hunk headers; drop context lines to control token usage.
    """
    lines = []
    for ln in text.splitlines():
        if ln.startswith(("+", "-", "@@", "diff --git", "index", "---", "+++")):
            lines.append(ln)
    return "\n".join(lines[:max_lines])


def ollama_chat(messages: List[dict], timeout: int) -> str:
    """Call Ollama /api/chat with provided messages and return content string."""
    url = f"{OLLAMA_ENDPOINT}/api/chat"
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    logger.info("Calling Ollama: model=%s timeout=%s", OLLAMA_MODEL, timeout)
    try:
        r = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        content = (data.get("message", {}) or {}).get("content", "").strip()
        logger.info("Ollama response length: %d chars", len(content))
        return content
    except Exception:
        logger.exception("Ollama chat request failed")
        raise


def summarize_file(fname: str, minimal_patch: str) -> str:
    """Ask the model to produce a concise summary for ONE file's patch."""
    system = textwrap.dedent("""
        You are a senior code reviewer. Given a code diff for ONE file, produce:
        - Intent & high-level change
        - file name
        - code changes
        Keep it concise (<= 12 lines). Avoid reprinting the patch verbatim.
    """).strip()
    user = f"FILE: {fname}\nPATCH:\n{trim_patch_text(minimal_patch)}"
    logger.info("Summarizing file: %s", fname)
    return ollama_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        timeout=TIMEOUT_PER_FILE,
    )


def synthesize_overall(repo: str, branch: str, sha: str,
                       per_file_sections: List[Tuple[str, str]]) -> str:
    """Ask the model to produce an overall summary across all per-file summaries."""
    system = textwrap.dedent("""
        You are an engineering lead. Given multiple per-file summaries, produce an overall commit summary:
        - Overall intent & scope
        - file name
        - code changes
        Keep it scannable; use concise bullets.
    """).strip()
    user = f"Repository: {repo}\nBranch: {branch}\nCommit: {sha}\n\n" + \
           "\n\n".join([f"FILE: {f}\nSUMMARY:\n{s}" for f, s in per_file_sections])
    logger.info("Synthesizing overall summary for %d file(s)", len(per_file_sections))
    return ollama_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        timeout=TIMEOUT_OVERALL,
    )


def read_patch_utf_best_effort(patch_path: str) -> PatchSet:
    """
    Read a patch robustly across UTF encodings. Falls back to binary read + fromstring.
    """
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            with open(patch_path, "r", encoding=enc, errors="strict") as f:
                return PatchSet(f)
        except Exception:
            continue
    with open(patch_path, "rb") as fb:
        txt = fb.read().decode("utf-8", errors="ignore")
    return PatchSet.fromstring(txt)


def pick_filename(pf) -> str:
    """
    Heuristic to pick a display filename; prefer normalized path, else target/source.
    Avoid /dev/null for added/deleted files.
    """
    for cand in (getattr(pf, "path", None),
                 getattr(pf, "target_file", None),
                 getattr(pf, "source_file", None)):
        if cand and cand != "/dev/null":
            return cand
    return "(unknown)"


def _bool_attr(pf, name: str) -> bool:
    """Safely handle unidiff attrs that might be properties or methods."""
    attr = getattr(pf, name, None)
    if callable(attr):
        try:
            return bool(attr())
        except Exception:
            return False
    return bool(attr)


def classify_patch_file(pf) -> str:
    """Classify patch file change type for audit."""
    if _bool_attr(pf, "is_added_file"):
        return "added"
    if _bool_attr(pf, "is_removed_file"):
        return "deleted"
    if _bool_attr(pf, "is_rename"):
        return "renamed"
    if _bool_attr(pf, "is_modified_file"):
        return "modified"
    return "unknown"


def snapshot_patch(patches_dir: str, sha: str, fname: str, minimal_patch: str) -> None:
    """Write a per-file minimal patch snapshot under .changes/patches/<sha>/<path>.patch"""
    safe_rel = fname.replace("\\", "/")
    out_path = os.path.join(patches_dir, safe_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path + ".patch", "w", encoding="utf-8") as f:
        f.write(minimal_patch)
    logger.info("Wrote patch snapshot: %s", out_path + ".patch")


def start_audit(repo: str, branch: str, sha: str) -> dict:
    """Initialize an audit structure."""
    return {
        "repo": repo,
        "branch": branch,
        "sha": sha,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "ollama": {
            "endpoint": OLLAMA_ENDPOINT,
            "model": OLLAMA_MODEL,
            "timeout_per_file_sec": TIMEOUT_PER_FILE,
            "timeout_overall_sec": TIMEOUT_OVERALL,
        },
        "files": [],
        "overall": {
            "status": "pending",
            "summary": "",
            "error": "",
        },
        "ended_at": None,
        "status": "pending",
    }


def append_file_audit(audit: dict, fname: str, pf_class: str, hunk_count: int,
                      minimal_patch: str, summary: str, error: str) -> None:
    """Append a per-file audit record."""
    audit["files"].append({
        "file": fname,
        "change_type": pf_class,
        "hunks": hunk_count,
        "summary_first_line": (summary or "").splitlines()[0] if summary else "",
        "summary_full": summary or "",
        "error": error or "",
        "patch_excerpt_lines": minimal_patch.splitlines()[:40],  # quick view sample
    })


def write_audit(base_dir: str, sha: str, audit: dict) -> None:
    """Persist the audit JSON under .changes/audit_<sha>.json."""
    audit["ended_at"] = datetime.utcnow().isoformat() + "Z"
    out_path = os.path.join(base_dir, f"audit_{sha}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    logger.info("Wrote audit JSON: %s", out_path)


# ---------------- Main Script ----------------
def main():
    ap = argparse.ArgumentParser(description="Summarize a diff patch via Ollama with audit logging.")
    ap.add_argument("--patch", required=True, help="Path to diff.patch (UTF-8 recommended)")
    ap.add_argument("--repo", required=True, help="Repository owner/name")
    ap.add_argument("--branch", required=True, help="Branch name")
    ap.add_argument("--sha", required=True, help="Commit SHA")
    ap.add_argument("--out", default="summary.txt", help="Output summary file path")
    args = ap.parse_args()

    logger.info("Starting script repo=%s branch=%s sha=%s out=%s",
                args.repo, args.branch, args.sha, args.out)

    base_dir, patches_dir = ensure_dirs(args.sha)
    audit = start_audit(args.repo, args.branch, args.sha)

    # Load patch with best-effort encoding
    try:
        patch = read_patch_utf_best_effort(args.patch)
        logger.info("Loaded patch file: %s (files in patch: %d)", args.patch, len(patch))
    except Exception:
        logger.exception("Failed to load/parse patch: %s", args.patch)
        audit["status"] = "error"
        write_audit(base_dir, args.sha, audit)
        sys.exit(2)

    per_file: List[Tuple[str, str]] = []

    for pf in patch:
        fname = pick_filename(pf)
        pf_class = classify_patch_file(pf)

        # Skip ignored directories early
        if fname:
            parts = fname.split("/")
            if any(d in SKIP_DIRS for d in parts):
                logger.info("Skipping file in ignored directory: %s", fname)
                append_file_audit(audit, fname, pf_class, 0, "", "", "skipped: ignored directory")
                continue

        # Reconstruct a minimal unified text per file (header + changes from hunks)
        text_chunks: List[str] = []
        hcount = 0

        for h in pf:
            hcount += 1
            if hcount > MAX_HUNKS_PER_FILE:
                logger.warning("Hunk cap reached for %s (%d > %d)", fname, hcount, MAX_HUNKS_PER_FILE)
                break
            hdr = f"@@ -{h.source_start},{h.source_length} +{h.target_start},{h.target_length} @@"
            changes = "\n".join(l.value.rstrip("\n") for l in h)
            text_chunks.append(hdr + "\n" + changes)

        minimal = "\n".join(text_chunks) if text_chunks else ""
        minimal = "\n".join(minimal.splitlines()[:MAX_LINES_PER_FILE])

        # Snapshot patch (ground truth) if we have textual hunks
        if minimal:
            try:
                snapshot_patch(patches_dir, args.sha, fname, minimal)
            except Exception:
                logger.exception("Failed to write patch snapshot for %s", fname)

        # Produce a per-file summary (or meaningful message for non-textual changes)
        summary_out = ""
        error_msg = ""

        try:
            if text_chunks:
                summary_out = summarize_file(fname, minimal)
            else:
                # No textual hunks -> annotate appropriately
                if _bool_attr(pf, "is_rename"):
                    summary_out = "Rename only (no content changes)."
                elif _bool_attr(pf, "is_removed_file"):
                    summary_out = "File deleted (no content hunks)."
                elif _bool_attr(pf, "is_added_file"):
                    summary_out = "File added (no textual hunks; possibly binary or empty)."
                else:
                    summary_out = "No textual hunks (possibly binary or metadata/perms change)."
        except Exception as e:
            logger.exception("Error summarizing file: %s", fname)
            error_msg = str(e)
            summary_out = f"(Error summarizing {fname}: {e})"

        append_file_audit(audit, fname, pf_class, hcount, minimal, summary_out, error_msg)
        per_file.append((fname, summary_out))
        time.sleep(0.2)  # small pacing to avoid hammering the model

    # Overall synthesis across files
    try:
        overall = synthesize_overall(args.repo, args.branch, args.sha, per_file)
        audit["overall"]["status"] = "ok"
        audit["overall"]["summary"] = overall
    except Exception as e:
        logger.exception("Error synthesizing overall summary")
        overall = f"(Error synthesizing overall summary: {e})"
        audit["overall"]["status"] = "error"
        audit["overall"]["error"] = str(e)

    # Human-readable summary.txt
    report_lines: List[str] = []
    report_lines.append(f"Repository: {args.repo}")
    report_lines.append(f"Branch:     {args.branch}")
    report_lines.append(f"Commit:     {args.sha}")
    report_lines.append("")
    report_lines.append("Overall")
    report_lines.append(overall)
    report_lines.append("")
    report_lines.append("Changes")
    for f, s in per_file:
        first_line = (s or "").splitlines()[0] if s else ""
        report_lines.append(f"- {f}: {first_line}")

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines).strip() + "\n")
        logger.info("Wrote summary to %s", args.out)
    except Exception:
        logger.exception("Failed writing output file: %s", args.out)
        audit["status"] = "error"
        write_audit(base_dir, args.sha, audit)
        sys.exit(3)

    audit["status"] = "ok"
    write_audit(base_dir, args.sha, audit)
    logger.info("Completed successfully")


if __name__ == "__main__":
    logger.info("Entrypoint reached")
    main()
