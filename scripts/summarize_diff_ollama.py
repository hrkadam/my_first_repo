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

    return "".join(out)
