
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import re
import time
import unicodedata
from pathlib import Path

import difflib
import requests
from unidiff import PatchSet, UnidiffParseError

# ------------------------------
# OLLAMA SETTINGS
# ------------------------------
OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5-coder:7b"

# ------------------------------
# LOGGING SETUP
# ------------------------------
def setup_logging(log_path: Path, level: str = "INFO") -> None:
    """
    Configure logging with a rotating file handler + console warnings.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lvl = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(lvl)

    # Avoid duplicate handlers on re-run
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(lvl)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.WARNING)  # only warnings+ to console by default
    logger.addHandler(ch)

    logging.info("Logging configured. Level=%s, file=%s", level.upper(), str(log_path))


# ------------------------------
# LLM CALL
# ------------------------------
def ollama_chat(messages, timeout=120):
    url = f"{OLLAMA_ENDPOINT}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 1,
        "num_ctx": 8192,
        "stream": False,
    }
    logging.info("Calling Ollama: model=%s timeout=%s", OLLAMA_MODEL, timeout)
    t0 = time.perf_counter()
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - t0
        logging.info("Ollama status=%s elapsed=%.2fs", r.status_code, elapsed)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "") or ""
        logging.debug("Ollama response length: %d chars", len(content))
        return content.strip()
    except Exception as e:
        logging.exception("Ollama chat request failed: %s", e)
        raise


# ------------------------------
# DIFF SANITIZER (for copy/paste artifacts)
# ------------------------------
def sanitize_unified_diff_bytes(data_bytes: bytes) -> bytes:
    """
    Repair common artifacts in a unified diff:
    - Normalize Unicode
    - Convert en/em-dash/ fullwidth plus at line starts to ASCII
    - Ensure hunk body lines begin with ' ', '+', or '-'
    """
    text = data_bytes.decode("utf-8", errors="replace")
    text = unicodedata.normalize("NFC", text)

    fixed_lines = []
    in_hunk = False

    for ln in text.splitlines():
        if ln.startswith("@@ ") or ln.startswith("@@-") or ln.startswith("@@"):
            fixed_lines.append(ln)
            in_hunk = True
            continue

        if ln.startswith("diff --git "):
            in_hunk = False
            fixed_lines.append(ln)
            continue

        if ln.startswith(("--- ", "+++ ", "index ", "new file mode ", "deleted file mode ",
                          "similarity index ", "rename from ", "rename to ", "Binary files ")):
            in_hunk = False
            fixed_lines.append(ln)
            continue

        if in_hunk:
            if ln:
                first = ln[0]
                if first in ("–", "—"):  # en/em dash
                    ln = "-" + ln[1:]
                elif first in ("＋",):    # fullwidth plus
                    ln = "+" + ln[1:]
                elif first == "\t":      # tab → treat as context
                    ln = " " + ln[1:]
                if ln[0] not in (" ", "+", "-"):
                    ln = " " + ln  # make it context
            fixed_lines.append(ln)
        else:
            fixed_lines.append(ln)

    sanitized = "\n".join(fixed_lines)
    if not sanitized.endswith("\n"):
        sanitized += "\n"
    logging.debug("Sanitized diff size: %d bytes", len(sanitized.encode("utf-8")))
    return sanitized.encode("utf-8")


# ------------------------------
# DIFF UTILITIES
# ------------------------------
def pair_modified(removals, additions, sim_threshold=0.6):
    """
    Greedy pairing of removed and added lines using similarity.
    Returns:
      modified: list of (old, new)
      remaining_removed: list[str]
      remaining_added: list[str]
    """
    modified = []
    used_add_idx = set()

    for old in removals:
        best_idx = None
        best_score = 0.0
        for j, new in enumerate(additions):
            if j in used_add_idx:
                continue
            s = difflib.SequenceMatcher(a=old, b=new).ratio()
            if s > best_score:
                best_score = s
                best_idx = j
        if best_idx is not None and best_score >= sim_threshold:
            modified.append((old, additions[best_idx]))
            used_add_idx.add(best_idx)

    remaining_removed = [old for old in removals if all(old != o for (o, _) in modified)]
    remaining_added = [new for j, new in enumerate(additions) if j not in used_add_idx]

    logging.debug(
        "pair_modified: pairs=%d rem_removed=%d rem_added=%d",
        len(modified), len(remaining_removed), len(remaining_added)
    )
    return modified, remaining_removed, remaining_added


def extract_changes_from_hunk(hunk, sample_added=5, sample_removed=5):
    removals = []
    additions = []
    for line in hunk:
        if line.is_removed:
            removals.append(line.value.rstrip("\n"))
        elif line.is_added:
            additions.append(line.value.rstrip("\n"))
        else:
            pass  # skip context

    removed_total = len(removals)
    added_total = len(additions)

    logging.debug(
        "Hunk %s->%s: +%d/-%d",
        f"{hunk.source_start}:{hunk.source_length}",
        f"{hunk.target_start}:{hunk.target_length}",
        added_total, removed_total
    )

    return {
        "old_range": f"{hunk.source_start}:{hunk.source_length}",
        "new_range": f"{hunk.target_start}:{hunk.target_length}",
        "context_example": (hunk.section_header or "").strip() or None,
        "removed": removals[:sample_removed],
        "added": additions[:sample_added],
        "removed_total": removed_total,
        "added_total": added_total
    }


# ------------------------------
# LANGUAGE / FILE HEURISTICS
# ------------------------------
def detect_language(path):
    ext = Path(path).suffix.lower()
    return {
        '.py': 'python', '.java': 'java', '.ts': 'typescript', '.js': 'javascript',
        '.tsx': 'typescript', '.jsx': 'javascript', '.go': 'go', '.rb': 'ruby',
        '.php': 'php', '.cs': 'csharp', '.md': 'markdown', '.rst': 'rst',
        '.yaml': 'yaml', '.yml': 'yaml', '.json': 'json', '.sh': 'shell',
        '.dockerfile': 'dockerfile'
    }.get(ext, 'text')


def is_test_file(path):
    return bool(
        re.search(r'(^|/)(tests?|spec|__tests__)/', path, re.I) or
        re.search(r'\.(test|spec)\.(js|ts|py|java)$', path, re.I)
    )


def is_config_file(path):
    return bool(
        re.search(r'\.(ya?ml|json|toml|ini|cfg|conf)$', path, re.I) or
        re.search(r'(Dockerfile|package\.json|pyproject\.toml)$', path)
    )


def mine_identifiers(lines, language):
    ids = []
    if language == 'python':
        for ln in lines:
            m = re.search(r'^\s*(def|class)\s+([A-Za-z_]\w*)', ln)
            if m: ids.append(m.group(2))
    elif language == 'java':
        for ln in lines:
            m = re.search(r'\b(class|interface)\s+([A-Za-z_]\w*)', ln)
            if m: ids.append(m.group(2))
            m2 = re.search(r'(public|protected|private)\s+[\w<>\[\]]+\s+([A-Za-z_]\w*)\s*\(', ln)
            if m2: ids.append(m2.group(2))
    elif language in ('javascript', 'typescript'):
        for ln in lines:
            m = re.search(r'\bclass\s+([A-Za-z_]\w*)', ln)
            if m: ids.append(m.group(1))
            m2 = re.search(r'\bfunction\s+([A-Za-z_]\w*)\s*\(', ln)
            if m2: ids.append(m2.group(1))
            m3 = re.search(r'\bconst\s+([A-Za-z_]\w*)\s*=\s*\(', ln)
            if m3: ids.append(m3.group(1))
    elif language == 'go':
        for ln in lines:
            m = re.search(r'^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\(', ln)
            if m: ids.append(m.group(1))
            m2 = re.search(r'^\s*type\s+([A-Za-z_]\w*)\s+struct', ln)
            if m2: ids.append(m2.group(1))
    elif language == 'markdown':
        for ln in lines:
            m = re.search(r'^\s*#+\s+(.*)$', ln)
            if m: ids.append(m.group(1).strip())
    return ids


def change_type_for_file(pf):
    if getattr(pf, "is_binary_file", False): return "binary"
    if getattr(pf, "is_added_file", False):   return "added"
    if getattr(pf, "is_removed_file", False): return "deleted"

    src = (getattr(pf, "source_file", "") or "").replace("a/", "")
    tgt = (getattr(pf, "target_file", "") or "").replace("b/", "")
    if src and tgt and src != tgt:
        return "renamed"
    return "modified"


def detect_mode_change(pf):
    info = []
    hdr = getattr(pf, "patch_info", None)
    if hdr:
        for ln in hdr:
            if ln.strip().startswith("old mode ") or ln.strip().startswith("new mode "):
                info.append(ln.strip())
    return info or None


# ------------------------------
# FILE-LEVEL STRUCTURING
# ------------------------------
def build_file_struct(pf, sample_added=5, sample_removed=5):
    old_path = getattr(pf, "source_file", pf.path) or pf.path
    new_path = getattr(pf, "target_file", pf.path) or pf.path

    language = detect_language(new_path)
    ctype = change_type_for_file(pf)
    mode_info = detect_mode_change(pf)

    hunks = []
    total_added = 0
    total_removed = 0
    all_added_lines = []
    all_removed_lines = []

    for h in pf:
        hmeta = extract_changes_from_hunk(h, sample_added=sample_added, sample_removed=sample_removed)
        hunks.append(hmeta)
        total_added += hmeta["added_total"]
        total_removed += hmeta["removed_total"]
        all_added_lines.extend(hmeta["added"])
        all_removed_lines.extend(hmeta["removed"])

    modified_pairs, remaining_removed, remaining_added = pair_modified(all_removed_lines, all_added_lines, sim_threshold=0.6)

    identifiers_added = mine_identifiers(all_added_lines, language)
    identifiers_removed = mine_identifiers(all_removed_lines, language)

    logging.info(
        "File %s: type=%s +%d/-%d ids_added=%s ids_removed=%s",
        new_path.replace("b/", ""),
        ctype,
        total_added,
        total_removed,
        identifiers_added,
        identifiers_removed,
    )

    return {
        "old_path": old_path.replace("a/", ""),
        "new_path": new_path.replace("b/", ""),
        "language": language,
        "change_type": ctype,
        "added_count": total_added,
        "removed_count": total_removed,
        "modified_estimate": min(total_added, total_removed) if ctype != "binary" else 0,
        "is_test_file": is_test_file(new_path),
        "is_config_file": is_config_file(new_path),
        "mode_change_info": mode_info,
        "hunks": hunks,
        "identifiers_added": identifiers_added,
        "identifiers_removed": identifiers_removed,
        "modified_pairs_sample": [{"old": o, "new": n} for (o, n) in modified_pairs[:5]],
        "added_sample": remaining_added[:5],
        "removed_sample": remaining_removed[:5]
    }


# ------------------------------
# PATCH-LEVEL STRUCTURING
# ------------------------------
def build_structured_patch(patch):
    files = []
    totals = {
        "files_changed": 0,
        "total_added": 0,
        "total_removed": 0,
        "renames": 0,
        "binary_changes": 0
    }

    for pf in patch:
        fobj = build_file_struct(pf, sample_added=5, sample_removed=5)
        files.append(fobj)
        totals["files_changed"] += 1
        totals["total_added"] += fobj["added_count"]
        totals["total_removed"] += fobj["removed_count"]
        if fobj["change_type"] == "renamed":
            totals["renames"] += 1
        if fobj["change_type"] == "binary":
            totals["binary_changes"] += 1

    logging.info(
        "Patch roll-up: files_changed=%d +%d/-%d renames=%d binary=%d",
        totals["files_changed"], totals["total_added"], totals["total_removed"],
        totals["renames"], totals["binary_changes"]
    )

    return {
        "patch_summary": totals,
        "files": files,
        "notes": []
    }


# ------------------------------
# LLM SUMMARIZATION
# ------------------------------
def summarize_patch_structured(structured_json):
    system = (
        "You are a precise code change summarizer. "
        "Only summarize what the structured diff JSON contains. "
        "Do not infer or speculate beyond provided fields. "
        "Highlight renames, binary changes, and mode changes if present. "
        "Call out identifiers added/removed/modified when available. "
        "Group tests/config changes separately."
    )
    user = (
        "Given this structured diff JSON, produce:\n"
        "1) A short roll‑up (files_changed, total_added/removed, renames, binary_changes).\n"
        "2) Per‑file bullets: change_type, counts, identifiers impacted, and a one‑line intent.\n"
        "3) Separate note for tests/config files.\n"
        "Keep it under ~8 sentences total; strictly factual.\n\n"
        f"{json.dumps(structured_json, indent=2)}"
    )
    logging.debug("Sending structured JSON to LLM (size=%d chars)", len(json.dumps(structured_json)))
    return ollama_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ], timeout=120)


def summarize_file_changes_structured(changes):
    system = (
        "You are a senior code reviewer.\n"
        "Summarize ONLY the changes provided. Do NOT invent changes.\n"
        "Treat paired old/new lines as MODIFIED. Unpaired '+' are ADDED, unpaired '-' are REMOVED.\n"
        "Output format:\n"
        "File: <filename>\n"
        "- Purpose: <one line>\n"
        "- Changes:\n"
        "  • Modified: old → new\n"
        "  • Added: ...\n"
        "  • Removed: ...\n"
        "Keep it concise and accurate."
    )
    user = (
        "Here are structured changes extracted from a diff. "
        "Summarize them as per the format.\n\n"
        f"{json.dumps(changes, indent=2)}"
    )
    logging.debug("Sending per-file changes to LLM for %s", changes.get("file", "unknown"))
    return ollama_chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ], timeout=120)


# ------------------------------
# MAIN
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Summarize a unified git diff patch with LLM (with logging).")
    parser.add_argument("--patch", type=str, required=True, help="Path to patch file (unified Git diff)")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to save summary file")
    parser.add_argument("--log", type=str, default="logs/summarize_patch.log", help="Path to log file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Log level")
    parser.add_argument("--save-json", action="store_true", help="Also save structured JSON next to summary")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite summary (use hash-only filename) instead of unique timestamped name")
    args = parser.parse_args()

    setup_logging(Path(args.log), args.log_level)

    patch_path = Path(args.patch)
    if not patch_path.exists():
        logging.error("Patch file not found: %s", str(patch_path))
        print(f"❌ Patch file not found: {patch_path}")
        return

    logging.info("Reading patch file: %s", str(patch_path))
    try:
        raw = patch_path.read_bytes()
    except Exception as e:
        logging.exception("Failed to read patch: %s", e)
        print(f"❌ Failed to read patch: {e}")
        return

    # Sanitize and parse
    sanitized = sanitize_unified_diff_bytes(raw)
    try:
        patch = PatchSet(iter(sanitized.splitlines(keepends=True)), encoding="utf-8")
    except UnidiffParseError as e:
        logging.exception("Unidiff parse error: %s", e)
        print(f"❌ Unidiff parse error: {e}")
        return
    except Exception as e:
        logging.exception("Unexpected parse error: %s", e)
        print(f"❌ Unexpected parse error: {e}")
        return

    logging.info("Patch parsed. Files changed: %d", len(patch))

    # Build structured JSON and (optionally) save it
    structured = build_structured_patch(patch)
    if args.save_json:
        json_path = (Path(args.outdir) / patch_path.with_suffix(".structured.json").name)
        try:
            json_path.write_text(json.dumps(structured, indent=2), encoding="utf-8")
            logging.info("Structured JSON saved to %s", str(json_path))
        except Exception as e:
            logging.exception("Failed to save structured JSON: %s", e)

    # Summarize with LLM
    try:
        overall_summary = summarize_patch_structured(structured)
        logging.info("Overall summary generated (len=%d)", len(overall_summary))
    except Exception as e:
        logging.exception("LLM summarization failed: %s", e)
        print(f"❌ LLM summarization failed: {e}")
        return

    # Save summary to unique filename
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hash_digest = hashlib.md5(raw).hexdigest()[:8]
    if args.overwrite:
        summary_name = f"summary_{hash_digest}.txt"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_name = f"summary_{timestamp}_{hash_digest}.txt"
    summary_path = outdir / summary_name

    try:
        summary_path.write_text(overall_summary, encoding="utf-8")
        logging.info("Overall summary saved to %s", str(summary_path))
        print(f"\n✅ Overall summary saved to: {summary_path}")
    except Exception as e:
        logging.exception("Failed to save summary: %s", e)
        print(f"❌ Failed to save summary: {e}")


if __name__ == "__main__":
    main()
