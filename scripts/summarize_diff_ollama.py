
#!/usr/bin/env python3
"""
Summarize a unified diff using Ollama.
- Parses patch with unidiff
- Extracts structured changes (modified/added/removed) for ONE file
- Calls Ollama once to generate summary
- Writes summary.txt
"""

import os, sys, argparse, json, difflib, logging
from logging.handlers import RotatingFileHandler
import requests
from unidiff import PatchSet

# --- Config ---
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_API_TOKEN = os.getenv("OLLAMA_API_TOKEN", "")
TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
SKIP_DIRS = {"node_modules", "vendor", "dist", "build", ".venv", ".git"}

# --- Logging ---
def _setup_logging():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "summarize_patch.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = _setup_logging()

def _headers():
    h = {"Content-Type": "application/json"}
    if OLLAMA_API_TOKEN:
        h["Authorization"] = f"Bearer {OLLAMA_API_TOKEN}"
    return h

# --- Ollama Chat ---
def ollama_chat(messages):
    url = f"{OLLAMA_ENDPOINT}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 1,
        "num_ctx": 8192,
        "stream": False
    }
    r = requests.post(url, json=payload, headers=_headers(), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()

# --- Structured Change Extraction ---
def pair_modified(removals, additions, sim_threshold=0.6):
    modified, used_add_idx = [], set()
    for old in removals:
        best_idx, best_score = None, 0.0
        for j, new in enumerate(additions):
            if j in used_add_idx: continue
            score = difflib.SequenceMatcher(a=old, b=new).ratio()
            if score > best_score:
                best_score, best_idx = score, j
        if best_idx is not None and best_score >= sim_threshold:
            modified.append((old, additions[best_idx]))
            used_add_idx.add(best_idx)
    remaining_removed = [o for o in removals if o not in {x[0] for x in modified}]
    remaining_added = [a for i, a in enumerate(additions) if i not in used_add_idx]
    return modified, remaining_removed, remaining_added

def extract_changes_from_hunk(hunk):
    removals, additions = [], []
    for line in hunk:
        if line.is_removed: removals.append(line.value.rstrip("\n"))
        elif line.is_added: additions.append(line.value.rstrip("\n"))
    return removals, additions

def build_file_changes(file_patch):
    all_removed, all_added = [], []
    for h in file_patch:
        r, a = extract_changes_from_hunk(h)
        all_removed.extend(r)
        all_added.extend(a)
    modified, remaining_removed, remaining_added = pair_modified(all_removed, all_added)
    return {
        "file": file_patch.path,
        "modified": [{"old": o, "new": n} for o, n in modified],
        "added": remaining_added,
        "removed": remaining_removed
    }

# --- Summarize File ---
def summarize_file(changes):
    system = (
        "You are a senior code reviewer.\n"
        "Summarize ONLY the changes provided. Do NOT invent changes.\n"
        "Output format:\n"
        "File: <filename>\n"
        "- Purpose: <one line>\n"
        "- Changes:\n"
        "  • Modified: old → new\n"
        "  • Added: ...\n"
        "  • Removed: ...\n"
        "Keep it concise and accurate."
    )
    user = f"Here are structured changes:\n{json.dumps(changes, indent=2)}"
    return ollama_chat([{"role": "system", "content": system}, {"role": "user", "content": user}])

# --- Main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", default="summary.txt")
    args = ap.parse_args()

    with open(args.patch, "r", encoding="utf-8", errors="ignore") as f:
        patch = PatchSet(f)

    if len(patch) == 0:
        print("No files in patch.")
        sys.exit(0)

    pf = patch[0]  # Only first file
    fname = pf.path
    if any(d in fname.split("/") for d in SKIP_DIRS):
        print("File in ignored directory.")
        sys.exit(0)

    changes = build_file_changes(pf)
    summary = summarize_file(changes)

    report = [
        f"Repository: {args.repo}",
        f"Branch:     {args.branch}",
        f"Commit:     {args.sha}",
        "",
        "Summary",
        summary
    ]

    with open(args.out, "w", encoding="utf-8") as out:
        out.write("\n".join(report))

    print("\n".join(report))

if __name__ == "__main__":
    main()
