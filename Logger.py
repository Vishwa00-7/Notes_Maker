import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Root log file path
ROOT_DUMP_FILE = Path("dump.txt")
_ACTIVE_TOPIC_DIR: Optional[Path] = None


def set_active_topic(topic: str) -> None:
    """Register current active topic folder to mirror dump.txt into it."""
    global _ACTIVE_TOPIC_DIR
    if topic:
        clean_topic = "".join(c for c in topic if c.isalnum() or c in (" ", "-", "_")).strip()
        clean_topic = clean_topic.replace(" ", "_")
        _ACTIVE_TOPIC_DIR = Path(clean_topic)
    else:
        _ACTIVE_TOPIC_DIR = None


def _format_content(content: Any) -> str:
    """Format any python object or string into a clean string representation."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, indent=2, default=str)
    except Exception:
        return str(content)


def log_dump(
    category: str,
    title: str,
    content: Any = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Append an entry to dump.txt (and active topic folder dump.txt if available).
    Flushes immediately to ensure complete persistence against crashes.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"[{now_str}] [{category.upper()}] {title}"
    separator = "-" * 75

    lines = [header, separator]

    if metadata:
        for k, v in metadata.items():
            lines.append(f"  • {k}: {v}")
        lines.append("")

    formatted = _format_content(content)
    if formatted:
        lines.append(formatted)

    lines.append("-" * 75)
    lines.append("\n")

    entry_text = "\n".join(lines)

    # 1. Write to root dump.txt
    try:
        with open(ROOT_DUMP_FILE, "a", encoding="utf-8") as f:
            f.write(entry_text)
            f.flush()
    except Exception as e:
        print(f"[WARN] Failed to write to {ROOT_DUMP_FILE}: {e}", file=sys.stderr)

    # 2. Write to topic directory dump.txt if topic folder exists
    global _ACTIVE_TOPIC_DIR
    if _ACTIVE_TOPIC_DIR and _ACTIVE_TOPIC_DIR.exists() and _ACTIVE_TOPIC_DIR.is_dir():
        try:
            topic_dump = _ACTIVE_TOPIC_DIR / "dump.txt"
            with open(topic_dump, "a", encoding="utf-8") as f:
                f.write(entry_text)
                f.flush()
        except Exception:
            pass


def log_session_start() -> None:
    """Log the initial start of a Notes Maker workflow execution."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    banner = "\n".join([
        "=" * 80,
        f"[{now_str}] [SESSION START] NOTES MAKER WORKFLOW RUN",
        "=" * 80,
        ""
    ])
    try:
        with open(ROOT_DUMP_FILE, "a", encoding="utf-8") as f:
            f.write(banner)
            f.flush()
    except Exception:
        pass


def log_decision(decision_name: str, choice: Any, details: Optional[Dict[str, Any]] = None) -> None:
    """Log user inputs, routing decisions, and human intervention choices."""
    log_dump(
        category="DECISION",
        title=decision_name,
        content=details,
        metadata={"User Selection": choice}
    )


def log_prompt(node_name: str, model_name: str, prompt_text: Any, attempt: int = 1) -> None:
    """Log the exact prompt sent to the LLM."""
    raw_str = _format_content(prompt_text)
    log_dump(
        category="PROMPT",
        title=f"Prompt Sent by '{node_name}' (Attempt {attempt})",
        content=raw_str,
        metadata={
            "Target Model": model_name,
            "Node": node_name,
            "Attempt": attempt,
            "Prompt Character Count": len(raw_str)
        }
    )


def log_model_output(
    node_name: str,
    model_name: str,
    raw_output: str,
    attempt: int = 1,
    finish_reason: Optional[str] = None
) -> None:
    """Log the exact, unedited raw output returned by the LLM."""
    metadata = {
        "Model": model_name,
        "Node": node_name,
        "Attempt": attempt,
        "Output Character Count": len(raw_output or ""),
        "Output Line Count": len((raw_output or "").splitlines())
    }
    if finish_reason:
        metadata["Finish Reason"] = finish_reason

    log_dump(
        category="MODEL OUTPUT",
        title=f"Raw Model Output for '{node_name}' (Attempt {attempt})",
        content=raw_output,
        metadata=metadata
    )


def log_parsed_result(node_name: str, summary: str, data: Any = None) -> None:
    """Log the parsed or processed result from model output."""
    log_dump(
        category="PARSED RESULT",
        title=f"Parsed Result in '{node_name}': {summary}",
        content=data,
        metadata={"Node": node_name}
    )


def log_error(
    node_name: str,
    error: Exception,
    attempt: int = 1,
    raw_output: Optional[str] = None
) -> None:
    """Log errors, exceptions, and stack traces alongside raw output if available."""
    tb = traceback.format_exc()
    details = f"Exception: {type(error).__name__}: {error}\n\nStack Trace:\n{tb}"
    if raw_output:
        details += f"\n\nRaw Output that triggered error:\n{raw_output}"

    log_dump(
        category="ERROR",
        title=f"Error in '{node_name}' (Attempt {attempt})",
        content=details,
        metadata={
            "Node": node_name,
            "Attempt": attempt,
            "Error Type": type(error).__name__
        }
    )


def log_file_created(file_path: str, char_count: int) -> None:
    """Log the successful generation and saving of a markdown file."""
    log_dump(
        category="FILE CREATED",
        title=f"Generated Note Saved: {file_path}",
        content=None,
        metadata={
            "File Path": str(file_path),
            "Character Count": char_count
        }
    )


def log_session_end(status: str, summary: Optional[Dict[str, Any]] = None) -> None:
    """Log workflow completion or termination."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dump(
        category="SESSION END",
        title=f"Workflow Status: {status.upper()}",
        content=summary,
        metadata={"Timestamp": now_str, "Final Status": status}
    )
