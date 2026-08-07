#!/usr/bin/env python3
"""Trace Analysis & Budget Summary Script for Kaggle-in-Kaggle Local Evaluation.

This script parses the rich execution trace files (``output/trace_*.json``) generated
by ``run_local_eval.py`` and presents a clean terminal summary of the agent's run.

The trace JSON follows the ``SessionTrace.to_dict()`` schema::

    {
      "trace_version": "1.0",
      "duration_s": 123.456,
      "num_entries": 42,
      "entries": [ { "elapsed_s", "type", "author", "tool", "args", ... }, ... ],
      "summary": {
        "total_events", "tool_calls", "tool_call_breakdown",
        "thinking_entries", "compaction_entries", "text_entries",
        "total_prompt_tokens", "total_cached_prompt_tokens",
        "total_completion_tokens", "total_tokens"
      }
    }

## Features
1. Displays high-level stats: duration, event counts, and token usage.
2. Shows per-tool call breakdown.
3. Scans the entry timeline for error events and tool-response failures.

## Usage
    python parse_eval_trace.py [--trace-file output/trace_kaggle_in_kaggle_train_01.json]
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def parse_trace(trace_path: Path) -> None:
    """Parse and print a summary of a single trace file."""
    print(f"\n{'=' * 60}")
    print(f"  Trace Analysis: {trace_path.name}")
    print(f"{'=' * 60}\n")

    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load trace JSON: {e}")
        sys.exit(1)

    # --- 1. High-Level Stats ---
    trace_version = data.get("trace_version", "unknown")
    duration_s = data.get("duration_s", 0.0)
    num_entries = data.get("num_entries", 0)
    summary = data.get("summary", {})

    print(f"Trace Version  : {trace_version}")
    print(f"Duration       : {duration_s:.1f}s ({duration_s / 60:.1f} min)")
    print(f"Total Events   : {num_entries}")
    print()

    # --- 2. Event Type Breakdown ---
    print("--- Event Breakdown ---")
    tool_calls = summary.get("tool_calls", 0)
    thinking = summary.get("thinking_entries", 0)
    compactions = summary.get("compaction_entries", 0)
    text = summary.get("text_entries", 0)
    print(f"  Tool calls        : {tool_calls}")
    print(f"  Thinking entries  : {thinking}")
    print(f"  Text entries      : {text}")
    print(f"  Compaction events : {compactions}")
    print()

    # --- 3. Tool Call Breakdown ---
    breakdown = summary.get("tool_call_breakdown", {})
    if breakdown:
        print("--- Tool Call Breakdown ---")
        max_name_len = max(len(name) for name in breakdown)
        for name, count in sorted(breakdown.items(), key=lambda x: -x[1]):
            print(f"  {name:<{max_name_len}} : {count}")
        print()

    # --- 4. Token Usage ---
    print("--- Token Usage ---")
    prompt_tokens = summary.get("total_prompt_tokens", 0)
    cached_tokens = summary.get("total_cached_prompt_tokens", 0)
    completion_tokens = summary.get("total_completion_tokens", 0)
    total_tokens = summary.get("total_tokens", 0)
    print(f"  Prompt tokens     : {prompt_tokens:,}")
    print(f"  Cached tokens     : {cached_tokens:,}")
    print(f"  Completion tokens : {completion_tokens:,}")
    print(f"  Total tokens      : {total_tokens:,}")
    if prompt_tokens > 0:
        cache_rate = (cached_tokens / prompt_tokens) * 100
        print(f"  Cache hit rate    : {cache_rate:.1f}%")
    print()

    # --- 5. Error Scanning ---
    print("--- Error Scan ---")
    entries = data.get("entries", [])
    error_count = 0

    for entry in entries:
        entry_type = entry.get("type", "")
        elapsed = entry.get("elapsed_s", 0.0)

        # Explicit error events
        if entry_type == "error":
            error_count += 1
            content = entry.get("content", "Unknown error")
            author = entry.get("author", "unknown")
            print(f"  [{elapsed:7.1f}s] ERROR ({author}): {content[:200]}")

        # Tool responses that contain error indicators
        elif entry_type == "tool_response":
            result_str = entry.get("result", "")
            tool_name = entry.get("tool", "unknown")
            # Check for common error patterns in tool responses
            if any(
                marker in result_str.lower()
                for marker in ['"error"', "traceback", "exception", "command failed"]
            ):
                error_count += 1
                preview = result_str[:150].replace("\n", " ")
                print(f"  [{elapsed:7.1f}s] TOOL ERROR ({tool_name}): {preview}")

    if error_count == 0:
        print("  No errors detected in trace.")
    else:
        print(f"\n  Total errors found: {error_count}")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse and summarize Kaggle-in-Kaggle evaluation trace logs."
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help="Path to the specific trace JSON file. If omitted, finds the most recent trace in output/",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Path to an experiment directory (e.g., submissions/01_baseline) to find its most recent trace.",
    )
    args = parser.parse_args()

    cwd_dir = Path.cwd()

    if args.trace_file:
        trace_path = (cwd_dir / args.trace_file).resolve()
    elif args.experiment_dir:
        exp_dir = (cwd_dir / args.experiment_dir).resolve()
        json_files = glob.glob(str(exp_dir / "output" / "trace_*.json"))
        if not json_files:
            print(f"Error: No trace files found in {exp_dir / 'output'}.")
            sys.exit(1)
        trace_path = Path(max(json_files, key=os.path.getmtime))
    else:
        # Search both structured experiment outputs and legacy root output
        structured_files = glob.glob(str(cwd_dir / "submissions" / "*" / "output" / "trace_*.json"))
        legacy_files = glob.glob(str(cwd_dir / "output" / "trace_*.json"))
        json_files = structured_files + legacy_files
        if not json_files:
            print("Error: No trace files found in submissions/*/output/ or output/. Please specify --trace-file.")
            sys.exit(1)
        trace_path = Path(max(json_files, key=os.path.getmtime))

    if not trace_path.exists():
        print(f"Error: Trace file not found at {trace_path}")
        sys.exit(1)

    parse_trace(trace_path)


if __name__ == "__main__":
    main()
