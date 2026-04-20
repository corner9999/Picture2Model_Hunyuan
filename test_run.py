#!/usr/bin/env python3
"""Minimal real-run example: set API key and input JSON, call one function."""

from __future__ import annotations

import json
from pathlib import Path

from task_model_service import process_generation_task


API_KEY = "sk-pq4NDCdPZtPGherSWQXA4TkHJ3LyIAJAVFtnO7OsiYw3kfI0"

INPUT_JSON = {
    "task_id": "task_001",
    "objects": [
        {
            "object_id": "obj_001",
            "label": "laptop",
            "crop_url": "input_images/pencil.png",
            "need_generation": True,
            "model_url": "",
        },
        
    ],
}

OUTPUT_JSON_PATH = Path("task_result_real.json")


def progress_printer(event: dict) -> None:
    phase = str(event.get("phase", "")).strip() or "progress"
    object_id = str(
        event.get("task_object_id") or event.get("object_id") or ""
    ).strip()
    status = str(event.get("status", "")).strip()
    message = str(event.get("message", "")).strip()

    prefix = f"[{phase}]"
    if object_id:
        prefix += f"[{object_id}]"

    parts = [prefix]
    if status:
        parts.append(f"status={status}")
    if message:
        parts.append(message)

    print(" ".join(parts), flush=True)


def main() -> None:
    print("开始执行 3D 生成任务...", flush=True)
    result = process_generation_task(
        INPUT_JSON,
        api_key=API_KEY,
        progress_callback=progress_printer,
    )
    OUTPUT_JSON_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"结果已写入: {OUTPUT_JSON_PATH.resolve()}", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
