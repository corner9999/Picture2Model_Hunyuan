#!/usr/bin/env python3
"""Batch-generate Tencent Hunyuan 3D models from images in one folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from hunyuan_3d import (
    DEFAULT_MODEL,
    IMAGE_EXTENSIONS,
    GenerationOptions,
    Hunyuan3DError,
    generate_3d,
    get_api_key,
)

DEFAULT_INPUT_DIR = Path("input_images")
DEFAULT_OUTPUT_DIR = Path("generated_models")
MANIFEST_NAME = "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D models for every image inside one input folder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder that contains input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder that receives generated 3D assets.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["hy-3d-3.0", "hy-3d-3.1"],
        help="Tencent Hunyuan 3D model version.",
    )
    parser.add_argument(
        "--generate-type",
        default="Normal",
        choices=["Normal", "LowPoly", "Geometry", "Sketch"],
        help="Generation mode.",
    )
    parser.add_argument(
        "--prompt",
        help="Optional prompt. Only combine with images in Sketch mode.",
    )
    parser.add_argument(
        "--face-count",
        type=int,
        help="Target face count. Effective range per docs: 3000-1500000.",
    )
    parser.add_argument(
        "--enable-pbr",
        action="store_true",
        help="Enable PBR material generation.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=8.0,
        help="Seconds between status checks.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Maximum seconds to wait for each job.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate images even if the same file hash already exists in the manifest.",
    )
    return parser.parse_args()


def relpath(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), Path.cwd().resolve())).as_posix()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug or "image"


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"updated_at": 0, "items": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "items" not in data or not isinstance(data["items"], dict):
        data["items"] = {}
    return data


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = int(time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def list_input_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise Hunyuan3DError(f"Input folder does not exist: {input_dir}")
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_first_by_suffix(paths: list[Path], suffix: str) -> Path | None:
    return next((path for path in paths if path.suffix.lower() == suffix.lower()), None)


def progress_for(image_name: str):
    def _callback(event: dict[str, Any]) -> None:
        phase = str(event.get("phase", ""))
        if phase == "polling":
            print(f"[{image_name}] Current status: {event.get('status', 'unknown')}")
            return
        message = event.get("message")
        if message:
            print(f"[{image_name}] {message}")

    return _callback


def build_record(
    *,
    image_path: Path,
    fingerprint: str,
    bucket_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    saved_paths = [Path(path).resolve() for path in result["saved_paths"]]
    glb_path = find_first_by_suffix(saved_paths, ".glb")
    preview_path = next(
        (path for path in saved_paths if "_preview" in path.name.lower()),
        None,
    )

    return {
        "status": "completed",
        "source_image": relpath(image_path),
        "fingerprint": fingerprint,
        "job_id": result["job_id"],
        "output_dir": relpath(bucket_dir),
        "model_file": relpath(glb_path) if glb_path else "",
        "preview_file": relpath(preview_path) if preview_path else "",
        "files": [relpath(path) for path in saved_paths],
        "generated_at": int(time.time()),
    }


def main() -> None:
    args = parse_args()
    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / MANIFEST_NAME
    manifest = read_manifest(manifest_path)

    try:
        images = list_input_images(args.input_dir)
        api_key = get_api_key(allow_prompt=True)
    except Hunyuan3DError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not images:
        print(f"No images found in {args.input_dir.resolve()}")
        return

    completed = 0
    skipped = 0
    failed = 0

    for image_path in images:
        fingerprint = sha256_file(image_path)
        existing = manifest["items"].get(fingerprint)
        if existing and existing.get("status") == "completed" and not args.force:
            skipped += 1
            print(f"[{image_path.name}] Skipped, already generated: {existing.get('model_file', '')}")
            continue

        bucket_name = f"{slugify(image_path.stem)}_{fingerprint[:8]}"
        bucket_dir = args.output_dir / bucket_name
        bucket_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = generate_3d(
                api_key=api_key,
                options=GenerationOptions(
                    model=args.model,
                    generate_type=args.generate_type,
                    prompt=args.prompt,
                    image_path=image_path,
                    face_count=args.face_count,
                    enable_pbr=args.enable_pbr,
                    poll_interval=args.poll_interval,
                    timeout=args.timeout,
                    out_dir=bucket_dir,
                    metadata_path=bucket_dir / "latest_model.json",
                ),
                progress_callback=progress_for(image_path.name),
            )
        except Hunyuan3DError as exc:
            failed += 1
            manifest["items"][fingerprint] = {
                "status": "failed",
                "source_image": relpath(image_path),
                "fingerprint": fingerprint,
                "error": str(exc),
                "generated_at": int(time.time()),
            }
            write_manifest(manifest_path, manifest)
            print(f"[{image_path.name}] Failed: {exc}")
            continue

        record = build_record(
            image_path=image_path,
            fingerprint=fingerprint,
            bucket_dir=bucket_dir,
            result=result,
        )
        manifest["items"][fingerprint] = record
        write_manifest(manifest_path, manifest)
        (bucket_dir / "result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        completed += 1
        print(f"[{image_path.name}] Done: {record['model_file']}")

    print(
        "\nSummary:"
        f"\n - completed: {completed}"
        f"\n - skipped: {skipped}"
        f"\n - failed: {failed}"
        f"\n - input folder: {args.input_dir.resolve()}"
        f"\n - output folder: {args.output_dir.resolve()}"
        f"\n - manifest: {manifest_path.resolve()}"
    )


if __name__ == "__main__":
    main()
