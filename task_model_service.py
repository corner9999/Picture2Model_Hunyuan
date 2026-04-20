#!/usr/bin/env python3
"""Backend-facing JSON wrapper for object-wise 3D generation tasks."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error, parse, request

from hunyuan_3d import (
    DEFAULT_MODEL,
    GenerationOptions,
    Hunyuan3DError,
    generate_3d,
    get_api_key,
)

DEFAULT_TASK_OUTPUT_DIR = Path("generated_models/tasks")


ProgressCallback = Callable[[dict[str, Any]], None]
ObjectGenerator = Callable[["TaskObject", bytes, Path], Path]
ModelReferenceBuilder = Callable[[Path, "TaskObject", str], str]


class TaskPayloadError(ValueError):
    """Raised when the input JSON payload shape is invalid."""


class TaskGenerationError(RuntimeError):
    """Raised when one object fails during generation."""


@dataclass(slots=True)
class TaskObject:
    object_id: str
    label: str
    crop_url: str
    need_generation: bool
    model_url: str = ""


@dataclass(slots=True)
class TaskRequest:
    task_id: str
    objects: list[TaskObject]


@dataclass(slots=True)
class TaskGenerationConfig:
    output_dir: Path = DEFAULT_TASK_OUTPUT_DIR
    model: str = DEFAULT_MODEL
    generate_type: str = "Normal"
    prompt: str | None = None
    face_count: int | None = None
    enable_pbr: bool = False
    poll_interval: float = 8.0
    timeout: int = 900


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskPayloadError(f"Field '{field_name}' must be a non-empty string.")
    return value.strip()


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TaskPayloadError(f"Field '{field_name}' must be a boolean.")
    return value


def safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or fallback


def parse_task_payload(payload: dict[str, Any] | str) -> TaskRequest:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TaskPayloadError(f"Invalid JSON string: {exc}") from exc

    if not isinstance(payload, dict):
        raise TaskPayloadError("Payload must be a JSON object.")

    task_id = _require_non_empty_str(payload.get("task_id"), "task_id")
    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, list):
        raise TaskPayloadError("Field 'objects' must be a list.")

    objects: list[TaskObject] = []
    for index, raw_object in enumerate(raw_objects):
        if not isinstance(raw_object, dict):
            raise TaskPayloadError(f"objects[{index}] must be a JSON object.")

        object_id = _require_non_empty_str(
            raw_object.get("object_id"),
            f"objects[{index}].object_id",
        )
        label = _require_non_empty_str(
            raw_object.get("label"),
            f"objects[{index}].label",
        )
        need_generation = _require_bool(
            raw_object.get("need_generation"),
            f"objects[{index}].need_generation",
        )
        crop_url = str(raw_object.get("crop_url") or "").strip()
        model_url = str(raw_object.get("model_url") or "").strip()

        if need_generation and not crop_url:
            raise TaskPayloadError(
                f"objects[{index}].crop_url is required when need_generation=true."
            )

        objects.append(
            TaskObject(
                object_id=object_id,
                label=label,
                crop_url=crop_url,
                need_generation=need_generation,
                model_url=model_url,
            )
        )

    return TaskRequest(task_id=task_id, objects=objects)


def notify(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def read_source_bytes(source: str) -> bytes:
    if not source:
        raise TaskPayloadError("Missing crop_url for object that needs generation.")

    if source.startswith(("http://", "https://")):
        req = request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TaskGenerationError(
                f"Failed to download crop_url '{source}': HTTP {exc.code} {detail}"
            ) from exc
        except error.URLError as exc:
            raise TaskGenerationError(
                f"Failed to download crop_url '{source}': {exc.reason}"
            ) from exc

    if source.startswith("file://"):
        path = Path(parse.unquote(parse.urlparse(source).path)).expanduser()
    else:
        path = Path(source).expanduser()

    if not path.exists():
        raise TaskGenerationError(f"Local crop image not found: {path}")
    if not path.is_file():
        raise TaskGenerationError(f"Local crop image is not a file: {path}")

    return path.read_bytes()


def find_generated_glb(path: Path) -> Path:
    if path.suffix.lower() == ".glb" and path.exists():
        return path

    if path.is_dir():
        matches = sorted(path.rglob("*.glb"))
        if matches:
            return matches[0]

    raise TaskGenerationError(f"No .glb model found under: {path}")


def default_model_reference_builder(
    model_path: Path,
    task_object: TaskObject,
    task_id: str,
) -> str:
    del task_object, task_id
    return str(model_path.resolve())


def build_task_model_url_builder(base_url: str) -> ModelReferenceBuilder:
    normalized = base_url.rstrip("/")

    def _builder(model_path: Path, task_object: TaskObject, task_id: str) -> str:
        del task_object
        return (
            f"{normalized}/tasks/{parse.quote(task_id)}/models/"
            f"{parse.quote(model_path.name)}"
        )

    return _builder


def build_hunyuan_object_generator(
    *,
    api_key: str,
    config: TaskGenerationConfig,
    progress_callback: ProgressCallback | None = None,
) -> ObjectGenerator:
    def _generate(task_object: TaskObject, image_bytes: bytes, work_dir: Path) -> Path:
        def _forward_progress(event: dict[str, Any]) -> None:
            payload = {
                "task_object_id": task_object.object_id,
                "task_label": task_object.label,
            }
            payload.update(event)
            notify(progress_callback, **payload)

        try:
            result = generate_3d(
                api_key=api_key,
                options=GenerationOptions(
                    model=config.model,
                    generate_type=config.generate_type,
                    prompt=config.prompt,
                    image_bytes=image_bytes,
                    image_name=f"{task_object.object_id}.png",
                    face_count=config.face_count,
                    enable_pbr=config.enable_pbr,
                    poll_interval=config.poll_interval,
                    timeout=config.timeout,
                    out_dir=work_dir,
                    metadata_path=work_dir / "latest_model.json",
                ),
                progress_callback=_forward_progress,
            )
        except Hunyuan3DError as exc:
            raise TaskGenerationError(str(exc)) from exc

        return find_generated_glb(result["saved_paths"][0].parent)

    return _generate


def build_mock_object_generator() -> ObjectGenerator:
    def _generate(task_object: TaskObject, image_bytes: bytes, work_dir: Path) -> Path:
        del image_bytes
        work_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"{safe_name(task_object.object_id, 'object')}.glb"
        model_path = work_dir / model_name
        model_path.write_bytes(
            (
                "mock glb placeholder for local JSON pipeline testing\n"
                f"object_id={task_object.object_id}\n"
                f"label={task_object.label}\n"
            ).encode("utf-8")
        )
        (work_dir / "latest_model.json").write_text(
            json.dumps(
                {
                    "job_id": f"mock_{safe_name(task_object.object_id, 'object')}",
                    "model_url": model_name,
                    "preview_url": "",
                    "files": [model_name],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return model_path

    return _generate


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def process_generation_task(
    payload: dict[str, Any] | str,
    *,
    api_key: str | None = None,
    config: TaskGenerationConfig | None = None,
    object_generator: ObjectGenerator | None = None,
    model_reference_builder: ModelReferenceBuilder | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    task_request = parse_task_payload(payload)
    config = config or TaskGenerationConfig()
    model_reference_builder = model_reference_builder or default_model_reference_builder

    if object_generator is None:
        resolved_api_key = get_api_key(api_key)
        object_generator = build_hunyuan_object_generator(
            api_key=resolved_api_key,
            config=config,
            progress_callback=progress_callback,
        )

    task_root = config.output_dir / safe_name(task_request.task_id, "task")
    work_root = task_root / "work"
    models_root = task_root / "models"
    work_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    generated_models: list[dict[str, str]] = []

    for task_object in task_request.objects:
        if not task_object.need_generation:
            continue

        notify(
            progress_callback,
            phase="loading_input",
            task_id=task_request.task_id,
            object_id=task_object.object_id,
            label=task_object.label,
            message=f"Loading source image for {task_object.object_id}",
        )

        try:
            image_bytes = read_source_bytes(task_object.crop_url)
            object_work_dir = work_root / safe_name(task_object.object_id, "object")
            source_model_path = object_generator(task_object, image_bytes, object_work_dir)
            source_model_path = find_generated_glb(source_model_path)

            final_model_path = models_root / (
                f"{safe_name(task_object.object_id, 'object')}.glb"
            )
            shutil.copy2(source_model_path, final_model_path)

            object_result = {
                "object_id": task_object.object_id,
                "label": task_object.label,
                "model_url": model_reference_builder(
                    final_model_path,
                    task_object,
                    task_request.task_id,
                ),
            }
            generated_models.append(object_result)

            _write_json(
                object_work_dir / "result.json",
                {
                    "task_id": task_request.task_id,
                    "object_id": task_object.object_id,
                    "label": task_object.label,
                    "source_crop_url": task_object.crop_url,
                    "model_path": str(final_model_path.resolve()),
                    "response_item": object_result,
                },
            )

            notify(
                progress_callback,
                phase="completed",
                task_id=task_request.task_id,
                object_id=task_object.object_id,
                label=task_object.label,
                model_path=str(final_model_path.resolve()),
                message=f"Finished {task_object.object_id}",
            )
        except (TaskPayloadError, TaskGenerationError) as exc:
            raise TaskGenerationError(
                f"Failed to generate model for object_id={task_object.object_id}: {exc}"
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive wrapper
            raise TaskGenerationError(
                f"Unexpected error for object_id={task_object.object_id}: {exc}"
            ) from exc

    response = {
        "task_id": task_request.task_id,
        "generated_models": generated_models,
    }
    _write_json(task_root / "response.json", response)
    return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process task JSON and generate object-wise 3D models."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Input task JSON file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional output file for the response JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TASK_OUTPUT_DIR,
        help="Directory that stores generated task artifacts.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use local mock generation instead of calling Tencent Hunyuan.",
    )
    parser.add_argument(
        "--base-model-url",
        help="Optional base URL, for example https://example.com .",
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
        help="Optional target face count.",
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
        help="Maximum seconds to wait for each object.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    config = TaskGenerationConfig(
        output_dir=args.output_dir,
        model=args.model,
        generate_type=args.generate_type,
        prompt=args.prompt,
        face_count=args.face_count,
        enable_pbr=args.enable_pbr,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
    generator = build_mock_object_generator() if args.mock else None
    model_reference_builder = (
        build_task_model_url_builder(args.base_model_url)
        if args.base_model_url
        else None
    )

    response = process_generation_task(
        payload,
        config=config,
        object_generator=generator,
        model_reference_builder=model_reference_builder,
    )

    if args.output_json:
        _write_json(args.output_json, response)

    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
