#!/usr/bin/env python3
"""Shared Tencent Hunyuan 3D generation helpers for CLI and local web app."""

from __future__ import annotations

import base64
import io
import json
import os
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error, parse, request

from PIL import Image
import certifi

SUBMIT_URL = "https://tokenhub.tencentmaas.com/v1/api/3d/submit"
QUERY_URL = "https://tokenhub.tencentmaas.com/v1/api/3d/query"
DEFAULT_MODEL = "hy-3d-3.1"
DEFAULT_GENERATE_TYPE = "Normal"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LATEST_MODEL_METADATA = Path("latest_model.json")
SUCCESS_STATUSES = {"completed", "done"}
FAILED_STATUSES = {"failed", "fail", "cancelled", "canceled"}
MAX_IMAGE_BINARY_BYTES = 1_450_000

ProgressCallback = Callable[[dict[str, Any]], None]


class Hunyuan3DError(RuntimeError):
    """Raised when Tencent Hunyuan 3D generation fails."""


@dataclass(slots=True)
class GenerationOptions:
    model: str = DEFAULT_MODEL
    generate_type: str = DEFAULT_GENERATE_TYPE
    prompt: str | None = None
    image_path: Path | None = None
    image_bytes: bytes | None = None
    image_name: str | None = None
    image_url: str | None = None
    face_count: int | None = None
    polygon_type: str | None = None
    enable_pbr: bool = False
    result_format: str | None = None
    poll_interval: float = 8.0
    timeout: int = 900
    out_dir: Path = Path("outputs")
    metadata_path: Path = LATEST_MODEL_METADATA


def fail(message: str) -> None:
    raise Hunyuan3DError(message)


def get_api_key(explicit_api_key: str | None = None, allow_prompt: bool = False) -> str:
    if explicit_api_key:
        return explicit_api_key.strip()

    for key_name in ("TENCENT_HUNYUAN_API_KEY", "HUNYUAN_API_KEY"):
        value = os.getenv(key_name)
        if value:
            return value

    if allow_prompt and sys.stdin.isatty():
        value = input("Tencent Hunyuan API Key: ").strip()
        if value:
            return value

    fail("Missing API key. Set TENCENT_HUNYUAN_API_KEY or provide api_key explicitly.")
    raise AssertionError("unreachable")


def find_single_local_image(search_dir: Path) -> Path | None:
    candidates = [
        path
        for path in search_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def validate_options(options: GenerationOptions) -> None:
    if options.image_path and options.image_url:
        fail("Use either image_path or image_url, not both.")

    if options.image_path and options.image_bytes:
        fail("Use either image_path or image_bytes, not both.")

    if not any([options.image_path, options.image_bytes, options.image_url, options.prompt]):
        fail("Provide an image, an image URL, or a prompt.")

    if (
        (options.image_path or options.image_bytes or options.image_url)
        and options.prompt
        and options.generate_type != "Sketch"
    ):
        fail("Prompt can only be combined with an image when generate_type is Sketch.")

    if options.image_path and not options.image_path.exists():
        fail(f"Input image not found: {options.image_path}")

    if options.face_count is not None and not (3000 <= options.face_count <= 1_500_000):
        fail("face_count must be between 3000 and 1500000.")

    if options.polygon_type and options.generate_type != "LowPoly":
        fail("polygon_type only works with generate_type=LowPoly.")

    if options.model == "hy-3d-3.1" and options.generate_type == "LowPoly":
        fail("hy-3d-3.1 does not support LowPoly according to the official docs.")


def encode_image_bytes(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def flatten_transparency(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.getchannel("A")
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    if image.mode == "P":
        return flatten_transparency(image.convert("RGBA"))
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def compress_image_to_limit(
    image_bytes: bytes,
    *,
    target_bytes: int = MAX_IMAGE_BINARY_BYTES,
) -> bytes:
    if len(image_bytes) <= target_bytes:
        return image_bytes

    source = flatten_transparency(Image.open(io.BytesIO(image_bytes)))
    width, height = source.size
    scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    qualities = [88, 82, 76, 70, 64, 58, 52]
    best = image_bytes

    for scale in scales:
        resized = source
        if scale != 1.0:
            resized = source.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )

        for quality in qualities:
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=quality, optimize=True)
            candidate = buffer.getvalue()
            if len(candidate) < len(best):
                best = candidate
            if len(candidate) <= target_bytes:
                return candidate

    if len(best) <= target_bytes:
        return best

    fail(
        "Image is still too large after compression. Try a smaller image or crop it first."
    )
    raise AssertionError("unreachable")


def load_image_bytes(options: GenerationOptions) -> bytes | None:
    if options.image_bytes is not None:
        return compress_image_to_limit(options.image_bytes)
    if options.image_path is not None:
        return compress_image_to_limit(options.image_path.read_bytes())
    return None


def build_submit_payload(options: GenerationOptions) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": options.model,
        "generateType": options.generate_type,
    }

    if options.prompt:
        payload["prompt"] = options.prompt

    image_bytes = load_image_bytes(options)
    if image_bytes is not None:
        payload["imageBase64"] = encode_image_bytes(image_bytes)

    if options.image_url:
        payload["imageUrl"] = options.image_url

    if options.face_count is not None:
        payload["faceCount"] = options.face_count

    if options.polygon_type:
        payload["polygonType"] = options.polygon_type

    if options.enable_pbr:
        payload["enablePBR"] = True

    if options.result_format:
        payload["resultFormat"] = options.result_format

    return payload


def post_json(url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    req = request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120, context=ssl_context) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        fail(f"HTTP {exc.code} while calling API: {detail}")
    except error.URLError as exc:
        fail(f"Unable to reach Tencent API: {exc.reason}")
    raise AssertionError("unreachable")


def normalize_status(status: str | None) -> str:
    return (status or "").strip().lower()


def notify(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def wait_for_completion(
    api_key: str,
    model: str,
    job_id: str,
    poll_interval: float,
    timeout: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    started = time.time()
    while True:
        response = post_json(QUERY_URL, {"model": model, "id": job_id}, api_key)
        status = normalize_status(response.get("status"))
        notify(
            progress_callback,
            phase="polling",
            job_id=job_id,
            status=status or "unknown",
            message=f"腾讯云状态：{status or 'unknown'}",
        )

        if status in SUCCESS_STATUSES:
            return response
        if status in FAILED_STATUSES:
            fail(f"Job failed: {json.dumps(response, ensure_ascii=False)}")

        if time.time() - started > timeout:
            fail(f"Timed out after {timeout} seconds waiting for job {job_id}.")

        time.sleep(poll_interval)


def guess_suffix(url: str, fallback: str) -> str:
    path = parse.urlparse(url).path
    suffix = Path(path).suffix
    if suffix:
        return suffix
    if fallback:
        return f".{fallback.lower()}"
    return ""


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    insecure_context = ssl._create_unverified_context()
    with request.urlopen(req, timeout=120, context=insecure_context) as resp:
        target.write_bytes(resp.read())


def save_outputs(result: dict[str, Any], output_dir: Path) -> list[Path]:
    files = result.get("data")
    if not isinstance(files, list) or not files:
        fail(f"Job completed but no downloadable data found: {json.dumps(result)}")

    saved_paths: list[Path] = []
    for index, item in enumerate(files, start=1):
        asset_type = str(item.get("type", "asset")).lower()
        asset_url = item.get("url")
        preview_url = item.get("preview_image_url")
        if not asset_url:
            continue

        asset_suffix = guess_suffix(asset_url, asset_type)
        asset_path = output_dir / f"{index:02d}_{asset_type}{asset_suffix}"
        download_file(asset_url, asset_path)
        saved_paths.append(asset_path)

        if preview_url:
            preview_suffix = guess_suffix(preview_url, "png")
            preview_path = output_dir / f"{index:02d}_{asset_type}_preview{preview_suffix}"
            download_file(preview_url, preview_path)
            saved_paths.append(preview_path)

    if not saved_paths:
        fail(f"Job completed but no valid download URLs found: {json.dumps(result)}")

    return saved_paths


def to_relative_posix(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()


def write_latest_model_metadata(
    job_id: str,
    saved_paths: list[Path],
    *,
    metadata_path: Path = LATEST_MODEL_METADATA,
) -> tuple[Path, dict[str, Any]]:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = metadata_path.parent.resolve()

    glb_path = next((path for path in saved_paths if path.suffix.lower() == ".glb"), None)
    preview_path = next(
        (
            path
            for path in saved_paths
            if path.name.lower().endswith("_preview.png")
            or path.name.lower().endswith("_preview.jpg")
            or path.name.lower().endswith("_preview.jpeg")
            or path.name.lower().endswith("_preview.webp")
        ),
        None,
    )

    metadata = {
        "job_id": job_id,
        "model_url": to_relative_posix(glb_path, base_dir) if glb_path else "",
        "preview_url": to_relative_posix(preview_path, base_dir) if preview_path else "",
        "files": [to_relative_posix(path, base_dir) for path in saved_paths],
        "updated_at": int(time.time()),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path, metadata


def generate_3d(
    *,
    api_key: str,
    options: GenerationOptions,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    validate_options(options)

    notify(progress_callback, phase="submitting", message="正在提交 3D 任务...")
    submit_result = post_json(SUBMIT_URL, build_submit_payload(options), api_key)

    if submit_result.get("error"):
        fail(f"Submit failed: {json.dumps(submit_result, ensure_ascii=False)}")

    job_id = str(submit_result.get("id") or "")
    if not job_id:
        fail(f"Submit succeeded but no job id was returned: {json.dumps(submit_result)}")

    notify(
        progress_callback,
        phase="submitted",
        job_id=job_id,
        status=normalize_status(submit_result.get("status")) or "submitted",
        message=f"任务已提交，job_id={job_id}",
    )

    final_result = wait_for_completion(
        api_key=api_key,
        model=options.model,
        job_id=job_id,
        poll_interval=options.poll_interval,
        timeout=options.timeout,
        progress_callback=progress_callback,
    )

    job_output_dir = options.out_dir / job_id
    notify(progress_callback, phase="downloading", job_id=job_id, message="正在下载结果...")
    saved_paths = save_outputs(final_result, job_output_dir)
    metadata_path, metadata = write_latest_model_metadata(
        job_id,
        saved_paths,
        metadata_path=options.metadata_path,
    )

    notify(
        progress_callback,
        phase="completed",
        job_id=job_id,
        status="completed",
        message="3D 模型已生成。",
        result=metadata,
    )

    return {
        "job_id": job_id,
        "saved_paths": saved_paths,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }
