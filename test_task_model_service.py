#!/usr/bin/env python3
"""Focused local tests for the backend-facing task wrapper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from task_model_service import (
    TaskGenerationConfig,
    TaskPayloadError,
    build_mock_object_generator,
    process_generation_task,
)


class TaskModelServiceTests(unittest.TestCase):
    def test_process_generation_task_with_mock_generator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "generated"
            payload = {
                "task_id": "task_001",
                "objects": [
                    {
                        "object_id": "obj_001",
                        "label": "laptop",
                        "crop_url": "input_images/pencil.png",
                        "need_generation": True,
                        "model_url": "",
                    },
                    {
                        "object_id": "obj_002",
                        "label": "cup",
                        "crop_url": "",
                        "need_generation": False,
                        "model_url": "https://example.com/assets/models/cup_001.glb",
                    },
                ],
            }

            result = process_generation_task(
                payload,
                config=TaskGenerationConfig(output_dir=output_dir),
                object_generator=build_mock_object_generator(),
            )

            self.assertEqual(result["task_id"], "task_001")
            self.assertEqual(len(result["generated_models"]), 1)
            generated = result["generated_models"][0]
            self.assertEqual(generated["object_id"], "obj_001")
            self.assertEqual(generated["label"], "laptop")

            model_path = Path(generated["model_url"])
            self.assertTrue(model_path.exists())
            self.assertEqual(model_path.suffix, ".glb")
            self.assertTrue((output_dir / "task_001" / "response.json").exists())

    def test_parse_invalid_payload_raises(self) -> None:
        payload = {
            "task_id": "task_001",
            "objects": [
                {
                    "object_id": "",
                    "label": "laptop",
                    "crop_url": "input_images/pencil.png",
                    "need_generation": True,
                    "model_url": "",
                }
            ],
        }

        with self.assertRaises(TaskPayloadError):
            process_generation_task(
                payload,
                config=TaskGenerationConfig(output_dir=Path("unused")),
                object_generator=build_mock_object_generator(),
            )


if __name__ == "__main__":
    unittest.main()
