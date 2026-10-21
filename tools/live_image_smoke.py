"""Run F03/F04 vision mapping locally, without a business backend.

Use only consented or synthetic images. The API key comes from GCA_API_KEY and
is never printed. Replay output is kept in a temporary directory and removed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gutcare_agents.config import load_settings  # noqa: E402
from gutcare_agents.image_adapter import ImageProcessingError, _run_one  # noqa: E402


def _image(path: Path, max_bytes: int) -> tuple[bytes, str]:
    data = path.read_bytes()
    if not data or len(data) > max_bytes:
        raise ValueError("image is empty or exceeds the configured size limit")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return data, "image/png"
    if data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9"):
        return data, "image/jpeg"
    raise ValueError("only PNG and JPEG images are supported")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test F03/F04 image agents in live mode without the backend")
    parser.add_argument("--diet-image", type=Path, help="consented or synthetic meal image")
    parser.add_argument("--stool-image", type=Path, help="consented or synthetic stool image")
    args = parser.parse_args()
    if not args.diet_image and not args.stool_image:
        parser.error("provide --diet-image and/or --stool-image")
    settings = load_settings(mode="live")
    if not settings.api_key:
        parser.error("GCA_API_KEY is not set; configure it locally, not in the repository")
    if not os.environ.get("GCA_BASE_URL") or not os.environ.get("GCA_VISION_MODEL"):
        parser.error("set GCA_BASE_URL and GCA_VISION_MODEL explicitly for live testing")
    results = []
    with tempfile.TemporaryDirectory(prefix="gutcare-live-replay-") as replay_dir:
        settings.replay_dir = Path(replay_dir)
        for recognition_type, path in (("diet_image_recognition", args.diet_image),
                                       ("stool_image_recognition", args.stool_image)):
            if path is None:
                continue
            image, mime = _image(path, settings.max_image_bytes)
            started = time.monotonic()
            request_id, attachment_id = str(uuid.uuid4()), str(uuid.uuid4())
            result = _run_one(settings, request_id, recognition_type,
                              {"attachment_id": attachment_id, "content_type": mime}, image)
            results.append({"recognition_type": recognition_type, "duration_seconds": round(time.monotonic() - started, 2),
                            "model": settings.vision_model, "result": result})
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImageProcessingError) as exc:
        print(f"Live image smoke test failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1) from exc
