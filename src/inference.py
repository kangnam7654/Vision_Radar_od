"""Run radar object detection on the public sample image (or a user-supplied image/folder).

Default (from the repository root)::

    uv run python src/inference.py

uses ``src/yaml/radar_config.yaml`` -> ``inference`` section:
``sample_images/radar2.png`` + ``runs/detect/train4/weights/best.pt`` +
``src/test_inference/test_circle_coordinates_old.json`` and writes
``outputs/inference/radar2.png.annotated.png`` and ``radar2.png.detections.json``.

Relative paths are resolved from the repository root. No GUI window is opened unless
``--show`` (or ``show: true``) is set explicitly.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.inference import inference_modules as im  # noqa: E402

DEFAULT_CONFIG = SRC_DIR / "yaml" / "radar_config.yaml"
WEIGHT_HINT = (
    "Expected the committed checkpoint runs/detect/train4/weights/best.pt. "
    "Weights are never downloaded or retrained automatically."
)


@dataclass
class InferenceSettings:
    source: Any
    weight: Any
    circle_json: Any
    save_dir: Any = "outputs/inference"
    save: bool = True
    show: bool = False
    conf: float = 0.25
    grayscale: bool = False


def load_settings(config_path: Any = None) -> InferenceSettings:
    path = DEFAULT_CONFIG if config_path is None else im.resolve_path(config_path, label="--config")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise im.InferenceConfigError(f"Config file is not valid YAML: {path} ({exc}).") from exc
    section = config.get("inference") if isinstance(config, dict) else None
    if not isinstance(section, dict):
        raise im.InferenceConfigError(f"Config {path} has no 'inference' mapping.")

    allowed = set(InferenceSettings.__dataclass_fields__)
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise im.InferenceConfigError(
            f"Unknown inference config key(s) {unknown} in {path}; allowed: {sorted(allowed)}."
        )
    missing = [k for k in ("source", "weight", "circle_json") if section.get(k) in (None, "")]
    if missing:
        raise im.InferenceConfigError(f"Config {path} is missing inference.{', inference.'.join(missing)}.")
    return InferenceSettings(**section)


def validate_settings(settings: InferenceSettings) -> None:
    for name in ("save", "show", "grayscale"):
        if not isinstance(getattr(settings, name), bool):
            raise im.InferenceConfigError(f"inference.{name} must be true or false, got {getattr(settings, name)!r}.")
    conf = settings.conf
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf) or not 0.0 <= conf <= 1.0:
        raise im.InferenceConfigError(f"inference.conf must be a number in [0, 1], got {conf!r}.")
    if not settings.save and not settings.show:
        raise im.InferenceConfigError("Nothing to do: both save and show are disabled.")


def run(settings: InferenceSettings, model: Any = None) -> list[dict[str, Any]]:
    """Run inference and return one summary dict per image.

    ``model`` may be injected (used by unit tests with a fake model); otherwise the
    configured local checkpoint is loaded.
    """
    validate_settings(settings)
    source = im.resolve_path(settings.source, label="inference.source / --source", kind="any")
    weight = im.resolve_path(settings.weight, label="inference.weight / --weights", hint=WEIGHT_HINT)
    circle_path = im.resolve_path(settings.circle_json, label="inference.circle_json / --circle-json")
    geometry = im.load_circle_geometry(circle_path)
    output_dir = (
        im.resolve_path(settings.save_dir, label="inference.save_dir / --output-dir", must_exist=False)
        if settings.save
        else None
    )
    if output_dir is not None and output_dir.exists() and not output_dir.is_dir():
        raise im.InferenceConfigError(f"Output path exists and is not a directory: {output_dir}")

    images = im.collect_images(source, exclude_dir=output_dir)
    if model is None:
        model = im.load_model(weight)

    summaries = []
    try:
        for image_path in images:
            annotated, record = im.process_image(
                model, image_path, geometry, conf=float(settings.conf),
                grayscale=settings.grayscale, weight_path=weight,
            )
            summary = {"image": image_path, "detection_count": record["detection_count"], "record": record}
            if output_dir is not None:
                png_path, json_path = im.output_paths(image_path, output_dir, source if source.is_dir() else None)
                im.save_outputs(annotated, record, png_path, json_path)
                summary.update(png=png_path, json=json_path)
                print(f"{im.display_path(image_path)}: {record['detection_count']} detection(s) -> "
                      f"{im.display_path(png_path)}, {im.display_path(json_path)}")
            else:
                print(f"{im.display_path(image_path)}: {record['detection_count']} detection(s)")
            if settings.show:
                im.show_image(annotated, image_path.name)
            summaries.append(summary)
    finally:
        if settings.show:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:  # headless build: nothing to close
                pass
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", help="YAML config (default: src/yaml/radar_config.yaml).")
    parser.add_argument("--source", help="Image file or directory (relative to repo root, or absolute).")
    parser.add_argument("--weights", help="Existing Ultralytics .pt checkpoint.")
    parser.add_argument("--circle-json", help="Circle JSON: flat {x_centre, y_centre, radius} or nested 'old_radar'.")
    parser.add_argument("--output-dir", help="Output folder for annotated PNG + JSON (default: outputs/inference).")
    parser.add_argument("--conf", type=float, help="Confidence threshold in [0, 1].")
    parser.add_argument("--show", action="store_true", help="Opt-in: open a preview window per image (needs a GUI).")
    parser.add_argument("--no-save", action="store_true", help="Do not write outputs (only useful with --show).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
        overrides = {
            "source": args.source, "weight": args.weights, "circle_json": args.circle_json,
            "save_dir": args.output_dir, "conf": args.conf,
        }
        for key, value in overrides.items():
            if value is not None:
                setattr(settings, key, value)
        if args.show:
            settings.show = True
        if args.no_save:
            settings.save = False
        run(settings)
    except (im.InferenceError, FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
