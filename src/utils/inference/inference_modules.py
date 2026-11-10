"""Inference helpers for the radar detector (Ultralytics YOLO26 checkpoint).

Coordinate conventions used throughout this module and in saved JSON:

* ``xyxy_px``       -- absolute pixel edge coordinates ``(x1, y1, x2, y2)`` in the
                       *original* image, origin at the top-left, clipped to
                       ``0 <= x <= width`` and ``0 <= y <= height``.
* ``crop_xyxy_px``  -- the same box in pixel coordinates of the masked circle crop
                       that is fed to the model (before the offset is added back).
* Normalized ratios (``xyxyn``, values in 0..1) are never written to outputs.

Paths given as relative strings are resolved against the repository root (not the
current working directory). Absolute paths are accepted as explicit external paths.

Ultralytics is imported lazily inside :func:`load_model`, so the geometry/path/
saving logic can be unit-tested without torch or a model checkpoint.
"""
from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
CIRCLE_KEYS = ("x_centre", "y_centre", "radius")
LEGACY_CIRCLE_KEY = "old_radar"
ANNOTATED_SUFFIX = ".annotated.png"
DETECTIONS_SUFFIX = ".detections.json"
COORDINATE_NOTE = (
    "xyxy_px: absolute pixel coordinates (x1, y1, x2, y2) in the original image, "
    "clipped to the image bounds. crop_xyxy_px: pixel coordinates inside the masked "
    "circle crop given to the model. No normalized (0..1) ratios are stored."
)


class InferenceError(Exception):
    """Base class for actionable inference errors."""


class InferenceConfigError(InferenceError, ValueError):
    """Invalid or incomplete configuration / path value."""


class GeometryError(InferenceError, ValueError):
    """Invalid circle geometry or crop that does not fit inside the image."""


class ImageReadError(InferenceError, ValueError):
    """An input image is empty or cannot be decoded."""


# --------------------------------------------------------------------------- paths
def resolve_path(
    value: Any,
    *,
    label: str,
    kind: str = "file",
    must_exist: bool = True,
    base: Path = REPO_ROOT,
    hint: str = "",
) -> Path:
    """Resolve a configured path consistently.

    Relative paths are resolved against ``base`` (the repository root) and may not
    use ``..`` to leave it; pass an absolute path for files outside the repository.

    :param kind: ``"file"``, ``"dir"`` or ``"any"`` (checked only if ``must_exist``).
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        raise InferenceConfigError(f"{label} is not set. {hint}".strip())
    if not isinstance(value, (str, Path)):
        raise InferenceConfigError(f"{label} must be a path string, got {value!r}.")

    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        if ".." in path.parts:
            raise InferenceConfigError(
                f"{label}={value!r}: relative paths are resolved from the repository root "
                f"({base}) and must not contain '..'. Use an absolute path for files "
                "outside the repository."
            )
        path = base / path
    path = path.resolve()

    suffix = f" {hint}" if hint else ""
    if must_exist:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}.{suffix}")
        if kind == "file" and not path.is_file():
            raise InferenceConfigError(f"{label} must be a file: {path}.{suffix}")
        if kind == "dir" and not path.is_dir():
            raise InferenceConfigError(f"{label} must be a directory: {path}.{suffix}")
    return path


def display_path(path: Path, base: Path = REPO_ROOT) -> str:
    """Repository-relative POSIX path when possible, otherwise the absolute path."""
    path = Path(path)
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def collect_images(source: Path, exclude_dir: Path | None = None) -> list[Path]:
    """Return the image(s) to process from a single file or a directory (recursive).

    Files inside ``exclude_dir`` (e.g. a nested output folder) are skipped so that
    previous annotated outputs are never re-processed.
    """
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise InferenceConfigError(
                f"Unsupported image extension for {source}; expected one of {IMAGE_EXTENSIONS}."
            )
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Inference source not found: {source}")

    exclude = exclude_dir.resolve() if exclude_dir is not None else None
    images = []
    for candidate in sorted(source.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if exclude is not None and candidate.resolve().is_relative_to(exclude):
            continue
        images.append(candidate)
    if not images:
        raise InferenceConfigError(
            f"No input images ({', '.join(IMAGE_EXTENSIONS)}) found in directory {source}."
        )
    return images


def read_image(path: Path) -> np.ndarray:
    """Read an image as a 3-channel BGR array; raise :class:`ImageReadError` on failure."""
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ImageReadError(f"Cannot read image file {path}: {exc}") from exc
    if data.size == 0:
        raise ImageReadError(f"Image file is empty: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageReadError(f"Image file is unreadable or not a supported image: {path}")
    return image


# ------------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class CircleGeometry:
    """Radar scan circle in integer pixel coordinates of the original image."""

    x_centre: int
    y_centre: int
    radius: int

    def crop_box(self) -> tuple[int, int, int, int]:
        """Crop box ``(x0, y0, x1, y1)`` in pixels; ``x1``/``y1`` are exclusive."""
        r = self.radius
        return (self.x_centre - r, self.y_centre - r, self.x_centre + r + 1, self.y_centre + r + 1)

    def to_dict(self) -> dict[str, int]:
        return {"x_centre": self.x_centre, "y_centre": self.y_centre, "radius": self.radius}


def _as_pixel_int(name: str, value: Any, source: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise GeometryError(f"{source}: '{name}' must be an integer pixel value, got {value!r}.")
    if isinstance(value, numbers.Integral):
        return int(value)
    value = float(value)
    if not math.isfinite(value) or not value.is_integer():
        raise GeometryError(f"{source}: '{name}' must be a finite integer pixel value, got {value!r}.")
    return int(value)


def parse_circle_geometry(data: Any, source: str = "circle JSON") -> CircleGeometry:
    """Accept the flat layout ``{"x_centre", "y_centre", "radius"}`` or the prior nested
    layout ``{"old_radar": {...}}`` (or any single nested entry with those keys)."""
    if not isinstance(data, Mapping):
        raise GeometryError(f"{source}: expected a JSON object, got {type(data).__name__}.")

    if all(key in data for key in CIRCLE_KEYS):
        entry = data
    elif isinstance(data.get(LEGACY_CIRCLE_KEY), Mapping):
        entry = data[LEGACY_CIRCLE_KEY]
    else:
        nested = [v for v in data.values() if isinstance(v, Mapping) and all(k in v for k in CIRCLE_KEYS)]
        if len(nested) != 1:
            raise GeometryError(
                f"{source}: expected keys {list(CIRCLE_KEYS)} at top level or under "
                f"'{LEGACY_CIRCLE_KEY}'; found top-level keys {sorted(map(str, data.keys()))}."
            )
        entry = nested[0]

    missing = [k for k in CIRCLE_KEYS if k not in entry]
    if missing:
        raise GeometryError(f"{source}: missing circle keys {missing}.")

    x_centre = _as_pixel_int("x_centre", entry["x_centre"], source)
    y_centre = _as_pixel_int("y_centre", entry["y_centre"], source)
    radius = _as_pixel_int("radius", entry["radius"], source)
    if radius <= 0:
        raise GeometryError(f"{source}: 'radius' must be > 0, got {radius}.")
    if x_centre < 0 or y_centre < 0:
        raise GeometryError(f"{source}: circle centre must be non-negative, got ({x_centre}, {y_centre}).")
    return CircleGeometry(x_centre, y_centre, radius)


def load_circle_geometry(path: Path) -> CircleGeometry:
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise InferenceConfigError(f"Circle coordinate JSON is not valid JSON: {path} ({exc}).") from exc
    return parse_circle_geometry(data, source=str(path))


def validate_crop(geometry: CircleGeometry, image_shape: Sequence[int]) -> tuple[int, int, int, int]:
    """Return the crop box if it lies fully inside the image, else raise GeometryError."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x0, y0, x1, y1 = geometry.crop_box()
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise GeometryError(
            f"Circle {geometry.to_dict()} needs crop box xyxy_px=({x0}, {y0}, {x1}, {y1}) "
            f"which is outside the {width}x{height} image. Check that the circle JSON "
            "matches this radar layout/resolution."
        )
    return x0, y0, x1, y1


def crop_and_mask(image: np.ndarray, geometry: CircleGeometry) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop the circle's bounding square and zero every pixel outside the scan circle.

    The input image is not modified. Returns ``(masked_crop, crop_box)``.
    """
    box = validate_crop(geometry, image.shape)
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    r = geometry.radius
    cv2.circle(mask, (r, r), r, 255, thickness=-1)
    crop[mask == 0] = 0
    return crop, box


# ---------------------------------------------------------------------- detections
@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy_px: tuple[float, float, float, float]
    crop_xyxy_px: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy_px": list(self.xyxy_px),
            "crop_xyxy_px": list(self.crop_xyxy_px),
        }


def _to_list(values: Any) -> list:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, class_id))
    if isinstance(names, Sequence) and not isinstance(names, str) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def raw_detections_from_result(result: Any) -> list[dict[str, Any]]:
    """Extract crop-pixel boxes from an Ultralytics ``Results`` object (or a look-alike
    with ``boxes.xyxy``, ``boxes.conf``, ``boxes.cls`` and ``names``)."""
    if result is None:
        return []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _to_list(boxes.xyxy)
    conf = _to_list(boxes.conf)
    cls = _to_list(boxes.cls)
    if not (len(xyxy) == len(conf) == len(cls)):
        raise ValueError(f"Inconsistent model output lengths: xyxy={len(xyxy)}, conf={len(conf)}, cls={len(cls)}.")
    names = getattr(result, "names", None) or {}

    raw = []
    for coords, confidence, class_value in zip(xyxy, conf, cls):
        coords = list(coords)
        if len(coords) != 4:
            raise ValueError(f"Expected 4 xyxy pixel values per box, got {coords!r}.")
        class_id = int(class_value)
        raw.append({
            "crop_xyxy_px": [float(v) for v in coords],
            "confidence": float(confidence),
            "class_id": class_id,
            "class_name": _class_name(names, class_id),
        })
    return raw


def crop_to_original(
    raw_detections: Sequence[Mapping[str, Any]],
    crop_box: Sequence[int],
    image_shape: Sequence[int],
) -> list[Detection]:
    """Map crop-pixel xyxy boxes back to original-image pixels.

    Adds the crop offset ``(x0, y0)``, clips to ``[0, width] x [0, height]`` and drops
    boxes with non-finite values or zero area after clipping.
    """
    x_off, y_off = int(crop_box[0]), int(crop_box[1])
    height, width = int(image_shape[0]), int(image_shape[1])
    detections = []
    for raw in raw_detections:
        cx1, cy1, cx2, cy2 = (float(v) for v in raw["crop_xyxy_px"])
        confidence = float(raw["confidence"])
        if not all(math.isfinite(v) for v in (cx1, cy1, cx2, cy2, confidence)):
            continue
        cx1, cx2 = min(cx1, cx2), max(cx1, cx2)
        cy1, cy2 = min(cy1, cy2), max(cy1, cy2)
        x1 = min(max(cx1 + x_off, 0.0), float(width))
        x2 = min(max(cx2 + x_off, 0.0), float(width))
        y1 = min(max(cy1 + y_off, 0.0), float(height))
        y2 = min(max(cy2 + y_off, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection(
            class_id=int(raw["class_id"]),
            class_name=str(raw["class_name"]),
            confidence=round(confidence, 6),
            xyxy_px=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
            crop_xyxy_px=(round(cx1, 2), round(cy1, 2), round(cx2, 2), round(cy2, 2)),
        ))
    return detections


def draw_detections(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Return an annotated copy of ``image``; the original pixels are left untouched."""
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det.xyxy_px
        p1 = (min(int(round(x1)), width - 1), min(int(round(y1)), height - 1))
        p2 = (min(int(round(x2)), width - 1), min(int(round(y2)), height - 1))
        cv2.rectangle(annotated, p1, p2, (0, 0, 255), 2)
        text_y = p1[1] - 4 if p1[1] >= 14 else min(p2[1] + 14, height - 1)
        cv2.putText(
            annotated,
            f"{det.class_name} {det.confidence:.2f}",
            (p1[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


# ------------------------------------------------------------------ model + output
def load_model(weight_path: Path):
    """Load an existing local Ultralytics checkpoint. Never downloads or trains."""
    weight_path = Path(weight_path)
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weight_path}. This tool does not download or "
            "retrain weights; point inference.weight / --weights to an existing .pt file."
        )
    if weight_path.suffix != ".pt":
        raise InferenceConfigError(f"Expected a .pt Ultralytics checkpoint, got {weight_path}.")
    from ultralytics import YOLO  # lazy: keeps unit tests independent of torch

    return YOLO(str(weight_path))


def predict(model: Any, model_input: np.ndarray, conf: float) -> Any:
    """Run the model on one BGR array and return the first result (or ``None``)."""
    results = model.predict(model_input, conf=conf, verbose=False)
    if results is None:
        return None
    results = list(results)
    return results[0] if results else None


def output_paths(image_path: Path, output_dir: Path, source_root: Path | None = None) -> tuple[Path, Path]:
    """Stable output names that keep the full input filename (including dots).

    ``radar2.png`` -> ``radar2.png.annotated.png`` and ``radar2.png.detections.json``.
    For directory inputs, sub-directories are flattened with ``__`` to avoid collisions.
    """
    image_path = Path(image_path)
    if source_root is not None and Path(source_root).is_dir():
        name = "__".join(image_path.resolve().relative_to(Path(source_root).resolve()).parts)
    else:
        name = image_path.name
    return output_dir / f"{name}{ANNOTATED_SUFFIX}", output_dir / f"{name}{DETECTIONS_SUFFIX}"


def build_record(
    image_path: Path,
    image_shape: Sequence[int],
    geometry: CircleGeometry,
    crop_box: Sequence[int],
    detections: Sequence[Detection],
    weight_path: Path | None,
    grayscale: bool,
    conf: float,
) -> dict[str, Any]:
    return {
        "source_image": display_path(image_path),
        "image_size_px": {"width": int(image_shape[1]), "height": int(image_shape[0])},
        "weights": display_path(weight_path) if weight_path is not None else None,
        "circle_px": geometry.to_dict(),
        "crop_box_xyxy_px": [int(v) for v in crop_box],
        "model_input": {"masked_outside_circle": True, "grayscale": bool(grayscale), "conf_threshold": conf},
        "coordinate_space": COORDINATE_NOTE,
        "detection_count": len(detections),
        "detections": [d.to_dict() for d in detections],
    }


def save_outputs(annotated: np.ndarray, record: Mapping[str, Any], png_path: Path, json_path: Path) -> None:
    """Write the annotated PNG and a strict (NaN-free) JSON sidecar. No GUI involved."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", annotated)
    if not ok:
        raise OSError(f"Failed to encode annotated PNG for {png_path}.")
    payload = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
    png_path.write_bytes(buffer.tobytes())
    json_path.write_text(payload + "\n", encoding="utf-8")


def show_image(image: np.ndarray, title: str) -> None:
    """Opt-in preview window. Fails with an actionable message on headless OpenCV."""
    try:
        cv2.imshow(title, image)
        cv2.waitKey(0)
    except cv2.error as exc:
        raise RuntimeError(
            "Showing images needs a GUI-capable OpenCV build and a display. "
            "Run without --show (outputs are still saved)."
        ) from exc


def process_image(
    model: Any,
    image_path: Path,
    geometry: CircleGeometry,
    *,
    conf: float,
    grayscale: bool = False,
    weight_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one image end-to-end and return ``(annotated_image, json_record)``."""
    original = read_image(image_path)
    if grayscale:
        model_source = cv2.cvtColor(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    else:
        model_source = original
    crop, crop_box = crop_and_mask(model_source, geometry)
    result = predict(model, crop, conf)
    detections = crop_to_original(raw_detections_from_result(result), crop_box, original.shape)
    annotated = draw_detections(original, detections)
    record = build_record(image_path, original.shape, geometry, crop_box, detections, weight_path, grayscale, conf)
    return annotated, record
