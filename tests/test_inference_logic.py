"""Deterministic unit tests for the inference *logic* (paths, geometry, coordinate
mapping, saving). They use temporary synthetic images and a fake model output only.

These are NOT real model inference tests: no checkpoint is loaded, Ultralytics/torch
are not imported, and nothing here says anything about detection quality.

Run from the repository root::

    uv run python -m unittest discover -s tests -v
"""
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import inference  # noqa: E402
from utils.inference import inference_modules as im  # noqa: E402


# ----------------------------------------------------------------------- fakes
class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)


class FakeResult:
    def __init__(self, xyxy=(), conf=(), cls=(), names=None):
        self.boxes = FakeBoxes(xyxy, conf, cls)
        self.names = names if names is not None else {0: "boat_rectangle", 1: "boat_triangle", 2: "end", 3: "start"}


class FakeModel:
    """Mimics ``YOLO.predict`` returning a list with one Results-like object."""

    def __init__(self, result=None):
        self.result = result if result is not None else FakeResult()
        self.inputs = []

    def predict(self, image, conf=0.25, verbose=False):
        self.inputs.append(image.copy())
        return [self.result]


def write_image(path, width=120, height=100, value=200):
    image = np.full((height, width, 3), value, dtype=np.uint8)
    image[:, :, 0] = 50  # make channels differ so grayscale conversion is observable
    assert cv2.imwrite(str(path), image)
    return image


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def make_settings(self, source, circle=None, **kwargs):
        weight = self.tmp / "fake_weights.pt"
        weight.write_bytes(b"not a real checkpoint; model is injected")
        if circle is None:
            circle = self.tmp / "circle.json"
            circle.write_text(json.dumps({"x_centre": 50, "y_centre": 40, "radius": 30}))
        values = dict(source=str(source), weight=str(weight), circle_json=str(circle),
                      save_dir=str(self.tmp / "out"), save=True, show=False, conf=0.25, grayscale=False)
        values.update(kwargs)
        return inference.InferenceSettings(**values)


# ------------------------------------------------------------------ path tests
class PathResolutionTests(TempDirTestCase):
    def test_relative_path_is_resolved_from_repo_root(self):
        path = im.resolve_path("sample_images/radar2.png", label="source")
        self.assertEqual(path, (REPO_ROOT / "sample_images" / "radar2.png").resolve())

    def test_absolute_external_path_is_accepted(self):
        target = self.tmp / "external.png"
        write_image(target)
        self.assertEqual(im.resolve_path(str(target), label="source"), target)

    def test_parent_escape_in_relative_path_is_rejected(self):
        with self.assertRaises(im.InferenceConfigError) as ctx:
            im.resolve_path("../outside.png", label="source")
        self.assertIn("absolute path", str(ctx.exception))

    def test_missing_file_has_actionable_message(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            im.resolve_path(str(self.tmp / "nope.pt"), label="inference.weight", hint="never downloaded")
        self.assertIn("inference.weight", str(ctx.exception))
        self.assertIn("never downloaded", str(ctx.exception))

    def test_empty_value_and_wrong_kind_are_rejected(self):
        with self.assertRaises(im.InferenceConfigError):
            im.resolve_path("", label="source")
        with self.assertRaises(im.InferenceConfigError):
            im.resolve_path(str(self.tmp), label="weight", kind="file")

    def test_default_config_points_to_committed_public_files(self):
        settings = inference.load_settings()
        source = im.resolve_path(settings.source, label="source")
        weight = im.resolve_path(settings.weight, label="weight")
        circle = im.resolve_path(settings.circle_json, label="circle")
        self.assertEqual(im.display_path(source), "sample_images/radar2.png")
        self.assertEqual(im.display_path(weight), "runs/detect/train4/weights/best.pt")
        self.assertEqual(im.load_circle_geometry(circle), im.CircleGeometry(438, 348, 339))
        self.assertFalse(settings.show)
        self.assertTrue(settings.save)

    def test_default_circle_crop_fits_the_committed_sample(self):
        image = im.read_image(REPO_ROOT / "sample_images" / "radar2.png")
        box = im.validate_crop(im.CircleGeometry(438, 348, 339), image.shape)
        self.assertEqual(box, (99, 9, 778, 688))


# -------------------------------------------------------------- geometry tests
class GeometryTests(TempDirTestCase):
    def test_flat_and_nested_layouts(self):
        flat = {"x_centre": 438, "y_centre": 348, "radius": 339}
        expected = im.CircleGeometry(438, 348, 339)
        self.assertEqual(im.parse_circle_geometry(flat), expected)
        self.assertEqual(im.parse_circle_geometry({"old_radar": flat}), expected)
        self.assertEqual(im.parse_circle_geometry({"x_centre": 438.0, "y_centre": 348, "radius": 339.0}), expected)

    def test_invalid_values_are_rejected(self):
        bad_values = [10.5, float("nan"), float("inf"), True, "10", None]
        for bad in bad_values:
            with self.subTest(bad=bad), self.assertRaises(im.GeometryError):
                im.parse_circle_geometry({"x_centre": bad, "y_centre": 10, "radius": 5})
        for radius in (0, -3):
            with self.subTest(radius=radius), self.assertRaises(im.GeometryError):
                im.parse_circle_geometry({"x_centre": 10, "y_centre": 10, "radius": radius})

    def test_missing_or_ambiguous_layout_is_rejected(self):
        with self.assertRaises(im.GeometryError):
            im.parse_circle_geometry({"x_centre": 1, "radius": 2})
        entry = {"x_centre": 1, "y_centre": 1, "radius": 1}
        with self.assertRaises(im.GeometryError):
            im.parse_circle_geometry({"a": entry, "b": entry})
        with self.assertRaises(im.GeometryError):
            im.parse_circle_geometry([entry])

    def test_invalid_json_file(self):
        path = self.tmp / "broken.json"
        path.write_text("{not json")
        with self.assertRaises(im.InferenceConfigError):
            im.load_circle_geometry(path)

    def test_crop_shape_mask_and_original_untouched(self):
        image = np.full((100, 120, 3), 200, dtype=np.uint8)
        before = image.copy()
        geometry = im.CircleGeometry(50, 40, 30)
        crop, box = im.crop_and_mask(image, geometry)
        self.assertEqual(box, (20, 10, 81, 71))
        self.assertEqual(crop.shape, (61, 61, 3))
        self.assertTrue((crop[0, 0] == 0).all())            # outside circle -> masked
        self.assertTrue((crop[60, 60] == 0).all())
        self.assertTrue((crop[30, 30] == 200).all())        # centre kept
        self.assertTrue((crop[30, 2] == 200).all())         # just inside the circle edge kept
        np.testing.assert_array_equal(image, before)

    def test_crop_touching_image_edge_is_allowed(self):
        image = np.zeros((61, 61, 3), dtype=np.uint8)
        crop, box = im.crop_and_mask(image, im.CircleGeometry(30, 30, 30))
        self.assertEqual(box, (0, 0, 61, 61))
        self.assertEqual(crop.shape[:2], (61, 61))

    def test_out_of_frame_crop_is_rejected(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        for geometry in (im.CircleGeometry(10, 40, 30), im.CircleGeometry(50, 80, 30), im.CircleGeometry(100, 40, 30)):
            with self.subTest(geometry=geometry), self.assertRaises(im.GeometryError) as ctx:
                im.crop_and_mask(image, geometry)
            self.assertIn("120x100", str(ctx.exception))


# -------------------------------------------------------- coordinate transforms
class CoordinateTransformTests(unittest.TestCase):
    def test_offset_clipping_and_filtering(self):
        raw = [
            {"crop_xyxy_px": [5, 6, 15, 16], "confidence": 0.9, "class_id": 0, "class_name": "boat_rectangle"},
            {"crop_xyxy_px": [-30, -30, 10, 10], "confidence": 0.5, "class_id": 3, "class_name": "start"},
            {"crop_xyxy_px": [90, 80, 200, 200], "confidence": 0.4, "class_id": 2, "class_name": "end"},
            {"crop_xyxy_px": [15, 16, 5, 6], "confidence": 0.3, "class_id": 1, "class_name": "boat_triangle"},
            {"crop_xyxy_px": [1, 1, float("nan"), 3], "confidence": 0.8, "class_id": 0, "class_name": "x"},
            {"crop_xyxy_px": [1, 1, 3, 3], "confidence": float("inf"), "class_id": 0, "class_name": "x"},
            {"crop_xyxy_px": [-50, -50, -25, -25], "confidence": 0.7, "class_id": 0, "class_name": "x"},
        ]
        dets = im.crop_to_original(raw, (20, 10, 81, 71), (100, 120, 3))
        self.assertEqual(len(dets), 4)
        self.assertEqual(dets[0].xyxy_px, (25.0, 16.0, 35.0, 26.0))
        self.assertEqual(dets[0].crop_xyxy_px, (5.0, 6.0, 15.0, 16.0))
        self.assertEqual(dets[1].xyxy_px, (0.0, 0.0, 30.0, 20.0))           # clipped at 0
        self.assertEqual(dets[2].xyxy_px, (110.0, 90.0, 120.0, 100.0))      # clipped at width/height
        self.assertEqual(dets[3].xyxy_px, (25.0, 16.0, 35.0, 26.0))         # swapped corners normalised
        self.assertEqual([d.class_name for d in dets], ["boat_rectangle", "start", "end", "boat_triangle"])
        self.assertEqual([d.class_id for d in dets], [0, 3, 2, 1])
        self.assertTrue(all(math.isfinite(d.confidence) for d in dets))

    def test_raw_detections_from_fake_result(self):
        result = FakeResult(xyxy=[[1, 2, 3, 4], [5, 6, 7, 8]], conf=[0.9, 0.1], cls=[2, 7])
        raw = im.raw_detections_from_result(result)
        self.assertEqual(raw[0]["crop_xyxy_px"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(raw[0]["class_name"], "end")
        self.assertEqual(raw[1]["class_name"], "7")  # unknown id falls back to the number
        list_names = FakeResult(xyxy=[[1, 2, 3, 4]], conf=[0.5], cls=[1], names=["a", "b"])
        self.assertEqual(im.raw_detections_from_result(list_names)[0]["class_name"], "b")

    def test_empty_and_inconsistent_results(self):
        self.assertEqual(im.raw_detections_from_result(None), [])
        self.assertEqual(im.raw_detections_from_result(FakeResult()), [])
        bad = FakeResult(xyxy=[[1, 2, 3, 4]], conf=[0.5, 0.6], cls=[0])
        with self.assertRaises(ValueError):
            im.raw_detections_from_result(bad)


# ------------------------------------------------------------- pipeline + saving
class PipelineTests(TempDirTestCase):
    def test_empty_result_writes_valid_json_and_png_without_gui(self):
        source = self.tmp / "scan.v2.png"
        original = write_image(source)
        model = FakeModel(FakeResult())
        with mock.patch.object(cv2, "imshow", side_effect=AssertionError("imshow called")), \
                mock.patch.object(cv2, "waitKey", side_effect=AssertionError("waitKey called")), \
                redirect_stdout(io.StringIO()):
            summaries = inference.run(self.make_settings(source), model=model)

        png = self.tmp / "out" / "scan.v2.png.annotated.png"
        sidecar = self.tmp / "out" / "scan.v2.png.detections.json"
        self.assertEqual(summaries[0]["png"], png)
        self.assertTrue(png.is_file() and sidecar.is_file())
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(record["detections"], [])
        self.assertEqual(record["detection_count"], 0)
        self.assertEqual(record["crop_box_xyxy_px"], [20, 10, 81, 71])
        self.assertEqual(record["image_size_px"], {"width": 120, "height": 100})
        saved = cv2.imread(str(png))
        np.testing.assert_array_equal(saved, original)  # no boxes -> original pixels
        self.assertEqual(model.inputs[0].shape, (61, 61, 3))

    def test_detections_saved_in_original_pixels_and_drawn_on_copy(self):
        source = self.tmp / "frame.png"
        write_image(source)
        model = FakeModel(FakeResult(xyxy=[[5, 6, 15, 16]], conf=[0.875], cls=[3]))
        with redirect_stdout(io.StringIO()):
            summaries = inference.run(self.make_settings(source), model=model)
        record = json.loads(summaries[0]["json"].read_text(encoding="utf-8"))
        det = record["detections"][0]
        self.assertEqual(det["xyxy_px"], [25.0, 16.0, 35.0, 26.0])
        self.assertEqual(det["crop_xyxy_px"], [5.0, 6.0, 15.0, 16.0])
        self.assertEqual((det["class_id"], det["class_name"]), (3, "start"))
        self.assertAlmostEqual(det["confidence"], 0.875)
        saved = cv2.imread(str(summaries[0]["png"]))
        self.assertFalse((saved == cv2.imread(str(source))).all())  # a box was drawn

    def test_grayscale_option_changes_model_input_only(self):
        source = self.tmp / "frame.png"
        write_image(source)
        model = FakeModel()
        with redirect_stdout(io.StringIO()):
            inference.run(self.make_settings(source, grayscale=True), model=model)
        centre = model.inputs[0][30, 30]
        self.assertTrue(centre[0] == centre[1] == centre[2])

    def test_show_is_opt_in(self):
        source = self.tmp / "frame.png"
        write_image(source)
        with mock.patch.object(im, "show_image") as show, redirect_stdout(io.StringIO()):
            inference.run(self.make_settings(source), model=FakeModel())
            show.assert_not_called()
            inference.run(self.make_settings(source, show=True), model=FakeModel())
            show.assert_called_once()

    def test_directory_input_skips_nested_output_and_flattens_names(self):
        data = self.tmp / "data"
        (data / "sub").mkdir(parents=True)
        write_image(data / "a.png")
        write_image(data / "sub" / "a.png")
        out = data / "out"
        out.mkdir()
        write_image(out / "old.png.annotated.png")
        with redirect_stdout(io.StringIO()):
            summaries = inference.run(self.make_settings(data, save_dir=str(out)), model=FakeModel())
        self.assertEqual([s["png"].name for s in summaries], ["a.png.annotated.png", "sub__a.png.annotated.png"])

    def test_empty_directory_unreadable_and_empty_files(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(im.InferenceConfigError):
            inference.run(self.make_settings(empty), model=FakeModel())
        garbage = self.tmp / "garbage.png"
        garbage.write_bytes(b"this is not a png")
        with self.assertRaises(im.ImageReadError):
            inference.run(self.make_settings(garbage), model=FakeModel())
        blank = self.tmp / "blank.png"
        blank.write_bytes(b"")
        with self.assertRaises(im.ImageReadError):
            im.read_image(blank)

    def test_out_of_frame_circle_fails_before_saving(self):
        source = self.tmp / "small.png"
        write_image(source, width=40, height=40)
        with self.assertRaises(im.GeometryError):
            inference.run(self.make_settings(source), model=FakeModel())
        self.assertFalse((self.tmp / "out" / "small.png.annotated.png").exists())

    def test_missing_weight_is_never_downloaded(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            im.load_model(self.tmp / "missing.pt")
        self.assertIn("does not download", str(ctx.exception))
        source = self.tmp / "frame.png"
        write_image(source)
        settings = self.make_settings(source, weight=str(self.tmp / "missing.pt"))
        with self.assertRaises(FileNotFoundError):
            inference.run(settings, model=FakeModel())

    def test_cli_reports_missing_source_with_exit_code(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = inference.main(["--source", str(self.tmp / "nope.png")])
        self.assertEqual(code, 2)
        self.assertIn("inference.source", stderr.getvalue())

    def test_invalid_settings_are_rejected(self):
        source = self.tmp / "frame.png"
        write_image(source)
        for kwargs in ({"conf": 1.5}, {"conf": float("nan")}, {"save": "yes"}, {"save": False, "show": False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(im.InferenceConfigError):
                inference.run(self.make_settings(source, **kwargs), model=FakeModel())


if __name__ == "__main__":
    unittest.main()
