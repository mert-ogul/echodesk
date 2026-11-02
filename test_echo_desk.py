"""Tests without the audio board or the LiDAR: synthetic echoes and a fake serial port."""

import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
import echo_desk as app

CFG = {"input": 0, "output": 0, "channels": 2, "speaker": 1, "amplitude": 0.12, "feature_version": 2}


def recording(distance_m, rng):
    """Direct chirp at 100 ms plus one echo delayed by the round trip to the target."""
    raw = rng.normal(0, 2e-5, (16800, 2)).astype("float32")
    pulse = app.pulse()
    start = 4800
    raw[start : start + len(pulse), 0] += 0.01 * pulse
    echo = start + round(2 * distance_m / 343 * app.FS)
    raw[echo : echo + len(pulse), 0] += 0.004 * pulse
    return raw


def features_at(distance_m, rng, count=app.CHIRPS):
    return np.stack([app.feature(recording(distance_m, rng))[0] for _ in range(count)])


class FakeLink:
    """Answers AT commands like the SEN0682 and returns a frame of the given Z values (mm)."""

    def __init__(self, z_mm):
        self.z_mm = z_mm
        self.sent = []
        self.buffer = bytearray()

    def reset_input_buffer(self):
        self.buffer.clear()

    def write(self, data):
        text = data.decode("ascii").strip()
        self.sent.append(text)
        self.buffer += b"\nOK\n"
        if text == "AT+SPAD_TRIG_ONE_FRAME=1":
            for z in self.z_mm:
                self.buffer += struct.pack("<hhhh", 3, -2, z, 900)

    def read(self, n):
        out = bytes(self.buffer[:n])
        del self.buffer[:n]
        return out

    def close(self):
        pass


class TempProject(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "data").mkdir()
        self.patcher = patch.multiple(
            app,
            ROOT=root,
            DATA=root / "data",
            CFG=root / "config.json",
            MODEL=root / "model.npz",
            REPORT=root / "report.json",
            PLOT=root / "media" / "test.svg",
        )
        self.patcher.start()
        self.root = root
        self.rng = np.random.default_rng(7)
        with (root / "data" / "baseline.npz").open("xb") as stream:
            np.savez(stream, features=features_at(2.0, self.rng, 5), signature=app.fingerprint(CFG))

    def tearDown(self):
        self.patcher.stop()
        self.tmp.cleanup()

    def write_runs(self, count, samples=25):
        for n in range(1, count + 1):
            distances = self.rng.uniform(app.MIN_M, app.MAX_M, samples)
            rows = [
                {"features": features_at(d, self.rng), "lidar_m": d, "spread": 0.004, "time": 0.0}
                for d in distances
            ]
            app.save_run(app.DATA / f"run{n:02}.npz", CFG, rows, skipped=2)


class SoundTests(unittest.TestCase):
    def test_bad_recordings_rejected(self):
        nan = np.zeros((16800, 2))
        nan[100, 1] = np.nan
        cases = {
            "Clipped": np.ones((16800, 2)),
            "silent": np.zeros((16800, 2)),
            "invalid": nan,
            "shape/length": np.zeros((1000, 2)),
        }
        for text, raw in cases.items():
            with self.subTest(text), self.assertRaisesRegex(ValueError, text):
                app.feature(raw)

    def test_profile_depends_on_echo_distance(self):
        rng = np.random.default_rng(1)
        near, _ = app.feature(recording(0.15, rng))
        far, quality = app.feature(recording(0.60, rng))
        self.assertEqual(near.shape, (app.BINS,))
        self.assertGreater(np.abs(near - far).max(), 0.5)
        self.assertGreater(quality["chirp_ratio"], 8)


class LidarTests(unittest.TestCase):
    def lidar(self, z_mm):
        with patch.object(app.time, "sleep"):
            return app.Lidar(FakeLink(z_mm))

    def test_setup_selects_middle_of_line_4(self):
        lidar = self.lidar([400, 401, 402, 403])
        self.assertIn("AT+SPAD_FRAME_MODE=1", lidar.link.sent)
        self.assertIn("AT+SPAD_OUTPUT_LINE_DATA=4,31,34", lidar.link.sent)
        self.assertEqual(lidar.link.sent[-1], "AT+STREAM_CONTROL=1")

    def test_distance_is_median_of_valid_points(self):
        self.assertAlmostEqual(self.lidar([412, 0, 418, 420]).distance(), 0.418)
        self.assertIsNone(self.lidar([412, 0, 6000, 420]).distance())

    def test_labels(self):
        self.assertEqual(app.label([0.40, 0.41], [0.41, 0.40])[1], "Saved.")
        self.assertAlmostEqual(app.label([0.40, 0.41], [0.41, 0.40])[0], 0.405)
        self.assertIsNone(app.label([0.40, 0.40], [0.45, 0.45])[0])
        self.assertIsNone(app.label([0.40], [0.40, 0.40])[0])
        self.assertIsNone(app.label([0.90, 0.90], [0.90, 0.90])[0])


class PipelineTests(TempProject):
    def test_collect_saves_lidar_labelled_samples(self):
        app.CFG.write_text(json.dumps(CFG))
        positions = [0.30] * 6 + [0.52] * 6

        class FakeLidar:
            def distance(self):
                return positions.pop(0) if positions else 0.52

            def close(self):
                pass

        calls = []

        def fake_repeats(cfg, count):
            if len(calls) == 2:
                raise KeyboardInterrupt
            calls.append(count)
            return features_at(0.30, self.rng), [{"peak": 0.01, "anchor_ms": 98.0, "chirp_ratio": 12.0}] * count

        real_server = app.make_server
        with patch.object(app, "open_lidar", lambda port: FakeLidar()), patch.object(
            app, "repeats", fake_repeats
        ), patch.object(app, "make_server", lambda snapshot: real_server(snapshot, port=0)), patch(
            "builtins.print"
        ):
            with self.assertRaises(KeyboardInterrupt):
                app.collect(CFG, "/dev/null")
        with np.load(app.DATA / "run01.npz") as z:
            np.testing.assert_allclose(z["lidar_m"], [0.30, 0.52], atol=1e-6)
            self.assertEqual(z["features"].shape, (2, app.CHIRPS, app.BINS))
            self.assertEqual(int(z["skipped"]), 0)

    def test_train_on_lidar_runs(self):
        self.write_runs(4)
        with patch("builtins.print"):
            app.train(CFG)
        report = json.loads(app.REPORT.read_text())
        self.assertEqual(report["test_run"], "run04")
        self.assertEqual(report["validation_run"], "run03")
        self.assertLess(report["test_mae_cm"], 2.0)
        self.assertLess(report["test_mae_cm"], report["median_guess_mae_cm"])
        self.assertTrue(report["distance_display"])
        model = app.load_model(CFG)
        self.assertEqual(len(model["targets"]), 75)
        ET.parse(app.PLOT)
        with self.assertRaisesRegex(ValueError, "already exists"):
            app.train(CFG)

    def test_train_needs_three_runs(self):
        self.write_runs(2)
        with self.assertRaisesRegex(ValueError, "at least 3 runs"):
            app.train(CFG)

    def test_model_rejects_other_settings(self):
        self.write_runs(3, samples=12)
        with patch("builtins.print"):
            app.train(CFG)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            app.load_model({**CFG, "amplitude": 0.2})

    def test_baseline_never_overwritten(self):
        with patch.object(app, "repeats") as capture:
            with self.assertRaisesRegex(ValueError, "already exists"):
                app.save_baseline(CFG)
            capture.assert_not_called()


class ServerTests(unittest.TestCase):
    def test_routes(self):
        server = app.make_server(lambda: {"status": "signal_error", "distance_m": None}, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/") as response:
                self.assertIn("EchoDesk", response.read().decode())
                self.assertEqual(response.headers["Cache-Control"], "no-store")
            with urlopen(base + "/state.json") as response:
                self.assertIsNone(json.load(response)["distance_m"])
            with self.assertRaises(HTTPError) as error:
                urlopen(base + "/../config.json")
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
