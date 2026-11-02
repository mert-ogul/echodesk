"""EchoDesk: estimate how far an object is (10-75 cm) from a speaker and a microphone.

A DFRobot SEN0682 LiDAR labels the training recordings. Live mode uses sound only.
"""

import argparse
import glob
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from scipy.signal import chirp, correlate, correlation_lags

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CFG = ROOT / "config.json"
MODEL = ROOT / "model.npz"
REPORT = ROOT / "report.json"
PLOT = ROOT / "media" / "test.svg"
PAGE = ROOT / "index.html"

FS = 48000
BINS = 256
FEATURE_VERSION = 2
CHIRPS = 3  # averaged per reading, in collect and in live
MIN_M = 0.10  # the LiDAR can't measure closer than 10 cm
MAX_M = 0.75
KS = (1, 3, 5, 9)
MAX_TEST_MAE_M = 0.07
BASELINE_ERROR_FACTOR = 0.8

# Four points in the middle of line 4, straight ahead of the sensor.
LIDAR_LINE = 4
LIDAR_POINTS = (31, 34)
LIDAR_FRAMES = 3  # read before and again after the chirps
STILL_M = 0.02  # before/after readings further apart than this: the target moved


# Sound


def pulse():
    t = np.arange(round(0.003 * FS)) / FS
    return (chirp(t, 2000, 0.003, 12000) * np.hanning(len(t))).astype("float32")


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def feature(raw):
    """Align the response to its strongest peak and sample the 0.25-12 ms after it.

    The strongest peak isn't always the direct sound, so delay is not read as distance.
    """
    if raw.ndim != 2 or raw.shape[1] < 1 or len(raw) < int(0.30 * FS):
        raise ValueError("Recording shape/length incorrect")
    x = raw[:, 0].astype(float)
    if not np.isfinite(raw).all() or np.max(np.abs(x)) >= 0.98:
        raise ValueError("Clipped or invalid input. Lower playback/capture gain.")
    x -= np.mean(x)
    if np.sqrt(np.mean(x * x)) < 1e-5:
        raise ValueError("Almost silent input. Check input device/mute/capture gain.")
    c = np.abs(correlate(x, pulse(), mode="full", method="fft"))
    lags = correlation_lags(len(x), len(pulse()), mode="full")
    window = np.flatnonzero((lags >= 0.02 * FS) & (lags <= 0.2 * FS))
    anchor = window[np.argmax(c[window])]
    level = c[anchor]
    # Late part of the recording (240-290 ms) is used as the noise floor.
    noise = np.median(c[(lags >= 0.24 * FS) & (lags <= 0.29 * FS)])
    if level < max(1e-8, 8 * noise):
        raise ValueError("No clear chirp peak. Reduce room noise/check speaker.")
    # These windows define feature_version 2. Changing them breaks the saved model.
    offsets = np.linspace(0.00025 * FS, 0.012 * FS, BINS)
    profile = np.interp(anchor + offsets, np.arange(len(c)), c)
    profile = np.log1p(100 * profile / (level + 1e-12)).astype("float32")
    quality = {
        "peak": float(np.max(np.abs(raw))),
        "anchor_ms": float(lags[anchor] / FS * 1000),
        "chirp_ratio": float(level / (noise + 1e-12)),
    }
    return profile, quality


def capture(cfg):
    import sounddevice as sd

    out = np.zeros((round(0.35 * FS), 2), dtype="float32")
    start = round(0.05 * FS)
    out[start : start + len(pulse()), cfg["speaker"] - 1] = cfg["amplitude"] * pulse()
    try:
        raw = sd.playrec(
            out,
            samplerate=FS,
            channels=cfg["channels"],
            device=(cfg["input"], cfg["output"]),
            dtype="float32",
            blocking=True,
        )
    except sd.PortAudioError as exc:
        raise ValueError(f"Audio device unavailable: {exc}") from exc
    status = sd.get_status()
    if status:
        raise ValueError("Audio dropout: " + str(status))
    f, quality = feature(raw)
    return raw, f, quality


def repeats(cfg, count=5):
    features, quality = [], []
    for _ in range(count):
        _, f, q = capture(cfg)
        features.append(f)
        quality.append(q)
        time.sleep(0.15)
    return np.stack(features), quality


# LiDAR (DFRobot SEN0682)
#
# Same protocol as DFRobot's DFRobot_64x8DTOF library: ASCII AT commands at
# 921600 baud, each answered with b"\nOK\n". A triggered frame follows its OK
# as 8 bytes per point: x, y, z, intensity, little-endian int16, in mm.


class Lidar:
    OK = b"\nOK\n"

    def __init__(self, link):
        self.link = link
        first, last = LIDAR_POINTS
        self.points = last - first + 1
        time.sleep(0.4)
        for text in (
            "AT+STREAM_CONTROL=1",
            "AT+STREAM_CONTROL=0",
            "AT+SPAD_FRAME_MODE=1",  # one frame per trigger
            "AT+SAVE_CONFIG",
            "AT+STREAM_CONTROL=1",
            "AT+STREAM_CONTROL=0",
            f"AT+SPAD_OUTPUT_LINE_DATA={LIDAR_LINE},{first},{last}",
            "AT+SAVE_CONFIG",
            "AT+STREAM_CONTROL=1",
        ):
            ok = any(self.command(text) for _ in range(3))
            if not ok and text != "AT+SAVE_CONFIG":
                raise ValueError(f"LiDAR didn't answer {text}. Check wiring and port.")
            if text != "AT+STREAM_CONTROL=1":
                time.sleep(0.7)  # DFRobot's library waits this long between settings
        time.sleep(0.5)

    def command(self, text):
        self.link.reset_input_buffer()
        self.link.write((text + "\n").encode("ascii"))
        seen = b""
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            byte = self.link.read(1)
            if byte:
                seen = (seen + byte)[-4:]
                if seen == self.OK:
                    return True
        return False

    def frame(self):
        """Z of the middle points in metres. Points outside 0.1-5 m are NaN."""
        if not any(self.command("AT+SPAD_TRIG_ONE_FRAME=1") for _ in range(3)):
            raise ValueError("LiDAR stopped answering. Check wiring and port.")
        size = 8 * self.points
        payload = self.link.read(size)
        if len(payload) != size:
            raise ValueError("LiDAR frame was cut short.")
        z = np.frombuffer(payload, dtype="<i2").reshape(-1, 4)[:, 2] / 1000
        return np.where((z >= 0.1) & (z <= 5.0), z, np.nan)

    def distance(self):
        """Median of the middle points, or None if fewer than three are valid."""
        z = self.frame()
        return float(np.nanmedian(z)) if np.isfinite(z).sum() >= 3 else None

    def close(self):
        self.link.close()


def open_lidar(port=None):
    """USB-C shows up as ttyUSB or ttyACM. On a Pi 5, GPIO 14/15 are ttyAMA0."""
    import serial

    if port is None:
        patterns = ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyAMA0")
        found = [path for pattern in patterns for path in sorted(glob.glob(pattern))]
        if not found:
            raise ValueError("No LiDAR found. Plug in its USB-C cable or pass --port.")
        port = found[0]
    print(f"LiDAR on {port}")
    try:
        link = serial.Serial(port, 921600, timeout=0.5)
    except serial.SerialException as exc:
        raise ValueError(f"Can't open the LiDAR on {port}: {exc}") from exc
    return Lidar(link)


def readings(lidar):
    values = [lidar.distance() for _ in range(LIDAR_FRAMES)]
    return [v for v in values if v is not None]


def label(before, after):
    """LiDAR label for one sample: (metres, message), or (None, reason)."""
    if len(before) < 2 or len(after) < 2:
        return None, "The LiDAR sees nothing straight ahead. Not saved."
    if abs(np.median(before) - np.median(after)) > STILL_M:
        return None, "The target moved during the chirps. Not saved."
    value = float(np.median(before + after))
    if not MIN_M <= value <= MAX_M:
        return None, f"LiDAR reads {value * 100:.1f} cm, outside 10-75 cm. Not saved."
    return value, "Saved."


# Data and model


def get_config():
    if not CFG.exists():
        raise ValueError("Run devices, then configure first.")
    return json.loads(CFG.read_text())


def background(cfg):
    path = DATA / "baseline.npz"
    if not path.exists():
        raise ValueError("Record the empty desk first: python echo_desk.py baseline")
    with np.load(path, allow_pickle=False) as z:
        if str(z["signature"]) != fingerprint(cfg):
            raise ValueError("Baseline was recorded with other audio settings.")
        return z["features"].mean(axis=0)


def save_baseline(cfg):
    path = DATA / "baseline.npz"
    if path.exists():
        raise ValueError("data/baseline.npz already exists. Move it away first.")
    features, quality = repeats(cfg)
    with path.open("xb") as stream:
        np.savez(stream, features=features, signature=fingerprint(cfg))
    print("Saved baseline.npz\n", json.dumps(quality, indent=2))


def next_run():
    n = 1
    while (DATA / f"run{n:02}.npz").exists():
        n += 1
    return DATA / f"run{n:02}.npz"


def save_run(path, cfg, samples, skipped):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez(
            stream,
            features=np.array([s["features"] for s in samples], dtype="float32"),
            lidar_m=np.array([s["lidar_m"] for s in samples], dtype="float32"),
            lidar_spread_m=np.array([s["spread"] for s in samples], dtype="float32"),
            time=np.array([s["time"] for s in samples]),
            skipped=skipped,
            signature=fingerprint(cfg),
        )
    os.replace(tmp, path)


def load_runs(cfg):
    runs = []
    for path in sorted(DATA.glob("run*.npz")):
        with np.load(path, allow_pickle=False) as z:
            if str(z["signature"]) != fingerprint(cfg):
                raise ValueError(f"{path.name} was recorded with other audio settings.")
            features, y = z["features"], z["lidar_m"].astype(float)
            if (
                features.ndim != 3
                or features.shape[1:] != (CHIRPS, BINS)
                or len(features) != len(y)
                or not np.isfinite(features).all()
            ):
                raise ValueError(f"Unexpected arrays in {path.name}")
            runs.append(
                {
                    "name": path.stem,
                    "x": features.mean(axis=1),
                    "y": y,
                    "spread": z["lidar_spread_m"].astype(float),
                    "skipped": int(z["skipped"]),
                }
            )
    return runs


def normalized(features, background):
    x = np.atleast_2d(features).astype(float) - background
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def predict(features, templates, targets, background, k):
    x = normalized(features, background)
    t = normalized(templates, background)
    # Squared distance between unit vectors, without a (samples x templates x bins) array.
    d = np.maximum(2 - 2 * x @ t.T, 0)
    idx = np.argsort(d, axis=1)[:, : int(k)]
    w = 1 / (np.take_along_axis(d, idx, axis=1) + 0.01)
    return np.sum(np.asarray(targets)[idx] * w, axis=1) / w.sum(axis=1)


def train(cfg):
    if MODEL.exists():
        raise ValueError("model.npz already exists. Move it and report.json away first.")
    runs = load_runs(cfg)
    if len(runs) < 3 or min(len(r["y"]) for r in runs) < 10:
        raise ValueError("Need at least 3 runs with 10+ samples each. Collect more.")
    bg = background(cfg)
    *fit, val, test = runs
    fx = np.concatenate([r["x"] for r in fit])
    fy = np.concatenate([r["y"] for r in fit])
    errors = {
        k: float(np.mean(np.abs(predict(val["x"], fx, fy, bg, k) - val["y"])))
        for k in KS
        if k <= len(fy)
    }
    k = min(errors, key=errors.get)
    # The live model also learns from the validation run. The test run stays unseen.
    tx = np.concatenate([fx, val["x"]])
    ty = np.concatenate([fy, val["y"]])
    pred = predict(test["x"], tx, ty, bg, k)
    err = np.abs(pred - test["y"])
    guess = float(np.mean(np.abs(np.median(ty) - test["y"])))
    mae = float(np.mean(err))
    passed = bool(mae <= MAX_TEST_MAE_M and mae <= BASELINE_ERROR_FACTOR * guess)
    report = {
        "runs": {r["name"]: {"samples": len(r["y"]), "skipped": r["skipped"]} for r in runs},
        "fit_runs": [r["name"] for r in fit],
        "validation_run": val["name"],
        "test_run": test["name"],
        "validation_mae_cm": {str(key): round(v * 100, 3) for key, v in errors.items()},
        "k": k,
        "test_samples": len(err),
        "test_mae_cm": round(mae * 100, 3),
        "test_rmse_cm": round(float(np.sqrt(np.mean(err**2))) * 100, 3),
        "test_p90_cm": round(float(np.percentile(err, 90)) * 100, 3),
        "test_max_cm": round(float(np.max(err)) * 100, 3),
        "median_guess_mae_cm": round(guess * 100, 3),
        "lidar_spread_cm": round(
            float(np.median(np.concatenate([r["spread"] for r in runs]))) * 100, 3
        ),
        "distance_display": passed,
        "test_lidar_vs_sound_cm": [
            [round(a * 100, 2), round(p * 100, 2)] for a, p in zip(test["y"], pred)
        ],
    }
    np.savez(
        MODEL,
        templates=tx.astype("float32"),
        targets=ty.astype("float32"),
        background=bg.astype("float32"),
        k=k,
        passed=passed,
        signature=fingerprint(cfg),
    )
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    write_plot(report["test_lidar_vs_sound_cm"], PLOT)
    print(summary(report))
    print(
        "Distance display enabled."
        if passed
        else "Test error too high. Live mode will show the echo only."
    )


def summary(report):
    samples = sum(r["samples"] for r in report["runs"].values())
    skipped = sum(r["skipped"] for r in report["runs"].values())
    return "\n".join(
        [
            f"{len(report['runs'])} runs, {samples} samples kept, {skipped} skipped "
            f"because the target moved. Test run: {report['test_run']} "
            f"({report['test_samples']} samples), k = {report['k']}.",
            "",
            "| | cm |",
            "|---|---:|",
            f"| Mean error, sound vs LiDAR | {report['test_mae_cm']:.2f} |",
            f"| RMSE | {report['test_rmse_cm']:.2f} |",
            f"| 90% of samples within | {report['test_p90_cm']:.2f} |",
            f"| Worst sample | {report['test_max_cm']:.2f} |",
            f"| Always guessing the median | {report['median_guess_mae_cm']:.2f} |",
            f"| LiDAR spread within a sample (median) | {report['lidar_spread_cm']:.2f} |",
        ]
    )


def write_plot(pairs, path):
    """Test run as an SVG scatter: LiDAR on x, sound on y."""
    size, pad, lo, hi = 420, 52, 10, 75

    def sx(v):
        return pad + (v - lo) / (hi - lo) * (size - 1.5 * pad)

    def sy(v):
        return size - pad - (v - lo) / (hi - lo) * (size - 1.5 * pad)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
        f'<rect width="{size}" height="{size}" fill="#fff"/>',
    ]
    for v in range(10, 80, 10):
        out.append(
            f'<line x1="{sx(v):.1f}" y1="{sy(lo):.1f}" x2="{sx(v):.1f}" y2="{sy(hi):.1f}" stroke="#e6e6e6"/>'
            f'<line x1="{sx(lo):.1f}" y1="{sy(v):.1f}" x2="{sx(hi):.1f}" y2="{sy(v):.1f}" stroke="#e6e6e6"/>'
            f'<text x="{sx(v):.1f}" y="{size - pad + 18}" text-anchor="middle" fill="#555">{v}</text>'
            f'<text x="{pad - 8}" y="{sy(v) + 4:.1f}" text-anchor="end" fill="#555">{v}</text>'
        )
    out.append(
        f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" '
        'stroke="#999" stroke-dasharray="4 4"/>'
    )
    for a, p in pairs:
        p = min(max(p, lo), hi)
        out.append(f'<circle cx="{sx(a):.1f}" cy="{sy(p):.1f}" r="3.2" fill="#1b1b1b" fill-opacity="0.7"/>')
    out.append(
        f'<text x="{sx(42.5):.1f}" y="{size - 12}" text-anchor="middle" fill="#1b1b1b">LiDAR (cm)</text>'
        f'<text transform="translate(16 {sy(42.5):.1f}) rotate(-90)" text-anchor="middle" '
        'fill="#1b1b1b">Sound estimate (cm)</text></svg>'
    )
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n")


def load_model(cfg):
    if not MODEL.exists():
        return None
    with np.load(MODEL, allow_pickle=False) as stored:
        model = {key: stored[key].copy() for key in stored.files}
    required = {"templates", "targets", "background", "k", "passed", "signature"}
    if not required.issubset(model):
        raise ValueError("Model is missing required arrays.")
    if str(model["signature"]) != fingerprint(cfg):
        raise ValueError("Model settings differ. Restore settings or recollect.")
    t, y, b = model["templates"], model["targets"], model["background"]
    if t.ndim != 2 or t.shape[1] != BINS or y.shape != (len(t),) or b.shape != (BINS,):
        raise ValueError("Model array dimensions are wrong.")
    if not all(np.isfinite(a).all() for a in (t, y, b)):
        raise ValueError("Model contains invalid values.")
    if not np.all((y >= MIN_M) & (y <= MAX_M)):
        raise ValueError("Model targets fall outside 10-75 cm.")
    if model["k"].shape != () or int(model["k"]) not in KS or model["passed"].shape != ():
        raise ValueError("Invalid model settings.")
    return model


# Web page


def make_server(snapshot, port=8000):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path == "/state.json":
                body = json.dumps(snapshot(), allow_nan=False).encode()
                content_type = "application/json"
            elif self.path == "/":
                body = PAGE.read_bytes()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A closed browser tab should not interrupt recording.

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


class Page:
    """State shown on http://localhost:8000, updated by the recording loop."""

    def __init__(self, **state):
        self.lock = threading.Lock()
        self.state = {"sequence": 0, "time": time.time(), **state}
        self.updated = time.monotonic()
        self.server = make_server(self.snapshot)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return {**self.state, "age_s": time.monotonic() - self.updated}

    def update(self, **changes):
        with self.lock:
            self.state.update(changes, time=time.time())
            self.state["sequence"] += 1
            self.updated = time.monotonic()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def coverage(labels):
    """Samples per 1 cm bin from 10 to 75 cm."""
    counts, _ = np.histogram(np.array(labels) * 100, bins=65, range=(10, 75))
    return counts.tolist()


def quality_summary(quality):
    return {
        "peak": max(q["peak"] for q in quality),
        "chirp_ratio": min(q["chirp_ratio"] for q in quality),
    }


def collect(cfg, port):
    bg = background(cfg)
    lidar = open_lidar(port)
    path = next_run()
    samples, skipped = [], 0
    page = Page(
        mode="collect",
        run=path.stem,
        status="starting",
        message="Waiting for the first reading.",
        lidar_m=None,
        distance_m=None,
        profile=[],
        baseline_profile=bg.tolist(),
        quality=None,
        saved=0,
        skipped=0,
        coverage=coverage([]),
    )
    print(f"Recording {path.name}. Open http://localhost:8000. Ctrl+C when done.")
    try:
        while True:
            try:
                before = readings(lidar)
                features, quality = repeats(cfg, CHIRPS)
                after = readings(lidar)
            except ValueError as exc:
                page.update(status="signal_error", message=str(exc), profile=[], quality=None)
                time.sleep(0.5)
                continue
            value, message = label(before, after)
            if value is not None:
                message = "Move the target a step, let go and wait."
            if value is None:
                skipped += 1
            else:
                samples.append(
                    {
                        "features": features,
                        "lidar_m": value,
                        "spread": float(np.std(before + after)),
                        "time": time.time(),
                    }
                )
                save_run(path, cfg, samples, skipped)
            page.update(
                status="saved" if value is not None else "skipped",
                message=message,
                lidar_m=float(np.median(after)) if after else None,
                profile=features.mean(axis=0).tolist(),
                quality=quality_summary(quality),
                saved=len(samples),
                skipped=skipped,
                coverage=coverage([s["lidar_m"] for s in samples]),
            )
    finally:
        page.close()
        lidar.close()
        if samples:
            save_run(path, cfg, samples, skipped)
            print(f"\n{len(samples)} samples saved to {path.name}, {skipped} skipped.")


def live(cfg):
    model = load_model(cfg)
    evaluation = None
    if model is not None and REPORT.exists():
        report = json.loads(REPORT.read_text())
        evaluation = {
            key: report[key]
            for key in ("test_run", "test_samples", "test_mae_cm", "median_guess_mae_cm", "k")
        }
    page = Page(
        mode="live",
        status="starting",
        message="Waiting for the first reading.",
        distance_m=None,
        profile=[],
        baseline_profile=model["background"].tolist() if model is not None else [],
        quality=None,
        evaluation=evaluation,
    )
    print("Sound only. Open http://localhost:8000 through your SSH tunnel. Ctrl+C stops.")
    try:
        while True:
            try:
                features, quality = repeats(cfg, CHIRPS)
                profile = features.mean(axis=0)
                estimate, status = None, "echo_only"
                message = (
                    "No trained model. Showing the echo only."
                    if model is None
                    else "The model's test error was too high. Showing the echo only."
                )
                if model is not None and bool(model["passed"]):
                    estimate = float(
                        predict(
                            profile,
                            model["templates"],
                            model["targets"],
                            model["background"],
                            int(model["k"]),
                        )[0]
                    )
                    status, message = "measuring", "Speaker and microphone only."
                page.update(
                    status=status,
                    message=message,
                    distance_m=estimate,
                    profile=profile.tolist(),
                    quality=quality_summary(quality),
                )
            except (ValueError, RuntimeError) as exc:
                page.update(
                    status="signal_error",
                    message=str(exc),
                    distance_m=None,
                    profile=[],
                    quality=None,
                )
            time.sleep(0.1)
    finally:
        page.close()


def lidar_check(port):
    lidar = open_lidar(port)
    print("Middle points of line 4, in cm. Ctrl+C stops.")
    try:
        while True:
            z = lidar.frame()
            points = " ".join("  --" if np.isnan(v) else f"{v * 100:4.0f}" for v in z)
            middle = f"{np.nanmedian(z) * 100:5.1f} cm" if np.isfinite(z).sum() >= 3 else "   -- cm"
            print(f"{middle}   points: {points}")
            time.sleep(0.25)
    finally:
        lidar.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices")
    s = sub.add_parser("configure")
    s.add_argument("--input", type=int, required=True)
    s.add_argument("--output", type=int, required=True)
    s.add_argument("--channels", type=int, choices=(1, 2), default=2)
    s.add_argument("--speaker", type=int, choices=(1, 2), default=1)
    s.add_argument("--amplitude", type=float, default=0.08)
    sub.add_parser("check")
    for name in ("lidar", "collect"):
        s = sub.add_parser(name)
        s.add_argument("--port", help="default: first ttyUSB, ttyACM or ttyAMA0")
    sub.add_parser("baseline")
    sub.add_parser("train")
    sub.add_parser("live")
    a = p.parse_args()
    DATA.mkdir(exist_ok=True)
    if a.cmd == "devices":
        import sounddevice as sd

        print(sd.query_devices())
        return
    if a.cmd == "lidar":
        lidar_check(a.port)
        return
    if a.cmd == "configure":
        import sounddevice as sd

        if not 0.005 <= a.amplitude <= 0.25:
            raise ValueError("Use amplitude 0.005 to 0.25.")
        sd.check_input_settings(
            device=a.input, channels=a.channels, samplerate=FS, dtype="float32"
        )
        sd.check_output_settings(device=a.output, channels=2, samplerate=FS, dtype="float32")
        cfg = {k: v for k, v in vars(a).items() if k != "cmd"}
        cfg["feature_version"] = FEATURE_VERSION
        if CFG.exists() and get_config() != cfg and list(DATA.glob("*.npz")):
            raise ValueError("Data already exists. Start a separate folder for changed settings.")
        CFG.write_text(json.dumps(cfg, indent=2) + "\n")
        print("Configured. Now run: python echo_desk.py check")
        return
    cfg = get_config()
    if a.cmd == "check":
        _, q = repeats(cfg)
        print(json.dumps(q, indent=2))
        print("Five chirps recorded. This checks the signal, not the distance.")
    elif a.cmd == "baseline":
        save_baseline(cfg)
    elif a.cmd == "collect":
        collect(cfg, a.port)
    elif a.cmd == "train":
        train(cfg)
    elif a.cmd == "live":
        live(cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit("STOP: " + str(exc))
