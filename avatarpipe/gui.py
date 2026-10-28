"""Local browser GUI for avatarpipe:  avatarpipe gui [shot.yaml]

Edits the same YAML file the command line uses, runs the pipeline as a child
process (so Stop really stops it, Blender included), streams the log, and
plays the tracking preview and the final video. Listens on 127.0.0.1 only.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml

from . import config

try:
    import gradio as gr
except ImportError as e:  # pragma: no cover
    raise SystemExit('The GUI needs Gradio:  pip install -e ".[gui]"') from e

TEMPLATE = Path(__file__).resolve().parent / "templates" / "config.yaml"
CACHE = Path(tempfile.gettempdir()) / "avatarpipe_gui"

# (dotted key, widget, choices or (min, max, step), label, help)
SETTINGS = [
    ("composite.mode", "dropdown", ["new_scene", "replace", "overlay"], "Composite mode",
     "new_scene: avatar over your background plate · replace: you painted out of your own shot · overlay: alignment check"),
    ("solve.focal_mm", "number", None, "Lens focal length (mm, 35mm equivalent)", "Most phone main cameras: 24-28"),
    ("solve.camera_height_m", "number", None, "Camera height (m)", "Lens to floor"),
    ("solve.camera_tilt_deg", "number", None, "Camera tilt (°)", "+ = tilted down, 0 = level"),
    ("input.max_height", "dropdown", [480, 720, 1080], "Working resolution (height)", "Lower = faster"),
    ("track.pose_model", "dropdown", ["lite", "full", "heavy"], "Pose model", "heavy is best and slowest"),
    ("track.face", "checkbox", None, "Track facial expressions", None),
    ("track.face_crop", "checkbox", None, "Face close-up tracking",
     "Tracks your face on a full-resolution crop of the original video: much more expression detail"),
    ("avatar.face_gain", "slider", (0.5, 3.0, 0.1), "Expression strength", "Above 1 exaggerates expressions"),
    ("lipsync.enabled", "checkbox", None, "Lip sync from your voice",
     "Needs Rhubarb Lip Sync (see README). Skipped with a note if it isn't installed"),
    ("lipsync.recognizer", "dropdown", ["pocketSphinx", "phonetic"], "Lip sync language",
     "pocketSphinx: English, most accurate · phonetic: any language"),
    ("lipsync.mode", "dropdown", ["max", "audio"], "Mouth source",
     "max: camera or voice, whichever opens the mouth more · audio: voice only"),
    ("lipsync.executable", "text", None, "Rhubarb program",
     "Leave as 'rhubarb' if it's on PATH or unzipped in a tools folder; otherwise the full path to rhubarb.exe"),
    ("track.smoothing.min_cutoff", "slider", (0.2, 4.0, 0.1), "Smoothing", "Lower = smoother, floatier"),
    ("track.smoothing.beta", "slider", (0.0, 0.2, 0.005), "Fast-move response", "Higher = snappier on fast moves"),
    ("avatar.rig_preset", "dropdown", ["auto", "vrm", "vroid", "mixamo", "custom"], "Rig preset", None),
    ("avatar.face_preset", "dropdown", ["auto", "arkit", "vroid", "none", "custom"], "Face preset", None),
    ("avatar.twist_axis", "dropdown", ["X", "-X", "Z", "-Z"], "Twist axis", "If torso or head faces backwards, change this"),
    ("blender.engine", "dropdown", ["EEVEE", "CYCLES", "WORKBENCH"], "Render engine",
     "EEVEE fast · CYCLES real floor shadows · WORKBENCH drafts"),
    ("blender.samples", "slider", (1, 128, 1), "Render samples", None),
    ("blender.shadow_catcher", "checkbox", None, "Shadows on the floor (Cycles)", None),
    ("blender.sun_angle_deg.0", "slider", (0, 90, 1), "Sun height (°)", "0 = straight down, 90 = at the horizon"),
    ("blender.sun_angle_deg.2", "slider", (-180, 180, 1), "Sun direction (°)", "Flip the sign to swap sides"),
    ("blender.sun_strength", "slider", (0, 10, 0.1), "Sun strength", None),
    ("matte.dilate_px", "slider", (0, 40, 1), "Matte grow (px)", "replace mode: raise if your edges peek out"),
    ("composite.grain", "slider", (0, 20, 1), "Grain", "Ties the avatar into camera noise"),
]
FILES = [  # (dotted key, label, folder for uploads)
    ("input.performance", "Your performance video", "footage"),
    ("input.clean_plate", "Empty set (clean plate)", "footage"),
    ("input.background", "Background plate (new scene)", "footage"),
    ("avatar.path", "Avatar (.vrm .glb .fbx .blend)", "avatars"),
]


# ---------------------------------------------------------------- yaml helpers
def _get(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)] if int(part) < len(cur) else None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _set(d: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = d
    for i, part in enumerate(parts[:-1]):
        nxt = parts[i + 1]
        if isinstance(cur, list):
            cur = cur[int(part)]
            continue
        if part not in cur or cur[part] is None:
            cur[part] = [] if nxt.isdigit() else {}
        cur = cur[part]
    last = parts[-1]
    if isinstance(cur, list):
        idx = int(last)
        while len(cur) <= idx:
            cur.append(0)
        cur[idx] = value
    else:
        cur[last] = value


def _effective(raw: dict, dotted: str):
    """User value if set, else the default."""
    v = _get(raw, dotted)
    if v is None:
        v = _get(config.DEFAULTS, dotted)
    if dotted.startswith("blender.sun_angle_deg") and v is None:
        v = [50, 0, 30][int(dotted.rsplit(".", 1)[1])]
    return v


def _read_raw(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _clean_number(v):
    if isinstance(v, float):
        v = round(v, 6)
        return int(v) if v.is_integer() else v
    return v


# ---------------------------------------------------------------- process control
class Runner:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stream(self, args: list[str], cwd: Path):
        """Run `python -m avatarpipe <args>`, yielding the growing log text."""
        if self.busy():
            yield "Already running. Press Stop first."
            return
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        cmd = [sys.executable, "-m", "avatarpipe", *args]
        lines = [f"$ avatarpipe {' '.join(args)}"]
        with self.lock:
            self.proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace", env=env, **kw)
        last = 0.0
        for line in self.proc.stdout:
            lines.append(line.rstrip("\n"))
            if time.time() - last > 0.25:
                last = time.time()
                yield "\n".join(lines[-500:])
        code = self.proc.wait()
        lines.append("— finished —" if code == 0 else f"— stopped or failed (exit {code}) —")
        yield "\n".join(lines[-500:])

    def stop(self) -> str:
        p = self.proc
        if not p or p.poll() is not None:
            return "Nothing is running."
        if os.name == "nt":  # take Blender (a grandchild) down too
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        else:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return "Stopped."


RUNNER = Runner()


# ---------------------------------------------------------------- app
def build(default_config: str) -> "gr.Blocks":
    def resolve(cfg_path: str) -> Path:
        return Path(cfg_path).expanduser().resolve()

    def status_rows(cfg_path: str):
        try:
            from .pipeline import Context
            from .stages import build as build_pipeline
            cfg = config.load(resolve(cfg_path))
            ctx = Context(cfg=cfg, workdir=Path(cfg["workdir"]))
            return [list(r) for r in build_pipeline().status(ctx)]
        except Exception as e:  # noqa: BLE001
            return [["—", f"config problem: {str(e).splitlines()[0][:80]}", ""]]

    def cached_copy(src: Path | None):
        """Gradio only serves files from safe folders; copy results into a temp cache."""
        if not src or not src.exists():
            return None
        CACHE.mkdir(parents=True, exist_ok=True)
        dst = CACHE / f"{src.stem}_{src.stat().st_mtime_ns}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
        return str(dst)

    def results(cfg_path: str):
        try:
            cfg = config.load(resolve(cfg_path))
        except Exception:  # noqa: BLE001
            return None, None, ""
        work = Path(cfg["workdir"])
        blend = work / "render" / "scene.blend"
        note = f"Output folder: `{work}`"
        if blend.exists():
            note += f"  \nEditable Blender scene: `{blend}`"
        return (cached_copy(work / "preview" / "tracking_preview.mp4"),
                cached_copy(Path(cfg["composite"]["output"])), note)

    def load(cfg_path: str):
        p = resolve(cfg_path)
        if not p.exists():
            msg = f"⚠ `{p}` doesn't exist yet. Click **New shot** to create it."
            return [msg] + [""] * len(FILES) + [_effective({}, k) for k, *_ in SETTINGS] + \
                [status_rows(cfg_path), None, None, ""]
        raw = _read_raw(p)
        try:
            config.load(p)
            msg = f"Loaded `{p}`"
        except config.ConfigError as e:
            msg = f"Loaded `{p}`, but it needs fixing:\n```\n{e}\n```"
        files = [_get(raw, k) or "" for k, *_ in FILES]
        vals = [_effective(raw, k) for k, *_ in SETTINGS]
        return [msg, *files, *vals, status_rows(cfg_path), *results(cfg_path)]

    def new_shot(cfg_path: str):
        p = resolve(cfg_path)
        if p.exists():
            return f"`{p}` already exists. Click **Load** to open it."
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = _read_raw(TEMPLATE)
        raw["project"] = p.stem
        raw["composite"]["mode"] = "new_scene"
        raw["input"]["background"] = "footage/background.mp4"
        raw["input"]["clean_plate"] = None
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("# avatarpipe shot config (written by the GUI; the CLI reads it too)\n")
            yaml.safe_dump(raw, fh, sort_keys=False)
        for d in ("footage", "avatars"):
            (p.parent / d).mkdir(exist_ok=True)
        return f"Created `{p}`. Add your clips and avatar below, then **Save**."

    def save(cfg_path: str, *values):
        p = resolve(cfg_path)
        if not p.exists():
            return f"⚠ `{p}` doesn't exist. Click **New shot** first."
        raw = _read_raw(p)
        files, settings = values[:len(FILES)], values[len(FILES):]
        for (key, *_), v in zip(FILES, files):
            _set(raw, key, (v or "").strip() or None)
        for (key, *_), v in zip(SETTINGS, settings):
            if key.startswith("blender.sun_angle_deg") and _get(raw, "blender.sun_angle_deg") is None:
                _set(raw, "blender.sun_angle_deg", [50, 0, 30])
            _set(raw, key, _clean_number(v))
        tmp = p.with_suffix(".yaml.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("# avatarpipe shot config (written by the GUI; the CLI reads it too)\n")
            yaml.safe_dump(raw, fh, sort_keys=False)
        tmp.replace(p)
        try:
            config.load(p)
            return f"Saved `{p}`"
        except config.ConfigError as e:
            return f"Saved, but fix these before running:\n```\n{e}\n```"

    def upload(cfg_path: str, file, folder: str):
        if not file:
            return gr.update()
        p = resolve(cfg_path)
        dest_dir = p.parent / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = Path(file)
        name = src.name
        dest = dest_dir / name
        shutil.copy2(src, dest)
        return f"{folder}/{name}"

    def run(cfg_path: str, *args):
        p = resolve(cfg_path)
        yield from RUNNER.stream([a.replace("{cfg}", str(p)) for a in args], p.parent)

    def open_folder(cfg_path: str):
        try:
            work = Path(config.load(resolve(cfg_path))["workdir"])
        except Exception as e:  # noqa: BLE001
            return f"⚠ {e}"
        work.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(work)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(work)])
        else:
            subprocess.Popen(["xdg-open", str(work)])
        return f"Opened `{work}`"

    with gr.Blocks(title="avatarpipe") as app:
        gr.Markdown("## avatarpipe\nFilm yourself, get the same shot with your avatar in your place.")
        with gr.Row():
            cfg_box = gr.Textbox(value=default_config, label="Shot config file", scale=5,
                                 info="Relative to the folder you launched from")
            load_btn = gr.Button("Load", scale=1)
            new_btn = gr.Button("New shot", scale=1)
            save_btn = gr.Button("Save", variant="primary", scale=1)
        msg = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("1 · Footage & avatar"):
                file_boxes = []
                for key, label, folder in FILES:
                    with gr.Row(equal_height=True):
                        box = gr.Textbox(label=label, scale=5, placeholder="(none)")
                        up = gr.UploadButton("Choose file…", scale=1, size="md")
                        up.upload(lambda c, f, folder=folder: upload(c, f, folder), [cfg_box, up], box)
                    file_boxes.append(box)
                gr.Markdown("Chosen files are copied into the shot's `footage/` or `avatars/` folder. "
                            "You can also type a path. Click **Save** when done.")

            with gr.Tab("2 · Settings"):
                setting_widgets = []
                with gr.Row():
                    cols = [gr.Column(), gr.Column()]
                for i, (key, kind, opts, label, info) in enumerate(SETTINGS):
                    with cols[i % 2]:
                        if kind == "dropdown":
                            w = gr.Dropdown(opts, label=label, info=info)
                        elif kind == "number":
                            w = gr.Number(label=label, info=info)
                        elif kind == "checkbox":
                            w = gr.Checkbox(label=label, info=info)
                        elif kind == "text":
                            w = gr.Textbox(label=label, info=info)
                        else:
                            lo, hi, step = opts
                            w = gr.Slider(lo, hi, step=step, label=label, info=info)
                    setting_widgets.append(w)
                gr.Markdown("The shooting guide covers how to measure lens, height and tilt. "
                            "Anything not shown here can still be set in the YAML file.")

            with gr.Tab("3 · Run"):
                with gr.Row():
                    b_track = gr.Button("Track & solve", variant="primary")
                    b_prev = gr.Button("Make tracking preview")
                    b_all = gr.Button("Render everything", variant="primary")
                    b_stop = gr.Button("Stop", variant="stop")
                with gr.Row():
                    force_stage = gr.Dropdown(["ingest", "track", "lipsync", "solve", "matte", "render", "composite"],
                                              value="render", label="Rerun one stage even if cached", scale=3)
                    b_force = gr.Button("Rerun stage", scale=1)
                    b_doctor = gr.Button("Check setup (doctor)", scale=1)
                status = gr.Dataframe(headers=["Stage", "State", "Last run"], interactive=False,
                                      label="Stage status")
                log = gr.Textbox(label="Log", lines=18, max_lines=18, autoscroll=True, interactive=False)

            with gr.Tab("4 · Results"):
                with gr.Row():
                    v_prev = gr.Video(label="Tracking preview (green = tracked, orange = 3D solve)",
                                      interactive=False)
                    v_final = gr.Video(label="Final video", interactive=False)
                where = gr.Markdown()
                with gr.Row():
                    b_refresh = gr.Button("Refresh")
                    b_open = gr.Button("Open output folder")

        form = [*file_boxes, *setting_widgets]
        loaded = [msg, *form, status, v_prev, v_final, where]
        after = [status, v_prev, v_final, where]

        def refresh(c):
            return [status_rows(c), *results(c)]

        load_btn.click(load, cfg_box, loaded)
        new_btn.click(new_shot, cfg_box, msg).then(load, cfg_box, loaded)
        save_btn.click(save, [cfg_box, *form], msg).then(refresh, cfg_box, after)
        app.load(load, cfg_box, loaded)

        # Gradio streams only from real generator functions, so each button gets its own
        def wire(btn, *args):
            def go(c):
                yield from run(c, *args)
            ev = btn.click(save, [cfg_box, *form], msg)
            ev = ev.then(go, cfg_box, log)
            return ev.then(refresh, cfg_box, after)

        wire(b_track, "run", "{cfg}", "--to", "solve")
        wire(b_prev, "preview", "{cfg}")
        wire(b_all, "run", "{cfg}")
        wire(b_doctor, "doctor", "{cfg}")

        def go_force(c, s):
            yield from run(c, "run", "{cfg}", "--force", s)
        (b_force.click(save, [cfg_box, *form], msg)
         .then(go_force, [cfg_box, force_stage], log)
         .then(refresh, cfg_box, after))
        b_stop.click(lambda: RUNNER.stop(), None, msg).then(refresh, cfg_box, after)
        b_refresh.click(refresh, cfg_box, after)
        b_open.click(open_folder, cfg_box, msg)
    return app


def launch(default_config: str = "shots/take1.yaml", port: int = 7860, open_browser: bool = True) -> int:
    app = build(default_config)
    app.queue(default_concurrency_limit=4)
    app.launch(server_name="127.0.0.1", server_port=port, inbrowser=open_browser, share=False,
               allowed_paths=[str(CACHE)], footer_links=[], theme=gr.themes.Soft())
    return 0
