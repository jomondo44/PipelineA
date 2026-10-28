"""avatarpipe command line.

  avatarpipe init [shot.yaml]           write a starter config
  avatarpipe doctor [shot.yaml]         check ffmpeg, Blender, MediaPipe, models, inputs
  avatarpipe run shot.yaml              run everything that is out of date
      --only track,solve  --from solve  --to render  --force render|all  --dry-run
  avatarpipe status shot.yaml           what's done / stale
  avatarpipe preview shot.yaml          tracking overlay video to sanity-check the solve
  avatarpipe clean shot.yaml [stage..]  forget cached results
  avatarpipe gui [shot.yaml]            browser GUI (pip install -e ".[gui]")
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__, config, models
from .pipeline import Context

log = logging.getLogger("avatarpipe")
TEMPLATE = Path(__file__).resolve().parent / "templates" / "config.yaml"


def _setup_logging(verbose: bool) -> None:
    # Windows pipes default to cp1252, which can't print ✓ ▶ → ; never crash on output
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s",
                        stream=sys.stdout)
    for noisy in ("absl", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _ctx(path: str, dry: bool = False) -> Context:
    cfg = config.load(path)
    return Context(cfg=cfg, workdir=Path(cfg["workdir"]), dry_run=dry)


def _split(v):
    return [s.strip() for s in v.split(",") if s.strip()] if v else []


def cmd_init(a) -> int:
    dest = Path(a.path)
    if dest.exists() and not a.overwrite:
        print(f"{dest} exists (use --overwrite)")
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, dest)
    for d in ("footage", "avatars"):
        (dest.parent / d).mkdir(exist_ok=True)
    print(f"wrote {dest}\nput your clips in {dest.parent / 'footage'} and an avatar in {dest.parent / 'avatars'}, "
          f"edit the config, then:  avatarpipe doctor {dest}")
    return 0


def cmd_doctor(a) -> int:
    ok = True

    def check(label, good, hint=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'✓' if good else '✗'} {label}" + ("" if good or not hint else f"\n      → {hint}"))

    print(f"avatarpipe {__version__}  (python {sys.version.split()[0]})")
    check("ffmpeg", shutil.which("ffmpeg"), "install FFmpeg and put it on PATH")
    check("ffprobe", shutil.which("ffprobe"), "comes with FFmpeg")
    try:
        import mediapipe  # noqa: F401
        check(f"mediapipe {mediapipe.__version__}", True)
    except ImportError:
        check("mediapipe", False, "pip install mediapipe")
    cached = sorted(p.name for p in models.cache_dir().glob("*") if p.suffix in (".task", ".tflite"))
    print(f"  · models cached in {models.cache_dir()}: {', '.join(cached) or 'none yet (downloaded on first run)'}")

    cfg = None
    if a.config:
        try:
            cfg = config.load(a.config)
            check(f"config {a.config}", True)
        except (config.ConfigError, OSError) as e:
            check(f"config {a.config}", False, str(e))
    blender_cfg = cfg["blender"]["executable"] if cfg else "blender"
    try:
        from .stages.render import find_blender
        exe = find_blender(blender_cfg)
        ver = subprocess.run([exe, "--version"] if not Path(exe).name.startswith("python") else
                             [exe, "-c", "import bpy;print('Blender', bpy.app.version_string)"],
                             capture_output=True, text=True, timeout=120).stdout.strip().splitlines()
        check(f"blender: {ver[0] if ver else exe}", True)
    except Exception as e:  # noqa: BLE001
        check("blender", False, str(e))
    if cfg:
        for sec, key in config.PATH_KEYS:
            p = cfg[sec][key]
            if p:
                check(f"{sec}.{key}: {p}", Path(p).exists(), "file not found")
        if cfg["avatar"]["path"] and cfg["avatar"]["path"].lower().endswith(".vrm"):
            print("  · .vrm avatars need the free 'VRM format' Blender add-on "
                  "(Blender: Edit > Preferences > Get Extensions > search VRM)")
        if not cfg["avatar"]["path"]:
            print("  · no avatar.path: run stops after tracking/solve (useful for checking the preview first)")
    print("all good" if ok else "fix the ✗ items above")
    return 0 if ok else 1


def cmd_run(a) -> int:
    from .stages import build
    ctx = _ctx(a.config, a.dry_run)
    pipe = build()
    names = pipe.names()
    only, force = _split(a.only), _split(a.force)
    for n in only + [x for x in force if x != "all"] + [x for x in (a.start, a.stop) if x]:
        if n not in names:
            print(f"unknown stage '{n}'; stages: {', '.join(names)}")
            return 2
    log.info("project %s → %s", ctx.cfg["project"], ctx.workdir)
    pipe.run(ctx, only=only or None, start=a.start, stop=a.stop, force=force)
    out = Path(ctx.cfg["composite"]["output"])
    if out.exists() and not a.dry_run:
        log.info("output: %s", out)
    return 0


def cmd_status(a) -> int:
    from .stages import build
    ctx = _ctx(a.config)
    for name, state, when in build().status(ctx):
        print(f"  {name:<10} {state:<11} {when}")
    return 0


def cmd_preview(a) -> int:
    from .stages import preview
    ctx = _ctx(a.config)
    if not (ctx.workdir / "track" / "tracking.npz").exists():
        print("no tracking yet: avatarpipe run <config> --to solve")
        return 1
    preview.run(ctx)
    return 0


def cmd_gui(a) -> int:
    try:
        from . import gui
    except SystemExit as e:
        print(e)
        return 1
    return gui.launch(a.config, port=a.port, open_browser=not a.no_browser)


def cmd_clean(a) -> int:
    ctx = _ctx(a.config)
    state = ctx.workdir / ".state"
    targets = a.stages or [p.stem for p in state.glob("*.json")]
    for s in targets:
        (state / f"{s}.json").unlink(missing_ok=True)
        print(f"  cleared {s}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="avatarpipe", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="write a starter config")
    s.add_argument("path", nargs="?", default="shot.yaml")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("doctor", help="check dependencies and config")
    s.add_argument("config", nargs="?")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("run", help="run the pipeline")
    s.add_argument("config")
    s.add_argument("--only", help="comma-separated stages to run")
    s.add_argument("--from", dest="start", help="start at this stage")
    s.add_argument("--to", dest="stop", help="stop after this stage")
    s.add_argument("--force", default="", help="comma-separated stages to rerun even if cached, or 'all'")
    s.add_argument("--dry-run", action="store_true", help="show what would run")
    s.set_defaults(fn=cmd_run)

    for name, fn, hlp in (("status", cmd_status, "show stage state"),
                          ("preview", cmd_preview, "render tracking diagnostic video")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("config")
        s.set_defaults(fn=fn)

    s = sub.add_parser("gui", help="open the browser GUI")
    s.add_argument("config", nargs="?", default="shots/take1.yaml")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    s.set_defaults(fn=cmd_gui)

    s = sub.add_parser("clean", help="forget cached stage results")
    s.add_argument("config")
    s.add_argument("stages", nargs="*")
    s.set_defaults(fn=cmd_clean)

    a = p.parse_args(argv)
    _setup_logging(a.verbose)
    try:
        return a.fn(a)
    except (config.ConfigError, RuntimeError, FileNotFoundError, ValueError) as e:
        log.error("error: %s", e)
        return 1
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
