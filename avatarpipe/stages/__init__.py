"""The pipeline definition: order, dependencies, cache keys."""
from __future__ import annotations

from ..pipeline import Pipeline, Stage
from . import composite, ingest, lipsync, matte, preview, render, solve, track


def build() -> Pipeline:
    return Pipeline([
        Stage("ingest", ingest.run, sections=("input", "composite.background_scale"),
              inputs=ingest.inputs, describe="normalize footage, clean plate, background", version="2"),
        Stage("track", track.run, sections=("track.pose_model", "track.face", "track.min_confidence",
                                            "track.face_crop", "track.face_crop_size"),
              inputs=track.inputs, deps=("ingest",), describe="MediaPipe pose + face on every frame",
              version="2"),
        Stage("lipsync", lipsync.run, sections=("lipsync.recognizer",), inputs=lipsync.inputs,
              deps=("ingest",), enabled=lipsync.enabled, describe="mouth shapes from your voice (Rhubarb)",
              version="2"),
        Stage("solve", solve.run, sections=("solve", "track.smoothing", "lipsync.enabled", "lipsync.mode",
                                            "lipsync.strength"),
              inputs=solve.inputs, deps=("track", "lipsync"), describe="smooth + place body in 3D camera space",
              version="3"),
        Stage("preview", lambda ctx: [preview.run(ctx)], deps=("ingest", "solve"),
              describe="tracking preview video (green = tracked, orange = 3D solve)"),
        Stage("matte", matte.run, sections=("matte",), inputs=matte.inputs, deps=("ingest",),
              enabled=matte.enabled, describe="person matte for paint-out"),
        Stage("render", render.run, sections=("avatar", "blender"), inputs=render.inputs,
              deps=("solve",), enabled=render.enabled, describe="drive avatar in Blender, render RGBA"),
        Stage("composite", composite.run, sections=("composite", "matte.enabled"),
              inputs=composite.inputs, deps=("ingest", "matte", "render"),
              enabled=lambda cfg: bool(cfg["avatar"]["path"]), describe="layer everything into the final video"),
    ])
