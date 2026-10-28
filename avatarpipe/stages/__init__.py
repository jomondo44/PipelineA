"""The pipeline definition: order, dependencies, cache keys."""
from __future__ import annotations

from ..pipeline import Pipeline, Stage
from . import composite, ingest, matte, render, solve, track


def build() -> Pipeline:
    return Pipeline([
        Stage("ingest", ingest.run, sections=("input", "composite.background_scale"),
              inputs=ingest.inputs, describe="normalize footage, clean plate, background"),
        Stage("track", track.run, sections=("track.pose_model", "track.face", "track.min_confidence"),
              inputs=track.inputs, deps=("ingest",), describe="MediaPipe pose + face on every frame"),
        Stage("solve", solve.run, sections=("solve", "track.smoothing"),
              inputs=solve.inputs, deps=("track",), describe="smooth + place body in 3D camera space"),
        Stage("matte", matte.run, sections=("matte",), inputs=matte.inputs, deps=("ingest",),
              enabled=matte.enabled, describe="person matte for paint-out"),
        Stage("render", render.run, sections=("avatar", "blender"), inputs=render.inputs,
              deps=("solve",), enabled=render.enabled, describe="drive avatar in Blender, render RGBA"),
        Stage("composite", composite.run, sections=("composite", "matte.enabled"),
              inputs=composite.inputs, deps=("ingest", "matte", "render"),
              enabled=lambda cfg: bool(cfg["avatar"]["path"]), describe="layer everything into the final video"),
    ])
