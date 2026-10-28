"""Stage runner with content-keyed caching.

Each stage declares which config sections and input files it depends on.
Its cache key = hash(code version, config sections, input fingerprints,
upstream keys). If the key matches the last successful run and all its
outputs still exist, the stage is skipped.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger("avatarpipe")


@dataclass
class Context:
    cfg: dict
    workdir: Path
    dry_run: bool = False
    keys: dict = field(default_factory=dict)

    def dir(self, name: str) -> Path:
        d = self.workdir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, stage: str, name: str) -> Path:
        return self.dir(stage) / name

    def read_json(self, stage: str, name: str):
        with open(self.path(stage, name), "r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_json(self, stage: str, name: str, data) -> Path:
        p = self.path(stage, name)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        tmp.replace(p)
        return p


@dataclass
class Stage:
    name: str
    run: Callable[[Context], list]          # returns list of output paths
    sections: tuple = ()                     # config sections it depends on
    inputs: Callable[[Context], list] = lambda ctx: []
    deps: tuple = ()
    enabled: Callable[[dict], bool] = lambda cfg: True
    version: str = "1"
    describe: str = ""


def _canon(v):
    """Make equal settings hash equally: 28 == 28.0, 1.2000000001 == 1.2."""
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, float):
        v = round(v, 6)
        return int(v) if v.is_integer() else v
    return v


def _cfg_get(cfg: dict, dotted: str):
    """'track' -> whole section; 'composite.mode' -> one key."""
    cur = cfg
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def fingerprint(path: str | Path | None) -> str:
    if not path:
        return "none"
    p = Path(path)
    if not p.exists():
        return f"missing:{p}"
    if p.is_dir():
        parts = [fingerprint(c) for c in sorted(p.rglob("*")) if c.is_file()]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()
    st = p.stat()
    h = hashlib.sha256(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode())
    # Small files (configs, maps, avatars < 64 MB): hash content too, so edits are caught
    if st.st_size < 64 * 1024 * 1024:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages
        self.by_name = {s.name: s for s in stages}

    def names(self) -> list[str]:
        return [s.name for s in self.stages]

    def key_for(self, stage: Stage, ctx: Context) -> str:
        payload = {
            "stage": stage.name,
            "version": stage.version,
            "cfg": {s: _canon(_cfg_get(ctx.cfg, s)) for s in stage.sections},
            "inputs": [fingerprint(p) for p in stage.inputs(ctx)],
            "deps": {d: ctx.keys.get(d) for d in stage.deps},
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def _state_path(self, ctx: Context, name: str) -> Path:
        d = ctx.workdir / ".state"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}.json"

    def load_state(self, ctx: Context, name: str) -> dict | None:
        p = self._state_path(ctx, name)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def is_fresh(self, ctx: Context, stage: Stage, key: str) -> bool:
        st = self.load_state(ctx, stage.name)
        if not st or st.get("key") != key:
            return False
        return all(Path(o).exists() for o in st.get("outputs", []))

    def plan(self, ctx: Context, only=None, start=None, stop=None, force=()):
        """Yield (stage, key, action) where action is run|skip|disabled."""
        names = self.names()
        lo = names.index(start) if start else 0
        hi = names.index(stop) if stop else len(names) - 1
        rerun: set[str] = set()   # anything that runs makes everything downstream of it run too
        for i, stage in enumerate(self.stages):
            if not stage.enabled(ctx.cfg):
                ctx.keys[stage.name] = "disabled"
                yield stage, "disabled", "disabled"
                continue
            key = self.key_for(stage, ctx)
            ctx.keys[stage.name] = key
            selected = (stage.name in only) if only else (lo <= i <= hi)
            upstream_ran = any(d in rerun for d in stage.deps)
            if not selected:
                fresh = self.is_fresh(ctx, stage, key) and not upstream_ran
                yield stage, key, "skip" if fresh else "stale"
            elif (stage.name in force or "all" in force or upstream_ran
                  or not self.is_fresh(ctx, stage, key)):
                rerun.add(stage.name)
                yield stage, key, "run"
            else:
                yield stage, key, "skip"

    def run(self, ctx: Context, only=None, start=None, stop=None, force=()) -> int:
        ctx.workdir.mkdir(parents=True, exist_ok=True)
        planned: set[str] = set()   # stages that ran (or would run, in a dry run) this time
        for stage, key, action in self.plan(ctx, only, start, stop, force):
            if action == "disabled":
                log.info("· %-10s disabled for this config", stage.name)
                continue
            if action == "skip":
                log.info("✓ %-10s up to date (%s)", stage.name, key)
                continue
            if action == "stale":
                log.warning("! %-10s out of date but not selected; downstream may use old data", stage.name)
                continue
            for d in stage.deps:
                dep = self.by_name[d]
                if not dep.enabled(ctx.cfg) or d in planned:
                    continue
                if not self.is_fresh(ctx, dep, ctx.keys[d]):
                    raise RuntimeError(
                        f"stage '{stage.name}' needs '{d}', which is not up to date. "
                        f"Run without --only, or include '{d}'.")
            log.info("▶ %-10s %s", stage.name, stage.describe)
            planned.add(stage.name)
            if ctx.dry_run:
                continue
            # forget the old result first: an interrupted rerun must not look finished
            self._state_path(ctx, stage.name).unlink(missing_ok=True)
            t0 = time.time()
            outputs = [str(o) for o in (stage.run(ctx) or [])]
            missing = [o for o in outputs if not Path(o).exists()]
            if missing:
                raise RuntimeError(f"stage '{stage.name}' did not produce: {missing}")
            self._state_path(ctx, stage.name).write_text(json.dumps({
                "key": key, "outputs": outputs, "finished": time.time(),
                "seconds": round(time.time() - t0, 1),
            }, indent=2))
            log.info("✓ %-10s done in %.1fs", stage.name, time.time() - t0)
        return 0

    def status(self, ctx: Context) -> list[tuple[str, str, str]]:
        rows = []
        for stage, key, action in self.plan(ctx, only=None):
            if action == "disabled":
                rows.append((stage.name, "disabled", ""))
                continue
            st = self.load_state(ctx, stage.name)
            fresh = self.is_fresh(ctx, stage, key)
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(st["finished"])) if st else ""
            rows.append((stage.name, "up to date" if fresh else ("stale" if st else "not run"), when))
        return rows
