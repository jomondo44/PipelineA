"""Runs INSIDE Blender (headless):  blender -b --python drive_avatar.py -- job.json

Builds the shot from scratch: imports the avatar, creates a camera matching the real
one, animates target empties from the solved motion, constrains the rig to them,
drives facial shape keys, adds a shadow-catcher floor and light, and renders an
RGBA PNG sequence. The saved .blend is a normal, editable Blender file.

Also works with the `bpy` Python module:  python drive_avatar.py -- job.json
"""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

ARKIT_HINT = ("eyeBlinkLeft", "jawOpen", "mouthSmileLeft", "browInnerUp", "mouthFunnel")


def log(msg):
    print(f"[avatarpipe] {msg}", flush=True)


def fail(msg):
    print(f"[avatarpipe] ERROR: {msg}", flush=True)
    sys.exit(2)


def load_job():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not argv:
        fail("usage: blender -b --python drive_avatar.py -- job.json")
    with open(argv[0], "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- scene setup
def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.cameras, bpy.data.lights,
                 bpy.data.actions, bpy.data.materials):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def import_avatar(path, armature_name=None):
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext == ".vrm":
        if not hasattr(bpy.ops.import_scene, "vrm"):
            fail("VRM import needs the 'VRM format' add-on: Edit > Preferences > Get Extensions > search 'VRM'")
        bpy.ops.import_scene.vrm(filepath=path)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        if hasattr(bpy.ops.import_scene, "fbx"):
            bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False)
        else:
            bpy.ops.wm.fbx_import(filepath=path)
    elif ext == ".blend":
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = list(src.objects)
        for ob in dst.objects:
            if ob is not None and ob.type not in ("CAMERA", "LIGHT"):
                bpy.context.scene.collection.objects.link(ob)
    else:
        fail(f"unsupported avatar format '{ext}' (use .vrm, .glb, .gltf, .fbx or .blend)")
    new = [o for o in bpy.data.objects if o not in before or ext == ".blend"]
    arms = [o for o in new if o.type == "ARMATURE" and o.name in bpy.context.scene.objects]
    if armature_name:
        arms = [o for o in arms if o.name == armature_name]
    if not arms:
        fail(f"no armature found in {path}" + (f" named '{armature_name}'" if armature_name else ""))
    if len(arms) > 1:
        arms.sort(key=lambda o: len(o.data.bones), reverse=True)
        log(f"several armatures; using '{arms[0].name}' (set avatar.armature to choose)")
    return arms[0]


def vrm_humanoid_map(arm):
    """Ask the VRM add-on which bone is which (works for any VRM 0.x / 1.0 model)."""
    ext = getattr(arm.data, "vrm_addon_extension", None)
    if ext is None:
        return {}
    out = {}
    try:  # VRM 1.0
        hb = ext.vrm1.humanoid.human_bones
        for name in ("hips", "spine", "chest", "upperChest", "neck", "head",
                     "leftUpperArm", "leftLowerArm", "leftHand", "rightUpperArm", "rightLowerArm", "rightHand",
                     "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"):
            b = getattr(hb, _snake(name), None)  # add-on uses snake_case: left_upper_arm
            bone = getattr(getattr(b, "node", None), "bone_name", "") if b is not None else ""
            if bone:
                out[name] = bone
    except AttributeError:
        pass
    if not out:
        try:  # VRM 0.x
            for item in ext.vrm0.humanoid.human_bones:
                if item.node.bone_name:
                    out[item.bone] = item.node.bone_name
        except AttributeError:
            pass
    return out


def _snake(name):
    return "".join("_" + c.lower() if c.isupper() else c for c in name)


def resolve_rig(arm, job):
    bones = set(arm.data.bones.keys())
    preset = job["avatar"]["rig_preset"]
    maps = job["rigs"]
    if preset == "custom":
        mapping = job["custom_bone_map"]
    elif preset == "vrm":
        mapping = vrm_humanoid_map(arm)
    elif preset in maps:
        mapping = maps[preset]
    else:  # auto
        mapping = vrm_humanoid_map(arm)
        if len(mapping) < 10:
            for name, m in maps.items():
                if sum(b in bones for b in m.values()) >= 12:
                    mapping, preset = m, name
                    break
        else:
            preset = "vrm"
    mapping = {k: v for k, v in mapping.items() if v in bones}
    needed = [k for k in job["drive"] if k not in mapping]
    if "hips" in needed or len(mapping) < 10:
        sample = ", ".join(sorted(bones)[:40])
        fail(f"could not map the rig (preset '{preset}'). Missing: {needed}. "
             f"Set avatar.rig_preset: custom with a bone_map file. Bones in this rig: {sample} ...")
    if needed:
        log(f"rig '{preset}': not driving {needed} (bones not found)")
    log(f"rig preset: {preset} ({len(mapping)} bones mapped)")
    return mapping


# ---------------------------------------------------------------- camera / targets
def make_camera(job):
    c = job["motion"]["camera"]
    cam_data = bpy.data.cameras.new("RealCamera")
    cam_data.lens = c["focal_mm"]
    cam_data.sensor_width = c["sensor_mm"]
    cam_data.sensor_fit = "AUTO"
    cam_data.clip_start, cam_data.clip_end = 0.05, 500
    cam = bpy.data.objects.new("RealCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (0.0, 0.0, c["height_m"])
    cam.rotation_euler = (math.radians(90.0 - c["tilt_deg"]), 0.0, 0.0)  # looking along +Y
    bpy.context.scene.camera = cam
    bpy.context.view_layer.update()
    return cam


def make_targets(job, cam, names):
    """One empty per driven point, parented to the camera, keyed every frame."""
    pts = job["motion"]["points"]
    coll = bpy.data.collections.new("MocapTargets")
    bpy.context.scene.collection.children.link(coll)
    out = {}
    for n in names:
        e = bpy.data.objects.new(f"T_{n}", None)
        e.empty_display_type = "SPHERE"
        e.empty_display_size = 0.03
        coll.objects.link(e)
        e.parent = cam
        e.matrix_parent_inverse = Matrix.Identity(4)
        e.hide_render = True
        track = pts[n]
        for f, p in enumerate(track, start=1):
            e.location = p
            e.keyframe_insert("location", frame=f)
        out[n] = e
    return out


def world_track(cam, local_track):
    m = cam.matrix_world
    return [m @ Vector(p) for p in local_track]


# ---------------------------------------------------------------- rig
def bone_head_world(arm, bone):
    return arm.matrix_world @ arm.data.bones[bone].head_local


def avatar_leg_length(arm, mapping):
    try:
        a = bone_head_world(arm, mapping["leftUpperLeg"])
        b = bone_head_world(arm, mapping["leftLowerLeg"])
        c = bone_head_world(arm, mapping["leftFoot"])
    except KeyError:
        return None
    return (b - a).length + (c - b).length


def constrain(arm, mapping, targets, drive, twist_axis):
    axis = {"X": "TRACK_X", "-X": "TRACK_NEGATIVE_X", "Z": "TRACK_Z", "-Z": "TRACK_NEGATIVE_Z"}[twist_axis]
    for hname, spec in drive.items():
        bname = mapping.get(hname)
        if not bname:
            continue
        pb = arm.pose.bones[bname]
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        pb.rotation_mode = "QUATERNION"
        if "copy_location" in spec:
            c = pb.constraints.new("COPY_LOCATION")
            c.name = "ap_loc"
            c.target = targets[spec["copy_location"]]
        if "track" in spec:
            c = pb.constraints.new("DAMPED_TRACK")
            c.name = "ap_aim"
            c.target = targets[spec["track"]]
            c.track_axis = "TRACK_Y"
        if "twist" in spec:
            c = pb.constraints.new("LOCKED_TRACK")
            c.name = "ap_twist"
            c.target = targets[spec["twist"]]
            c.lock_axis = "LOCK_Y"
            c.track_axis = axis


# ---------------------------------------------------------------- face
def face_mapping(arm, job):
    shapes = job["motion"]["shapes"]
    preset = job["avatar"]["face_preset"]
    if preset == "none" or not shapes:
        return [], {}
    meshes = [o for o in bpy.context.scene.objects
              if o.type == "MESH" and o.data.shape_keys and
              (o.find_armature() == arm or o.parent == arm)]
    keys = set()
    for m in meshes:
        keys |= set(m.data.shape_keys.key_blocks.keys())
    lower = {k.lower(): k for k in keys}
    if preset == "auto":
        if sum(h.lower() in lower for h in ARKIT_HINT) >= 3:
            preset = "arkit"
        elif any(k.startswith("Fcl_") for k in keys):
            preset = "vroid"
        else:
            log("face: no ARKit or VRoid shape keys found; skipping facial animation")
            return [], {}
    if preset == "arkit":
        fmap = {lower[n.lower()]: {n: 1.0} for n in shapes if n.lower() in lower}
    elif preset == "custom":
        fmap = job["custom_face_map"]
    else:
        fmap = job["faces"][preset]
    fmap = {k: v for k, v in fmap.items() if k in keys}
    log(f"face preset: {preset} ({len(fmap)} shape keys driven)")
    return meshes, fmap


def key_face(meshes, fmap, shapes, frames):
    for m in meshes:
        kb = m.data.shape_keys.key_blocks
        for key, sources in fmap.items():
            if key not in kb:
                continue
            block = kb[key]
            for f in range(frames):
                v = sum(w * shapes[s][f] for s, w in sources.items() if s in shapes)
                block.value = max(0.0, min(1.0, v))
                block.keyframe_insert("value", frame=f + 1)


# ---------------------------------------------------------------- floor / light / render
def percentile(vals, q):
    s = sorted(vals)
    if not s:
        return 0.0
    i = (len(s) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def make_floor(job, cam, engine):
    pts = job["motion"]["points"]
    zs = []
    for n in ("l_heel", "r_heel", "l_foot_index", "r_foot_index"):
        zs += [p.z for p in world_track(cam, pts[n])]
    floor_z = percentile(zs, 0.05) - 0.01
    if abs(floor_z) > 0.25:
        log(f"floor estimated at z={floor_z:.2f}m: solve.camera_height_m may be off by about {-floor_z:.2f}m")
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 15, floor_z))
    floor = bpy.context.active_object
    floor.name = "ShadowCatcher"
    if engine == "CYCLES" and job["blender"]["shadow_catcher"]:
        floor.is_shadow_catcher = True
    else:
        floor.hide_render = True
    return floor_z


def make_light(job):
    b = job["blender"]
    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = b["sun_strength"]
    sun.angle = math.radians(8)
    ob = bpy.data.objects.new("Sun", sun)
    bpy.context.scene.collection.objects.link(ob)
    ob.rotation_euler = [math.radians(a) for a in b["sun_angle_deg"]]
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    if not world.node_tree:
        world.use_nodes = True  # Blender < 5 creates the node tree on demand
    bg = world.node_tree.nodes.get("Background") if world.node_tree else None
    if bg:
        bg.inputs[0].default_value = (0.6, 0.62, 0.65, 1.0)
        bg.inputs[1].default_value = b["world_strength"]


def set_engine(scene, engine, samples):
    if engine == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        try:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        except TypeError:
            pass
    elif engine == "WORKBENCH":
        scene.render.engine = "BLENDER_WORKBENCH"
    else:
        for ident in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
            try:
                scene.render.engine = ident
                break
            except TypeError:
                continue
        eevee = getattr(scene, "eevee", None)
        if eevee is not None and hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = samples


def setup_render(job):
    m = job["motion"]
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, m["frames"]
    fps = m["fps"]
    scene.render.fps = max(1, int(round(fps)))
    scene.render.fps_base = scene.render.fps / fps if fps else 1.0
    scene.render.resolution_x, scene.render.resolution_y = m["width"], m["height"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = os.path.join(job["frames_dir"], "avatar_")
    scene.render.use_file_extension = True
    try:
        scene.view_settings.view_transform = "Standard"  # match camera footage, not AgX/Filmic
    except TypeError:
        pass
    set_engine(scene, job["blender"]["engine"], job["blender"]["samples"])


def main():
    job = load_job()
    with open(job["motion_path"], "r", encoding="utf-8") as fh:
        job["motion"] = json.load(fh)
    m = job["motion"]
    log(f"{m['frames']} frames {m['width']}x{m['height']} @ {m['fps']:.3f} fps, engine {job['blender']['engine']}")

    clear_scene()
    arm = import_avatar(job["avatar"]["path"], job["avatar"].get("armature"))
    mapping = resolve_rig(arm, job)
    cam = make_camera(job)

    # scale avatar so its legs match yours: keeps feet planted on your footprints
    scale = job["avatar"]["scale"]
    if scale == "auto":
        leg = avatar_leg_length(arm, mapping)
        perf = m["performer"]["leg_m"]
        scale = (perf / leg) if leg and perf and math.isfinite(perf) else 1.0
        log(f"auto scale: avatar legs {leg or 0:.2f}m -> yours {perf:.2f}m (x{scale:.3f})")
    arm.scale = [s * float(scale) for s in arm.scale]
    bpy.context.view_layer.update()

    used = sorted({v for spec in job["drive"].values() for v in spec.values()} |
                  {"l_heel", "r_heel", "l_foot_index", "r_foot_index"})
    targets = make_targets(job, cam, used)
    constrain(arm, mapping, targets, job["drive"], job["avatar"]["twist_axis"])
    meshes, fmap = face_mapping(arm, job)
    if fmap:
        key_face(meshes, fmap, m["shapes"], m["frames"])

    make_light(job)
    setup_render(job)
    make_floor(job, cam, job["blender"]["engine"])

    if job.get("blend_out"):
        bpy.ops.wm.save_as_mainfile(filepath=job["blend_out"], check_existing=False)
        log(f"saved {job['blend_out']}")
    if job.get("no_render"):
        return
    log("rendering...")
    bpy.ops.render.render(animation=True)
    log("render finished")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        fail(str(e))
