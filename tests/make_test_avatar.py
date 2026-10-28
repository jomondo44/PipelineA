"""Build a tiny VRoid-named test avatar (.blend) with Blender: python make_test_avatar.py -- out.blend

Capsule body parts rigidly skinned to J_Bip_* bones, plus a Fcl_MTH_A mouth shape key,
so the rig preset 'vroid' and face preset 'vroid' are both exercised.
"""
import sys

import bpy
from mathutils import Vector

out = sys.argv[sys.argv.index("--") + 1]
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)

B = {  # name: (head, tail, parent, radius, color)
    "J_Bip_C_Hips": ((0, 0, 0.95), (0, 0, 1.05), None, 0.15, (0.2, 0.3, 0.8)),
    "J_Bip_C_Spine": ((0, 0, 1.05), (0, 0, 1.25), "J_Bip_C_Hips", 0.14, (0.9, 0.4, 0.1)),
    "J_Bip_C_Chest": ((0, 0, 1.25), (0, 0, 1.35), "J_Bip_C_Spine", 0.16, (0.9, 0.4, 0.1)),
    "J_Bip_C_UpperChest": ((0, 0, 1.35), (0, 0, 1.45), "J_Bip_C_Chest", 0.17, (0.9, 0.4, 0.1)),
    "J_Bip_C_Neck": ((0, 0, 1.45), (0, 0, 1.55), "J_Bip_C_UpperChest", 0.05, (0.95, 0.8, 0.7)),
    "J_Bip_C_Head": ((0, 0, 1.55), (0, 0, 1.78), "J_Bip_C_Neck", 0.11, (0.95, 0.8, 0.7)),
}
for side, sx in (("L", 1), ("R", -1)):
    B.update({
        f"J_Bip_{side}_UpperArm": ((0.18 * sx, 0, 1.43), (0.20 * sx, 0, 1.16), "J_Bip_C_UpperChest", 0.05, (0.9, 0.4, 0.1)),
        f"J_Bip_{side}_LowerArm": ((0.20 * sx, 0, 1.16), (0.22 * sx, 0, 0.92), f"J_Bip_{side}_UpperArm", 0.04, (0.95, 0.8, 0.7)),
        f"J_Bip_{side}_Hand": ((0.22 * sx, 0, 0.92), (0.23 * sx, -0.01, 0.82), f"J_Bip_{side}_LowerArm", 0.045, (0.95, 0.8, 0.7)),
        f"J_Bip_{side}_UpperLeg": ((0.10 * sx, 0, 0.93), (0.10 * sx, 0, 0.50), "J_Bip_C_Hips", 0.075, (0.2, 0.3, 0.8)),
        f"J_Bip_{side}_LowerLeg": ((0.10 * sx, 0, 0.50), (0.10 * sx, 0.01, 0.08), f"J_Bip_{side}_UpperLeg", 0.06, (0.2, 0.3, 0.8)),
        f"J_Bip_{side}_Foot": ((0.10 * sx, 0.01, 0.08), (0.10 * sx, -0.13, 0.02), f"J_Bip_{side}_LowerLeg", 0.05, (0.1, 0.1, 0.1)),
    })

arm_data = bpy.data.armatures.new("Armature")
arm = bpy.data.objects.new("Armature", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
for name, (h, t, parent, *_r) in B.items():
    eb = arm_data.edit_bones.new(name)
    eb.head, eb.tail = h, t
    if parent:
        eb.parent = arm_data.edit_bones[parent]
        eb.use_connect = False
bpy.ops.object.mode_set(mode="OBJECT")

parts = []
for name, (h, t, _p, r, col) in B.items():
    h, t = Vector(h), Vector(t)
    d = t - h
    if name == "J_Bip_C_Head":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(h + t) / 2, segments=24, ring_count=12)
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, location=(h + t) / 2, vertices=12)
        ob = bpy.context.active_object
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    ob = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    mat = bpy.data.materials.new(name + "_mat")
    mat.diffuse_color = (*col, 1)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*col, 1)
    ob.data.materials.append(mat)
    vg = ob.vertex_groups.new(name=name)
    vg.add(range(len(ob.data.vertices)), 1.0, "REPLACE")
    parts.append(ob)

# a nose (on the head bone) so head turns are visible in renders; the avatar faces -Y
bpy.ops.mesh.primitive_cone_add(radius1=0.035, depth=0.09, location=(0, -0.14, 1.67), rotation=(1.5708, 0, 0))
nose = bpy.context.active_object
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
nm = bpy.data.materials.new("nose_mat")
nm.diffuse_color = (0.85, 0.1, 0.1, 1)
nose.data.materials.append(nm)
nose.vertex_groups.new(name="J_Bip_C_Head").add(range(len(nose.data.vertices)), 1.0, "REPLACE")
parts.append(nose)

bpy.ops.object.select_all(action="DESELECT")
for ob in parts:
    ob.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.object.join()
body = bpy.context.active_object
body.name = "Body"
body.shape_key_add(name="Basis")
sk = body.shape_key_add(name="Fcl_MTH_A")
for i, v in enumerate(body.data.vertices):
    if v.co.z < 1.64 and v.co.z > 1.56 and v.co.y < -0.05:   # front lower face: open "mouth"
        sk.data[i].co = v.co + Vector((0, 0.03, -0.04))
mod = body.modifiers.new("Armature", "ARMATURE")
mod.object = arm
body.parent = arm
bpy.ops.wm.save_as_mainfile(filepath=out)
print("saved", out)
