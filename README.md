# PipelineA: avatarpipe

Film yourself, get back the same shot with an animated character standing where you were.
It runs on CPU only and uses open-source tools for every step.

```
your video ─▶ ingest ─▶ track ─▶ solve ─▶ render ─▶ composite ─▶ final.mp4
                 │       MediaPipe   3D body    Blender      ffmpeg
                 └──────────────────▶ matte ─────────────────┘
               clean plate   (paints you out of the shot)
```

| Stage | What it does | Tool |
|---|---|---|
| `ingest` | Normalizes all footage to one size and frame rate. Builds a clean plate from your empty-scene clip as a per-pixel median. | FFmpeg |
| `track` | Runs body pose (33 joints, metric 3D) and 52 ARKit-style face blendshapes on every frame. | MediaPipe (CPU) |
| `solve` | Fills gaps, applies One Euro smoothing, and places the body in 3D camera space with a least-squares solve against the real lens. | NumPy |
| `matte` | Generates a person matte, grown and feathered, so you can be painted out. | MediaPipe segmenter |
| `render` | Imports your avatar, matches the real camera, constrains the rig to the motion, drives facial shape keys, adds a shadow-catcher floor, and renders RGBA. | Blender (headless) |
| `composite` | Combines live footage, the clean plate through the matte, the avatar, a grade and grain. | FFmpeg |

Every stage is cached. If you change only `blender.engine`, only `render` and `composite` rerun.

## Install

1. **Python 3.10+** and **FFmpeg** on your PATH.
2. **Blender 4.2+** from blender.org (free). If your avatar is a `.vrm`, also add the free *VRM format* add-on in Blender under Edit → Preferences → Get Extensions.
3. Install this package:
   ```bash
   pip install -e .
   avatarpipe doctor
   ```
   The MediaPipe models (about 30 MB) download automatically on the first run to `~/.cache/avatarpipe/models`. To use a different location, set `AVATARPIPE_MODELS`.

## Shoot it

- **Lock off the camera.** A tripod or a phone propped against something works. The pipeline assumes a static camera.
- **Record the clean plate first:** 5–10 seconds of the empty scene, without moving the camera. This is what lets `mode: replace` remove you from the shot.
- Keep your whole body in frame, use even light, and wear fitted clothes that contrast with the background.
- Note your lens and roughly how high the camera is. Most phone main cameras are about 24–28 mm equivalent.

## Get an avatar

| Source | Format | Preset |
|---|---|---|
| [VRoid Studio](https://vroid.com/en/studio) (free) | `.vrm` | `auto` (uses the VRM humanoid map) |
| Mixamo characters | `.fbx` | `mixamo` |
| Anything else rigged | `.glb` / `.fbx` / `.blend` | `custom` + `bone_map.yaml` (template in `avatarpipe/templates/`) |

For facial animation, the avatar needs ARKit-named shape keys ("perfect sync" models) or VRoid's `Fcl_*` keys. With `face_preset: auto`, the pipeline detects which set the model has.

## Run it in the browser (GUI)

```bash
pip install -e ".[gui]"
avatarpipe gui shots/take1.yaml
```

This opens a page in your browser at http://127.0.0.1:7860. It only listens on your own machine. It has four tabs:

1. **Footage & avatar**: choose your clips and avatar. They're copied into the shot's `footage/` and `avatars/` folders.
2. **Settings**: mode, lens, camera height, render engine, lighting, smoothing.
3. **Run**: *Track & solve*, *Make tracking preview*, *Render everything*, rerun one stage, *Stop*, a live log and the stage status.
4. **Results**: the tracking preview and the final video, plus a button that opens the output folder.

*New shot* creates a config file and its folders. The GUI and the command line read and write the same YAML file, so you can switch between them freely. Saving from the GUI drops the comments from the YAML file.

## Run it from the command line

```bash
avatarpipe init shots/take1.yaml                 # starter config + footage/ avatars/ folders
# copy me.mp4 + empty.mp4 into shots/footage, your avatar into shots/avatars, edit the yaml
avatarpipe doctor shots/take1.yaml               # checks everything before you wait on a render
avatarpipe run shots/take1.yaml --to solve       # track + solve only (fast)
avatarpipe preview shots/take1.yaml              # green = tracked joints, orange = 3D solve reprojected
avatarpipe run shots/take1.yaml                  # everything else
```

Output goes to `runs/<project>/final.mp4`. A `runs/<project>/render/scene.blend` file is also saved. It's a normal Blender scene with the camera, targets and constraints, so you can open it to clean up hands, relight, or add props.

Other commands:

```bash
avatarpipe status shots/take1.yaml               # which stages are done or stale
avatarpipe run shots/take1.yaml --force render   # rerun one stage even if cached
avatarpipe run shots/take1.yaml --only composite # just recomposite (e.g. after a grade change)
avatarpipe run shots/take1.yaml --dry-run        # show the plan
```

## Face and lip sync

Facial expressions are tracked markerless, without dots or a mask:

- **Face close-up** (`track.face_crop`, on by default): the face is tracked on a crop of your head cut from the *original* video at full resolution. In a full-body shot this gives the face tracker several times more pixels than the working-size frame. Shoot at 1080p or 4K to get the most out of it. The track log reports `face found in N/M frames`.
- **Head rotation**: the head's turn, nod and tilt come from the face tracker. If the face is lost, the pipeline fades back to an estimate from the body landmarks. The log reports `head: from the face tracker on N/M frames`.
- **Expressions**: blinks, brows, smiles, frowns, eye squints and mouth vowels drive the avatar's expression shape keys. VRoid models use about 20 keys, and "perfect sync" (ARKit) models get all 52 one to one. Use `avatar.face_gain` (default 1.4) to exaggerate or soften them.
- **Lip sync from your voice** (`lipsync`): [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) (MIT license, CPU, offline) turns your speech into mouth shapes. By default (`mode: max`) the pipeline uses whichever opens the mouth more, the camera or your voice, so talking reads clearly even when your face is tiny.

Install Rhubarb once:

1. Download the zip for your system from the [releases page](https://github.com/DanielSWolf/rhubarb-lip-sync/releases), for example `Rhubarb-Lip-Sync-1.14.0-Windows.zip`.
2. Unzip the **whole folder** into a `tools` folder in the project, so you have `tools/Rhubarb-Lip-Sync-1.14.0-Windows/rhubarb.exe`. Keep its `res` folder next to it.
3. Run `avatarpipe doctor shots/take1.yaml`. It should report `lip sync: Rhubarb at …`.

If Rhubarb isn't installed, lip sync is skipped with a note and everything else still runs. Use `recognizer: pocketSphinx` for English (most accurate) or `phonetic` for any language.

## Composite modes

- **`replace`**: The avatar stands where you stood in your own shot. You are painted out using the clean plate and the matte. Needs `input.clean_plate`.
- **`new_scene`**: The avatar is placed over a different video or still. Needs `input.background`. For the avatar to sit on that scene's floor, the background should be shot at a similar camera height and angle.
- **`overlay`**: The avatar is drawn over your original footage. Use this to check alignment.

## Tuning

| Symptom | Fix |
|---|---|
| In `preview`, the orange skeleton drifts off the green one | Set `solve.focal_mm` to your real lens |
| Avatar floats above or sinks into the floor | Set `solve.camera_height_m`. The render log reports an estimate when it's off |
| Torso or head is twisted backwards | Set `avatar.twist_axis: -X` (or `Z` / `-Z`, depending on the rig's bone roll) |
| Jittery limbs | Lower `track.smoothing.min_cutoff` (0.6–1.0) |
| Fast moves feel laggy | Raise `track.smoothing.beta` |
| Edges of you peek out | Raise `matte.dilate_px` |
| Avatar looks pasted on | Use `blender.engine: CYCLES` for real floor shadows, match `sun_angle_deg` to your key light, and add a little `grain` |

## Speed on CPU (rough)

Measured on a 2-core cloud machine with no GPU at all, 640×360:

- **Cycles, 4 samples:** about 3.6 s per frame.
- **EEVEE:** about 4 s per frame, using software OpenGL. Any integrated GPU (Intel, AMD, Apple) makes EEVEE many times faster, so it's the default.

Render time scales with pixel count, so 720p takes about 4× as long and 1080p about 9×. Tracking is usually not the bottleneck.

- **Drafts:** use `max_height: 480` and `engine: WORKBENCH`.

## Limits worth knowing

- **Single camera.** Depth and foot contact are estimated, not measured. Arms crossing the body or turning sideways are the weakest cases.
- **Static camera only for now.** A moving camera would need a solved camera track, which is a planned stage.
- **No fingers.** Hands are driven as a single bone. Use the saved `.blend` to hand-key fingers if needed.

## Development

```bash
pip install -e ".[dev]"
pytest -q tests                                  # unit tests (no models, no Blender needed)
# full end to end on synthetic footage, offline (needs Blender or `pip install bpy`):
python tests/make_test_avatar.py -- /tmp/test_avatar.blend          # run with Blender's python / bpy
python tests/e2e_synthetic.py --blender blender --avatar /tmp/test_avatar.blend --out /tmp/e2e
```

`blender.executable` can also point at a Python interpreter that has the `bpy` module installed (`pip install bpy`). That setup is how the tests run headless.

## License

MIT
