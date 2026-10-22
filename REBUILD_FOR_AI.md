# Rebuild guide (for an AI or a developer starting cold)

This file is written so that someone with **no prior context** can install,
verify, understand and modify the system. Read it top to bottom once; each
section ends with a way to check that the step is done.

Related repositories: `stg-recorder` (Raspberry Pi recorder that produces the
input) and `stg-player` (plays the trained models).

---

## 1. What the system does

A rig of four USB cameras on a rigid frame (about 15 x 9 cm) records
synchronized MJPEG video plus stereo audio. The product is **a window**: the
viewer looks through the rig frame and may move slightly, but only inside the
convex hull of the four cameras. Nothing outside that hull is a goal.

Processing chain:

```
calibration set --calibrate.py--> rig/ --+
                                         +-- prep_4d.py --> stg<N>/ (dataset) --trainer--> <name>_out/ (model) --> stg-player
recording (rec_*) -----------------------+
```

All three steps run from the studio (tkinter) or from the command line.

## 2. Installation, step by step

Everything Python runs in the repository's own `.venv`, created **without**
`--system-site-packages`: the system Python is used only to create it, and
nothing is installed into the system Python. Another project upgrading a
system package (for example NumPy to 2) therefore cannot break the trainer.

`setup.ps1` automates this; the manual equivalent, for when it fails or for
another platform:

| # | Action | Done when |
|---|---|---|
| 1 | Install NVIDIA driver, CUDA Toolkit 13.0, Visual Studio with C++ workload, Python 3.12, git | `nvcc --version` shows 13.0, `py -3.12 --version` works |
| 2 | `py -3.12 -m venv .venv` then `.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130` | `import torch; torch.cuda.is_available()` is `True`, `numpy.__version__ == "1.26.4"` |
| 3 | `git -c core.longpaths=true -c core.autocrlf=false clone https://github.com/oppo-us-research/SpacetimeGaussians.git third_party\SpacetimeGaussians`, `git checkout 427abfc`, `git -c core.autocrlf=false apply --whitespace=nowarn upstream\stg_427abfc.patch` | `git status --short` in the clone shows exactly 12 `M` lines |
| 4 | In a shell after `vcvars64.bat`, with `CUDA_HOME` set and `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0+PTX"`, `DISTUTILS_USE_SDK=1`: `pip install --no-build-isolation` each of `thirdparty\gaussian_splatting\submodules\{gaussian_rasterization_ch3,forward_lite,simple-knn}` | `python -c "import torch, simple_knn._C, forward_lite, diff_gaussian_rasterization_ch3"` succeeds (**torch first**: it adds the DLL search path the extensions need) |
| 5 | Unpack a COLMAP Windows CUDA release (4.1.0 was used) so that `third_party\colmap\bin\colmap.exe` exists | `colmap.exe version` prints a version |
| 6 | `.venv\Scripts\python.exe tools\check_time_range.py`, then the same for `check_resume_mode.py` and `check_studio_buttons.py` (the last two need a desktop) | each ends with `OK` |
| 7 | Start `.venv\Scripts\pythonw.exe studio\stg_studio.py` | the window opens; lower right shows `v<version> (<hash>)` |

Not compiled: the `ch9` rasterizer and `forward_full` (the "lite" model is
used), and the `thirdparty/mmcv` submodule (the patch makes it a lazy import).

### Pitfalls already met

| Problem | Reason |
|---|---|
| NumPy 2 breaks the trainer | it uses `np.NaN`; pin `numpy==1.26.4` |
| two OpenCV packages installed | the wrong one may load; install only `opencv-python` |
| `Filename too long` on clone | glm test files under `realtimedemolite/`; enable `core.longpaths` |
| CUDA extension compile rejects the MSVC version | the patch adds `-allow-unsupported-compiler -std=c++20` to the five `setup.py` files |
| black console windows flash during preprocessing | every subprocess (including COLMAP called by the pipeline) needs `creationflags=CREATE_NO_WINDOW` |

## 3. Data formats

### Recording (`rec_YYYYMMDD_HHMMSS/`, from the recorder)

```
session.json            {"mode": "rec", "streams": [{"name": "cam_A", ...}, {"name": "audio", "kind": "audio"}],
                         "capture": {"resolution": [W, H], "fps", "exposure_us", "gain"}, "created_at", ...}
cam_A/stream.mjpg       concatenated raw JPEG frames
cam_A/index.jsonl       one line per frame: {"i": index, "ts_ns": timestamp, "off": byte offset, "len": bytes}
cam_B/ cam_C/ cam_D/    same
audio/audio.wav         stereo PCM
audio/index.json        {"start_ts_ns", "sample_rate_measured_hz", "frames", "mic_spacing_mm", "offset_ms", "xruns"}
```

Camera names are assigned by sort order of the USB ports; changing ports
requires a new calibration.

### Calibration set

`session.json` with `"mode": "calib"`, and `cam_X/shot_NNN.jpg`: still sets
taken by all cameras at once while a textured board is moved. 15 or more sets.

### Shooting root

The folder the studio works on. One recording plus its calibration:

```
<shooting root>/
    rec_YYYYMMDD_HHMMSS/    the recording (exactly one)
    rig/                    copy of the calibration used for this shoot
    stg<N>/                 dataset, written by prep_4d.py
    <name>_out/             model, written by the trainer
```

### `rig/` (from calibrate.py, about 5 KB, portable)

| File | Content |
|---|---|
| `rig_config.json` | per camera: `image_prefix`, OPENCV intrinsics, `cam_from_rig_rotation` (w,x,y,z) / `cam_from_rig_translation` in metres; the reference camera has `ref_sensor: true` |
| `cameras.txt`, `rigs.txt` | the same in COLMAP text format |
| `prefixes.json` | COLMAP camera_id -> `cam_X/` |
| `scale.json` | `{"metric": true, "unit": "m", "mm_per_unit", "edges": [...]}`; always written, so "deliberately unscaled" differs from "file lost" |
| `calib_report.json` | acceptance numbers and `passed` |

### Dataset `stg<N>/` (from prep_4d.py; N = number of valid time steps)

| File | Content |
|---|---|
| `colmap_0/ ... colmap_{N-1}/` | per time step: `images/cam_X.jpg` (undistorted, principal point centred) and `sparse/0/` (shared poses, a disjoint slice of the initial point cloud) |
| `times_group.json` | `{"<n>/<cam>.jpg": t}` normalized time per image = midpoint of its group (used) |
| `times.json` | same with each camera's own timestamp (continuous time; kept for comparison) |
| `groups.jsonl`, `phase_report.md` | how frames were grouped; per-camera phase and dropped frames |
| `poses_bounds.npy`, `cfg_lite.json` | trainer inputs |
| `window.json` | everything the trainer and player need: `cameras`, `nsteps`, `duration_s`, `background`, `wh`, `fov`, `ctr`, `fwd`, `win` (window corners, 9 decimals), `win_order`, `near_far`, `unit_mm`, `audio` block, `tri_reproj_px` |

`window.json` from stage 1 only (no calibration) lacks the geometry fields
and contains a `_geometry` note instead; such a folder has no `colmap_*` and
is not treated as a dataset.

### Model `<name>_out/` (from the trainer)

```
point_cloud/iteration_<k>/point_cloud.ply     the model
_studio/                                      control folder used by the studio
    train.log, progress.jsonl, preview.jpg, preview.mp4, train_meta.json, pid.txt
    cmd.txt (state: "" or "pause"), save.flag, render.flag, stop.flag (one-shot actions)
    state.pth                                 checkpoint for resuming (written on save/autosave)
window.json                                   only when a time range was trained (describes the clip)
```

## 4. Code map

### pipeline/

| File | Role |
|---|---|
| `calibrate.py` | 8 steps: check set, sharpness filter, initial focal length from HFOV, stage A (SIMPLE_RADIAL, several fixed mapper seeds), stage B (OPENCV), rig calibration + bundle adjustment with principal point refinement, metric scale from `--edge` distances, write `rig/`, acceptance table |
| `prep_4d.py` | stage 1: group frames by smallest span, phase report, time labels; stage 2: undistortion, poses taken from the rig, triangulation, MVS initial point cloud (sparse fallback), rectify world frame to the window, write `colmap_N`, `window.json`, audio block |
| `audio_level.py` | EBU R128 loudness and gain via ffmpeg (from `imageio-ffmpeg`) |

Output lines that the studio parses (keep them exact if you edit prep_4d.py):
`Valid groups X/Y (Z%)  sampled N time steps` and
`Dataset -> stgN (...)`; phase report headings `## Group time span` and
`## Discarded groups`.

### studio/

| File | Role |
|---|---|
| `stg_studio.py` | main window, custom tab bar, shooting-root picker, settings, version tag, no window icon (`_no_icon`) |
| `prep_tab.py` / `prep_runner.py` | preprocessing tab; read-only pre-check runs stage 1 into a temporary folder |
| `train_tab.py` / `runner.py` | training tab; `runner.build()` assembles the trainer command and environment; `Runner` is the state machine `idle -> running <-> paused -> finished (holding GPU memory) -> stopping -> idle` |
| `combo_tab.py` | combined training: split a time range into segments, train them one after another, merge with `tools/merge_segments.py` |
| `calib_tab.py` / `calib_runner.py` | calibration tab; six measured distances are required |
| `params.py` | every training parameter with tier defaults (`fast`, `balanced`, `max`), tooltips, and `RESUME_FROZEN` (parameters that cannot change when resuming) |
| `curve.py`, `preview_win.py`, `video_win.py`, `tooltip.py` | widgets |
| `joblimit.py` | Windows job object: child processes die with the studio |
| `orphans.py` | lists training runs left over from a previous session (never kills) |
| `version.py` | `STUDIO_VERSION` |

UI rules followed throughout: no logo in the title bar, no emoji in UI
strings (state is shown by colour), no console windows for subprocesses,
results shown inside the main window, displayed state recomputed from the
current state on every poll rather than written once on an event.

### tools/

| File | Role |
|---|---|
| `train_cmd.py` | finds the dataset, computes `STG_CAMERAS_EXTENT`, `--duration`, background, and prints a ready-to-run command; also defines the paths of the interpreter (`STG_PYTHON`) and trainer (`STG_HOME`) |
| `merge_segments.py` | merges segment PLYs, remapping time fields, and writes a `segments` table |
| `check_*.py` | self-checks (section 2 step 6) |

### Trainer changes (`upstream/stg_427abfc.patch`)

All behaviour changes are opt-in environment variables; without them the
trainer behaves like upstream.

| Variable | Effect |
|---|---|
| `STG_TIMES` | per-image time from a JSON file instead of the folder index |
| `STG_TIME_RANGE` | train only time steps `a,b`, renormalized to 0..1 |
| `STG_HOLDOUT_T` | hold out time steps instead of a camera (a corner camera would be extrapolation) |
| `STG_CAMERAS_EXTENT` | scene scale for density control (a narrow rig would give a far too small value) |
| `STG_ZNEAR_PRUNE` | near-point pruning threshold (upstream 4.5 deletes every point of a metric scene) |
| `STG_GNUM_MAX` | hard cap on the number of Gaussians |
| `STG_OMEGA_ITER` | iteration of the omega/rotation split (must come after densification ends) |
| `STG_EMS_VIEWS` | number of views guided sampling acts on (upstream hard-codes 2) |
| `STG_BACKGROUND`, `STG_BG_RANDOM` | background colour, or random per iteration |
| `STG_EARLYSTOP` | `interval:patience[:tol]` loss-based stop |
| `STG_CHECKPOINT`, `STG_CHECKPOINT_EVERY`, `STG_CKPT_ITERS`, `STG_RESUME` | control folder, autosave, resuming from `state.pth` |
| `STG_HOLD` | after finishing, stay loaded and wait for commands |
| `STG_PROGRESS`, `STG_PROGRESS_EVERY`, `STG_PREVIEW`, `STG_PREVIEW_EVERY`, `STG_PREVIEW_MM` | progress JSONL and preview image for the UI |

Marker lines in `train.log` read by the studio: `[checkpoint] finished`,
`saved at iteration N`, `stopped by request at iteration N`.

To change the trainer: edit files in `third_party/SpacetimeGaussians`, then
regenerate the patch so the change survives a reinstall:

```powershell
cd third_party\SpacetimeGaussians
git -c core.autocrlf=false diff --binary --diff-filter=M 427abfc -- . ":(exclude)thirdparty/mmcv" > ..\..\upstream\stg_427abfc.patch
```

## 5. Building a similar application

To build another UI on the same pipeline, the minimal contract is:

1. Run `pipeline/prep_4d.py <shooting root>` and parse the two output lines above.
2. Run `tools/train_cmd.py <shooting root> --tier <tier>` to get the trainer
   command, or call `runner.build()` to get it as a list with an environment.
3. Give the trainer an `STG_CHECKPOINT` folder; write `pause` / `""` to
   `cmd.txt` and create `save.flag` / `render.flag` / `stop.flag` for actions;
   read `progress.jsonl` (one JSON object per line: `iter`, `loss`,
   `n_gauss`, `max_scale`, `elapsed_s`, ...) and `preview.jpg`.
4. A flag file is deleted by the trainer **after** the action finishes;
   "flag still present" means "still busy".

## 6. Verification checklist

| # | Check | Pass |
|---|---|---|
| 1 | `.venv\Scripts\python.exe tools\check_time_range.py` | `OK` |
| 2 | `.venv\Scripts\python.exe tools\check_resume_mode.py` | `OK` |
| 3 | `.venv\Scripts\python.exe tools\check_studio_buttons.py` | `OK` |
| 4 | studio opens | version tag visible |
| 5 | calibrate -> preprocess -> train (`fast`) -> view in stg-player | **visual inspection**; numeric metrics such as PSNR do not reveal depth errors |
