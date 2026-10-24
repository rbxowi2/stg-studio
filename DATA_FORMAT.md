# Input data format

What stg-studio reads, so recordings can come from something other than
`stg-recorder`. Fields marked **required** are read by the pipeline; anything
else in the files is ignored.

There are two kinds of input:

| Input | Used by | Produces |
|---|---|---|
| **Calibration set**: still images of a static scene, all cameras at once | Calibrate tab (`pipeline/calibrate.py`) | `rig/` |
| **Recording**: synchronized video of all cameras, optional audio | Preprocess tab (`pipeline/prep_4d.py`) | dataset `stg<N>/` |

If the intrinsics and extrinsics of the cameras are already known, the
calibration set can be skipped: write the `rig/` folder directly (section 3)
and go straight to preprocessing.

## 1. Rules that apply to both

- **Camera names** are folder names such as `cam_A`, `cam_B`, ... Any names
  work, but the calibration set and the recording **must use the same
  names**: preprocessing looks each camera up in the calibration by name.
- **Same cameras, same settings**: the same physical cameras in the same
  positions, the same resolution, and an untouched focus. The undistortion
  maps are built at the calibration resolution and applied to every recorded
  frame, so **the recording must have the same resolution as the
  calibration set**.
- Exposure should be fixed during a recording (no auto exposure), so the
  cameras agree on brightness.

## 2. Calibration set

```
calib_<anything>/
    session.json
    cam_A/shot_0001.jpg
    cam_A/shot_0002.jpg
    ...
    cam_B/shot_0001.jpg
    ...
```

- One folder per camera, named as in `streams`.
- Image files are named `shot_*` (any image extension COLMAP reads; JPEG is
  used). **The same file name must exist in every camera folder**: `shot_0007`
  in all cameras is one set, taken from one rig pose.

`session.json` (minimal):

```json
{
  "mode": "calib",
  "capture": {"resolution": [1920, 1080]},
  "streams": [
    {"name": "cam_A"}, {"name": "cam_B"}, {"name": "cam_C"}, {"name": "cam_D"}
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `streams[].name` | yes | camera folder names |
| `capture.resolution` | yes | `[width, height]` of the images; used for the initial focal length |
| `mode` | no | `"calib"`; anything else only prints a warning |

### 2.1 What the images must be (required)

Calibration estimates each camera's focal length, principal point and lens
distortion (OPENCV model: k1, k2, p1, p2) plus the rig geometry, **as they
appear in the images it is given**. Preprocessing applies exactly these
values to the recorded frames. So the one rule behind most of the list below
is: **the calibration images must come from the same source as the
recording.**

| Requirement | Why |
|---|---|
| **Same source as the recording**: the same cameras, the same camera mode and resolution, and the same processing (in-camera lens correction on or off, any software step applied afterwards) | What is measured is the geometry of the images, not of the bare lens. If both inputs went through the same fixed correction, the calibration simply measures what remains, which is still correct; if they differ, the values do not fit the recording |
| **No processing that changes from frame to frame**: electronic stabilization, automatic cropping or zoom | Each frame then has a different geometry, which one set of calibration values cannot describe |
| **Fixed focus and fixed zoom**, the same as for the recording | Focus and zoom change the intrinsics |
| **Set the horizontal field of view** (Calibrate tab, advanced parameters, "Horizontal FOV"; `--hfov`) | It gives the initial focal length. The default 86 deg fits the cameras used with stg-recorder; an initial value off by about 2x makes calibration fail |
| **A rigid rig: the cameras must not move relative to each other**, from calibration through the recording | The calibration describes where the cameras are relative to each other; if any camera shifts or turns on the rig, it no longer fits |
| Ordinary lenses up to wide angle | The OPENCV model covers normal and wide-angle lenses; strong fisheye lenses are not modelled well |
| Sharp images | Blurred sets are detected and dropped automatically, which leaves fewer sets |

### 2.2 How to shoot (recommended, not required)

This procedure worked for the rig used with stg-recorder (four cameras on a
153 x 93 mm frame). Other procedures can work; each point says what it is for.

| Recommendation | Purpose |
|---|---|
| Calibrate on the day of shooting, and again after anything on the rig changed (camera moved, ports swapped, focus touched) | Accuracy was observed to degrade over time even with the rig untouched |
| No calibration board: photograph a static scene | A board is expensive and bulky; a textured static scene works as well |
| Nothing in the scene moves (people, pets, screens, leaves, shadows) | Moving content cannot be matched between images and pulls the solution |
| Scene about 2.5 m from the rig, ideally indoors, nothing at infinity (sky, distant street through a window) | Points at a known, moderate distance constrain the geometry; points at infinity carry no depth |
| Texture everywhere, including the image corners; avoid a single flat wall and repetitive patterns such as tiles | Distortion is largest in the corners; a flat wall cannot separate focal length from distance; repeated patterns cause false matches |
| At least 3 rig positions; at each position a 3 x 3 grid of orientations (left / centre / right x up / centre / down), neighbouring orientations overlapping by roughly a third of the frame | Varied positions and rotations constrain focal length and distortion; overlap lets the images be matched. This gives 27 sets or more (15 is the minimum the pre-check accepts without a warning) |
| Keep the rig still while one set is taken (or trigger all cameras at the same instant), and move it between sets | The rig geometry is derived from each set, so a set whose images were taken at different rig poses adds error. With a static scene the cameras need not fire at once, as long as the rig stays still until the set is complete |

Then measure the six distances between the lens centres (not the housings)
with calipers; they give the metric scale. Run the Calibrate tab; the result
must show PASSED.

## 3. Providing `rig/` directly (skipping the calibration set)

If the intrinsics (focal length, principal point, distortion) and the
extrinsics (each camera's pose on the rig) are already known, for example
from another calibration tool, there is no need to shoot a calibration set.
Write the `rig/` folder by hand, put it in the shooting root (section 4) and
run the Preprocess tab.

The values must describe **the recording's images**: the same rule as in
section 2.1 applies (same source, same resolution, fixed focus and zoom, a
rigid rig).

```
rig/
    cameras.txt        intrinsics
    prefixes.json      camera id -> camera name
    rig_config.json    extrinsics
    scale.json         metric scale and measured edge lengths
```

These four files are all preprocessing reads. The Calibrate tab also writes
`calib_report.json`, `rigs.txt` and `summary.txt`; they are not needed.

### 3.1 `cameras.txt`

COLMAP text format, one line per camera; lines starting with `#` are ignored.

```
# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy k1 k2 p1 p2
1 OPENCV 1920 1080 973.30 966.79 932.12 539.56 0.11230 -0.08146 -0.00076 0.00351
2 OPENCV 1920 1080 981.82 980.57 958.48 512.23 0.11395 -0.08316 0.00146 0.00140
...
```

| Field | Meaning |
|---|---|
| `CAMERA_ID` | integer, the key used in `prefixes.json` |
| `MODEL` | **must be `OPENCV`**: exactly eight parameters in the order shown. The name itself is not checked, so other models (with a different parameter count or order) are silently misread |
| `WIDTH HEIGHT` | image size in pixels, **the same for all cameras and equal to the recording's resolution** |
| `fx fy cx cy` | focal lengths and principal point **in pixels at that resolution**; the origin is the top-left corner of the image |
| `k1 k2 p1 p2` | OpenCV distortion coefficients (radial k1, k2; tangential p1, p2). Use `0 0 0 0` for images that are already undistorted |

### 3.2 `prefixes.json`

Maps each `CAMERA_ID` to the camera's folder name in the recording:

```json
{"1": "cam_A/", "2": "cam_B/", "3": "cam_C/", "4": "cam_D/"}
```

A trailing `/` is optional. Every camera of the recording must appear here.

### 3.3 `rig_config.json`

The pose of every camera relative to the rig, in COLMAP's rig format: a list
with one object.

```json
[
  {
    "cameras": [
      {"image_prefix": "cam_A/", "ref_sensor": true},
      {"image_prefix": "cam_D/",
       "cam_from_rig_rotation": [0.999907, -0.002458, -0.002186, -0.013257],
       "cam_from_rig_translation": [0.153362, -0.000109, 0.002234]},
      ...
    ]
  }
]
```

| Field | Meaning |
|---|---|
| `image_prefix` | camera name, as in `prefixes.json` |
| `ref_sensor` | `true` for **exactly one** camera, the reference. The rig frame is that camera's frame, so it has no rotation or translation |
| `cam_from_rig_rotation` | unit quaternion **`[w, x, y, z]`** (w first) |
| `cam_from_rig_translation` | `[x, y, z]` **in metres** |

The pose maps a point from the rig frame (= the reference camera's frame)
into this camera's frame:

    X_cam = R(q) * X_rig + t

So `t` is **not** the camera's position; the position is `-R(q)^T * t`. Axes
follow COLMAP / OpenCV: x right, y down, z forward (the viewing direction).
In the example (values from a real calibration) cam_D sits 153 mm to the
left of cam_A, seen from behind the rig, so its `t` has x = +0.153.

Other fields that COLMAP writes here (`camera_model_name`, `camera_params`)
are ignored; the intrinsics come from `cameras.txt` only.

### 3.4 `scale.json`

```json
{
  "metric": true,
  "unit": "m",
  "edges": [
    {"pair": ["cam_A", "cam_B"], "measured_mm": 153.0},
    {"pair": ["cam_A", "cam_C"], "measured_mm": 93.5}
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `metric` | yes | `true`: the translations above are in metres |
| `unit` | yes | must be `"m"` |
| `edges` | **yes, at least one** | distances between lens centres, measured on the rig, in mm. Camera names in `pair` must match the recording. Preprocessing compares them with the distances computed from `rig_config.json` |

Without `scale.json` preprocessing stops. **With an empty `edges` list, or
with no edge whose two names match the recording, preprocessing crashes at
the end of the geometry step** and leaves an incomplete dataset folder
behind; do not train it. List all camera pairs if
they were measured; one is the minimum.

### 3.5 Checking a hand-written `rig/`

Run preprocessing and read the beginning of its log:

| Line | Must show |
|---|---|
| `OK poses from calibration (4/4 cameras)` | every camera was found in `rig_config.json` |
| `triangulated ... points, reprojection ... px` | at most 2.5 px; a `FAIL` line here means the intrinsics or extrinsics do not fit the images (wrong resolution, wrong axis convention, `t` given as a position) |
| `cam_A-cam_B: calibrated ... mm  measured ... mm  diff ...` | `OK` (within 2 mm) for every edge; `BAD` means the translations or their unit disagree with the rig |

Then open the trained model and **inspect it by eye**: a wrong rig often
still produces a dataset, only with doubled or smeared geometry.

## 4. Recording

**The rig must stay in a fixed position and must not move during the
recording.**

The Preprocess tab works on a **shooting root**: a folder that contains
**exactly one** recording folder (the only subfolder with a `session.json`)
and, next to it, the `rig/` folder from calibration (or written by hand,
section 3).

```
<shooting root>/
    rig/                         copied from <calibration set>/rig
    <recording>/
        session.json
        cam_A/stream.mjpg
        cam_A/index.jsonl
        cam_B/...
        audio/audio.wav          optional
        audio/index.json         optional
```

### 4.1 `session.json` (minimal)

```json
{
  "streams": [
    {"name": "cam_A", "kind": "video"},
    {"name": "cam_B", "kind": "video"},
    {"name": "cam_C", "kind": "video"},
    {"name": "cam_D", "kind": "video"},
    {"name": "audio", "kind": "audio"}
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `streams[].name` | yes | camera folder names; streams with `"kind": "audio"` are not treated as cameras |
| `created_at` | no | recording time; stored in the dataset so it can be traced after the folder is renamed. Without it (and without `session_id`) preprocessing prints a warning and continues |
| `capture.exposure_us`, `capture.gain`, `capture.fps`, `capture.resolution` | no | stored in the dataset for reference |
| `warmup_s`, `session_id` | no | stored for reference |

### 4.2 Video: `stream.mjpg` + `index.jsonl` per camera

- `stream.mjpg` is the frames' encoded images **concatenated** byte by byte
  (each frame a complete JPEG). Nothing is decoded as a video; each frame is
  read by its byte range.
- `index.jsonl` has one JSON object per line, one line per frame:

```
{"i": 0, "ts_ns": 5284452935000, "off": 0, "len": 303832}
{"i": 1, "ts_ns": 5284488908000, "off": 303832, "len": 304350}
```

| Field | Meaning |
|---|---|
| `i` | frame number, unique within the camera |
| `ts_ns` | **capture time in nanoseconds**, integer, ascending |
| `off`, `len` | byte offset and length of this frame in `stream.mjpg` |

**Timestamps are what synchronization rests on.** All cameras (and the audio)
must use **one common clock**; its origin does not matter. Frames are grouped
across cameras by time: a group is valid when the spread of its four
timestamps (latest minus earliest) is at most 25 ms (`--span-ms`, Preprocess
parameters). Each group's time label is the midpoint of its timestamps. Frame
rates may differ slightly between cameras; dropped frames are tolerated
(their groups are discarded and reported in `phase_report.md`).

### 4.3 Audio (optional)

Without `audio/audio.wav` the dataset simply has no sound.

- `audio.wav`: uncompressed PCM, 16, 24 or 32 bit, mono or stereo. For
  stereo, **channel 0 is the left microphone and channel 1 the right one, as
  seen from behind the rig** (looking where the cameras look); the player
  places them on the window frame accordingly.
- `audio/index.json`:

```json
{
  "start_ts_ns": 5284514176113,
  "sample_rate": 48000,
  "sample_rate_measured_hz": 47997.732,
  "frames": 510000,
  "mic_spacing_mm": 195,
  "offset_ms": null
}
```

| Field | Required | Meaning |
|---|---|---|
| `start_ts_ns` | **yes** | time of the first sample **on the same clock as the video `ts_ns`**. Without it audio cannot be aligned and is left out |
| `sample_rate` | no | nominal rate (48000 assumed) |
| `sample_rate_measured_hz` | no | real rate if measured; preferred over `sample_rate` |
| `frames` | no | number of samples per channel, for the printed duration |
| `mic_spacing_mm` | no | distance between the two microphones (200 mm assumed) |
| `offset_ms` | no | measured residual audio/video offset; positive means the audio timestamps are late. `null` = 0 |
| `xruns` | no | list of buffer overruns, only counted |

Loudness is measured during preprocessing (EBU R128) and a playback gain is
stored in the dataset, so very quiet raw recordings are fine.

## 5. Checking your own data

1. Calibrate tab: choose the calibration set. The pre-check lists the cameras,
   the number of sets and any camera with missing images. The result must
   show PASSED.
2. Copy `rig/` next to the recording, choose the shooting root in the
   Preprocess tab. The automatic pre-check runs stage 1 only (grouping by
   timestamps, no images decoded) and shows **the valid rate**: the share of
   frame groups within 25 ms. Close to 100% means the timestamps are
   consistent; a low rate means the clocks are not shared or frames are
   missing.
3. Run preprocessing. Errors such as "these cameras are not in the
   calibration" mean the camera names differ between the two inputs.

## 6. Converting footage from other cameras

Separate cameras (action cameras, phones) work if:

- the same cameras with the same settings also shoot the calibration set,
- each frame gets a timestamp on a common clock with an error well below
  25 ms. Cameras that are not synchronized in hardware need their videos
  aligned afterwards (for example with a clap or a flash visible in every
  camera), then timestamps assigned as `start + n / fps`,
- frames are re-encoded as JPEG, concatenated into `stream.mjpg`, with
  `index.jsonl` listing each frame's byte range.

Rolling-shutter cameras moving fast, or clocks that drift during a long take,
lower the valid rate and blur the result.
