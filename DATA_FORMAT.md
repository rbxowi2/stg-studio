# Input data format

What stg-studio reads, so recordings can come from something other than
`stg-recorder`. Fields marked **required** are read by the pipeline; anything
else in the files is ignored.

There are two kinds of input:

| Input | Used by | Produces |
|---|---|---|
| **Calibration set**: still images of a static scene, all cameras at once | Calibrate tab (`pipeline/calibrate.py`) | `rig/` |
| **Recording**: synchronized video of all cameras, optional audio | Preprocess tab (`pipeline/prep_4d.py`) | dataset `stg<N>/` |

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
| **Within one set, the rig must not move** | All cameras of a set must see the scene from one rig pose. The scene is static, so the cameras do not have to fire at the same instant, as long as the rig stays still until the whole set is taken |
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

Then measure the six distances between the lens centres (not the housings)
with calipers; they give the metric scale. Run the Calibrate tab; the result
must show PASSED.

## 3. Recording

The Preprocess tab works on a **shooting root**: a folder that contains
**exactly one** recording folder (the only subfolder with a `session.json`)
and, next to it, the `rig/` folder from calibration.

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

### 3.1 `session.json` (minimal)

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

### 3.2 Video: `stream.mjpg` + `index.jsonl` per camera

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

### 3.3 Audio (optional)

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

## 4. Checking your own data

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

## 5. Converting footage from other cameras

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
