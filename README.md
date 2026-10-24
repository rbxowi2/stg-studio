# stg-studio

Desktop application that turns a four-camera rig recording into a
4D Gaussian Splatting model ("a window": a scene that can be watched from
slightly different positions behind the rig frame).

It drives three steps, each in its own tab:

| Tab | What it does | Program behind it |
|---|---|---|
| Calibrate | calibration set -> portable `rig/` folder (intrinsics, extrinsics, metric scale) | `pipeline/calibrate.py` + COLMAP |
| Preprocess | recording + `rig/` -> training dataset `stg<N>/` | `pipeline/prep_4d.py` + COLMAP |
| Train / Combined training | dataset -> model, with live loss curve, preview, pause, save, resume | patched SpacetimeGaussians trainer |

Recordings come from the companion recorder (`stg-recorder`); models are
played with `stg-player`.

## Install

Windows 10/11 with an NVIDIA GPU. See [INSTALL.md](INSTALL.md); in short:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

then start **STG studio** from the desktop shortcut.

## Layout

```
studio/       the tkinter application (entry point: studio/stg_studio.py)
pipeline/     calibrate.py, prep_4d.py, audio_level.py (also usable from the command line)
tools/        train_cmd.py (training command builder), merge_segments.py, self-checks
upstream/     stg_427abfc.patch: every change made to the upstream trainer
THIRD_PARTY_LICENSES/   licenses of the upstream projects
```

[DATA_FORMAT.md](DATA_FORMAT.md) specifies the input folders and files
(calibration set, recording, audio), for preparing data without
stg-recorder.

[REBUILD_FOR_AI.md](REBUILD_FOR_AI.md) describes the whole system in enough
detail to rebuild or modify it without prior context.

## Upstream

* SpacetimeGaussians, https://github.com/oppo-us-research/SpacetimeGaussians (commit 427abfc), MIT
* gaussian-splatting (Inria / MPII), contained in SpacetimeGaussians, under its own license

Their license texts are in [THIRD_PARTY_LICENSES/](THIRD_PARTY_LICENSES/).
