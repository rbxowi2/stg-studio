# Installing STG studio

Everything except the system prerequisites is installed **inside this
repository folder** by `setup.ps1`. Moving or deleting the folder removes the
installation; nothing else on the machine is changed apart from one desktop
shortcut. The system Python is only used to create `.venv` (an independent
environment without access to system packages); nothing is installed into it.

## 1. System prerequisites (by hand)

| Software | Version used | Notes |
|---|---|---|
| Windows | 11 (10 also works) | |
| NVIDIA GPU + driver | RTX 20xx or newer, 12 GB VRAM recommended | `nvidia-smi` must work |
| CUDA Toolkit | **13.0** | default location `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0` |
| Visual Studio | 2022 or newer, workload **"Desktop development with C++"** | Community edition is enough; needed to compile the CUDA extensions |
| Python | **3.12** | from python.org; tick "Add to PATH" or keep the `py` launcher |
| git | any recent | |

**Check:**

```powershell
nvidia-smi
nvcc --version          # "release 13.0"
py -3.12 --version      # Python 3.12.x
git --version
```

## 2. Run setup.ps1

From this folder:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

It takes 20-40 minutes (mostly downloading PyTorch and compiling). The seven
steps print `OK` or `ERROR`; on an error, fix the cause and run the script
again - finished steps are skipped.

| Step | Creates | Check that it worked |
|---|---|---|
| 1 prerequisites | nothing | five `OK` lines |
| 2 Python environment | `.venv\` | prints torch / `True` / numpy 1.26.4 / OpenCV 4.11 |
| 3 trainer | `third_party\SpacetimeGaussians\` at commit 427abfc with `upstream\stg_427abfc.patch` applied | "12 modified files" |
| 4 CUDA extensions | `simple_knn`, `forward_lite`, `diff_gaussian_rasterization_ch3` inside `.venv` | "built and importable" |
| 5 COLMAP | `third_party\colmap\bin\colmap.exe` | "installed" |
| 6 checks | nothing | trainer starts, three self-checks pass |
| 7 shortcut | `STG studio.lnk` on the desktop | the file exists |

Options (`Get-Help .\setup.ps1 -Detailed` lists them all):

| Option | Use |
|---|---|
| `-CudaArch 8.6` | compile only for your GPU (8.6 = RTX 30xx, 8.9 = RTX 40xx, 12.0 = RTX 50xx); faster. The default covers RTX 20xx-50xx |
| `-Python C:\path\python.exe` | a Python 3.12 that is not found automatically |
| `-CudaHome <folder>` | CUDA 13.0 in a non-standard location |
| `-ColmapZip <zip>` | a COLMAP Windows CUDA zip downloaded by hand (no internet access to GitHub, or a specific version) |
| `-SkipColmap` | COLMAP is already on PATH |
| `-RebuildExtensions` | compile the CUDA extensions again (after changing GPU or CUDA) |
| `-NoShortcut` | no desktop shortcut |

## 3. Start

Double-click **STG studio** on the desktop, or:

```powershell
.venv\Scripts\pythonw.exe studio\stg_studio.py
```

**Check:** the main window opens with the tabs Preprocess / Train /
Combined training / Calibrate, and the lower right corner shows
`v<version> (<git hash>)`.

`studio\settings.json` (last used folders and edge lengths) is created on
first use; it is per machine and not tracked by git.

## 4. End-to-end check

The only proof that the whole chain works is a real run:

1. **Calibrate**: choose a calibration set from the recorder, enter the six
   measured camera distances, run. The result table must show PASSED.
   What the calibration images must be and how to shoot them is described in
   [DATA_FORMAT.md](DATA_FORMAT.md), section 2.
2. Copy the resulting `rig\` folder into a shooting root next to its recording.
3. **Preprocess**: choose the shooting root, run. A dataset `stg<N>\` appears.
4. **Train**: tier `fast`, start. The loss curve and the preview update.
5. Open the model in `stg-player` and **inspect it by eye**.

## Command line (without the studio)

```powershell
.venv\Scripts\python.exe pipeline\calibrate.py <calibration set> --edge cam_A,cam_B,91.0 --edge ...
.venv\Scripts\python.exe pipeline\prep_4d.py <shooting root>
.venv\Scripts\python.exe tools\train_cmd.py <shooting root> --tier fast
```

`train_cmd.py` prints a complete training command (environment variables
included) that can be pasted into PowerShell.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Filename too long` during step 3 | some upstream test files have long paths; setup enables `core.longpaths` for the clone. Delete `third_party\SpacetimeGaussians` and run again |
| `ImportError: DLL load failed` for `simple_knn._C` | `torch` must be imported before the extensions (the trainer does this). If it fails even then, rebuild with `-RebuildExtensions` |
| compile error mentioning the compiler version | the patch adds `-allow-unsupported-compiler`; make sure the patch was applied (step 3 reports 12 modified files) |
| `np.NaN` attribute error | NumPy 2 got installed; the trainer needs `numpy==1.26.4` (run step 2 again) |
| torch prints `False` for CUDA | NVIDIA driver missing or too old for CUDA 13.0 |
| the studio cannot find COLMAP | `third_party\colmap\bin\colmap.exe` missing; run setup again or put `colmap` on PATH |
