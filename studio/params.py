"""Single source of the training parameters: names, layers, tier values, help text.

The UI (train_tab.py) and the command builder both read from here; help text
is written nowhere else.

Four layers:
    A  filled in automatically and read-only (derived from window.json; changing it is an error)
    B  main knobs, pre-filled by the three tiers
    C  advanced; the defaults are fine, but they are exposed
    D  not exposed (not listed here)

Tier values are not defined here; they are read from TIERS in
tools/train_cmd.py, the command builder, so there is only one copy.

Help text answers "what is it" and "what happens if it is set wrong", briefly.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from tools.train_cmd import TIERS                       # noqa: E402

TIER_NAMES = list(TIERS)                                 # fast / balanced / max


def _tier(idx):
    """Split a TIERS tuple into {tier: value}. See the TIERS comment in train_cmd for idx."""
    return {name: TIERS[name][idx] for name in TIERS}


class Param:
    """One adjustable parameter.

    key      identifier in the code
    label    name shown in the UI
    kind     int / float / str / bool
    group    "A" / "B" / "C" (see the module docstring)
    cli      command-line flag (for example "--iterations")
    env      environment variable name (for example "STG_GNUM_MAX")
    tiers    {tier: value}, group B only
    default  default value for group C
    tip      hover help text (single source)
    """

    def __init__(self, key, label, tip, kind=int, group="C",
                 cli=None, env=None, tiers=None, default=None, choices=None):
        self.key, self.label, self.tip = key, label, tip
        self.kind, self.group = kind, group
        self.cli, self.env = cli, env
        self.tiers, self.default, self.choices = tiers, default, choices

    def value_for(self, tier):
        return self.tiers[tier] if self.tiers else self.default

    def __repr__(self):
        return f"<Param {self.key} {self.group}>"


# -- A  automatic, read-only ------------------------------------------------

AUTO = [
    Param("duration", "Time steps", kind=int, group="A", cli="--duration",
          tip="Number of time steps in the dataset, read from window.json."),
    Param("cameras_extent", "Scene radius", kind=float, group="A",
          env="STG_CAMERAS_EXTENT",
          tip="Scene scale; every densification threshold is multiplied by it.\n"
              "Computed from window.json (farthest surface x 1.1)."),
    Param("background", "Background", kind=float, group="A", env="STG_BACKGROUND",
          tip="Grey level of the training background. Must match playback, otherwise the image gets a white haze."),
    Param("times", "Timestamp table", kind=str, group="A", env="STG_TIMES",
          tip="Real timestamp of every image (times_group.json).\n"
              "Without it the time axis is assumed to be evenly spaced."),
]

# -- B  main knobs, pre-filled by the tiers ---------------------------------

TIERED = [
    Param("iterations", "Iterations", kind=int, group="B", cli="--iterations",
          tiers=_tier(0),
          tip="Total number of iterations. The learning-rate schedule follows it.\n"
              "More iterations do not guarantee a better image; judge by the preview."),
    Param("desicnt", "Densify count limit", kind=int, group="B", cli="--desicnt",
          tiers=_tier(1),
          tip="How many densification steps are allowed (cloning or splitting\n"
              "Gaussians with large gradients). It does not set the Gaussian\n"
              "count directly; the real gate is the densify end."),
    Param("densify_until_iter", "Densify end", kind=int, group="B",
          cli="--densify_until_iter", tiers=_tier(2),
          tip="Iteration at which densification stops.\n"
              "It also stops pruning and opacity resets; misshapen Gaussians\n"
              "are no longer cleaned up after it."),
    Param("gnum_max", "Gaussian limit", kind=int, group="B", env="STG_GNUM_MAX",
          tiers=_tier(3),
          tip="Hard limit on the total number of Gaussians.\n"
              "Peak GPU memory during training has not been measured, so\n"
              "exceeding it only warns."),
    # No TIERS column exists for this one (TIERS stores iteration lists), so the
    # tier values are written here: fast/balanced save once at the end, max every 10%.
    Param("autosave_pct", "Autosave interval (%)", kind=int, group="B",
          tiers={"fast": 100, "balanced": 100, "max": 10},
          tip="Write a model usable by the player every this many percent of training.\n"
              "100 = save once at the end. 10 saves at 10%, 20% ... 100%.\n"
              "This saves a PLY (for playback), not the resumable checkpoint.\n"
              "Each PLY is tens to hundreds of MB; saving too often fills the disk."),
    # Deliberately separate from the setting above: that one saves a PLY
    # (playback only, no optimizer state, cannot be resumed); this one saves
    # state.pth (resumable). Sharing one setting would force frequent
    # checkpoints on anyone who wants frequent playable models, and vice versa.
    Param("ckpt_pct", "Autosave checkpoint (%)", kind=int, group="B",
          tiers={"fast": 0, "balanced": 0, "max": 0},
          tip="Save a resumable checkpoint every this many percent of training.\n"
              "0 = off (default). After training, \"Save checkpoint\" still works.\n"
              "Only the latest one is kept and overwritten: 25% does not give four files.\n"
              "Training pauses while it writes. A checkpoint of 460k Gaussians\n"
              "is about 115 MB and takes about 0.1 s (smaller than the 160 MB PLY).\n"
              "Unlike the autosave interval above, which writes playback PLYs that cannot be resumed."),
    # Time window. Empty string = all time steps.
    Param("time_range", "Time range", kind=str, group="B", default="",
          tip="Train only part of the time steps, as start,end (both inclusive), e.g. 50,105.\n"
              "Empty = all. Time inside the range is renormalized, so the model is\n"
              "a complete short clip of its own that the player plays start to end.\n"
              "With fewer than 30 time steps, motion is poorly constrained and the\n"
              "image may distort."),
    Param("omega_iter", "Omega split iteration", kind=int, group="B",
          env="STG_OMEGA_ITER", tiers=_tier(4),
          tip="Iteration at which Gaussians are split into moving and static groups:\n"
              "static ones freeze their angular velocity and learn a fixed rotation.\n"
              "Three cases:\n"
              "  at or after the densify end -> no split at all, no message\n"
              "    (the split code sits inside the densification range)\n"
              "  before the densify end, densification still running after the split\n"
              "    -> training crashes (the frozen list no longer matches the count)\n"
              "  before the densify end, densification already stopped -> fine\n"
              "The second case was observed: the split recorded 70,986 Gaussians at\n"
              "1,500, densification grew to 135,594, and freezing the gradients failed.\n"
              "When densification stops depends on the dataset (count exhausted or\n"
              "the Gaussian limit reached), so safety is not decided by this field\n"
              "alone; watch the densify count on the curve."),
]

# -- C  advanced: defaults are fine, but exposed ----------------------------

ADVANCED = [
    Param("densify", "Densification branch", kind=int, default=2, choices=[1, 2, 3],
          cli="--densify",
          tip="Which densify / prune rule set to use. 2 is N3DV, 3 is Technicolor.\n"
              "Near-plane pruning runs only with 1 or 2."),
    Param("densify_grad_threshold", "Densify sensitivity", kind=float, default=0.0002,
          cli="--densify_grad_threshold",
          tip="Only Gaussians with a gradient above this are cloned or split.\n"
              "Lower values grow the count faster."),
    Param("densification_interval", "Densify interval", kind=int, default=100,
          cli="--densification_interval",
          tip="Check for densification every this many iterations."),
    Param("densify_from_iter", "Densify start", kind=int, default=500,
          cli="--densify_from_iter",
          tip="Iteration at which densification starts. The count does not change before it."),
    Param("opacity_reset_interval", "Opacity reset interval (branch 3 only)", kind=int,
          default=3000, cli="--opacity_reset_interval",
          tip="Every this many iterations all opacities are pushed down to 0.01 so\n"
              "Gaussians must prove themselves again: real surfaces become opaque\n"
              "again, floaters do not and are pruned. Without resets floaters only\n"
              "accumulate and never reach the prune threshold.\n"
              "Densification branch 2 (default) hard-codes 3000, so this field has\n"
              "no effect there; only branch 3 reads it.\n"
              "3000 is the original 3DGS default for 30,000 iterations; it has not\n"
              "been validated here."),
    Param("opthr", "Prune threshold", kind=float, default=0.005, cli="--opthr",
          tip="Gaussians with opacity below this are pruned."),
    Param("percent_dense", "Split size threshold", kind=float, default=0.01,
          cli="--percent_dense",
          tip="Gaussians larger than this fraction of the scene radius are split\n"
              "instead of cloned."),
    Param("lambda_dssim", "SSIM weight", kind=float, default=0.2,
          cli="--lambda_dssim",
          tip="Share of SSIM in the loss; the rest is L1.\n"
              "Higher values favour structure over absolute brightness."),
    Param("emsstart", "Guided sampling start", kind=int, default=1600,
          cli="--emsstart",
          tip="Iteration at which points start being added where the error is large."),
    Param("emsthr", "Guided sampling threshold", kind=float, default=0.6,
          cli="--emsthr",
          tip="Only regions with an error above this get new points."),
    Param("farray", "Points per ray", kind=int, default=2, cli="--farray",
          tip="How many new Gaussians are placed along each ray when adding points."),
    Param("znear_prune", "Near-plane prune", kind=float, default=0.1,
          env="STG_ZNEAR_PRUNE",
          tip="Initial points with z below this are removed.\n"
              "In scene units (metric); too large a value removes every point."),
    Param("holdout_t", "Held-out time steps", kind=str, default="", env="STG_HOLDOUT_T",
          tip="Time steps excluded from training and kept for scoring, comma separated.\n"
              "Empty = 5 evenly spaced steps.\n"
              "Scores are measured at the four camera positions only, not at the window center."),
    Param("bg_random", "Random background", kind=bool, default=False, env="STG_BG_RANDOM",
          tip="Use a random solid colour as background every iteration, forcing the\n"
              "model to learn opacity correctly. Enable only when the image has holes."),
    Param("gtisint8", "Store GT as uint8", kind=int, default=1, choices=[0, 1],
          cli="--gtisint8",
          tip="Keep training images as 8-bit integers in GPU memory, saving a lot of space.\n"
              "Off is slightly more precise but easily runs out of GPU memory."),
    Param("resolution", "Resolution divisor", kind=int, default=2, cli="--resolution",
          choices=[1, 2, 4, 8],
          tip="Downscale factor for the training images; 2 halves width and height.\n"
              "Scores at different resolutions cannot be compared."),
]

# -- Preprocessing tab (recording -> dataset) -------------------------------
#
# Behaviour = `python prep_4d.py <shooting root>` with no flags that change defaults.
# --rig is never passed: it would copy an outside calibration into the root
# and delete the existing one.

# Preprocessing has no tiers: it has a single main knob (number of time steps),
# so a set of presets would add nothing.
PREP_PARAMS = [
    Param("prep_steps", "Sampled time steps", kind=int, default=0, cli="--steps",
          tip="How many time steps to take. 0 = every valid time step.\n"
              "All of them took about 13 minutes for 301 steps (measured).\n"
              "More steps give finer time resolution and a larger dataset.\n"
              "This changes the dataset name (stg<steps>), so run the pre-check again."),
    Param("prep_span_ms", "Grouping threshold (ms)", kind=float, default=25.0,
          cli="--span-ms",
          tip="Frames whose timestamps span more than this across the cameras do not\n"
              "form a group. Loosen it only with many dropped frames and a low valid\n"
              "rate. Run the pre-check again after changing it."),
    Param("prep_background", "Background", kind=float, default=0.0,
          cli="--background",
          tip="Colour for areas no camera saw; 0 is black.\n"
              "Do not use white: it turns those areas into a glaring white haze.\n"
              "Training must use the same value (the training tab fills it in)."),
    Param("prep_anchor", "Reference camera", kind=str, default="", cli="--anchor",
          tip="Camera whose timestamps the others are matched against. Empty = the first.\n"
              "Grouping searches all combinations for the smallest span, so the\n"
              "reference matters little."),
    Param("prep_keep_work", "Keep intermediate files", kind=bool, default=False,
          cli="--keep-work",
          tip="Keep the MVS intermediate files. Large; only for diagnosing preprocessing."),
]

# -- Calibration tab (calibration set -> rig) -------------------------------
#
# Behaviour = calibrate.py <calibration set> --edge x6.

# The six measured distances; the camera pairs are fixed.
# (key in settings.json, camera 1, camera 2, default mm)
CALIB_EDGES = [
    ("A-D", "cam_A", "cam_D", 153.0),
    ("B-C", "cam_B", "cam_C", 153.0),
    ("A-B", "cam_A", "cam_B", 93.5),
    ("C-D", "cam_C", "cam_D", 93.5),
    ("A-C", "cam_A", "cam_C", 179.0),
    ("B-D", "cam_B", "cam_D", 179.0),
]
CALIB_EDGE_TIP = ("Measured distance between two cameras (mm), between lens centers,\n"
                  "not housing edges. Measure again after rebuilding the rig. All six are\n"
                  "required. It sets the metric scale; a wrong value fails the\n"
                  "\"max difference from measured edges\" check.")

CALIB_PARAMS = [
    Param("calib_hfov", "Horizontal FOV (deg)", kind=float, default=86.0,
          cli="--hfov",
          tip="Only used for the initial focal length estimate, which is then refined.\n"
              "About 86 for the cameras used here. Change it only for different lenses."),
    Param("calib_blur_ratio", "Sharpness threshold", kind=float, default=0.5,
          cli="--blur-ratio",
          tip="Sets sharper than this fraction of the median are discarded (camera shake).\n"
              "Too high discards too many sets."),
    Param("calib_keep_work", "Keep intermediate files", kind=bool, default=False,
          cli="--keep-work",
          tip="Keep the _work folder (COLMAP database and reconstructions).\n"
              "Only for diagnosing a failed calibration."),
]

ALL = AUTO + TIERED + ADVANCED + PREP_PARAMS + CALIB_PARAMS
BY_KEY = {p.key: p for p in ALL}


# No parameter validation here: invalid combinations are avoided by choosing
# safe tier defaults rather than by warning the user.


# position_lr_max_steps is not a separate parameter: it always equals
# iterations. A function instead of a Param, so the UI never shows a field
# that cannot be changed independently.
def position_lr_max_steps(values):
    return values.get("iterations")

# Parameters that cannot change when resuming, each for its own reason:
#   resolution   the ground-truth image size would change
#   densify      the branch controls behaviour such as the omega split; switching midway is inconsistent
#   znear_prune  applied once before the loop starts, so changing it on resume has no effect
#   holdout_t    a different test set means a different training set
#   time_range   a different range is a different dataset; the checkpoint's Gaussians fit the old one
# Layer A (dataset values) is read-only anyway. Everything else may change,
# including random background and uint8 GT: it still runs, at the cost of a
# model trained one way for the first half and another for the second.
RESUME_FROZEN = ("resolution", "densify", "znear_prune", "holdout_t",
                 "time_range")
