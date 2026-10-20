# config.py
# one place for all paths, dataset ids, and preprocessing constants.
# everything else imports from here so I never hard-code a path twice.

from pathlib import Path

# ---- project paths ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# make sure the folders exist whenever config is imported
DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ---- dataset ids on HuggingFace ----
# the report text and segmentation masks live here (cc-by-4.0, not gated)
RADGENOME_REPO = "RadGenome/RadGenome-ChestCT"
# the actual CT volumes live in the base dataset, which IS gated
CTRATE_REPO = "ibrahimhamamci/CT-RATE"

# the grounded reports subset has a space in the name, this is the exact config string
GROUNDED_REPORTS_CONFIG = "grounded reports"

# ---- CT preprocessing constants ----
# lung window in Hounsfield units, same range Muralidharan (2025) used so the
# baseline and this project stay directly comparable
HU_MIN = -1000
HU_MAX = 400

# value to fill voxels that fall outside the lung mask
HU_BACKGROUND = -1000

# target voxel spacing in mm for resampling (isotropic, sensible default for chest CT)
TARGET_SPACING = (1.5, 1.5, 1.5)

# how many CT volumes to pull for a first sample, keeps the download small
SAMPLE_VOLUME_COUNT = 3
