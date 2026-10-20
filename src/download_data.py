# download_data.py
# gets the data in two parts, because it lives in two repos:
#   part A - grounded report sentences from RadGenome (small, not gated)
#   part B - the actual CT volumes from CT-RATE (large, gated)
#
# part A runs every time and is the reliable win. part B is opt-in via a flag
# so a quick run does not accidentally pull gigabytes during a meeting.
#
# before running this, log in once in the terminal:
#   huggingface-cli login
# and accept the CT-RATE terms on its dataset page, or part B will be blocked.

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RADGENOME_REPO,
    CTRATE_REPO,
    GROUNDED_REPORTS_CONFIG,
    DATA_DIR,
    SAMPLE_VOLUME_COUNT,
)

# flip this to True only after you have seen the file tree and are ready to pull volumes
DOWNLOAD_VOLUME_SAMPLE = False


def download_grounded_reports():
    # pull the region-grounded report sentences and save the validation split.
    # validation is used because it is smaller and manually verified.
    print("=" * 60)
    print("PART A - GROUNDED REPORTS (RadGenome, not gated)")
    print("=" * 60)

    from datasets import load_dataset

    print(f"loading '{GROUNDED_REPORTS_CONFIG}' from {RADGENOME_REPO} ...")
    dataset = load_dataset(RADGENOME_REPO, GROUNDED_REPORTS_CONFIG)
    print(f"splits available: {list(dataset.keys())}")

    # save the validation split to disk as parquet so exploring it later is instant
    split_name = "validation" if "validation" in dataset else list(dataset.keys())[0]
    out_path = DATA_DIR / f"grounded_reports_{split_name}.parquet"
    dataset[split_name].to_parquet(str(out_path))

    print(f"rows in {split_name}: {len(dataset[split_name])}")
    print(f"saved to: {out_path}")


def list_repo_structure(repo_id):
    # show the first chunk of the file tree so I can see how a repo is laid out.
    # also surfaces gating problems early with a clear message.
    print("-" * 60)
    print(f"file tree: {repo_id}")
    print("-" * 60)
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        print(f"total files listed: {len(files)}")
        for f in files[:30]:
            print(f"  {f}")
        if len(files) > 30:
            print("  ...")
        return files
    except GatedRepoError:
        print("BLOCKED: this dataset is gated.")
        print("fix: open the dataset page in a browser, accept the terms,")
        print("then run 'huggingface-cli login' with a token that has access.")
        return []
    except HfHubHTTPError as error:
        print(f"could not list files ({error}).")
        print("check you are logged in and have accepted the dataset terms.")
        return []


def download_volume_sample(files):
    # pull a few CT volumes so I have real imaging data to inspect.
    # only runs when DOWNLOAD_VOLUME_SAMPLE is True.
    print("-" * 60)
    print(f"PART B - CT VOLUME SAMPLE ({SAMPLE_VOLUME_COUNT} volumes)")
    print("-" * 60)
    from huggingface_hub import hf_hub_download

    # keep only nifti volume files, then take the first few
    volume_files = [f for f in files if f.endswith(".nii.gz")][:SAMPLE_VOLUME_COUNT]
    if not volume_files:
        print("no .nii.gz files found in the listing, skipping.")
        return

    for f in volume_files:
        print(f"downloading {f} ...")
        local_path = hf_hub_download(
            repo_id=CTRATE_REPO,
            filename=f,
            repo_type="dataset",
            local_dir=str(DATA_DIR / "ctrate_sample"),
        )
        print(f"  saved to {local_path}")


def main():
    # part A always
    download_grounded_reports()

    # part B inspection: always show the trees so I understand the layout
    print("")
    print("=" * 60)
    print("PART B - VOLUME AND MASK STRUCTURE")
    print("=" * 60)
    ctrate_files = list_repo_structure(CTRATE_REPO)
    print("")
    list_repo_structure(RADGENOME_REPO)

    # only pull volumes when I have flipped the flag
    if DOWNLOAD_VOLUME_SAMPLE and ctrate_files:
        print("")
        download_volume_sample(ctrate_files)
    else:
        print("")
        print("volume download is off. set DOWNLOAD_VOLUME_SAMPLE = True to pull a sample.")


if __name__ == "__main__":
    main()
