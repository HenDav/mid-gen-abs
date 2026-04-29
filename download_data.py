"""
Download trajectory CSVs from HuggingFace into traj_csvs/.

Usage:
    python download_data.py

Requires:
    pip install huggingface_hub
"""

import os
from huggingface_hub import snapshot_download

HF_REPO_ID   = "HenDav/mid-gen-abs-data"
LOCAL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traj_csvs")

if __name__ == "__main__":
    print(f"Downloading trajectory CSVs from {HF_REPO_ID} → {LOCAL_DIR}")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
    )
    print("Done.")
