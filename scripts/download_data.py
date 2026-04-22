import kagglehub
import os
import sys
from pathlib import Path

def download_and_symlink_aeroscapes(link_path: Path = Path("data")) -> None:
    if link_path.is_symlink() or link_path.exists():
        print("Error: Link path already exists.")
        sys.exit(1)
        
    print("Downloading dataset to cache...")
    cache_path = kagglehub.dataset_download("kooaslansefat/uav-segmentation-aeroscapes")
            
    os.symlink(cache_path, link_path)
    print(f"Symlink created: {link_path} -> {cache_path}")

if __name__ == "__main__":
    download_and_symlink_aeroscapes()
