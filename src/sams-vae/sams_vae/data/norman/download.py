import os

NORMAN_URL = "s3://insitro-research-2023-sams-vae/data/norman.h5ad"


def download_norman_dataset(force: bool = False):
    cache_dir = os.environ.get("SAMS_VAE_DATASET_DIR", "datasets")
    path = os.path.join(cache_dir, "norman.h5ad")
    is_cached = os.path.exists(path)
    if not force and not is_cached:
        os.makedirs(cache_dir, exist_ok=True)
        os.system(f"aws s3 cp {NORMAN_URL} {path}")
    return path


if __name__ == "__main__":
    download_norman_dataset()
