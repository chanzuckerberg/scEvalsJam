import argparse

from sams_vae.data.norman.download import download_norman_dataset
from sams_vae.data.replogle.download import download_replogle_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replogle", action="store_true")
    parser.add_argument("--norman", action="store_true")
    args = parser.parse_args()

    print(args)

    if args.replogle:
        download_replogle_dataset()

    if args.norman:
        download_norman_dataset()
