import argparse

from huggingface_hub import snapshot_download

from src import paths

REPO_ID = '446f6e6e79/CVAE-Suffix-Generation'


def run() -> None:
    """Download every checkpoint published to the Hugging Face repo into `best-models/`,
    mirroring the repo's `dataset/model/tag.pt` layout exactly."""
    snapshot_download(repo_id=REPO_ID, repo_type='model', local_dir=paths.BEST_MODELS_DIR)


def main() -> None:
    argparse.ArgumentParser(
        description='Fetch every checkpoint published to the Hugging Face repo into `best-models/`.'
    ).parse_args()

    run()
    print(f'Fetched every published checkpoint into {paths.BEST_MODELS_DIR}')


if __name__ == '__main__':
    main()
