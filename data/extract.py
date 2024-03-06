import concurrent.futures
import logging
import requests
import shutil
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

import torch
import tqdm

from data.config import TRAIN_DATA_ROOT, TEST_DATA_ROOT
from data.scenario_tensor_converter import ScenarioTensorConverter


LOG = logging.getLogger(__name__)

TRAIN_URL = "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/train.tar"
TEST_URL = "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/val.tar"


def _download_and_extract(url, output_path) -> None:
    output_path.mkdir(exist_ok=True, parents=True)
    file_size = int(requests.head(url).headers.get('content-length', 0))
    with tempfile.NamedTemporaryFile() as f:
        LOG.info(f"Downloading {url}...")
        with requests.get(url, stream=True) as response:
            if response.status_code != 200:
                LOG.error(f"Failed to download {str(url)}: {str(response)}")
                return

            bar = tqdm.tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {url}",
            )
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
            f.flush()
            bar.close()

        LOG.info(f"Extracting to {str(output_path)}...")
        with tarfile.open(f.name) as tar:
            tar.extractall(path=str(output_path.parent))

        current_dir = output_path.parent / Path(url).stem
        current_dir.rename(output_path)


def _ensure_data_exists() -> None:
    LOG.info("Downloading testing data...")
    if not (TEST_DATA_ROOT.exists() and any(TEST_DATA_ROOT.iterdir())):
        _download_and_extract(TEST_URL, TEST_DATA_ROOT)
    else:
        LOG.info(f"Data exists at {str(TEST_DATA_ROOT)}")

    LOG.info("Downloading training data...")
    if not (TRAIN_DATA_ROOT.exists() and any(TRAIN_DATA_ROOT.iterdir())):
        _download_and_extract(TRAIN_URL, TRAIN_DATA_ROOT)
    else:
        LOG.info(f"Data exists at {str(TRAIN_DATA_ROOT)}")


def _process(scenario_dir: Path) -> Path | None:
    if not (scenario_dir.exists() and scenario_dir.is_dir()):
        LOG.error(f"Skipping: {str(scenario_dir)}")
        return scenario_dir

    try:
        converter = ScenarioTensorConverter(scenario_dir)
        converter.write_agent_history_tensor()
        converter.write_roadgraph_tensor()
    except Exception as err:
        LOG.error(f"Failed to extract: {str(scenario_dir)} \n {traceback.format_exc()}")
        return scenario_dir


def main():
    st = time.time()
    _ensure_data_exists()
    LOG.info(f"Finished download! Took {time.time() - st} seconds")

    st = time.time()
    futures = []
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as pool:
        for root in (TRAIN_DATA_ROOT, TEST_DATA_ROOT):
            for scenario_dir in root.iterdir():
                futures.append(pool.submit(_process, scenario_dir))
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            path = future.result()
            if path is not None:
                failures.append(path)

    LOG.info(f"Finished extraction! Took {time.time() - st:0.2f} seconds")
    LOG.info(f"{len(failures)} / {len(futures)} failed:")
    for path in failures:
        LOG.info(str(path))


if __name__ == "__main__":
    logging.basicConfig(
        filename='extract.log',
        encoding='utf-8',
        level=logging.INFO,
    )
    main()
