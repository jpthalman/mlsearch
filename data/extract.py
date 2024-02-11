import concurrent.futures
import logging
import time
import traceback
from pathlib import Path

import torch
import tqdm

from data.scenario_tensor_converter import ScenarioTensorConverter


ROOT = Path("/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2")
OVERWRITE = False

LOG = logging.getLogger(__name__)


def _process(scenario_dir: Path) -> Path | None:
    if not (scenario_dir.exists() and scenario_dir.is_dir()):
        LOG.error(f"Skipping: {str(scenario_dir)}")
        return scenario_dir

    out = scenario_dir / "data.pt"
    if out.exists():
        if not OVERWRITE:
            return
        out.unlink()

    try:
        converter = ScenarioTensorConverter(scenario_dir)
        torch.save(converter.tensors, out)
    except Exception as err:
        LOG.error(f"Failed to extract: {str(scenario_dir)} \n {traceback.format_exc()}")
        return scenario_dir


def main():
    st = time.time()
    futures = []
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as pool:
        for dataset in ("train", "val"):
            for scenario_dir in (ROOT / dataset).iterdir():
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
