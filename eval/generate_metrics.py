import argparse
import concurrent.futures
import logging
import time
import traceback
from pathlib import Path

import torch
import tqdm

from data import controls as control_utils
from data.config import Dim, EXPERIMENT_ROOT, TEST_DATA_ROOT
from data.data_module import load_tensors_from_scenario_dir
from train.module import MLSearchModule

LOG = logging.getLogger(__name__)


def generate_metrics(
    checkpoint_path: Path,
    scenario_dir: Path,
    output_root: Path,
) -> None:
    model = MLSearchModule.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device("cpu"),
    )
    model = model.eval()
    inputs = load_tensors_from_scenario_dir(scenario_dir)

    data = []
    steps_per_second = 5
    for start_time in range(0, Dim.T, steps_per_second):
        T = Dim.T - start_time

        agent_history = inputs["agent_history"][:, start_time:, :].to(model.device)
        roadgraph = inputs["roadgraph"].to(model.device)
        embedding = model.scene_encoder(
            agent_history=agent_history.unsqueeze(0),
            roadgraph=roadgraph.unsqueeze(0),
        )
        agent_state_gt = agent_history.clone()
        ego_state = agent_history[0, :, :].clone()
        for rollout_duration in range(1, T // steps_per_second):
            embedding = embedding[:, :, :-steps_per_second, :]
            agent_state_gt = agent_state_gt[:, steps_per_second:, :]
            ego_state = ego_state[:-steps_per_second, :]

            controls = model.control_predictor(
                scene_embedding=embedding,
            )
            ego_state = control_utils.integrate(
                ego_state,
                controls.reshape(-1, Dim.C),
            )
            embedding = model.world_model(
                scene_embedding=embedding,
                next_ego_state=ego_state.unsqueeze(0),
            )

            error = control_utils.relative_positions(
                agent_state_gt[0, :, :],
                ego_state,
            )
            for history_duration in range(error.shape[0]):
                longitudinal_error = error[history_duration, 0].item()
                lateral_error = error[history_duration, 1].item()
                data.append(
                    (
                        start_time,
                        history_duration,
                        rollout_duration * steps_per_second,
                        longitudinal_error,
                        lateral_error,
                    )
                )

    output_root.mkdir(exist_ok=True, parents=True)
    with open(output_root / f"{scenario_dir.name}.csv", "w") as f:
        for row in data:
            f.write(",".join([str(e) for e in row]) + "\n")


def process(
    checkpoint_path: Path,
    scenario_dir: Path,
    output_root: Path,
) -> None:
    try:
        generate_metrics(checkpoint_path, scenario_dir, output_root)
    except Exception as err:
        LOG.error(f"Failed to process: {str(scenario_dir)} \n {traceback.format_exc()}")
        return scenario_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    args = parser.parse_args()

    experiment_dir = EXPERIMENT_ROOT / args.experiment_name
    checkpoint_path = experiment_dir / "last.ckpt"
    metrics_root = experiment_dir / "metrics"

    st = time.time()
    futures = []
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for scenario_dir in TEST_DATA_ROOT.iterdir():
            futures.append(pool.submit(process, checkpoint_path, scenario_dir, metrics_root))
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            rslt = future.result()
            if rslt is not None:
                failures.append(rslt)

    LOG.info(f"Finished metrics computation! Took {time.time() - st:0.2f} seconds")
    LOG.info(f"{len(failures)} / {len(futures)} failed:")
    for path in failures:
        LOG.info(str(path))


if __name__ == "__main__":
    main()
