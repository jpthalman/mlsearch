import random
import torch
from pathlib import Path
from typing import List, Tuple
from typing_extensions import Self

from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    Track,
    ObjectState,
    ObjectType,
    TrackCategory,
)
from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet
)
from av2.map.map_api import ArgoverseStaticMap

from data import roadgraph
from data import controls
from data import config
from data.scenario_tensor_converter_utils import (
    distance_between_object_states,
    extract_state_features,
    object_state_at_timestep,
    object_state_to_string,
    min_distance_between_tracks,
    padded_object_state_iterator,
)

from data import config
from data.config import Dim

RANDOM = random.Random(42)

"""
The ScenarioTensorConverter class will populate tensors from a scenario parquet
file and map. The tensors will be input to the scene encoder and world
propagation models.

Parameters
__________
scenario_path : str
    The path to the scenario parquet file
map_path : path
    The path to the map json file
"""
class ScenarioTensorConverter:
    def __init__(self: Self, scenario_dir: Path):
        scenario_path = scenario_dir / f"scenario_{scenario_dir.name}.parquet"
        map_path = scenario_dir / f"log_map_archive_{scenario_dir.name}.json"
        self.scenario = load_argoverse_scenario_parquet(scenario_path)

        # The relevance of a track will be determined by the min distance the
        # track gets to the ego track across all timesteps. The focal track will
        # always be included and the tracks will be of random order with the
        # exception of ego always coming first.
        # Note: There will be Dim.A relevant tracks including the ego and focal
        # tracks.
        self.ego_track, self.relevant_tracks = self._ego_and_relevant_tracks()
        RANDOM.shuffle(self.relevant_tracks)
        self.relevant_tracks.insert(0, self.ego_track)

        # We need to pick a single reference position for the entire scenario.
        # Since we are trying to do a history-conditioned prediction task, we
        # want there to be some past and some future relative to the chosen
        # reference point. We also want to use 16-bit precision so the
        # magnitudes should be small. Since the scene is Ego focused, we pick
        # the time point at which the prediction task begins, 5sec, for the
        # central agent in the scene, Ego.
        REF_TIME = 50
        central_state = self.ego_track.object_states[REF_TIME]
        assert central_state.timestep == REF_TIME
        self.reference_point = central_state.position

        self.tensors = dict(
            scenario_name=scenario_dir.name,
            agent_history=torch.zeros([Dim.A, Dim.T, Dim.S]),
            agent_history_mask=torch.zeros([Dim.A, Dim.T]).bool(),
            agent_interactions=torch.zeros([Dim.A, Dim.T, Dim.Ai, Dim.S]),
            agent_interactions_mask=torch.zeros([Dim.A, Dim.T, Dim.Ai]),
            roadgraph=torch.zeros([Dim.R, Dim.Rd]),
            roadgraph_mask=torch.zeros([Dim.R]),
            ground_truth_controls=torch.zeros([Dim.T, Dim.C]),
        )

        self.lat_error = 0.0
        self.long_error = 0.0

        self._populate_ego_tensors()
        self._populate_agent_tensors()
        self._populate_agent_interaction_tensors()
        self._populate_roadgraph_tensors(map_path)
        self._normalize_tensors()

    """Returns the ego and relevant tracks separately"""
    def _ego_and_relevant_tracks(self:Self) -> Tuple[Track, List[Track]]:
        focal_track = None
        ego_track = None
        relevant_tracks = []
        for track in self.scenario.tracks:
            if track.track_id == self.scenario.focal_track_id:
                focal_track = track
                continue
            elif track.track_id == "AV":
                ego_track = track
                continue
            relevant_tracks.append(track)

        # Sort the tracks based on min distance to ego and then trim to size
        # Dim.A - 2 as ego and focal tracks still need to be included.
        relevant_tracks.sort(key=lambda track: min_distance_between_tracks(track, ego_track))
        if len(relevant_tracks) > Dim.A - 2:
            relevant_tracks = relevant_tracks[:Dim.A - 2]

        relevant_tracks.append(focal_track)
        if len(relevant_tracks) < Dim.A - 1:
            relevant_tracks.extend([None] * (Dim.A - 1 - len(relevant_tracks)))
        assert len(relevant_tracks) == Dim.A - 1
        return ego_track, relevant_tracks

    def _populate_ego_tensors(self: Self) -> None:
        out = controls.compute_from_track(self.ego_track)
        self.lat_error = out["lat_error"]
        self.long_error = out["long_error"]
        self.tensors["ground_truth_controls"].copy_(out["controls"])

        self.tensors["agent_history_mask"][0, :] = False
        path = out["path"]
        history = self.tensors["agent_history"][0, :, :]
        history[:, 0].copy_(path[:, 0] - self.reference_point[0]) # x
        history[:, 1].copy_(path[:, 1] - self.reference_point[1]) # y
        history[:, 2].copy_(path[:, 2]) # cos(yaw)
        history[:, 3].copy_(path[:, 3]) # sin(yaw)
        history[:, 4].copy_(path[:, 6]) # vx
        history[:, 5] *= 0.0 # vy
        history[:, 6] = 0.0 # object_type == VEHICLE

    def _populate_agent_tensors(self: Self) -> None:
        agent_history = self.tensors["agent_history"]
        agent_history_mask = self.tensors["agent_history_mask"]
        for a, track in enumerate(self.relevant_tracks):
            if track is not None and track.track_id == "AV":
                continue
            for t, state in enumerate(padded_object_state_iterator(track)):
                i = t // 10 - 1
                if t < 10 or t > 100:
                    # Drop the first and last second of data
                    continue
                elif t % 10 != 0:
                    # Downsample to 1hz
                    continue
                elif state is None:
                    agent_history_mask[a, i] = True
                    continue

                agent_history_mask[a, i] = False
                agent_history[a, i, :] = extract_state_features(
                    track,
                    state,
                    self.reference_point,
                )

    def _populate_agent_interaction_tensors(self: Self) -> None:
        agent_history = self.tensors["agent_history"]
        agent_history_mask = self.tensors["agent_history_mask"]

        agent_interactions = self.tensors["agent_interactions"]
        agent_interactions_mask = self.tensors["agent_interactions_mask"]
        for t in range(Dim.T):
            # Collect all valid agents at this timestep
            agents = []
            for a in range(Dim.A):
                if agent_history_mask[a, t]:
                    continue
                state = agent_history[a, t, :]
                agents.append(dict(pos=state[:2], idx=a))

            # Sort by distance to each agent and populate
            for a in range(Dim.A):
                if agent_history_mask[a, t]:
                    agent_interactions_mask[a, t, :] = True
                    continue
                state = agent_history[a, t, :]

                # Sort the agents by distance to current agent.
                agents.sort(key=lambda e: torch.norm(state[:2] - e["pos"]))
                for ai in range(Dim.Ai):
                    # The case that there are fewer available agents than the configured number
                    # of relevant agents for each agent.
                    if ai + 1 >= len(agents):
                        agent_interactions_mask[a, t, ai] = True
                        continue

                    idx = agents[ai + 1]["idx"]
                    agent_interactions[a, t, ai, :] = agent_history[idx, t, :]

                    # Shift reference frame to be relative to this agent
                    xr, yr = state[:2]
                    agent_interactions[a, t, ai, 0] -= xr
                    agent_interactions[a, t, ai, 1] -= yr

    def _populate_roadgraph_tensors(self: Self, map_path: Path) -> None:
        road, mask = roadgraph.extract(
            self.reference_point,
            map_path,
        )
        self.tensors["roadgraph"].copy_(road)
        self.tensors["roadgraph_mask"].copy_(mask)

    def _normalize_tensors(self: Self) -> None:
        # Scale positions such that 1.0 == 100m away
        self.tensors["agent_history"][:, :, (0,1)] /= config.POS_SCALE # Scale (x, y)
        self.tensors["agent_interactions"][:, :, :, (0,1)] /= config.POS_SCALE # Scale (x, y)
        self.tensors["roadgraph"][:, (0,1,2,3)] /= config.POS_SCALE # Scale (x1, y1, x2, y2) for each line segment

        # Scale velocities such that 1.0 == 25m/s
        self.tensors["agent_history"][:, :, (4,5)] /= config.VEL_SCALE # Scale (vx, vy)
        self.tensors["agent_interactions"][:, :, :, (4,5)] /= config.VEL_SCALE # Scale (vx, vy)


def main():
    import time

    torch.set_printoptions(precision=2, sci_mode=False)
    scenario_dir = Path("/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2/train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7")

    st = time.time()
    converter = ScenarioTensorConverter(scenario_dir)
    print(f"Conversion took: {time.time() - st: 0.2f} sec")

    for idx, item in enumerate(converter.tensors.items()):
        if idx == 0:
            continue
        print(item[0], item[1].shape)

    print()
    print("--- Controls ---")
    print(converter.tensors["ground_truth_controls"])
    print(controls.discretize(converter.tensors["ground_truth_controls"]))


if __name__ == "__main__":
    main()