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
from data.dimensions import Dim
from data.scenario_tensor_converter_utils import (
    distance_between_object_states,
    extract_state_features,
    object_state_at_timestep,
    object_state_to_string,
    min_distance_between_tracks,
    padded_object_state_iterator,
    transform_to_reference_frame,
)

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
            agent_history=torch.zeros([Dim.A, Dim.T, 1, Dim.S]),
            agent_mask=torch.zeros([Dim.A, Dim.T]).bool(),
            agent_interactions=torch.zeros([Dim.A, Dim.T, Dim.Ai, Dim.S]),
            agent_interactions_mask=torch.zeros([Dim.A, Dim.T, Dim.Ai]),
            roadgraph=torch.zeros([1, 1, Dim.R, Dim.Rd]),
            roadgraph_mask=torch.zeros([1, 1, Dim.R]),
            ground_truth_controls=torch.zeros([Dim.T - 1, Dim.C]),
        )

        self._populate_agent_tensors()
        self._populate_agent_interaction_tensors()
        self._populate_roadgraph_tensors(map_path)
        self._populate_controls()

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

    def _populate_agent_tensors(self: Self) -> None:
        agent_history = self.tensors["agent_history"]
        agent_mask = self.tensors["agent_mask"]
        for a, track in enumerate(self.relevant_tracks):
            for t, state in enumerate(padded_object_state_iterator(track)):
                # include the last state if it exists
                if t == 109:
                    t += 1

                # Downsample to 1hz
                if t % 10 != 0:
                    continue
                elif state is None:
                    agent_mask[a, t // 10] = True
                    continue

                agent_mask[a, t // 10] = False
                agent_history[a, t // 10, 0, :] = extract_state_features(
                    track,
                    state,
                    self.reference_point,
                )

    def _populate_agent_interaction_tensors(self: Self) -> None:
        agent_history = self.tensors["agent_history"]
        agent_mask = self.tensors["agent_mask"]

        agent_interactions = self.tensors["agent_interactions"]
        agent_interactions_mask = self.tensors["agent_interactions_mask"]
        for t in range(Dim.T):
            # Collect all valid agents at this timestep
            agents = []
            for a in range(Dim.A):
                if agent_mask[a, t]:
                    continue
                state = agent_history[a, t, 0, :]
                agents.append(dict(pos=state[:2], idx=a))

            # Sort by distance to each agent and populate
            for a in range(Dim.A):
                if agent_mask[a, t]:
                    agent_interactions_mask[a, t, :] = True
                    continue
                state = agent_history[a, t, 0, :]
                agents.sort(key=lambda e: torch.norm(state[:2] - e["pos"]))
                for ai in range(Dim.Ai):
                    if ai + 1 >= len(agents):
                        agent_interactions_mask[a, t, ai] = True
                        continue

                    idx = agents[ai + 1]["idx"]
                    agent_interactions[a, t, ai, :] = agent_history[idx, t, 0, :]
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

    def _populate_controls(self: Self) -> None:
        self.tensors["ground_truth_controls"] = controls.compute_from_track(
            self.ego_track,
        )


def main():
    import time

    torch.set_printoptions(precision=2, sci_mode=False)
    scenario_dir = Path("/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2/train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7")

    st = time.time()
    converter = ScenarioTensorConverter(scenario_dir)
    print(f"Conversion took: {time.time() - st: 0.2f} sec")

    for k, v in converter.tensors.items():
        print(k, v.shape)

    # print out ego track using the ego track object
    print(object_state_to_string(converter.ego_track.object_states[0]))
    # print out agent history of first row in the agent history tensor
    print("Ego Tensor Output: ")
    print(converter.tensors["agent_history"][0, 0, 0, :])

    print("Ai shape: " + str(converter.agent_interactions.shape))
    # print the agent interactions of the ego object at t=0
    ego_interactions = converter.agent_interactions[0][2]
    for index, interaction in enumerate(ego_interactions):
        print(str(index) + ": " + str(interaction))

    print()
    print("--- Controls ---")
    print(converter.tensors["ground_truth_controls"])
    print(controls.discretize(converter.tensors["ground_truth_controls"]))


if __name__ == "__main__":
    main()