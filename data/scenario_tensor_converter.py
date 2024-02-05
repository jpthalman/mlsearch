import heapq
import torch
from pathlib import Path
from typing import List
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

from scenario_tensor_converter_utils import (
    distance_between_object_states,
    Dim,
    state_feature_list,
    object_state_at_timestep,
    object_state_to_string,
)

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
    def __init__(self: Self, scenario_path: str, map_path: str):
        self.scenario = load_argoverse_scenario_parquet(Path(scenario_path))
        self.static_map = ArgoverseStaticMap.from_json(Path(map_path))

        # Ego has a predefined track id of "AV"
        self.ego_track = self.track_from_track_id("AV")

        # The relevant tracks for the scenario are the Dim.A - 1 closest tracks
        # to ego at the first time step. The ego track is always the first
        # element.
        # Note: We use the first timestep for selection of relevant tracks. This
        # has its drawbacks but will help us get started.
        self.relevant_tracks = self.n_closest_tracks_to_track(self.ego_track, Dim.A - 1, 0)
        self.relevant_tracks.insert(0, self.ego_track)

        # This tensor represents the trace histories of all relevant agents. If
        # there are fewer than Dim.A total agents, the remaining space is filled
        # with `0`s.
        # Shape: [Dim.A, Dim.T, 1, Dim.S]
        self.agent_history = self.construct_agent_history_tensor()

        # TODO: Populate below tensors. Default tensors values to zero
        self.agent_interactions=torch.zeros([Dim.A, Dim.T, Dim.Ai, Dim.S])
        self.agent_mask=torch.zeros([Dim.A, Dim.T])
        self.roadgraph=torch.zeros([Dim.A, 1, Dim.R, Dim.Rd])

        # TODO: Change to only use Ego controls.
        self.ground_truth_control=torch.zeros([Dim.A, Dim.T - 1, Dim.C])
        self.ground_truth_control_dist=torch.zeros([Dim.A, Dim.T - 1, Dim.Cd**2])

    """Returns the track with the associated track_id."""
    def track_from_track_id(self: Self, track_id: str) -> Track:
        for track in self.scenario.tracks:
            if track.track_id == track_id:
                return track

    def n_closest_tracks_to_track(self: Self, reference_track: Track, n: int, timestep: int) -> List[Track]:
        closest_tracks = []
        for track in self.scenario.tracks:
            if track.track_id != reference_track.track_id:
                current_object_state = object_state_at_timestep(track, timestep)
                if current_object_state is None:
                    # indicates state is not present at this timestep
                    continue
                distance_to_reference = distance_between_object_states(current_object_state, object_state_at_timestep(reference_track, timestep))
                heapq.heappush(closest_tracks, (distance_to_reference, track))

        if len(closest_tracks) > n:
            closest_tracks = heapq.nsmallest(n, closest_tracks)

        """Returns the track from a (distance_to_reference, Track) tuple"""
        def _get_track_from_tuple(tuple):
            return tuple[1]
        return list(map(_get_track_from_tuple, closest_tracks))

    """ Returns an agent history tensor of shape: [Dim.A, Dim.T, 1, Dim.S]"""
    def construct_agent_history_tensor(self: Self):
        agent_history_list = []

        for track_idx in range(Dim.A):
            agent_history_at_track_idx = []
            for timestep in range(Dim.T):
                agent_history_at_track_idx_at_time_idx = []
                if track_idx < len(self.relevant_tracks) - 1:
                    track = self.relevant_tracks[track_idx]
                    object_state = object_state_at_timestep(track, timestep)
                    if object_state is not None:
                        state_features = state_feature_list(object_state, track)
                        for feature in state_features:
                            agent_history_at_track_idx_at_time_idx.append(feature)
                    else:
                        for idx in range(Dim.S):
                            agent_history_at_track_idx_at_time_idx.append(0)
                else:
                    for idx in range(Dim.S):
                        agent_history_at_track_idx_at_time_idx.append(0)
                agent_history_at_track_idx.append(agent_history_at_track_idx_at_time_idx)
            agent_history_list.append(agent_history_at_track_idx)

        agent_history_tensor = torch.Tensor(agent_history_list)
        return torch.unsqueeze(agent_history_tensor, 2)

def main():
    parquet_file_path = "/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2/train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/scenario_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet"
    map_file_path = "/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2/train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/log_map_archive_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.json"
    scenario_tensor_converter = ScenarioTensorConverter(parquet_file_path, map_file_path)
    print(scenario_tensor_converter.agent_history.shape)

    # print out ev track using the ev track object
    # print(object_state_to_string(scenario_introspector.ev_track.object_states[-1]))
    # print out agent history of first row in the agent history tensor
    # print("Tensor Output: ")
    # print(scenario_introspector.agent_history[0])


if __name__ == "__main__":
    main()