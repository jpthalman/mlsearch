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

from data.dimensions import Dim
from data.scenario_tensor_converter_utils import (
    distance_between_object_states,
    state_feature_list,
    object_state_at_timestep,
    object_state_to_string,
    min_distance_between_tracks,
    padded_object_state_iterator,
    transform_to_reference_frame,
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

        # The relevance of a track will be determined by the min distance the
        # track gets to the ego track across all timesteps. The focal track will
        # always be included and the tracks will be of random order with the
        # exception of ego always coming first.
        # Note: There will be Dim.A relevant tracks including the ego and focal
        # tracks.
        self.ego_track, self.relevant_tracks = self.ego_and_relevant_tracks()
        random.shuffle(self.relevant_tracks)
        self.relevant_tracks.insert(0, self.ego_track)

        # This tensor represents the trace histories of all relevant agents. If
        # there are fewer than Dim.A total agents, the remaining space is filled
        # with `0`s.
        # Shape: [Dim.A, Dim.T, 1, Dim.S]
        self.agent_history = self.construct_agent_history_tensor()

        # This tensor represents the interactions between agents. If there are
        # fewer than Dim.Ai agents, the remaining space is filled with 0's.
        # Shape: [Dim.A, Dim.T, Dim.Ai, Dim.S]
        self.agent_interactions = self.construct_agent_interactions_tensor()

         # TODO: Populate below tensors. Default tensors values to zero
        self.agent_mask=torch.zeros([Dim.A, 10 * Dim.T])
        self.roadgraph=torch.zeros([Dim.A, 1, Dim.R, Dim.Rd])

        # TODO: Change to only use Ego controls.
        self.ground_truth_control=torch.zeros([Dim.A, (10 * Dim.T) - 1, Dim.C])
        self.ground_truth_control_dist=torch.zeros([Dim.A, (10 * Dim.T) - 1, Dim.Cd**2])

    """Returns the track with the associated track_id."""
    def track_from_track_id(self: Self, track_id: str) -> Track:
        for track in self.scenario.tracks:
            if track.track_id == track_id:
                return track

    """Returns the ego and relevant tracks separately"""
    def ego_and_relevant_tracks(self :Self) -> Tuple[Track, List[Track]]:
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

    """
    Returns the n most relevant tracks to a reference track. Relevance of a
    track is determined by how close it gets to the reference track at any
    point in time. The closer it gets, the higher the priority. The tracks are
    returned in random order.

    Only tracks within the scenario's high level relevant tracks are considered.
    """
    def relevant_n_tracks_to_reference_track(self: Self, n: int, reference_track: Track) -> List[Track]:
        relevant_tracks_to_reference = []
        for track in self.relevant_tracks:
            if track is not None and track.track_id != reference_track.track_id:
                relevant_tracks_to_reference.append(track)

        relevant_tracks_to_reference.sort(key=lambda track: min_distance_between_tracks(track, reference_track))
        num_tracks = len(relevant_tracks_to_reference)
        if num_tracks > n:
            relevant_tracks_to_reference = relevant_tracks_to_reference[:n]
        elif num_tracks < n:
            relevant_tracks_to_reference.extend([None] * (n - num_tracks))
        assert len(relevant_tracks_to_reference) == n
        random.shuffle(relevant_tracks_to_reference)
        return relevant_tracks_to_reference

    """ Returns an agent history tensor of shape: [Dim.A, Dim.T, 1, Dim.S]"""
    def construct_agent_history_tensor(self: Self) -> torch.Tensor:
        agent_history_list = []

        for track_idx in range(Dim.A):
            agent_history_at_track_idx = []
            for timestep in range(10 * Dim.T):
                agent_history_at_track_idx_at_time_idx = []
                track = self.relevant_tracks[track_idx]
                if track is not None:
                    # Nominal case of adding features.
                    object_state = object_state_at_timestep(track, timestep)
                    if object_state is not None:
                        state_features = state_feature_list(object_state, track)
                        for feature in state_features:
                            agent_history_at_track_idx_at_time_idx.append(feature)
                    else:
                        # Case where there is no object state for this track at
                        # this timestep.
                        for idx in range(Dim.S):
                            agent_history_at_track_idx_at_time_idx.append(0)
                else:
                    # Case where there were fewer available tracks than Dim.A
                    for idx in range(Dim.S):
                        agent_history_at_track_idx_at_time_idx.append(0)
                agent_history_at_track_idx.append(agent_history_at_track_idx_at_time_idx)
            agent_history_list.append(agent_history_at_track_idx)

        agent_history_tensor = torch.Tensor(agent_history_list)
        return torch.unsqueeze(agent_history_tensor, 2)

    """Returns an agent interaction tensor of shape: [Dim.A, Dim.T, Dim.Ai, Dim.S]"""
    def construct_agent_interactions_tensor(self: Self) -> torch.Tensor:
        a_dimension = []
        for track in self.relevant_tracks:
            if track is not None:
                closest_tracks = self.relevant_n_tracks_to_reference_track(Dim.Ai, track)
                t_dimension = []
                for timestep in range(Dim.T):
                    ai_dimension = []
                    for ai in closest_tracks:
                        s_dimension = []
                        if ai is not None:
                            object_state = object_state_at_timestep(ai, timestep)
                            if object_state is not None:
                                state_features = state_feature_list(object_state, track)
                                s_dimension.extend(state_features)
                            else:
                                # Case where there is no object state for this track at
                                # this timestep.
                                s_dimension.extend([0] * Dim.S)
                        else:
                            # Case where there were fewer available tracks than Dim.Ai
                            s_dimension.extend([0] * Dim.S)
                        ai_dimension.append(s_dimension)
                    t_dimension.append(ai_dimension)
                a_dimension.append(t_dimension)
            else:
                none_padding = [[[0 for _ in range(Dim.S)] for _ in range(Dim.Ai)] for _ in range(Dim.T)]
                a_dimension.append(none_padding)
        return torch.Tensor(a_dimension)


def main():
    parquet_file_path = "/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2/train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/scenario_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet"
    map_file_path = "/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2/train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/log_map_archive_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.json"
    scenario_tensor_converter = ScenarioTensorConverter(parquet_file_path, map_file_path)
    print(scenario_tensor_converter.agent_history.shape)

    # print out ego track using the ego track object
    print(object_state_to_string(scenario_tensor_converter.ego_track.object_states[0]))
    # print out agent history of first row in the agent history tensor
    print("Ego Tensor Output: ")
    print(scenario_tensor_converter.agent_history[0][0][0])

    # print out focal track using the focal track object
    focal_track = scenario_tensor_converter.track_from_track_id(scenario_tensor_converter.scenario.focal_track_id)
    focal_object_state = object_state_at_timestep(focal_track, 0)
    print(state_feature_list(focal_object_state, focal_track))

    print("Ai shape: " + str(scenario_tensor_converter.agent_interactions.shape))
    # print the agent interactions of the ego object at t=0
    ego_interactions = scenario_tensor_converter.agent_interactions[0][2]
    for index, interaction in enumerate(ego_interactions):
        print(str(index) + ": " + str(interaction))


if __name__ == "__main__":
    main()