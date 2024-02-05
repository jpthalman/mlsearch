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

from scenario_introspection_utils import (
    distance_between_states,
    Dim,
    track_object_state_to_list_at_timestep,
    dilate_track_object_states,
    object_state_to_string
)
from av2.map.map_api import ArgoverseStaticMap

class ScenarioIntrospector:
    def __init__(self: Self, scenario_path: str, map_path: str):
        self.scenario = load_argoverse_scenario_parquet(Path(scenario_path))
        self.static_map = ArgoverseStaticMap.from_json(Path(map_path))

        self.ego_track = self.get_track_from_track_id("AV")

        # The relevant tracks for the scenario. The ego track is always the first element.
        # Dim.A - 1 is used instead of Dim.A since the ego track must be
        # included in the number of agents.
        # We use the first timestep for selection of relevant tracks. This has
        # its drawbacks but will help us get started.
        self.relevant_tracks = self.find_n_closest_tracks_to_track(self.ego_track, Dim.A - 1, 0)
        self.relevant_tracks.insert(0, self.ego_track)

        # Default tensors values to zero
        self.agent_history = self.construct_agent_history_tensor()
        self.agent_interactions=torch.zeros([Dim.A, Dim.T, Dim.Ai, Dim.S])
        self.agent_mask=torch.zeros([Dim.A, Dim.T])
        self.roadgraph=torch.zeros([Dim.A, 1, Dim.R, Dim.Rd])
        self.ground_truth_control=torch.zeros([Dim.A, Dim.T - 1, Dim.C])
        self.ground_truth_control_dist=torch.zeros([Dim.A, Dim.T - 1, Dim.Cd**2])

    def get_track_from_track_id(self: Self, track_id: str) -> Track:
        for track in self.scenario.tracks:
            if track.track_id == track_id:
                return track

    def find_n_closest_tracks_to_track(self: Self, reference_track: Track, n: int, timestep: int) -> List[Track]:
        closest_tracks = []
        reference_dilated_track_object_states = dilate_track_object_states(reference_track)
        for track in self.scenario.tracks:
            if track.track_id != reference_track.track_id:
                dilated_track_object_states = dilate_track_object_states(track)
                current_object_state = dilated_track_object_states[timestep]
                if current_object_state == 0:
                    # indicates state is not present at this timestep
                    continue
                distance_to_reference = distance_between_states(current_object_state, reference_dilated_track_object_states[timestep])
                heapq.heappush(closest_tracks, (distance_to_reference, track))

        if len(closest_tracks) > n:
            closest_tracks = heapq.nsmallest(n, closest_tracks)

        def get_track_from_tuple(tuple):
            return tuple[1]

        return list(map(get_track_from_tuple, closest_tracks))

    # Agent history tensor shape: [Dim.A, Dim.T, 1, Dim.S]
    def construct_agent_history_tensor(self: Self):
        # Shape is [Dim.A, Dim.T, Dim.S] which will be unsqueezed to a tensor of
        # shape [Dim.A, Dim.T, 1, Dim.S]
        agent_history_list = []

        for track_idx in range(Dim.A):
            agent_history_at_track_idx = []
            for timestep in range(Dim.T):
                agent_history_at_track_idx_at_time_idx = []
                if track_idx < len(self.relevant_tracks) - 1:
                    track_object_state_at_timestep = track_object_state_to_list_at_timestep(self.relevant_tracks[track_idx], timestep)
                    for elem in track_object_state_at_timestep:
                        agent_history_at_track_idx_at_time_idx.append(elem)
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
    scenario_introspector = ScenarioIntrospector(parquet_file_path, map_file_path)
    print(scenario_introspector.agent_history.shape)

    # print out ev track using the ev track object
    # print(object_state_to_string(scenario_introspector.ev_track.object_states[-1]))
    # print out agent history of first row in the agent history tensor
    # print("Tensor Output: ")
    # print(scenario_introspector.agent_history[0])


if __name__ == "__main__":
    main()