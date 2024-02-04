import enum
import heapq
import math
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

class Dim(enum.IntEnum):
    # Max agents
    A = 128
    # Time dimension size
    T = 11
    # Agent state size
    S = 7
    # Max agent interactions
    Ai = 16
    # Number of roadgraph features per agent
    R = 32
    # Dimension of roadgraph features
    Rd = 32
    # Dimension of the controls that can be applied to the vehicle
    C = 2
    # Discretization of each control dimension
    Cd = 16

class ParsedScenario:
    def __init__(self: Self, scenario_path: str, map_path: str):
        self.scenario = load_argoverse_scenario_parquet(Path(scenario_path))
        self.static_map = ArgoverseStaticMap.from_json(Path(map_path))

        self.focal_track = self.find_focal_track()

        # The relevant tracks for the scenario. The focal track is always the first element.
        # Dim.A - 1 is used instead of Dim.A since the focal track must be
        # included in the number of agents.
        # We use the first timestep for selection of relevant tracks. This has
        # its drawbacks but will help us get started.
        self.relevant_tracks = self.find_n_closest_tracks_to_track(self.focal_track, Dim.A - 1, 0)
        self.relevant_tracks.insert(0, self.focal_track)

        # Default tensors values to zero
        self.agent_history = self.construct_agent_history_tensor()
        self.agent_interactions=torch.zeros([Dim.A, Dim.T, Dim.Ai, Dim.S])
        self.agent_mask=torch.zeros([Dim.A, Dim.T])
        self.roadgraph=torch.zeros([Dim.A, 1, Dim.R, Dim.Rd])
        self.ground_truth_control=torch.zeros([Dim.A, Dim.T - 1, Dim.C])
        self.ground_truth_control_dist=torch.zeros([Dim.A, Dim.T - 1, Dim.Cd**2])

    def find_focal_track(self: Self) -> Track:
        for track in self.scenario.tracks:
            if track.track_id == self.scenario.focal_track_id:
                return track

    def find_n_closest_tracks_to_track(self: Self, reference_track: Track, n: int, timestep_idx: int) -> List[Track]:
        closest_tracks = []
        for track in self.scenario.tracks:
            if track.track_id != reference_track.track_id:
                current_object_state = track.object_states[timestep_idx]
                if current_object_state.observed:
                    distance_to_reference = _distance_between_states(current_object_state, reference_track.object_states[timestep_idx])
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
            for timestep_idx in range(Dim.T):
                agent_history_at_track_idx_at_time_idx = []
                if track_idx < len(self.relevant_tracks) - 1:
                    track_object_state_at_timestep = _track_object_state_to_list_at_timestep_idx(self.relevant_tracks[track_idx], timestep_idx)
                    agent_history_at_track_idx_at_time_idx.append(track_object_state_at_timestep)
                else:
                    agent_history_at_track_idx_at_time_idx.append([0] * Dim.S)
                agent_history_at_track_idx.append(agent_history_at_track_idx_at_time_idx)
            agent_history_list.append(agent_history_at_track_idx)

        agent_history_tensor = torch.Tensor(agent_history_list)
        return torch.unsaueeze(agent_history_tensor, 2)




def _distance_between_states(object_state_1: ObjectState, object_state_2: ObjectState) -> float:
    position_1 = object_state_1.position
    position_2 = object_state_2.position
    return math.sqrt((position_1[0] - position_2[0])**2 + (position_1[1] - position_2[1])**2)

# Object state list: [focal_frame_x, focal_frame_y, heading, vx, vy, object_type, track_category]
def _track_object_state_to_list_at_timestep_idx(track: Track, timestep_idx: int) -> List:
    print("timestep idx: " + str(timestep_idx))
    print(type(track))
    print("len object states: " + str(len(track.object_states)))
    object_state = track.object_states[timestep_idx]
    object_state_list = []
    object_state_list.append(object_state.position[0])
    object_state_list.append(object_state.position[1])
    object_state_list.append(object_state.heading)
    object_state_list.append(object_state.velocity[0])
    object_state_list.append(object_state.velocity[1])
    object_state_list.append(_get_enum_int(track.object_type))
    object_state_list.append(track.category.value)

def _get_enum_int(enum_member):
    return list(ObjectType).index(enum_member) + 1

def main():
    parquet_file_path = "/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2/train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/scenario_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.parquet"
    map_file_path = "/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2/train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7/log_map_archive_0000b0f9-99f9-4a1f-a231-5be9e4c523f7.json"
    parsed_scenario = ParsedScenario(parquet_file_path, map_file_path)


if __name__ == "__main__":
    main()