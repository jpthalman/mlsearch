import random
import torch
from pathlib import Path
from typing import List, Tuple
from typing_extensions import Self

import numpy as np
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
from data.scenario_tensor_converter_utils import (
    distance_between_object_states,
    object_state_at_timestep,
    object_state_to_string,
    object_type_to_int,
    min_distance_between_tracks,
    padded_object_state_iterator,
)
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
"""
class ScenarioTensorConverter:
    def __init__(self: Self, scenario_dir: Path):
        self.scenario_path = scenario_dir / f"scenario_{scenario_dir.name}.parquet"
        self.map_path = scenario_dir / f"log_map_archive_{scenario_dir.name}.json"

        self.agent_history_path = scenario_dir / "agent_history.pt"
        self.agent_interactions_path = scenario_dir / "agent_interactions.pt"
        self.roadgraph_path = scenario_dir / "roadgraph.pt"

        self._scenario = None
        self._roadgraph = None
        self._ego_track = None
        self._relevant_tracks = None

    @property
    def scenario(self: Self):
        if self._scenario is None:
            self._scenario = load_argoverse_scenario_parquet(self.scenario_path)
        return self._scenario

    @property
    def roadgraph(self: Self) -> ArgoverseStaticMap:
        if self._roadgraph is None:
            self._roadgraph = ArgoverseStaticMap.from_json(self.map_path)
        return self._roadgraph

    @property
    def ego_track(self: Self) -> Track:
        if self._ego_track is None:
            self._load_ego_and_relevant_tracks()
        return self._ego_track

    @property
    def relevant_tracks(self: Self) -> List[Track]:
        if self._relevant_tracks is None:
            self._load_ego_and_relevant_tracks()
        return self._relevant_tracks

    @property
    def reference_point(self: Self) -> Tuple[float, float]:
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
        return central_state.position

    def write_agent_history_tensor(self: Self) -> None:
        agent_history = torch.zeros([Dim.A, Dim.T, Dim.S])
        for a, track in enumerate(self.relevant_tracks):
            for t, state in enumerate(padded_object_state_iterator(track)):
                if state is None:
                    continue

                agent_history[a, t, 0] = state.position[0] - self.reference_point[0]
                agent_history[a, t, 1] = state.position[1] - self.reference_point[1]
                agent_history[a, t, 2] = np.cos(state.heading)
                agent_history[a, t, 3] = np.sin(state.heading)
                agent_history[a, t, 4] = np.sqrt(state.velocity[0]**2 + state.velocity[1]**2)
                agent_history[a, t, 5] = object_type_to_int(track.object_type)
                agent_history[a, t, 6] = 1.0
        torch.save(agent_history, self.agent_history_path)

    def write_agent_interaction_tensor(self: Self) -> None:
        assert self.agent_history_path.exists()
        agent_history = torch.load(self.agent_history_path)
        agent_interactions = torch.zeros([Dim.A, Dim.T, Dim.Ai, Dim.S])
        for t in range(Dim.T):
            # Collect all valid agents at this timestep
            agents = []
            for a in range(Dim.A):
                if not agent_history[a, t, -1]:
                    continue
                state = agent_history[a, t, :]
                agents.append(dict(pos=state[:2], idx=a))

            # Sort by distance to each agent and populate
            for a in range(Dim.A):
                if not agent_history[a, t, -1]:
                    continue

                state = agent_history[a, t, :]

                # Sort the agents by distance to current agent.
                agents.sort(key=lambda e: torch.norm(state[:2] - e["pos"]))
                for ai in range(Dim.Ai):
                    # The case that there are fewer available agents than the configured number
                    # of relevant agents for each agent.
                    if ai + 1 >= len(agents):
                        break

                    idx = agents[ai + 1]["idx"]
                    agent_interactions[a, t, ai, :] = agent_history[idx, t, :]

                    # Shift reference frame to be relative to this agent
                    xr, yr = state[:2]
                    agent_interactions[a, t, ai, 0] -= xr
                    agent_interactions[a, t, ai, 1] -= yr
        torch.save(agent_interactions, self.agent_interactions_path)

    def write_roadgraph_tensor(self: Self) -> None:
        road = roadgraph.extract(self.reference_point, self.roadgraph)
        torch.save(road, self.roadgraph_path)

    def _load_ego_and_relevant_tracks(self:Self) -> None:
        # The relevance of a track will be determined by the min distance the
        # track gets to the ego track across all timesteps. The focal track will
        # always be included and the tracks will be of random order with the
        # exception of ego always coming first.
        # Note: There will be Dim.A relevant tracks including the ego and focal
        # tracks.
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

        RANDOM.shuffle(relevant_tracks)
        relevant_tracks.insert(0, ego_track)
        self._ego_track = ego_track
        self._relevant_tracks = relevant_tracks
