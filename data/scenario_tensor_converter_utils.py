from typing import Iterator

from av2.datasets.motion_forecasting.data_schema import (
    Track,
    ObjectState,
    ObjectType,
)

from data import config


"""
An iterator that goes through the timesteps and returns an ObjectState
if one is associated with the timestep and None otherwise.

Args:
track (Track): the track whose object states are to be iterated on
"""
def padded_object_state_iterator(track: Track) -> Iterator[ObjectState | None]:
    object_state_idx = 0
    for timestep in range(config.AV2_MAX_TIME):
        if track is None:
            yield None
            continue
        elif object_state_idx >= len(track.object_states):
            yield None
            continue

        object_state = track.object_states[object_state_idx]
        if timestep == object_state.timestep:
            object_state_idx += 1
            yield object_state
        else:
            yield None

"""
Returns the min distance across timesteps for two tracks.
"""
def min_distance_between_tracks(track_1: Track, track_2: Track) -> float:
    min_dist = float('inf')
    for track_1_os, track_2_os in zip(padded_object_state_iterator(track_1), padded_object_state_iterator(track_2)):
        if track_1_os is None or track_2_os is None:
            continue
        min_dist = min(min_dist, distance_between_object_states(track_1_os, track_2_os))
    return min_dist

"""
Converts an ObjectType enum to an integer value.

Args:
    object_type (ObjectType): Object type enum to convert
Returns:
    int: Integer representation of ObjectType enum.
"""
def object_type_to_int(object_type: ObjectType) -> int:
    return list(ObjectType).index(object_type) + 1
