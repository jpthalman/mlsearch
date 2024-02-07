import math
from typing import List, Iterator

from av2.datasets.motion_forecasting.data_schema import (
    Track,
    ObjectState,
    ObjectType,
)

"""
Computes the euclidean distance between two object states.

Parameters
----------
object_state_1 : ObjectState
object_state_2 : ObjectState

Returns
-------
Euclidean distance between the two object states.
"""
def distance_between_object_states(object_state_1: ObjectState, object_state_2: ObjectState) -> float:
    position_1 = object_state_1.position
    position_2 = object_state_2.position
    return math.sqrt((position_1[0] - position_2[0])**2 + (position_1[1] - position_2[1])**2)

"""
Constructs a state feature list from an object state and track.

Args:
    object_state (ObjectState) : Used for inertial data.
    track (Track) : Used for object type and track category.

Returns:
    List: [observed, x, y, heading, vx, vy, object_type, track_category]
"""
def state_feature_list(object_state: ObjectState, track: Track) -> List:
    object_state_list = []
    object_state_list.append(float(object_state.observed))
    object_state_list.append(object_state.position[0])
    object_state_list.append(object_state.position[1])
    object_state_list.append(object_state.heading)
    object_state_list.append(object_state.velocity[0])
    object_state_list.append(object_state.velocity[1])
    object_state_list.append(_object_type_to_int(track.object_type))
    object_state_list.append(track.category.value)
    return object_state_list

"""
Iterates through object states of a track to find the one associated with the
provided timestep.

Args:
    track (Track): track to query
    timestep (int): timestep to query

Returns:
    ObjectState or None: ObjectState if one is found and None otherwise.
"""
def object_state_at_timestep(track: Track, timestep: int) -> ObjectState:
    for object_state in track.object_states:
        if object_state.timestep == timestep:
            return object_state
    return None

"""
Converts an object state to a string.

Args:
    object_state (ObjectState): object state to convert.
Returns:
    str: String representation of object state.
"""
def object_state_to_string(object_state: ObjectState) -> str:
    object_state_str = "object_state: ["
    object_state_str += ("observed: " + str(object_state.observed))
    object_state_str += (", position_x: " + str(object_state.position[0]))
    object_state_str += (", position_y: " + str(object_state.position[1]))
    object_state_str += (", heading: " + str(object_state.heading))
    object_state_str += (", velocity_x: " + str(object_state.velocity[0]))
    object_state_str += (", velocity_y: " + str(object_state.velocity[1]))
    object_state_str += "]"
    return object_state_str

"""
An iterator that goes through the timesteps and returns an ObjectState
if one is associated with the timestep and None otherwise.

Args:
track (Track): the track whose object states are to be iterated on
"""
def padded_object_state_iterator(Track: track) -> Iterator[ObjectState | None]:
    object_state_idx = 0
    for timestep in range(Dim.T):
        object_state = track.object_states[object_state_idx]
        if timestep == object_state.timestep:
            object_state_idx += 1
            yield object_state
        else:
            yield None

"""
Converts an ObjectType enum to an integer value.

Args:
    object_type (ObjectType): Object type enum to convert
Returns:
    int: Integer representation of ObjectType enum.
"""
def _object_type_to_int(object_type: ObjectType) -> int:
    return list(ObjectType).index(object_type) + 1