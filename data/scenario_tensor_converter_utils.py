import enum
import math
from typing import List

from av2.datasets.motion_forecasting.data_schema import (
    Track,
    ObjectState,
    ObjectType,
)

class Dim(enum.IntEnum):
    # Max agents
    A = 128
    # Time dimension size
    T = 110
    # Agent state size
    S = 8
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

def distance_between_object_states(object_state_1: ObjectState, object_state_2: ObjectState) -> float:
    position_1 = object_state_1.position
    position_2 = object_state_2.position
    return math.sqrt((position_1[0] - position_2[0])**2 + (position_1[1] - position_2[1])**2)

# Object state list: [observed, ego_frame_x, ego_frame_y, heading, vx, vy, object_type, track_category]
def track_object_state_to_list_at_timestep(track: Track, timestep: int) -> List:
    object_state = dilate_track_object_states(track)[timestep]
    if object_state == 0:
        # This indicates the track has no object state at the timestep. Return all 0 values.
        return [0] * Dim.S
    object_state_list = []
    object_state_list.append(float(object_state.observed))
    object_state_list.append(object_state.position[0])
    object_state_list.append(object_state.position[1])
    object_state_list.append(object_state.heading)
    object_state_list.append(object_state.velocity[0])
    object_state_list.append(object_state.velocity[1])
    object_state_list.append(_get_enum_int(track.object_type))
    object_state_list.append(track.category.value)
    return object_state_list

def dilate_track_object_states(track: Track) -> List[ObjectState]:
    dilated_track_object_states = [0] * Dim.T
    for object_state in track.object_states:
        dilated_track_object_states[object_state.timestep] = object_state
    return dilated_track_object_states

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

def _get_enum_int(enum_member):
    return list(ObjectType).index(enum_member) + 1