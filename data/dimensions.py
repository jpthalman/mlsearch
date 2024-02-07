import enum


"""
Enum for configured dimensions of scenario tensors.
"""
class Dim(enum.IntEnum):
    # Max agents
    A = 128
    # Time dimension size
    T = 11
    # Agent state size
    S = 8
    # Max agent interactions
    Ai = 16
    # Number of roadgraph features
    R = 512
    # Dimension of roadgraph features
    Rd = 8
    # Dimension of the controls that can be applied to the vehicle
    C = 2
    # Discretization of each control dimension
    Cd = 16
