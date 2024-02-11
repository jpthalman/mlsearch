from pathlib import Path
from typing import Iterator, List, Tuple

from av2.map.map_api import (
    ArgoverseStaticMap,
    PedestrianCrossing,
)
from av2.map.lane_segment import (
    LaneMarkType,
    LaneSegment,
    LaneType,
)
from av2.map.map_primitives import Polyline
import numpy as np
import shapely
import torch

from data.config import Dim
from data import config

# Simple encoding for lane marks and lane types.
LANE_TYPE_ENCODING = {e: float(i) for i, e in enumerate(list(LaneType) + ["PED"])}
LANE_MARK_ENCODING = {e: float(i) for i, e in enumerate(list(LaneMarkType))}

"""
Extracts the road features from a map in the reference frame of the provided
reference point.

Args:
    reference_point (Tuple[float]): x, y point in map frame to use as origin
    map_path (Path): Path to load the ArgoverseStaticMap from

Returns:
    Roadgraph tensor: [Dim.R, Dim.Rd]
    Roadgraph mask tensor: [Dim.R]
"""
def extract(
    reference_point: Tuple[float],
    map_path: Path
) -> Tuple[torch.Tensor, torch.Tensor]:

    # Loads an SRTree for distance calculations.
    tree, data = _load_rtree(map_path, reference_point)

    roadgraph = torch.zeros([Dim.R, Dim.Rd])
    roadgraph_mask = torch.zeros([Dim.R]).bool()
    query = shapely.Point(reference_point[0], reference_point[1])

    relevant_tree_indices = []
    radius = config.ROADGRAPH_SEARCH_RADIUS

    # Double the search radius until the relevant_tree_indices is filled from the tree query
    while len(relevant_tree_indices) < min(Dim.R, len(data)):
        relevant_tree_indices = list(tree.query(query.buffer(radius)))
        radius *= 2

    points = tree.geometries.take(relevant_tree_indices)
    points_data = data.take(relevant_tree_indices, axis=0)
    R = min(Dim.R, len(relevant_tree_indices))
    for r in range(R):
        roadgraph[r, 0] = points[r].coords[0][0]
        roadgraph[r, 1] = points[r].coords[0][1]
        roadgraph[r, 2] = points[r].coords[1][0]
        roadgraph[r, 3] = points[r].coords[1][1]
        roadgraph[r, 4] = points_data[r][0]
        roadgraph[r, 5] = points_data[r][1]
        roadgraph[r, 6] = points_data[r][2]
    roadgraph_mask[:R] = False
    roadgraph_mask[R:] = True
    return roadgraph, roadgraph_mask

"""
Constructs and returns an STRtree representing the map. The elements of the STRtree
will all be line segments from the lane segments and pedestrian crossings. Metadata
on the semantics of the line segments will be stored separately and returned as a
numpy array.

Args:
    map_path (Path): path to the map for loading the ArgoverseStaticMap
    reference_point (Tuple[float]): point used for the origin
"""
def _load_rtree(map_path: Path, reference_point: Tuple[float]):
    roadgraph = ArgoverseStaticMap.from_json(map_path)

    geometries = []
    data = []
    for segment in roadgraph.vector_lane_segments.values():
        g, d = _extract_lane_segment(segment, reference_point)
        geometries.extend(g)
        data.extend(d)
    for crossing in roadgraph.vector_pedestrian_crossings.values():
        g, d = _extract_pedestrian_crossing(crossing, reference_point)
        geometries.extend(g)
        data.extend(d)

    return shapely.STRtree(geometries), np.array(data)

"""
Extracts the relevant information for a LaneSegment.

Args:
    segment (LaneSegment): lane segment to extract from
    reference_point (Tuple[float]): reference_frame_origin

Returns:
    geometries of the road features: List[shapely.LineString]
    metadata for road features: List[List[is_intersection, lane_type, lane_mark_type]]

    Note: The two lists will have the same length
"""
def _extract_lane_segment(
    segment: LaneSegment,
    reference_point: Tuple[float],
) -> Tuple[List[shapely.LineString], List[List[float]]]:
    geometries = []
    data = []
    for boundary, mark_type in (
        (segment.right_lane_boundary, segment.right_mark_type),
        (segment.left_lane_boundary, segment.left_mark_type)
    ):
        segment_data = [
            float(segment.is_intersection),
            LANE_TYPE_ENCODING[segment.lane_type],
            LANE_MARK_ENCODING[mark_type],
        ]
        for line in _polyline_to_shapely(boundary, reference_point):
            geometries.append(line)
            data.append(segment_data)
    return geometries, data

"""
Extract relevant information from a PedestrianCrossing using the reference_point
as the origin.

Args:
    crossing (PedestrianCrossing)
    reference_point (Tuple[float]): Reference point to use as the origin.
"""
def _extract_pedestrian_crossing(
    crossing: PedestrianCrossing,
    reference_point: Tuple[float],
) -> Tuple[List[shapely.LineString], List[List[float]]]:
    geometries = []
    data = []
    segment_data = [
        float(True),
        LANE_TYPE_ENCODING["PED"],
        LANE_MARK_ENCODING[LaneMarkType.UNKNOWN],
    ]
    for line in _polyline_to_shapely(crossing.edge1, reference_point):
        geometries.append(line)
        data.append(segment_data)
    for line in _polyline_to_shapely(crossing.edge2, reference_point):
        geometries.append(line)
        data.append(segment_data)
    return geometries, data

"""
An iterator that yields the LineString representations of a PolyLine's
waypoints in the reference frame of the given reference point.

Args:
    polyline (Polyline)
    reference_point (Tuple[float]): Reference frame origin

Returns:
    Iterator that yields the LineString representations of a PolyLine's
waypoints in the reference frame of the given reference point
"""
def _polyline_to_shapely(
    polyline: Polyline,
    reference_point: Tuple[float],
) -> Iterator[shapely.LineString]:
    for a, b in zip(polyline.waypoints[:-1], polyline.waypoints[1:]):
        x0 = a.x - reference_point[0]
        y0 = a.y - reference_point[1]
        x1 = b.x - reference_point[0]
        y1 = b.y - reference_point[1]
        yield shapely.LineString(
            [shapely.Point(x0, y0), shapely.Point(x1, y1)]
        )
