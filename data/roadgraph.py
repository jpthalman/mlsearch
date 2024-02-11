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

from data.dimensions import Dim


LANE_TYPE_ENCODING = {e: float(i) for i, e in enumerate(list(LaneType) + ["PED"])}
LANE_MARK_ENCODING = {e: float(i) for i, e in enumerate(list(LaneMarkType))}


def extract(
    reference_point: Tuple[float],
    map_path: Path
) -> torch.Tensor:
    """
    agent_history[A, T, 1, S]
    agent_mask[A, T]
    """
    tree, data = _load_rtree(map_path, reference_point)

    roadgraph = torch.zeros([Dim.R, Dim.Rd])
    roadgraph_mask = torch.zeros([Dim.R]).bool()
    query = shapely.Point(reference_point[0], reference_point[1])

    idx = []
    radius = 100.0
    while len(idx) < min(Dim.R, len(data)):
        idx = list(tree.query(query.buffer(radius)))
        radius *= 2

    points = tree.geometries.take(idx)
    points_data = data.take(idx, axis=0)
    R = min(Dim.R, len(idx))
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
