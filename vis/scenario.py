from pathlib import Path
from typing import Iterable

import torch
import plotly.graph_objects as go
import streamlit as st

from data import controls as control_utils
from data.config import TRAIN_DATA_ROOT, EXPERIMENT_ROOT, POS_SCALE, VEL_SCALE
from train.module import MLSearchModule


@st.cache_resource
def load_model(checkpoint_path: Path) ->  MLSearchModule | None:
    try:
        return MLSearchModule.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device("cpu"),
        )
    except Exception as err:
        st.warning(f"Failed to load model from checkpoint! error={str(err)}")
        return


def draw_roadgraph(roadgraph: torch.Tensor) -> Iterable[go.Scatter]:
    R, Rd = roadgraph.shape
    segments = []
    for r in range(R):
        x0, y0, x1, y1, intersection, lane_type, mark_type, valid = roadgraph[r, :]
        if valid != 1.0:
            continue

        # TODO: Change color based on lane type
        segment = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color="rgb(128,138,135)"),
        )
        segments.append(segment)
    return segments


def draw_agents(agent_tensor: torch.Tensor) -> Iterable[go.Scatter]:
    COLORS = {
        0: "rgb(0,255,127)", # AV
        1: "rgb(0,72,186)", # vehicle
        2: "rgb(255,0,255)", # ped
        3: "rgb(255,64,64)", # motorcycle
        4: "rgb(0,255,255)", # cyclist
        5: "rgb(0,46,99)", # bus
        6: "rgb(128,128,105)", # static
        7: "rgb(142,142,56)", # background
        8: "rgb(255,215,0)", # construction
        9: "rgb(0,139,139)", # riderless bicycle
        10: "rgb(255,255,51)", # unknown
    }

    agent_data = []
    for a in range(agent_tensor.shape[0]):
        x, y, c, s, v, t, valid = agent_tensor[a, :]
        if valid != 1.0:
            continue

        t = int(t.item() if a > 0 else 0)
        if t in (0, 1):
            L, W = 4, 2
        elif t == 2:
            L, W = 1, 1
        elif t in (3, 4, 9):
            L, W = 2, 1
        elif t == 5:
            L, W = 7, 2.5
        else:
            L, W = 2, 2

        fl = [
            x + c * L/2 - s * W/2,
            y + s * L/2 + c * W/2
        ]
        fr = [
            x + c * L/2 + s * W/2,
            y + s * L/2 - c * W/2
        ]
        rl = [
            x - c * L/2 - s * W/2,
            y - s * L/2 + c * W/2
        ]
        rr = [
            x - c * L/2 + s * W/2,
            y - s * L/2 - c * W/2
        ]

        bbox = go.Scatter(
            x=[fl[0], fr[0], rr[0], rl[0], fl[0]],
            y=[fl[1], fr[1], rr[1], rl[1], fl[1]],
            name="Hero" if a == 0 else f"Agent{a}",
            mode="lines",
            line=dict(color=COLORS[t]),
            fill="toself",
            fillcolor=COLORS[t],
            opacity=0.5,
        )
        agent_data.append(bbox)
    return agent_data


def draw_predictions(
    agent_history: torch.Tensor,
    roadgraph: torch.Tensor,
    checkpoint_path: Path,
    depth: int,
) -> Iterable[go.Scatter]:
    model = load_model(checkpoint_path)
    model.eval()

    agent_history = agent_history.clone().to(model.device)
    agent_history[:, :, :2] /= POS_SCALE
    agent_history[:, :, 4] /= VEL_SCALE

    roadgraph = roadgraph.clone().to(model.device)
    roadgraph[:, :4] /= POS_SCALE

    embedding = model.scene_encoder(
        agent_history=agent_history.unsqueeze(0),
        roadgraph=roadgraph.unsqueeze(0),
    )

    states = agent_history[0, :, :]
    states[:, :2] *= POS_SCALE
    states[:, 4] *= VEL_SCALE

    data = [[[], []] for _ in range(agent_history.shape[1])]
    for t in range(agent_history.shape[1]):
        data[t][0].append(states[t, 0].cpu().item())
        data[t][1].append(states[t, 1].cpu().item())

    for _ in range(depth):
        controls = model.control_predictor(embedding)
        states = control_utils.integrate(states, controls[0, :, :])
        for t in range(agent_history.shape[1]):
            data[t][0].append(states[t, 0].cpu().item())
            data[t][1].append(states[t, 1].cpu().item())

        states_norm = states.clone()
        states_norm[:, :2] /= POS_SCALE
        states_norm[:, 4] /= VEL_SCALE
        embedding = model.world_model(
            scene_embedding=embedding,
            next_ego_state=states_norm.unsqueeze(0),
        )

    predictions = []
    for e in data:
        pred = go.Scatter(
            x=e[0],
            y=e[1],
            mode="lines+markers",
            line=dict(color="rgb(255,64,64)"),
        )
        predictions.append([pred])
    return predictions


@st.cache_resource
def load_scenario(
    scenario_path: Path,
    checkpoint_path: Path | None,
    depth: int,
) -> Iterable[go.Figure]:
    agent_history = torch.load(scenario_path / "agent_history.pt")
    agent_history = agent_history[:64, ::2, :]
    roadgraph = torch.load(scenario_path / "roadgraph.pt")

    prediction_data = [[] for _ in range(agent_history.shape[1])]
    if checkpoint_path is not None:
        prediction_data = draw_predictions(
            agent_history,
            roadgraph,
            checkpoint_path,
            depth,
        )

    figures = []
    roadgraph_data = draw_roadgraph(roadgraph)
    for t in range(agent_history.shape[1]):
        agent_data = draw_agents(agent_history[:, t, :])
        figure = go.Figure(
            data=roadgraph_data + agent_data + prediction_data[t],
            layout=go.Layout(
                height=1000,
                width=1000,
                showlegend=False,
                yaxis=go.layout.YAxis(
                    autorange="reversed",
                    scaleanchor="x",
                    scaleratio=1,
                ),
                uirevision="42",
            ),
        )
        figures.append(figure)
    return figures


def main():
    st.set_page_config(layout="wide")
    st.title("Model Input Visualization")

    with st.sidebar:
        scenario_path = Path(
            st.text_input(
                "Scenario path",
                str(TRAIN_DATA_ROOT / "001cbde0-dfca-40ff-ab9c-38230e96d03b"),
            )
        )
        if not scenario_path.exists():
            st.write(f"Path does not exist! {str(scenario_path)}")
            return

        checkpoint_path = Path(st.text_input(
            "Checkpoint path",
            str(EXPERIMENT_ROOT / "best.ckpt"),
        ))

        depth = st.slider("Depth", 1, 11)

    scenes: Iterable[go.Figure] = load_scenario(scenario_path, checkpoint_path, depth)
    display_area = st.container()
    index = st.slider("Index", 0, len(scenes) - 1)
    display_area.plotly_chart(
        scenes[index],
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
