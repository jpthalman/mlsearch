import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from data.config import Dim
from data import config
from data.scenario_tensor_converter import (
    ScenarioTensorConverter,
)


COLORS = {
    0: "springgreen", # AV
    1: "b", # vehicle
    2: "m", # ped
    3: "r", # motorcycle
    4: "c", # cyclist
    5: "k", # bus
    6: "grey", # static
    7: "olivedrab", # background
    8: "gold", # construction
    9: "darkcyan", # riderless bicycle
    10: "y", # unknown
}


def plot_agent(converter: ScenarioTensorConverter, a: int):
    if converter.tensors["agent_mask"][a, :].all():
        print(f"Agent {a} has no data")
        return

    fig, ax = plt.subplots()

    def draw(t):
        ax.clear()

        # Plot the roadgraph
        roadgraph = converter.tensors["roadgraph"]
        roadgraph_mask = converter.tensors["roadgraph_mask"]
        for r in range(Dim.R):
            if roadgraph_mask[r]:
                continue
            segment = roadgraph[r, :]
            ax.plot([segment[0], segment[2]], [segment[1], segment[3]], "k", linewidth=0.1)

        def plot_bbox(state: torch.Tensor):
            x, y, c, s, vx, vy, t, _ = state
            t = int(t)
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

            L /= config.POS_SCALE
            W /= config.POS_SCALE

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
            get_xy = lambda a, b: ([a[0], b[0]], [a[1], b[1]])
            ax.plot(*get_xy(fl, fr), COLORS[t], linewidth=0.2)
            ax.plot(*get_xy(fr, rr), COLORS[t], linewidth=0.2)
            ax.plot(*get_xy(rr, rl), COLORS[t], linewidth=0.2)
            ax.plot(*get_xy(rl, fl), COLORS[t], linewidth=0.2)

        # Plot this agent
        agent_state = converter.tensors["agent_history"][a, t, 0, :].clone()
        agent_state[6] = 0
        plot_bbox(agent_state)

        # Plot other agents
        agent_interactions = converter.tensors["agent_interactions"]
        agent_interactions_mask = converter.tensors["agent_interactions_mask"]
        for ai in range(Dim.Ai):
            if agent_interactions_mask[a, t, ai]:
                continue
            state = agent_interactions[a, t, ai, :].clone()
            state[:2] += agent_state[:2]
            plot_bbox(state)

        ax.axis('equal')

    ani = FuncAnimation(fig, draw, frames=Dim.T, blit=False, interval=100)
    ani.save(f"agent{a}.gif", writer="imagemagick", fps=2, dpi=250)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/mnt/sun-tcs02/planner/shared/zRL/jthalman/av2/train/0000b0f9-99f9-4a1f-a231-5be9e4c523f7")
    parser.add_argument("--agent", type=int, default=0)
    args = parser.parse_args()

    scenario_dir = Path(args.path)
    converter = ScenarioTensorConverter(scenario_dir)
    print("agents with data:")
    for i, e in enumerate(converter.tensors["agent_mask"].logical_not().float().sum(dim=1).squeeze()):
        print(i, e.numpy())
    plot_agent(converter, args.agent)


if __name__ == "__main__":
    main()
