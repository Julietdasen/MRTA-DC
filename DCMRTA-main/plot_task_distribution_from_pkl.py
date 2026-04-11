import argparse
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib import patches

from env.task_env import TaskEnv

# Compat for old pickle files that store TaskEnv as __main__.TaskEnv
import __main__
setattr(__main__, "TaskEnv", TaskEnv)


def _load_env(pkl_path):
    with open(pkl_path, "rb") as f:
        env = pickle.load(f)
    return env


def _agent_overlap_color(agent_count):
    if agent_count >= 4:
        return "m"
    if agent_count == 3:
        return "c"
    if agent_count == 2:
        return "y"
    return "r"


def plot_task_distribution(env, save_path, show_agents=True):
    fig, ax = plt.subplots(dpi=100)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.02)

    # finished_tasks = sum(1 for t in env.task_dic.values() if t.get("finished", False))
    # total_tasks = max(len(env.task_dic), 1)
    # finished_rate = finished_tasks / total_tasks
    ax.set_title(f"Task Distribution") #

    # 删除这些图例，因为它们不太适合说明distribution，而且可能会引起误解
    # green_patch = patches.Patch(color="g", label="Finished task")
    # blue_patch = patches.Patch(color="b", label="Unfinished task")
    # red_patch = patches.Patch(color="r", label="Single agent")
    # yellow_patch = patches.Patch(color="y", label="Two agents")
    # cyan_patch = patches.Patch(color="c", label="Three agents")
    # magenta_patch = patches.Patch(color="m", label=">= Four agents")
    # ax.legend(
    #     handles=[green_patch, blue_patch, red_patch, yellow_patch, cyan_patch, magenta_patch],
    #     bbox_to_anchor=(0.99, 0.7),
    # )



    for task in env.task_dic.values():
        task_color = "g" if task.get("finished", False) else "b"
        vertices = int(task["requirements"].sum()) + 3
        ax.add_patch(
            patches.RegularPolygon(
                xy=(task["location"][0] * 10, task["location"][1] * 10),
                numVertices=vertices,
                radius=0.3,
                color=task_color,
            )
        )

    depot_xy = (env.depot["location"][0] * 10, env.depot["location"][1] * 10)
    ax.add_patch(patches.Circle(depot_xy, 0.2, color="r"))

    if show_agents:
        agent_color = _agent_overlap_color(len(env.agent_dic))
        for _ in env.agent_dic.values():
            ax.add_patch(
                patches.RegularPolygon(
                    xy=depot_xy,
                    numVertices=3,
                    radius=0.2,
                    color=agent_color,
                )
            )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize task distribution from pickle env file."
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default=None,
        help="Full path to pickle file, e.g. testSet_xxx/env_0.pkl",
    ) # 这个参数是添加我需要画出什么pkl文件的路径
    parser.add_argument(
        "--test-set",
        type=str,
        default=None,
        help="Test set folder, used with --env-id. e.g. testSet_10A_20T_CONDET_TEST",
    )
    parser.add_argument(
        "--env-id",
        type=int,
        default=0,
        help="Environment index, used with --test-set.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output png path.",
    )
    parser.add_argument(
        "--hide-agents",
        action="store_true",
        help="Only draw tasks + depot.",
    )
    args = parser.parse_args()

    if args.pkl is None and args.test_set is None:
        raise ValueError("Please provide either --pkl or --test-set.")

    if args.pkl is not None:
        pkl_path = args.pkl
    else:
        pkl_path = os.path.join(args.test_set, f"env_{args.env_id}.pkl")

    if args.out is None:
        base_dir = os.path.dirname(pkl_path)
        env_name = os.path.splitext(os.path.basename(pkl_path))[0]
        out_path = os.path.join(base_dir, f"{env_name}_task_distribution.png")
    else:
        out_path = args.out

    env = _load_env(pkl_path)
    plot_task_distribution(env, out_path, show_agents=not args.hide_agents)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
