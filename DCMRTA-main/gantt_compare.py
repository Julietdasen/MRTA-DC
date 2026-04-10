import argparse
import copy
import importlib.util
import os
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
from matplotlib import patches

from env.task_env import TaskEnv

# Compat for old pickle files that store TaskEnv as __main__.TaskEnv
import __main__
setattr(__main__, "TaskEnv", TaskEnv)


EMBEDDING_DIM = 128
AGENT_INPUT_DIM = 6
TASK_INPUT_DIM = 5
USE_GPU_GLOBAL = True
METHOD_NOTES = {
    "RL-LF": "RL Leader-Follower",
    "RL-IA": "RL Individaul Strategy",
    "OR-Tools": "OR-Tools grouped mTSP Problem optimization",
    "CTAS-D": "read Off-Line TeamPlanner Deterministic Path",
}
STATE_NOTE = ""


def _node_label(task_id):
    if task_id == -1:
        return "D"
    return f"T{int(task_id) + 1}"


def build_agent_route_notes(env, max_visits=6):
    notes = {}
    for aid in sorted(env.agent_dic.keys()):
        route = env.agent_dic[aid].get("route", [])
        arrivals = env.agent_dic[aid].get("arrival_time", [])
        if len(route) == 0:
            notes[aid] = "D (idle)"
            continue

        segments = []
        cap = min(len(route), max_visits)
        for idx in range(cap):
            node = _node_label(route[idx])
            if idx < len(arrivals):
                node = f"{node}@{float(arrivals[idx]):.1f}"
            segments.append(node)

        if len(route) > max_visits:
            segments.append("...")
        notes[aid] = " -> ".join(segments)
    return notes


def annotate_agent_routes(ax, env, max_visits=6, fontsize=7):
    notes = build_agent_route_notes(env, max_visits=max_visits)
    agent_ids = sorted(env.agent_dic.keys())
    y_ticks = ax.get_yticks()
    x_left, x_right = ax.get_xlim()
    x_span = max(x_right - x_left, 1.0)

    note_x = x_right + x_span * 0.02
    for i, aid in enumerate(agent_ids):
        if i >= len(y_ticks):
            break
        ax.text(
            note_x,
            y_ticks[i],
            f"R{aid}: {notes[aid]}",
            va="center",
            ha="left",
            fontsize=fontsize,
            family="monospace",
            color="black",
        )

    # Expand x-range so right-side route notes are visible.
    ax.set_xlim(x_left, x_right + x_span * 0.65)


def _segment_action_text(rec):
    aid = rec["agent_id"]
    task_id = rec["task_id"]
    task_name = "Depot" if task_id is None or task_id == -1 else f"Task {int(task_id) + 1}"
    state = rec["state"]

    # Keep semantics aligned with TaskEnv.build_gantt_records labels.
    if state == "travel":
        return f"Robot {aid} -> {task_name} (travel)"
    if state == "execute":
        return f"Robot {aid} @ {task_name} (execute)"
    if state == "wait":
        return f"Robot {aid} @ {task_name} (wait)"
    return f"Robot {aid} @ Depot (idle)"


def annotate_segment_actions(ax, env, fontsize=4.2, min_text_width=1.2):
    records = env.build_gantt_records()
    agent_ids = sorted(env.agent_dic.keys())
    row_h = 7
    row_gap = 4
    y_pos = {aid: idx * (row_h + row_gap) for idx, aid in enumerate(agent_ids)}

    for rec in records:
        start = float(rec["start"])
        end = float(rec["end"])
        duration = end - start
        if duration <= 0:
            continue
        if duration < float(min_text_width):
            # Keep consistent with TaskEnv.plot_gantt: ignore text on tiny blocks.
            continue

        x = start + duration / 2
        y = y_pos[rec["agent_id"]] + row_h / 2
        rot = 0
        ax.text(
            x,
            y,
            _segment_action_text(rec),
            ha="center",
            va="center",
            fontsize=fontsize,
            rotation=rot,
            color="black",
            bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "alpha": 0.68, "edgecolor": "none"},
        )


def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_case_env(test_set, env_id):
    pkl_path = os.path.join(test_set, f"env_{env_id}.pkl")
    with open(pkl_path, "rb") as f:
        env = pickle.load(f)

    tasks = copy.deepcopy(env.task_dic)
    agents = copy.deepcopy(env.agent_dic)
    depot = copy.deepcopy(env.depot)
    env.reset((tasks, agents, depot))
    env.clear_decisions()
    env.reactive_planning = False
    env.max_waiting_time = 10
    env.force_wait = True
    return env


def build_worker(model_folder, device):
    import torch
    from attention import AttentionNet
    from worker import Worker

    model_path = os.path.join("model", model_folder, "checkpoint.pth")
    network = AttentionNet(AGENT_INPUT_DIM, TASK_INPUT_DIM, EMBEDDING_DIM).to(device)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    network.load_state_dict(checkpoint["model"])
    return Worker(0, network, network, 0, device)


def run_rl_env(worker, env, rl_method):
    if rl_method == "IA":
        worker.run_test_IS(0, env)
    else:
        worker.run_test(0, env)
    return copy.deepcopy(worker.baseline_env)


def run_ortools_env(or_tools_module, env):
    solver = or_tools_module.TSPSolver()
    solver.VRP(env)
    env.execute_by_route(path="./", method="OR-Tools", plot_figure=False)
    return env


def run_ctasd_env(ctasd_module, env, test_set, env_id):
    result_dir = os.path.join(test_set, f"env_{env_id}") + "/"
    ok = ctasd_module.CTASD_read_results(env, result_dir)
    if ok is None:
        return None
    env.execute_by_route(path="./", method="CTAS-D", plot_figure=False)
    return env


def draw_gantt_on_ax(
    env,
    ax,
    title,
    show_text=False,
    min_text_width=1.2,
    text_fontsize=5,
    axis_fontsize=11,
    tick_fontsize=10,
):
    records = env.build_gantt_records()
    color_map = {
        "travel": "tab:blue",
        "wait": "tab:orange",
        "execute": "tab:green",
        "idle": "tab:gray",
    }

    agent_ids = sorted(env.agent_dic.keys())
    row_h = 7
    row_gap = 4
    y_pos = {aid: idx * (row_h + row_gap) for idx, aid in enumerate(agent_ids)}

    max_end = 0.0
    for rec in records:
        duration = rec["end"] - rec["start"]
        if duration <= 0:
            continue

        y = y_pos[rec["agent_id"]]
        max_end = max(max_end, rec["end"])
        ax.broken_barh(
            [(rec["start"], duration)],
            (y, row_h),
            facecolors=color_map.get(rec["state"], "tab:purple"),
            edgecolors="black",
            linewidth=0.8,
            alpha=0.9,
        )
        if show_text and duration >= min_text_width:
            ax.text(
                rec["start"] + duration / 2,
                y + row_h / 2,
                rec["label"],
                ha="center",
                va="center",
                fontsize=text_fontsize,
            )

    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Time", fontsize=axis_fontsize)
    ax.set_ylabel("Agent / Robot", fontsize=axis_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)

    ax.set_yticks([y_pos[aid] + row_h / 2 for aid in agent_ids])
    ax.set_yticklabels([f"R{aid}" for aid in agent_ids], fontsize=tick_fontsize)
    ax.set_ylim(-2, (len(agent_ids) - 1) * (row_h + row_gap) + row_h + 3 if agent_ids else 10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    return max(float(env.current_time), max_end, 1.0)


def save_single_gantt(method_name, env, output_dir, env_id, show_text):
    save_path = os.path.join(output_dir, f"gantt_{method_name}_env_{env_id}.png")
    fig, ax, _ = env.plot_gantt(
        save_path=save_path,
        title=f"{method_name} | env_{env_id} | makespan={env.current_time:.2f}",
        show_text=False,
        min_text_width=1.2,
        text_fontsize=4.5,
    )
    annotate_segment_actions(ax=ax, env=env, fontsize=4.2, min_text_width=1.2)
    method_note = METHOD_NOTES.get(method_name, method_name)
    fig.text(
        0.012,
        0.988,
        f"{method_note}\n{STATE_NOTE}\n每个色块内: Robot k ->/ @ Task t (travel/execute/wait)\n小时间块按阈值自动省略文字",
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.86, "edgecolor": "0.6"},
    )
    fig.savefig(save_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_comparison_gantt(method_envs, output_dir, env_id, show_text):
    valid = [(name, env) for name, env in method_envs if env is not None]
    if not valid:
        return None

    n_methods = len(valid)
    agent_num = len(valid[0][1].agent_dic)
    fig_h = max(8, n_methods * (2.2 + 0.18 * agent_num))
    fig, axes = plt.subplots(n_methods, 1, figsize=(22, fig_h), dpi=260, sharex=True)
    if n_methods == 1:
        axes = [axes]

    x_max = 1.0
    for ax, (name, env) in zip(axes, valid):
        x_local = draw_gantt_on_ax(
            env=env,
            ax=ax,
            title=f"{name} | makespan={env.current_time:.2f}",
            show_text=show_text,
        )
        x_max = max(x_max, x_local)

    x_pad = x_max * 0.08
    for ax in axes:
        ax.set_xlim(-x_pad, x_max + x_pad)

    legend_handles = [
        patches.Patch(color="tab:blue", label="Travel"),
        patches.Patch(color="tab:orange", label="Wait / Abandon"),
        patches.Patch(color="tab:green", label="Execute"),
        patches.Patch(color="tab:gray", label="Idle at Depot"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=9)
    fig.suptitle(f"Gantt Comparison on env_{env_id}", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    save_path = os.path.join(output_dir, f"gantt_compare_env_{env_id}.png")
    fig.savefig(save_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Plot RL / OR-Tools / CTAS-D gantt charts on one env case.")
    parser.add_argument("--test-set", default="testSet_20A_50T_CONDET")
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--rl-folder", default="REINFORCE_0408_multimode_test")
    parser.add_argument("--rl-method", choices=["LF", "IA"], default="IA")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.test_set
    os.makedirs(output_dir, exist_ok=True)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    or_tools_module = None
    ctasd_module = None
    try:
        or_tools_module = load_module_from_path(
            "ortools_baseline", os.path.join(repo_root, "baselines", "OR-Tools.py")
        )
    except ModuleNotFoundError as e:
        print(f"[WARN] Skip OR-Tools: missing dependency ({e})")

    try:
        ctasd_module = load_module_from_path(
            "ctasd_baseline", os.path.join(repo_root, "baselines", "CTAS-D.py")
        )
    except ModuleNotFoundError as e:
        print(f"[WARN] Skip CTAS-D: missing dependency ({e})")

    rl_env = None
    if not args.skip_rl:
        import torch

        use_cuda = USE_GPU_GLOBAL and torch.cuda.is_available()
        device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
        worker = build_worker(args.rl_folder, device)
        rl_env = run_rl_env(worker, load_case_env(args.test_set, args.env_id), args.rl_method)

    or_env = None
    ctas_env = None
    if or_tools_module is not None:
        or_env = run_ortools_env(or_tools_module, load_case_env(args.test_set, args.env_id))
    if ctasd_module is not None:
        ctas_env = run_ctasd_env(ctasd_module, load_case_env(args.test_set, args.env_id), args.test_set, args.env_id)

    method_envs = [
        (f"RL-{args.rl_method}", rl_env),
        ("OR-Tools", or_env),
        ("CTAS-D", ctas_env),
    ]

    for name, env in method_envs:
        if env is None:
            print(f"[WARN] Skip {name}: no valid result for env_{args.env_id}")
            continue
        path = save_single_gantt(name, env, output_dir, args.env_id, args.show_text)
        print(f"[OK] Saved: {path}")

    compare_path = save_comparison_gantt(method_envs, output_dir, args.env_id, args.show_text)
    if compare_path is not None:
        print(f"[OK] Saved: {compare_path}")
    else:
        print("[WARN] No valid method result; nothing was plotted.")


if __name__ == "__main__":
    main()
