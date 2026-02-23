#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.task_env import TaskEnv

try:
    from scipy.stats import ttest_rel  # Optional; regression still works without it.
except Exception:
    ttest_rel = None


def pick_metric(df, candidates):
    # df is a dict of numpy arrays.
    for col in candidates:
        if col in df:
            return df[col]
    return None


def compare_csv(current_csv, reference_csv, output_md):
    cur = load_csv_numeric(current_csv)
    ref = load_csv_numeric(reference_csv)
    metrics = [
        ("success_rate", ["success_rate"]),
        ("makespan", ["makespan"]),
        ("time_cost", ["time_cost"]),
        ("waiting_time", ["waiting_time"]),
        ("travel_dist", ["travel_dist"]),
        ("utilization_exec", ["utilization_exec", "efficiency"]),
        ("utilization_wait", ["utilization_wait"]),
        ("utilization_travel", ["utilization_travel"]),
    ]

    lines = []
    lines.append("# P2 Regression CSV Comparison")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- current_csv: `{current_csv}`")
    lines.append(f"- reference_csv: `{reference_csv}`")
    lines.append("")
    lines.append("| metric | current_mean | reference_mean | delta(current-ref) | paired_t | p_value |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for metric_name, candidates in metrics:
        cur_arr = pick_metric(cur, candidates)
        ref_arr = pick_metric(ref, candidates)
        if cur_arr is None or ref_arr is None:
            lines.append(f"| {metric_name} | NA | NA | NA | NA | NA |")
            continue
        n = min(len(cur_arr), len(ref_arr))
        if n == 0:
            lines.append(f"| {metric_name} | NA | NA | NA | NA | NA |")
            continue
        cur_arr = cur_arr[:n]
        ref_arr = ref_arr[:n]
        cur_mean = float(np.nanmean(cur_arr))
        ref_mean = float(np.nanmean(ref_arr))
        delta = cur_mean - ref_mean
        if ttest_rel is None:
            t_val = np.nan
            p_val = np.nan
        else:
            out = ttest_rel(cur_arr, ref_arr, nan_policy="omit")
            t_val = float(out.statistic) if out is not None else np.nan
            p_val = float(out.pvalue) if out is not None else np.nan
        lines.append(
            f"| {metric_name} | {cur_mean:.6f} | {ref_mean:.6f} | {delta:.6f} | "
            f"{t_val:.6f} | {p_val:.6f} |"
        )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_random_episode(seed, agents_num, tasks_num, max_time, max_steps):
    env = TaskEnv(agents_range=(agents_num, agents_num), tasks_range=(tasks_num, tasks_num), seed=seed)
    previous_remaining = {tid: float(task["remaining_workload"]) for tid, task in env.task_dic.items()}

    step_count = 0
    invariant_ok = True
    invariant_msg = "ok"
    while (not env.finished) and env.current_time < max_time and step_count < max_steps:
        decision_agents, current_time = env.next_decision()
        groups = env.get_unique_group(decision_agents) if len(decision_agents) > 0 else []
        env.current_time = float(current_time)
        env.task_update()
        env.agent_update()

        for task in env.task_dic.values():
            rid = task["ID"]
            rem = float(task["remaining_workload"])
            if rem < -1e-6:
                invariant_ok = False
                invariant_msg = f"task_{rid}_negative_remaining"
                break
            if rem > previous_remaining[rid] + 1e-6:
                invariant_ok = False
                invariant_msg = f"task_{rid}_remaining_increase"
                break
            previous_remaining[rid] = rem
        if not invariant_ok:
            break

        for group in groups:
            while len(group) > 0:
                leader = int(np.random.choice(group))
                if env.agent_dic[leader]["returned"]:
                    group.remove(leader)
                    continue
                mask = env.get_unfinished_task_mask()
                if np.sum(mask) == env.tasks_num:
                    action = 0
                else:
                    valid_actions = [0] + [tid + 1 for tid, m in enumerate(mask) if not bool(m)]
                    action = int(np.random.choice(valid_actions))
                group, _ = env.step(group, leader, action, step_count)
                env.task_update()
                env.agent_update()
                step_count += 1

        env.finished = env.check_finished()

    reward, finished_tasks = env.get_episode_reward()
    util_exec, util_wait, util_travel = env.get_utilization_metrics()

    if not np.isfinite(float(reward)):
        invariant_ok = False
        invariant_msg = "reward_nan_or_inf"
    if not (0.0 <= util_exec <= 1.0 + 1e-6 and 0.0 <= util_wait <= 1.0 + 1e-6 and 0.0 <= util_travel <= 1.0 + 1e-6):
        invariant_ok = False
        invariant_msg = "utilization_out_of_range"

    return {
        "seed": seed,
        "invariant_ok": invariant_ok,
        "invariant_msg": invariant_msg,
        "finished": bool(env.finished),
        "steps": int(step_count),
        "makespan": float(env.current_time),
        "success_rate": float(np.sum(finished_tasks) / len(finished_tasks)),
        "time_cost": float(np.nanmean(env.get_matrix(env.task_dic, "time_start"))),
        "waiting_time": float(np.mean(env.get_matrix(env.agent_dic, "sum_waiting_time"))),
        "travel_dist": float(np.sum(env.get_matrix(env.agent_dic, "travel_dist"))),
        "utilization_exec": float(util_exec),
        "utilization_wait": float(util_wait),
        "utilization_travel": float(util_travel),
        "efficiency": float(util_exec),
        "reward": float(reward),
    }


def run_env_regression(seeds, agents_num, tasks_num, max_time, max_steps, out_csv, out_json):
    rows = []
    for seed in seeds:
        rows.append(run_random_episode(seed, agents_num, tasks_num, max_time, max_steps))
    write_rows_csv(rows, out_csv)

    invariant_ok = np.array([float(row["invariant_ok"]) for row in rows], dtype=float) if rows else np.array([])
    finished = np.array([float(row["finished"]) for row in rows], dtype=float) if rows else np.array([])
    makespan = np.array([float(row["makespan"]) for row in rows], dtype=float) if rows else np.array([])
    waiting = np.array([float(row["waiting_time"]) for row in rows], dtype=float) if rows else np.array([])
    travel = np.array([float(row["travel_dist"]) for row in rows], dtype=float) if rows else np.array([])
    util_exec = np.array([float(row["utilization_exec"]) for row in rows], dtype=float) if rows else np.array([])

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_cases": int(len(rows)),
        "invariant_pass_rate": float(np.mean(invariant_ok)) if invariant_ok.size else np.nan,
        "finished_rate": float(np.mean(finished)) if finished.size else np.nan,
        "makespan_mean": float(np.nanmean(makespan)) if makespan.size else np.nan,
        "waiting_time_mean": float(np.nanmean(waiting)) if waiting.size else np.nan,
        "travel_dist_mean": float(np.nanmean(travel)) if travel.size else np.nan,
        "utilization_exec_mean": float(np.nanmean(util_exec)) if util_exec.size else np.nan,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)


def write_rows_csv(rows, out_csv):
    if not rows:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_csv_numeric(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    columns = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            try:
                columns[k].append(float(v))
            except Exception:
                columns[k].append(np.nan)
    for k in list(columns.keys()):
        columns[k] = np.array(columns[k], dtype=float)
    return columns


def parse_seeds(seed_text):
    if "," in seed_text:
        return [int(v.strip()) for v in seed_text.split(",") if v.strip()]
    if "-" in seed_text:
        s, e = seed_text.split("-")
        s_i = int(s.strip())
        e_i = int(e.strip())
        if e_i < s_i:
            s_i, e_i = e_i, s_i
        return list(range(s_i, e_i + 1))
    return [int(seed_text)]


def main():
    parser = argparse.ArgumentParser(description="P2 regression suite for dynamic-workload MRTA.")
    parser.add_argument("--output-dir", type=str, default="regression", help="Directory to store regression outputs.")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Seed list, e.g. '0,1,2' or range '0-9'.")
    parser.add_argument("--agents", type=int, default=6, help="Fixed number of agents for env regression.")
    parser.add_argument("--tasks", type=int, default=12, help="Fixed number of tasks for env regression.")
    parser.add_argument("--max-time", type=float, default=400.0, help="Episode max time for env regression.")
    parser.add_argument("--max-steps", type=int, default=8000, help="Step cap for env regression.")
    parser.add_argument("--current-csv", type=str, default="", help="Current run CSV for paired comparison.")
    parser.add_argument("--reference-csv", type=str, default="", help="Reference CSV for paired comparison.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_csv = os.path.join(args.output_dir, f"p2_env_regression_{stamp}.csv")
    env_json = os.path.join(args.output_dir, f"p2_env_regression_summary_{stamp}.json")
    seeds = parse_seeds(args.seeds)
    run_env_regression(
        seeds=seeds,
        agents_num=args.agents,
        tasks_num=args.tasks,
        max_time=args.max_time,
        max_steps=args.max_steps,
        out_csv=env_csv,
        out_json=env_json,
    )
    print(f"[P2] env regression csv: {env_csv}")
    print(f"[P2] env regression summary: {env_json}")

    if args.current_csv and args.reference_csv:
        compare_md = os.path.join(args.output_dir, f"p2_csv_compare_{stamp}.md")
        compare_csv(args.current_csv, args.reference_csv, compare_md)
        print(f"[P2] csv comparison report: {compare_md}")


if __name__ == "__main__":
    main()
