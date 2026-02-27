"""Simple Vehicles Routing Problem (VRP).

   This is a sample using the routing library python wrapper to solve a VRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""
import copy
import os.path
import time

from env.task_env import TaskEnv
from parameters import MAX_TIME, EVAL_MAX_WAITING_TIME
import math
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import yaml


def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = int(math.hypot((from_node[0] - to_node[0]),(from_node[1] - to_node[1])) * 5 + to_node[2])
    return distances


def routes2id(routes, task_dict):
    tasks = []
    for i in routes:
        if i == 0:
            tasks += [0]
        else:
            tasks += [task_dict[i - 1]['ID'] + 1]
    return tasks


class TSPSolver:
    def __init__(self, show_routes=True):
        self.magnify = 1000  # ease numerical calculation
        self.coords = None
        self.show_routes = bool(show_routes)

    def create_data_model(self, coords, num_vehicles=1, depot=0):
        """Stores the data for the problem."""
        data = dict()
        # Locations in block units
        data['locations'] = np.array(coords) * self.magnify
        data['num_vehicles'] = num_vehicles
        data['depot'] = depot
        return data

    def print_solution(self, data, manager, routing, solution):
        """Prints solution on console."""
        routes = {}
        max_route_distance = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            while not routing.IsEnd(index):
                plan_output += ' {} -> '.format(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
                routes[vehicle_id] = routes.get(vehicle_id, []) + [manager.IndexToNode(index)]
            plan_output += '{}\n'.format(manager.IndexToNode(index))
            plan_output += 'Distance of the route: {}m\n'.format(route_distance)
            if self.show_routes:
                print(plan_output)
            max_route_distance = max(route_distance, max_route_distance)
        if self.show_routes:
            print('Maximum of the route distances: {}m'.format(max_route_distance))
        return routes, max_route_distance

    def run_solver(self, coords, num_vehicles=1, depot=0):
        """Entry point of the program."""
        # Instantiate the data problem.
        data = self.create_data_model(coords, num_vehicles)
        distance_matrix = compute_euclidean_distance_matrix(data['locations'])
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Add Distance constraint.
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            200000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        # search_parameters.local_search_metaheuristic = (
        #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = 10
        # search_parameters.log_search = True

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Print solution on console.
        if solution:
            route, route_distance = self.print_solution(data, manager, routing, solution)
            return route, route_distance
        else:
            return None

    def VRP(self, env):
        """
        split agents into groups then execute mTSP for each group
        """
        task_groups, agent_groups = env.get_grouped_tasks()
        routes = {}
        sum_distance = []
        agent_id = 0
        for cat, tasks in task_groups.items():
            coords = env.get_matrix(tasks, 'location')
            time_ = env.get_matrix(tasks, 'time')
            coords = np.hstack([coords, np.array(time_).reshape(len(time_), -1)])
            coords = np.vstack([env.depot['location'].tolist() + [0], coords])
            result = self.run_solver(coords, agent_groups[cat])
            if result is None:
                print(f'[OR-Tools] Warning: no solution found for cat={cat}, skipping.')
                continue
            routes, route_distance = result
            for i in range(agent_groups[cat]):
                routes[i] = routes2id(routes[i], tasks)
                if routes[i] == [0]:
                    continue
                else:
                    for j in range(cat):
                        env.pre_set_route(copy.copy(routes[i])[:-1], agent_id)
                        agent_id += 1
                        if agent_id >= env.agents_num:
                            agent_id -= env.agents_num
        return routes


if __name__ == '__main__':
    import argparse
    import pickle
    import pandas as pd
    import glob
    from natsort import natsorted
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    parser = argparse.ArgumentParser(description='Run OR-Tools baseline with progress and timing.')
    parser.add_argument('--folder', type=str, default='testSet_v0_1_20A_50T_CONDET',
                        help='Test set directory containing env_*.pkl')
    parser.add_argument('--max-envs', type=int, default=0,
                        help='Limit number of env files for smoke test. 0 means all.')
    parser.add_argument('--show-routes', action='store_true',
                        help='Print detailed route output from OR-Tools solver.')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Only run OR-Tools route generation, skip env.execute_by_route().')
    parser.add_argument('--sim-timeout-sec', type=float, default=0.0,
                        help='Wall-time timeout for env.execute_by_route(). 0 means no timeout.')
    parser.add_argument('--sim-progress-sec', type=float, default=10.0,
                        help='Progress log interval (seconds) during env.execute_by_route(). 0 disables logs.')
    args = parser.parse_args()

    solver = TSPSolver(show_routes=args.show_routes)
    folder = args.folder
    method = 'OR-Tools'
    files = natsorted(glob.glob(f'./{folder}/env_*.pkl'), key=lambda y: y.lower())
    if args.max_envs > 0:
        files = files[:args.max_envs]
    if not files:
        raise FileNotFoundError(f'No env_*.pkl found under ./{folder}')
    perf_metrics = {
        'success_rate': [],
        'makespan': [],
        'time_cost': [],
        'waiting_time': [],
        'travel_dist': [],
        'utilization_exec': [],
        'utilization_wait': [],
        'utilization_travel': [],
        'efficiency': [],
    }
    total_start = time.perf_counter()
    iterator = tqdm(files, desc='OR-Tools baseline', unit='env') if tqdm is not None else files
    for idx, i in enumerate(iterator, start=1):
        env_start = time.perf_counter()
        env = pickle.load(open(i, 'rb'))
        agents = env.agent_dic
        tasks = env.task_dic
        depot = env.depot
        env.reactive_planning = False
        test_env = (tasks, agents, depot)
        env.reset(test_env)
        env.clear_decisions()
        vrp_start = time.perf_counter()
        if method == 'OR-Tools':
            solver.VRP(env)
        vrp_elapsed = time.perf_counter() - vrp_start
        print(f'[phase] vrp_done env={idx}/{len(files)} time={vrp_elapsed:.2f}s', flush=True)

        sim_elapsed = 0.0
        if not args.skip_sim:
            sim_start = time.perf_counter()
            env.force_wait = True
            env.max_waiting_time = EVAL_MAX_WAITING_TIME
            env.execute_by_route(
                i.replace('.pkl', '/'),
                method,
                False,
                max_time=MAX_TIME,
                max_waiting_time=EVAL_MAX_WAITING_TIME,
                max_wall_time_sec=(args.sim_timeout_sec if args.sim_timeout_sec > 0 else None),
                progress_log_interval_sec=args.sim_progress_sec,
            )
            sim_elapsed = time.perf_counter() - sim_start
            print(f'[phase] sim_done env={idx}/{len(files)} time={sim_elapsed:.2f}s', flush=True)
        else:
            print(f'[phase] sim_skipped env={idx}/{len(files)}', flush=True)

        reward, finished_tasks = env.get_episode_reward(MAX_TIME)
        if np.sum(finished_tasks) / len(finished_tasks) < 1:
            perf_metrics['success_rate'].append(np.sum(finished_tasks) / len(finished_tasks))
            perf_metrics['makespan'].append(np.nan)
            perf_metrics['time_cost'].append(np.nan)
            perf_metrics['waiting_time'].append(np.nan)
            perf_metrics['travel_dist'].append(np.nan)
            perf_metrics['utilization_exec'].append(np.nan)
            perf_metrics['utilization_wait'].append(np.nan)
            perf_metrics['utilization_travel'].append(np.nan)
            perf_metrics['efficiency'].append(np.nan)
        else:
            util_exec, util_wait, util_travel = env.get_utilization_metrics()
            perf_metrics['success_rate'].append(np.sum(finished_tasks) / len(finished_tasks))
            perf_metrics['makespan'].append(env.current_time)
            perf_metrics['time_cost'].append(np.nanmean(env.get_matrix(env.task_dic, 'time_start')))
            perf_metrics['waiting_time'].append(np.mean(env.get_matrix(env.agent_dic, 'sum_waiting_time')))
            perf_metrics['travel_dist'].append(np.sum(env.get_matrix(env.agent_dic, 'travel_dist')))
            perf_metrics['utilization_exec'].append(util_exec)
            perf_metrics['utilization_wait'].append(util_wait)
            perf_metrics['utilization_travel'].append(util_travel)
            # Backward-compatible alias.
            perf_metrics['efficiency'].append(util_exec)

        env_elapsed = time.perf_counter() - env_start
        if tqdm is not None:
            iterator.set_postfix({
                'idx': f'{idx}/{len(files)}',
                'env_s': f'{env_elapsed:.1f}',
                'vrp_s': f'{vrp_elapsed:.1f}',
                'sim_s': f'{sim_elapsed:.1f}',
                'mk': f'{perf_metrics["makespan"][-1]:.2f}' if not np.isnan(perf_metrics["makespan"][-1]) else 'nan',
                'sr': f'{perf_metrics["success_rate"][-1]:.2f}',
            })
        else:
            print(f'[{idx}/{len(files)}] {i} | env_time={env_elapsed:.2f}s | '
                  f'makespan={perf_metrics["makespan"][-1]} | success={perf_metrics["success_rate"][-1]:.3f}')

    df = pd.DataFrame(perf_metrics)
    out_csv = f'./{folder}/{method}.csv'
    df.to_csv(out_csv)
    total_elapsed = time.perf_counter() - total_start
    mean_makespan = float(np.nanmean(perf_metrics['makespan'])) if perf_metrics['makespan'] else float('nan')
    mean_success = float(np.nanmean(perf_metrics['success_rate'])) if perf_metrics['success_rate'] else float('nan')
    print(f'[done] envs={len(files)} | total_time={total_elapsed:.2f}s | '
          f'mean_makespan={mean_makespan:.4f} | mean_success={mean_success:.4f}')
    print(f'[saved] {out_csv}')
