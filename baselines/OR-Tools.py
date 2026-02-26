"""Simple Vehicles Routing Problem (VRP).

   This is a sample using the routing library python wrapper to solve a VRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""
import copy
import os.path

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
    def __init__(self):
        self.magnify = 1000  # ease numerical calculation
        self.coords = None

    def create_data_model(self, coords, num_vehicles=1, depot=0):
        """Stores the data for the problem."""
        data = dict()
        # Locations in block units
        data['locations'] = np.array(coords) * self.magnify
        data['num_vehicles'] = num_vehicles
        data['depot'] = depot
        return data

    @staticmethod
    def print_solution(data, manager, routing, solution):
        """Prints solution on console."""
        routes = {}
        max_route_distance = 0
        # print('Objective: {}'.format(solution.ObjectiveValue()))
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
            print(plan_output)
            max_route_distance = max(route_distance, max_route_distance)
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
    import pickle
    import pandas as pd
    import glob
    from natsort import natsorted
    solver = TSPSolver()
    time = []
    folder = 'testSet_v0_1_20A_50T_CONDET' # 这里是制定Test Set的文件夹，文件夹内应该有env_*.pkl的环境文件
    method = 'OR-Tools'
    files = natsorted(glob.glob(f'./{folder}/env_*.pkl'), key=lambda y: y.lower())
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
    for i in files[0:1]: # 这里可以小batch test一下
        env = pickle.load(open(i, 'rb'))
        agents = env.agent_dic
        tasks = env.task_dic
        depot = env.depot
        env.reactive_planning = False
        test_env = (tasks, agents, depot)
        env.reset(test_env)
        env.clear_decisions()
        if method == 'OR-Tools':
            solver.VRP(env)
        env.force_wait = True
        env.max_waiting_time = EVAL_MAX_WAITING_TIME
        env.execute_by_route(
            i.replace('.pkl', '/'),
            method,
            False,
            max_time=MAX_TIME,
            max_waiting_time=EVAL_MAX_WAITING_TIME,
        )
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
        print(i)
    df = pd.DataFrame(perf_metrics)
    df.to_csv(f'./{folder}/{method}.csv')
