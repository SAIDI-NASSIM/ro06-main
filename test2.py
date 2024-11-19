import os
import time
import math
import random
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import glob
import copy

class Client:
    def __init__(self, id, x, y, profit):
        self.id = id
        self.x = x
        self.y = y
        self.profit = profit

    def __repr__(self):
        return f"Client({self.id}, x={self.x}, y={self.y}, profit={self.profit})"


class TOPHeuristic:
    def __init__(self, start_point, end_point, clients, m, L, max_runtime=300, debug=False):
        self.start_point = start_point
        self.end_point = end_point
        self.clients = clients
        self.m = m
        self.L = L
        self.debug = debug
        self.max_runtime = max_runtime
        self.distances = self._precompute_distances()
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.stats_data = {
            'iteration': [], 
            'best_fitness': [], 
            'avg_fitness': [],  # Added avg_fitness
            'method': []
        }

    def _precompute_distances(self):
        distances = {}
        all_points = [self.start_point] + self.clients + [self.end_point]
        for i, p1 in enumerate(all_points):
            distances[p1.id] = {}
            for p2 in all_points:
                if p1 != p2:
                    distances[p1.id][p2.id] = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
        return distances

    def _calculate_solution_quality(self, solution):
        if not solution: return 0
        total_profit = 0; total_distance = 0; used_clients = set()
        for route in solution:
            route_profit = 0; route_distance = 0
            for i in range(len(route) - 1):
                route_distance += self.distances[route[i].id][route[i+1].id]
                if i > 0 and route[i].id not in used_clients:
                    route_profit += route[i].profit
                    used_clients.add(route[i].id)
            if route_distance <= self.L:
                total_profit += route_profit
                total_distance += route_distance
        if total_distance > self.L * self.m: return 0
        coverage = len(used_clients) / len(self.clients)
        efficiency = 1 - (total_distance / (self.L * self.m))
        return total_profit * (1 + 0.15 * coverage + 0.1 * efficiency)

    def _greedy_route(self, available_clients, method='profit_distance'):
        route = [self.start_point]; current = self.start_point; current_time = 0
        remaining = available_clients.copy()
        
        while remaining:
            best_score = float('-inf'); best_next = None
            candidates = sorted(remaining, key=lambda c: self.distances[current.id][c.id])[:10]
            
            for client in candidates:
                time_to_client = self.distances[current.id][client.id]
                time_to_end = self.distances[client.id][self.end_point.id]
                total_time = current_time + time_to_client + time_to_end
                
                if total_time <= self.L:
                    if method == 'profit_distance':
                        score = client.profit / time_to_client
                    elif method == 'profit':
                        score = client.profit
                    elif method == 'nearest':
                        score = 1 / time_to_client
                    elif method == 'savings':
                        direct = self.distances[current.id][self.end_point.id]
                        savings = direct - (time_to_client + time_to_end)
                        score = client.profit * max(0, savings)
                    
                    if score > best_score:
                        best_score = score
                        best_next = client
            
            if best_next is None: break
            
            route.append(best_next)
            current_time += self.distances[current.id][best_next.id]
            current = best_next
            remaining.remove(best_next)
            
        route.append(self.end_point)
        return route if len(route) > 2 else None

    def _construct_solution(self, method):
        solution = []; used_clients = set()
        available_clients = set(self.clients)
        
        for _ in range(self.m):
            remaining = available_clients - used_clients
            if not remaining: break
            
            route = self._greedy_route(remaining, method)
            if route:
                solution.append(route)
                used_clients.update(set(route[1:-1]))
            else:
                break
                
        return solution

    def _simple_improvement(self, solution):
        if not solution: return solution
        improved = True
        while improved:
            improved = False
            # Try to insert unvisited clients
            used_clients = set(c for route in solution for c in route[1:-1])
            unvisited = set(self.clients) - used_clients
            
            for client in sorted(unvisited, key=lambda c: c.profit, reverse=True):
                for route in solution:
                    for i in range(1, len(route)):
                        new_route = route[:i] + [client] + route[i:]
                        route_length = sum(self.distances[new_route[j].id][new_route[j+1].id] 
                                         for j in range(len(new_route)-1))
                        if route_length <= self.L:
                            route[:] = new_route
                            improved = True
                            break
                    if improved: break
                if improved: break
            
            if not improved:
                # Try 2-opt on each route
                for route in solution:
                    if len(route) <= 4: continue
                    best_length = sum(self.distances[route[i].id][route[i+1].id] 
                                    for i in range(len(route)-1))
                    for i in range(1, len(route)-2):
                        for j in range(i+1, len(route)-1):
                            new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                            new_length = sum(self.distances[new_route[k].id][new_route[k+1].id] 
                                           for k in range(len(new_route)-1))
                            if new_length < best_length:
                                route[:] = new_route
                                improved = True
                                break
                        if improved: break
                    if improved: break
        
        return solution

    def solve(self):
        start_time = time.time()
        methods = ['profit_distance', 'profit', 'nearest', 'savings']
        iteration = 0
        
        all_qualities = []  # Track all solution qualities for averaging
        
        for method in methods:
            if time.time() - start_time > self.max_runtime:
                break
            
            solution = self._construct_solution(method)
            solution = self._simple_improvement(solution)
            quality = self._calculate_solution_quality(solution)
            
            all_qualities.append(quality)  # Add current quality to list
            avg_quality = sum(all_qualities) / len(all_qualities)  # Calculate average
            
            self.stats_data['iteration'].append(iteration)
            self.stats_data['best_fitness'].append(quality)
            self.stats_data['avg_fitness'].append(avg_quality)  # Add average fitness
            self.stats_data['method'].append(method)
            
            if quality > self.best_fitness:
                self.best_fitness = quality
                self.best_solution = copy.deepcopy(solution)
                
            if self.debug:
                print(f"\nMethod: {method}")
                print(f"Quality: {quality:.2f}")
                
            iteration += 1
            
        return self.best_solution

    def get_stats(self):
        return pd.DataFrame(self.stats_data)

class AntColonyTOP:
    def __init__(self, start_point, end_point, clients, m, L, max_runtime=300, debug=False):
        self.start_point = start_point; self.end_point = end_point; self.clients = clients
        self.m = m; self.L = L; self.debug = debug; self.max_runtime = max_runtime
        n = len(clients)
        # Balanced parameters to match GA's computation budget
        self.n_colonies = min(2, m)
        self.n_ants_per_colony = min(75, n * 2)  # Matches GA population size
        self.max_iterations = min(125, n * 4)     # Matches GA generations
        self.colony_params = [
            {'alpha': 1.0, 'beta': 2.0, 'rho': 0.1},
            {'alpha': 2.0, 'beta': 1.0, 'rho': 0.15}
        ]
        self.q0_initial = 0.7; self.q0_final = 0.9
        self.local_search_freq = max(10, n // 10)  # Reduced frequency
        self.max_stagnation = max(40, n // 2)      # Matches GA stagnation
        self.tau_max = 2.0; self.tau_min = self.tau_max * 0.01
        self.distances = self._precompute_distances()
        self.pheromone_matrices = self._initialize_pheromone_matrices()
        self.eta = self._initialize_heuristic()
        self.best_solution = None; self.best_fitness = float('-inf')
        self.iteration_best_solutions = []
        self.stats_data = {'iteration': [], 'best_fitness': [], 'avg_fitness': [], 
                          'diversity': [], 'pheromone_avg': [], 'pheromone_max': [], 
                          'pheromone_min': []}

    def _precompute_distances(self):
        distances = {}
        all_points = [self.start_point] + self.clients + [self.end_point]
        for i, p1 in enumerate(all_points):
            distances[p1.id] = {}
            for p2 in all_points:
                if p1 != p2:
                    distances[p1.id][p2.id] = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
        return distances

    def _initialize_pheromone_matrices(self):
        matrices = []
        all_points = [self.start_point] + self.clients + [self.end_point]
        for _ in range(self.n_colonies):
            tau = {i.id: {j.id: self.tau_max for j in all_points if i != j} for i in all_points}
            matrices.append(tau)
        return matrices

    def _initialize_heuristic(self):
        eta = {}; all_points = [self.start_point] + self.clients + [self.end_point]
        max_profit = max((c.profit for c in self.clients), default=1)
        center_x = sum(c.x for c in self.clients) / len(self.clients)
        center_y = sum(c.y for c in self.clients) / len(self.clients)
        max_dist = max(((c.x - center_x)**2 + (c.y - center_y)**2)**0.5 for c in self.clients)
        for i in all_points:
            eta[i.id] = {}
            for j in all_points:
                if i != j:
                    dist = self.distances[i.id][j.id]
                    if j in self.clients:
                        profit_factor = j.profit / max_profit
                        distance_factor = 1.0 / (dist if dist > 0 else 0.1)
                        centrality = 1.0 - (((j.x - center_x)**2 + (j.y - center_y)**2)**0.5 / max_dist)
                        eta[i.id][j.id] = (1.0 + profit_factor) * distance_factor * (1.0 + 0.2 * centrality)
                    else:
                        eta[i.id][j.id] = 1.0 / (dist if dist > 0 else 0.1)
        return eta

    def _select_next_client(self, ant_route, current, available_clients, current_time, colony_idx, iteration):
        if not available_clients: return self.end_point
        q0 = self.q0_initial + (self.q0_final - self.q0_initial) * (iteration / self.max_iterations)
        params = self.colony_params[colony_idx]
        feasible = []
        for client in available_clients:
            time_to_client = self.distances[current.id][client.id]
            time_to_end = self.distances[client.id][self.end_point.id]
            if current_time + time_to_client + time_to_end <= self.L:
                route_position = len(ant_route) / self.L
                score = (self.pheromone_matrices[colony_idx][current.id][client.id]**params['alpha'] * 
                        self.eta[current.id][client.id]**params['beta'] * 
                        (1 + 0.2 * (1 - route_position)))
                feasible.append((client, score))
        if not feasible: return self.end_point
        if random.random() < q0:
            return max(feasible, key=lambda x: x[1])[0]
        total = sum(score for _, score in feasible)
        if total == 0: return random.choice([client for client, _ in feasible])
        r = random.random() * total; cum_prob = 0
        for client, score in feasible:
            cum_prob += score
            if cum_prob >= r: return client
        return feasible[-1][0]

    def _construct_route(self, colony_idx, iteration, used_clients=None):
        route = [self.start_point]; current = self.start_point; current_time = 0
        if used_clients is None: used_clients = set()
        available = set(self.clients) - used_clients
        while available:
            next_client = self._select_next_client(route, current, available, current_time, colony_idx, iteration)
            if next_client == self.end_point: break
            route.append(next_client)
            current_time += self.distances[current.id][next_client.id]
            current = next_client; available.remove(next_client)
            self.pheromone_matrices[colony_idx][route[-2].id][route[-1].id] *= (1 - self.colony_params[colony_idx]['rho'])
        route.append(self.end_point)
        return route

    def _construct_solution(self, colony_idx, iteration):
        solution = []; used_clients = set()
        for _ in range(self.m):
            if not (set(self.clients) - used_clients): break
            route = self._construct_route(colony_idx, iteration, used_clients)
            if len(route) > 2:
                solution.append(route)
                used_clients.update(set(route[1:-1]))
        return solution

    def _local_search(self, solution):
        if not solution: return solution
        max_local_time = self.max_runtime * 0.1  # 10% of total time budget
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time <= max_local_time:
            improved = False
            # Route optimization
            for route_idx, route in enumerate(solution):
                if len(route) <= 4: continue
                best_length = sum(self.distances[route[i].id][route[i+1].id] for i in range(len(route)-1))
                for i in range(1, len(route)-2):
                    if time.time() - start_time > max_local_time: break
                    for j in range(i+1, min(i+5, len(route)-1)):
                        new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                        new_length = sum(self.distances[new_route[k].id][new_route[k+1].id] for k in range(len(new_route)-1))
                        if new_length < best_length and new_length <= self.L:
                            solution[route_idx] = new_route
                            improved = True
                            break
                    if improved: break
        return solution

    def _calculate_solution_quality(self, solution):
        if not solution: return 0
        total_profit = 0; total_distance = 0; used_clients = set()
        for route in solution:
            route_profit = 0; route_distance = 0
            for i in range(len(route) - 1):
                route_distance += self.distances[route[i].id][route[i+1].id]
                if i > 0 and route[i].id not in used_clients:
                    route_profit += route[i].profit
                    used_clients.add(route[i].id)
            if route_distance <= self.L:
                total_profit += route_profit
                total_distance += route_distance
        if total_distance > self.L * self.m: return 0
        coverage = len(used_clients) / len(self.clients)
        efficiency = 1 - (total_distance / (self.L * self.m))
        return total_profit * (1 + 0.15 * coverage + 0.1 * efficiency)

    def _update_pheromone(self, colony_idx, iteration):
        params = self.colony_params[colony_idx]
        decay = 1 - params['rho']
        for i in self.pheromone_matrices[colony_idx]:
            for j in self.pheromone_matrices[colony_idx][i]:
                self.pheromone_matrices[colony_idx][i][j] *= decay
        if self.iteration_best_solutions:
            best_solution = self.iteration_best_solutions[-1]
            for route in best_solution:
                deposit = 1.0 / len(route)
                for i in range(len(route)-1):
                    self.pheromone_matrices[colony_idx][route[i].id][route[i+1].id] += params['rho'] * deposit
                    self.pheromone_matrices[colony_idx][route[i+1].id][route[i].id] = self.pheromone_matrices[colony_idx][route[i].id][route[i+1].id]
        for i in self.pheromone_matrices[colony_idx]:
            for j in self.pheromone_matrices[colony_idx][i]:
                self.pheromone_matrices[colony_idx][i][j] = min(self.tau_max, max(self.tau_min, self.pheromone_matrices[colony_idx][i][j]))

    def _calculate_diversity(self, solutions):
        if not solutions: return 0
        diversity = 0; comparisons = 0
        for i in range(len(solutions)):
            for j in range(i+1, len(solutions)):
                clients1 = set(c.id for route in solutions[i] for c in route[1:-1])
                clients2 = set(c.id for route in solutions[j] for c in route[1:-1])
                if clients1 or clients2:
                    hamming = len(clients1.symmetric_difference(clients2))
                    diversity += hamming / max(len(clients1), len(clients2))
                    comparisons += 1
        return diversity / max(1, comparisons)

    def solve(self):
        start_time = time.time()
        stagnation_counter = 0
        colony_solutions = [[] for _ in range(self.n_colonies)]
        colony_qualities = [[] for _ in range(self.n_colonies)]
        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.max_runtime: break
            iteration_solutions = []; iteration_qualities = []
            for colony_idx in range(self.n_colonies):
                colony_best_solution = None; colony_best_quality = float('-inf')
                for ant in range(self.n_ants_per_colony):
                    solution = self._construct_solution(colony_idx, iteration)
                    if iteration % self.local_search_freq == 0:
                        solution = self._local_search(solution)
                    quality = self._calculate_solution_quality(solution)
                    iteration_solutions.append(solution)
                    iteration_qualities.append(quality)
                    if quality > colony_best_quality:
                        colony_best_quality = quality
                        colony_best_solution = solution
                colony_solutions[colony_idx].append(colony_best_solution)
                colony_qualities[colony_idx].append(colony_best_quality)
                self._update_pheromone(colony_idx, iteration)
            best_idx = np.argmax(iteration_qualities)
            iteration_best = iteration_solutions[best_idx]
            iteration_best_quality = iteration_qualities[best_idx]
            self.iteration_best_solutions.append(iteration_best)
            if iteration_best_quality > self.best_fitness:
                self.best_fitness = iteration_best_quality
                self.best_solution = copy.deepcopy(iteration_best)
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            self._update_stats(iteration, iteration_qualities, iteration_solutions)
            if (stagnation_counter > self.max_stagnation and 
                iteration > self.max_iterations // 4): break
        return self.best_solution

    def _update_stats(self, iteration, qualities, solutions):
        avg_fitness = sum(qualities) / len(qualities)
        diversity = self._calculate_diversity(solutions)
        pheromone_values = [v for matrix in self.pheromone_matrices 
                           for d in matrix.values() for v in d.values()]
        self.stats_data['iteration'].append(iteration)
        self.stats_data['best_fitness'].append(self.best_fitness)
        self.stats_data['avg_fitness'].append(avg_fitness)
        self.stats_data['diversity'].append(diversity)
        self.stats_data['pheromone_avg'].append(statistics.mean(pheromone_values))
        self.stats_data['pheromone_max'].append(max(pheromone_values))
        self.stats_data['pheromone_min'].append(min(pheromone_values))
        if self.debug and iteration % 10 == 0:
            print(f"\nIteration {iteration}:")
            print(f"  Best Fitness = {self.best_fitness:.2f}")
            print(f"  Average Fitness = {avg_fitness:.2f}")
            print(f"  Diversity = {diversity:.3f}")

    def get_stats(self):
        return pd.DataFrame(self.stats_data)
    

class GeneticTOP:
    def __init__(self, start_point, end_point, clients, m, L, max_runtime=300, debug=False):
        self.start_point = start_point; self.end_point = end_point; self.clients = clients
        self.m = m; self.L = L; self.debug = debug; self.max_runtime = max_runtime
        n = len(clients)
        # Match ACO computation budget
        self.population_size = min(150, n * 2)     # Matches n_ants_per_colony * n_colonies
        self.generations = min(125, n * 4)         # Matches ACO iterations
        self.crossover_rate = 0.85
        self.mutation_rate = 0.15                  # Matches ACO exploration rate
        self.elite_size = max(2, self.population_size // 20)
        self.tournament_size = max(3, self.population_size // 25)
        self.local_search_freq = max(10, n // 10)  # Matches ACO frequency
        self.max_stagnation = max(40, n // 2)      # Matches ACO stagnation
        self.distances = self._precompute_distances()
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.stats_data = {'iteration': [], 'best_fitness': [], 'avg_fitness': [], 
                          'diversity': [], 'mutation_rate': [], 'crossover_rate': [], 
                          'generations_without_improvement': []}

    def _precompute_distances(self):
        distances = {}
        all_points = [self.start_point] + self.clients + [self.end_point]
        for i, p1 in enumerate(all_points):
            distances[p1.id] = {}
            for p2 in all_points:
                if p1 != p2:
                    distances[p1.id][p2.id] = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
        return distances

    def _create_initial_solution(self):
        solution = []; available_clients = set(self.clients)
        for _ in range(self.m):
            if not available_clients: break
            route = [self.start_point]; current = self.start_point; current_time = 0
            candidates = sorted(available_clients, 
                             key=lambda c: self.distances[current.id][c.id])[:10]
            while candidates:
                client = random.choice(candidates[:3])  # More randomness
                time_to_client = self.distances[current.id][client.id]
                time_to_end = self.distances[client.id][self.end_point.id]
                if current_time + time_to_client + time_to_end <= self.L:
                    route.append(client)
                    current_time += time_to_client
                    current = client
                    available_clients.remove(client)
                    candidates = sorted([c for c in available_clients if c not in route],
                                     key=lambda c: self.distances[current.id][c.id])[:10]
                else:
                    break
            if len(route) > 1:
                route.append(self.end_point)
                solution.append(route)
        return solution

    def _calculate_solution_quality(self, solution):
        if not solution: return 0
        total_profit = 0; total_distance = 0; used_clients = set()
        for route in solution:
            route_profit = 0; route_distance = 0
            for i in range(len(route) - 1):
                route_distance += self.distances[route[i].id][route[i+1].id]
                if i > 0 and route[i].id not in used_clients:
                    route_profit += route[i].profit
                    used_clients.add(route[i].id)
            if route_distance <= self.L:
                total_profit += route_profit
                total_distance += route_distance
        if total_distance > self.L * self.m: return 0
        coverage = len(used_clients) / len(self.clients)
        efficiency = 1 - (total_distance / (self.L * self.m))
        return total_profit * (1 + 0.15 * coverage + 0.1 * efficiency)

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        child1, child2 = [], []
        used_clients1, used_clients2 = set(), set()
        routes1 = [(r, sum(c.profit for c in r[1:-1]) / 
                   sum(self.distances[r[i].id][r[i+1].id] for i in range(len(r)-1)))
                  for r in parent1]
        routes2 = [(r, sum(c.profit for c in r[1:-1]) / 
                   sum(self.distances[r[i].id][r[i+1].id] for i in range(len(r)-1)))
                  for r in parent2]
        routes1.sort(key=lambda x: x[1], reverse=True)
        routes2.sort(key=lambda x: x[1], reverse=True)
        for route, _ in routes1:
            route_clients = set(c.id for c in route[1:-1])
            if not (route_clients & used_clients1) and len(child1) < self.m:
                child1.append(copy.deepcopy(route))
                used_clients1.update(route_clients)
        for route, _ in routes2:
            route_clients = set(c.id for c in route[1:-1])
            if not (route_clients & used_clients2) and len(child2) < self.m:
                child2.append(copy.deepcopy(route))
                used_clients2.update(route_clients)
        return child1, child2

    def _mutation(self, solution):
        if random.random() > self.mutation_rate or not solution: return solution
        mutated = copy.deepcopy(solution)
        for _ in range(2):
            if random.random() < 0.5 and len(mutated) >= 2:
                route1, route2 = random.sample(mutated, 2)
                if len(route1) > 3 and len(route2) > 3:
                    pos1 = random.randrange(1, len(route1) - 1)
                    pos2 = random.randrange(1, len(route2) - 1)
                    original_quality = (self._evaluate_route(route1) + 
                                     self._evaluate_route(route2))
                    route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
                    new_quality = (self._evaluate_route(route1) + 
                                 self._evaluate_route(route2))
                    if new_quality < original_quality * 0.95:
                        route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
            else:
                if mutated:
                    route = random.choice(mutated)
                    if len(route) > 4:
                        i = random.randrange(1, len(route) - 2)
                        j = random.randrange(i + 1, len(route) - 1)
                        if random.random() < 0.5:
                            route[i:j+1] = reversed(route[i:j+1])
                        else:
                            client = route.pop(i)
                            new_pos = random.randrange(1, len(route))
                            route.insert(new_pos, client)
        return mutated

    def _evaluate_route(self, route):
        if len(route) < 3: return 0
        route_time = sum(self.distances[route[i].id][route[i+1].id] 
                        for i in range(len(route)-1))
        if route_time > self.L: return 0
        return sum(c.profit for c in route[1:-1]) / route_time

    def _local_search(self, solution):
        if not solution: return solution
        max_local_time = self.max_runtime * 0.1  # 10% of total time budget
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time <= max_local_time:
            improved = False
            for route_idx, route in enumerate(solution):
                if len(route) <= 4: continue
                best_length = sum(self.distances[route[i].id][route[i+1].id] 
                                for i in range(len(route)-1))
                for i in range(1, len(route)-2):
                    if time.time() - start_time > max_local_time: break
                    for j in range(i+1, min(i+5, len(route)-1)):
                        new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                        new_length = sum(self.distances[new_route[k].id][new_route[k+1].id] 
                                       for k in range(len(new_route)-1))
                        if new_length < best_length and new_length <= self.L:
                            solution[route_idx] = new_route
                            improved = True
                            break
                    if improved: break
        return solution

    def _tournament_selection(self, population, fitnesses):
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitnesses[i] for i in tournament_indices]
        return copy.deepcopy(population[tournament_indices[tournament_fitness.index(max(tournament_fitness))]])

    def _calculate_diversity(self, population):
        if not population: return 0
        diversity = 0; comparisons = 0
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                clients1 = set(c.id for route in population[i] for c in route[1:-1])
                clients2 = set(c.id for route in population[j] for c in route[1:-1])
                if clients1 or clients2:
                    hamming = len(clients1.symmetric_difference(clients2))
                    diversity += hamming / max(len(clients1), len(clients2))
                    comparisons += 1
        return diversity / max(1, comparisons)

    def evolve(self):
        start_time = time.time()
        population = [self._create_initial_solution() 
                     for _ in range(self.population_size)]
        generations_without_improvement = 0
        for generation in range(self.generations):
            if time.time() - start_time > self.max_runtime: break
            fitnesses = [self._calculate_solution_quality(solution) 
                       for solution in population]
            max_fitness_idx = fitnesses.index(max(fitnesses))
            if fitnesses[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_fitness_idx]
                self.best_solution = copy.deepcopy(population[max_fitness_idx])
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            self._update_stats(generation, fitnesses, population, 
                             generations_without_improvement)
            if (generations_without_improvement > self.max_stagnation and 
                generation > self.generations // 4): break
            sorted_indices = sorted(range(len(fitnesses)), 
                                 key=lambda k: fitnesses[k], reverse=True)
            new_population = [copy.deepcopy(population[i]) 
                            for i in sorted_indices[:self.elite_size]]
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                if generation % self.local_search_freq == 0:
                    child1 = self._local_search(child1)
                    child2 = self._local_search(child2)
                child1, child2 = self._mutation(child1), self._mutation(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            population = new_population
        return self.best_solution

    def _update_stats(self, generation, fitnesses, population, generations_without_improvement):
        avg_fitness = sum(fitnesses) / len(fitnesses)
        diversity = self._calculate_diversity(population)
        self.stats_data['iteration'].append(generation)
        self.stats_data['best_fitness'].append(self.best_fitness)
        self.stats_data['avg_fitness'].append(avg_fitness)
        self.stats_data['diversity'].append(diversity)
        self.stats_data['mutation_rate'].append(self.mutation_rate)
        self.stats_data['crossover_rate'].append(self.crossover_rate)
        self.stats_data['generations_without_improvement'].append(generations_without_improvement)
        if self.debug and generation % 10 == 0:
            print(f"\nGeneration {generation}:")
            print(f"  Best Fitness = {self.best_fitness:.2f}")
            print(f"  Average Fitness = {avg_fitness:.2f}")
            print(f"  Diversity = {diversity:.3f}")

    def get_stats(self):
        return pd.DataFrame(self.stats_data)
    
class SimulatedAnnealingTOP:
    def __init__(self, start_point, end_point, clients, m, L, max_runtime=300, debug=False):
        self.start_point = start_point; self.end_point = end_point; self.clients = clients
        self.m = m; self.L = L; self.debug = debug; self.max_runtime = max_runtime
        n = len(clients)
        self.initial_temp = 100
        self.final_temp = 1
        self.max_iterations = min(125, n * 4)      # Match other methods
        self.iterations_per_temp = min(150, n * 2) # Match population/colony size
        self.alpha = pow(self.final_temp/self.initial_temp, 1.0/self.max_iterations)
        self.distances = self._precompute_distances()
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.stats_data = {'iteration': [], 'best_fitness': [], 'avg_fitness': [], 
                          'current_temp': [], 'acceptance_rate': []}

    def _precompute_distances(self):
        distances = {}
        all_points = [self.start_point] + self.clients + [self.end_point]
        for i, p1 in enumerate(all_points):
            distances[p1.id] = {}
            for p2 in all_points:
                if p1 != p2:
                    distances[p1.id][p2.id] = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
        return distances

    def _calculate_solution_quality(self, solution):
        if not solution: return 0
        total_profit = 0; total_distance = 0; used_clients = set()
        for route in solution:
            route_profit = 0; route_distance = 0
            for i in range(len(route) - 1):
                route_distance += self.distances[route[i].id][route[i+1].id]
                if i > 0 and route[i].id not in used_clients:
                    route_profit += route[i].profit
                    used_clients.add(route[i].id)
            if route_distance <= self.L:
                total_profit += route_profit
                total_distance += route_distance
        if total_distance > self.L * self.m: return 0
        coverage = len(used_clients) / len(self.clients)
        efficiency = 1 - (total_distance / (self.L * self.m))
        return total_profit * (1 + 0.15 * coverage + 0.1 * efficiency)

    def _create_initial_solution(self):
        solution = []; available_clients = set(self.clients)
        for _ in range(self.m):
            if not available_clients: break
            route = [self.start_point]; current = self.start_point; current_time = 0
            while available_clients:
                candidates = sorted(available_clients, 
                                 key=lambda c: self.distances[current.id][c.id])[:10]
                if not candidates: break
                client = random.choice(candidates[:3])
                time_to_client = self.distances[current.id][client.id]
                time_to_end = self.distances[client.id][self.end_point.id]
                if current_time + time_to_client + time_to_end <= self.L:
                    route.append(client)
                    current_time += time_to_client
                    current = client
                    available_clients.remove(client)
                else:
                    break
            if len(route) > 1:
                route.append(self.end_point)
                solution.append(route)
        return solution

    def _generate_neighbor(self, solution):
        if not solution: return solution
        neighbor = copy.deepcopy(solution)
        move_type = random.random()
        
        if move_type < 0.3 and len(neighbor) >= 2:  # Inter-route swap
            route1, route2 = random.sample(neighbor, 2)
            if len(route1) > 3 and len(route2) > 3:
                pos1 = random.randrange(1, len(route1) - 1)
                pos2 = random.randrange(1, len(route2) - 1)
                route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
        
        elif move_type < 0.6:  # Intra-route 2-opt
            if neighbor:
                route = random.choice(neighbor)
                if len(route) > 4:
                    i = random.randrange(1, len(route) - 2)
                    j = random.randrange(i + 1, len(route) - 1)
                    route[i:j+1] = reversed(route[i:j+1])
        
        else:  # Insert/remove client
            if random.random() < 0.5 and neighbor:  # Remove
                route = random.choice(neighbor)
                if len(route) > 3:
                    pos = random.randrange(1, len(route) - 1)
                    route.pop(pos)
            else:  # Insert
                used_clients = set(c for route in neighbor for c in route[1:-1])
                available = set(self.clients) - used_clients
                if available and neighbor:
                    client = random.choice(list(available))
                    route = random.choice(neighbor)
                    pos = random.randrange(1, len(route))
                    route.insert(pos, client)
        
        return neighbor

    def _local_search(self, solution):
        if not solution: return solution
        max_local_time = self.max_runtime * 0.1
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time <= max_local_time:
            improved = False
            for route_idx, route in enumerate(solution):
                if len(route) <= 4: continue
                best_length = sum(self.distances[route[i].id][route[i+1].id] 
                                for i in range(len(route)-1))
                for i in range(1, len(route)-2):
                    if time.time() - start_time > max_local_time: break
                    for j in range(i+1, min(i+5, len(route)-1)):
                        new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                        new_length = sum(self.distances[new_route[k].id][new_route[k+1].id] 
                                       for k in range(len(new_route)-1))
                        if new_length < best_length and new_length <= self.L:
                            solution[route_idx] = new_route
                            improved = True
                            break
                    if improved: break
        return solution

    def solve(self):
        start_time = time.time()
        current_solution = self._create_initial_solution()
        current_quality = self._calculate_solution_quality(current_solution)
        self.best_solution = copy.deepcopy(current_solution)
        self.best_fitness = current_quality
        
        temp = self.initial_temp
        iteration = 0
        accepted_moves = 0
        total_moves = 0
        
        while iteration < self.max_iterations and time.time() - start_time <= self.max_runtime:
            iteration_qualities = []
            
            for _ in range(self.iterations_per_temp):
                if time.time() - start_time > self.max_runtime: break
                
                neighbor = self._generate_neighbor(current_solution)
                if iteration % 10 == 0:
                    neighbor = self._local_search(neighbor)
                    
                neighbor_quality = self._calculate_solution_quality(neighbor)
                iteration_qualities.append(neighbor_quality)
                delta = neighbor_quality - current_quality
                
                if delta > 0 or random.random() < math.exp(delta / temp):
                    current_solution = neighbor
                    current_quality = neighbor_quality
                    accepted_moves += 1
                    
                    if current_quality > self.best_fitness:
                        self.best_solution = copy.deepcopy(current_solution)
                        self.best_fitness = current_quality
                
                total_moves += 1
            
            avg_fitness = sum(iteration_qualities) / len(iteration_qualities)
            acceptance_rate = accepted_moves / max(1, total_moves)
            
            self.stats_data['iteration'].append(iteration)
            self.stats_data['best_fitness'].append(self.best_fitness)
            self.stats_data['avg_fitness'].append(avg_fitness)
            self.stats_data['current_temp'].append(temp)
            self.stats_data['acceptance_rate'].append(acceptance_rate)
            
            if self.debug and iteration % 10 == 0:
                print(f"\nIteration {iteration}:")
                print(f"  Temperature = {temp:.2f}")
                print(f"  Best Fitness = {self.best_fitness:.2f}")
                print(f"  Average Fitness = {avg_fitness:.2f}")
                print(f"  Acceptance Rate = {acceptance_rate:.3f}")
            
            temp *= self.alpha
            iteration += 1
        
        return self.best_solution

    def get_stats(self):
        return pd.DataFrame(self.stats_data)

def lire_instance_chao(nom_fichier):
    try:
        with open(nom_fichier, "r") as f:
            lines = f.readlines()
        m = int(lines[1].split()[1])
        L = float(lines[2].split()[1])
        points = []
        for i, line in enumerate(lines[3:]):
            values = line.strip().split()
            if len(values) == 3:
                x, y, score = map(float, values)
                points.append(Client(i, x, y, score))
        if not points:
            raise ValueError("No valid points found in file")
        return points[0], points[-1], points[1:-1], m, L
    except FileNotFoundError:
        raise FileNotFoundError(f"Instance file not found: {nom_fichier}")
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid file format in {nom_fichier}: {str(e)}")

def create_experiment_structure(instance_name, algorithms):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    base_dir = f'experiments/{instance_name}_{timestamp}'
    folders = {
        'root': base_dir,
        'visualizations': f'{base_dir}/visualizations',
        'stats': f'{base_dir}/stats',
        'raw_data': f'{base_dir}/raw_data',
        'summary': f'{base_dir}/summary'
    }
    for path in folders.values():
        os.makedirs(path, exist_ok=True)
    for algo in algorithms:
        os.makedirs(f'{folders["visualizations"]}/{algo.lower()}', exist_ok=True)
        os.makedirs(f'{folders["stats"]}/{algo.lower()}', exist_ok=True)
    return folders

def visualize_solution(solution, start_point, end_point, clients, filename):
    plt.figure(figsize=(12, 8))
    visited_clients = {c.id for route in solution for c in route[1:-1]}
    unvisited_clients = [c for c in clients if c.id not in visited_clients]
    
    if unvisited_clients:
        plt.scatter([c.x for c in unvisited_clients], 
                   [c.y for c in unvisited_clients], 
                   c='gray', alpha=0.5, label='Unvisited')
    
    plt.scatter(start_point.x, start_point.y, c='green', marker='s', s=100, label='Start')
    plt.scatter(end_point.x, end_point.y, c='red', marker='s', s=100, label='End')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(solution)))
    for i, (route, color) in enumerate(zip(solution, colors)):
        route_x = [c.x for c in route]
        route_y = [c.y for c in route]
        plt.plot(route_x, route_y, c=color, linewidth=2)
        if len(route) > 2:
            plt.scatter(route_x[1:-1], route_y[1:-1], 
                       c=[color], s=100, 
                       label=f'Route {i+1} ({len(route)-2} clients)')
    
    plt.title('TOP Solution Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_algorithm_stats(stats_data, algorithm_name, output_dir):
    plt.style.use('seaborn-v0_8')
    plot_params = {
        'figsize': (12, 6),
        'grid': True,
        'alpha': 0.7,
        'linewidth': 2,
    }
    
    # Fitness Evolution Plot
    plt.figure(figsize=plot_params['figsize'])
    plt.plot(stats_data['iteration'], stats_data['best_fitness'], 
             label='Best Fitness', linewidth=plot_params['linewidth'])
    
    # Only plot average fitness if it exists
    if 'avg_fitness' in stats_data.columns:
        plt.plot(stats_data['iteration'], stats_data['avg_fitness'], 
                label='Average Fitness', linewidth=plot_params['linewidth'], alpha=0.7)
    
    plt.title(f'{algorithm_name} Fitness Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(plot_params['grid'])
    plt.savefig(f'{output_dir}/fitness_evolution.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Diversity Plot (if exists)
    if 'diversity' in stats_data.columns:
        plt.figure(figsize=plot_params['figsize'])
        plt.plot(stats_data['iteration'], stats_data['diversity'], 
                color='purple', linewidth=plot_params['linewidth'])
        plt.title(f'{algorithm_name} Population Diversity')
        plt.xlabel('Iteration')
        plt.ylabel('Diversity')
        plt.grid(plot_params['grid'])
        plt.savefig(f'{output_dir}/diversity.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    # Additional metric plots
    if isinstance(stats_data, pd.DataFrame):
        if 'acceptance_rate' in stats_data.columns:
            plt.figure(figsize=plot_params['figsize'])
            plt.plot(stats_data['iteration'], stats_data['acceptance_rate'], 
                    color='orange', linewidth=plot_params['linewidth'])
            plt.title(f'{algorithm_name} Acceptance Rate')
            plt.xlabel('Iteration')
            plt.ylabel('Acceptance Rate')
            plt.grid(plot_params['grid'])
            plt.savefig(f'{output_dir}/acceptance_rate.png', bbox_inches='tight', dpi=300)
            plt.close()
        elif 'pheromone_avg' in stats_data.columns:
            plt.figure(figsize=plot_params['figsize'])
            plt.plot(stats_data['iteration'], stats_data['pheromone_max'], 
                    label='Max', linewidth=plot_params['linewidth'])
            plt.plot(stats_data['iteration'], stats_data['pheromone_avg'], 
                    label='Average', linewidth=plot_params['linewidth'])
            plt.plot(stats_data['iteration'], stats_data['pheromone_min'], 
                    label='Min', linewidth=plot_params['linewidth'])
            plt.title(f'{algorithm_name} Pheromone Levels')
            plt.xlabel('Iteration')
            plt.ylabel('Pheromone Level')
            plt.legend()
            plt.grid(plot_params['grid'])
            plt.savefig(f'{output_dir}/pheromone_levels.png', bbox_inches='tight', dpi=300)
            plt.close()

def calculate_solution_metrics(solution, clients, L, m, execution_time, best_fitness):
    total_profit = 0
    total_distance = 0
    used_clients = set()
    route_times = []
    route_profits = []
    routes_exceeding_L = 0
    
    for route in solution:
        route_profit = 0
        route_time = 0
        route_clients = set()
        
        for i in range(len(route) - 1):
            dist = math.sqrt((route[i].x - route[i+1].x)**2 + (route[i].y - route[i+1].y)**2)
            route_time += dist
            
            if i > 0 and route[i].id not in used_clients:
                route_profit += route[i].profit
                route_clients.add(route[i].id)
        
        route_times.append(route_time)
        route_profits.append(route_profit)
        
        if route_time <= L:
            total_profit += route_profit
            total_distance += route_time
            used_clients.update(route_clients)
        else:
            routes_exceeding_L += 1
    
    avg_route_time = statistics.mean(route_times) if route_times else 0
    route_time_std = statistics.stdev(route_times) if len(route_times) > 1 else 0
    
    return {
        'Execution_Time': execution_time,
        'Best_Fitness': best_fitness,
        'Total_Profit': total_profit,
        'Unique_Clients_Visited': len(used_clients),
        'Total_Available_Clients': len(clients),
        'Coverage_Ratio': len(used_clients) / len(clients),
        'Number_of_Routes': len(solution),
        'Routes_Exceeding_TimeLimit': routes_exceeding_L,
        'Avg_Route_Time': avg_route_time,
        'Route_Time_Std': route_time_std,
        'Time_Efficiency': 1 - (avg_route_time / L) if L > 0 else 0,
        'Constraints_Respected': (routes_exceeding_L == 0 and 
                                len(solution) <= m)
    }

def run_experiment(instance_file, debug=False):
    start_point, end_point, clients, m, L = lire_instance_chao(instance_file)
    instance_name = os.path.splitext(os.path.basename(instance_file))[0]
    max_runtime = 300
    
    algorithms = [
        ('Baseline', TOPHeuristic(start_point, end_point, clients, m, L, max_runtime, debug)),
        ('SimulatedAnnealing', SimulatedAnnealingTOP(start_point, end_point, clients, m, L, max_runtime, debug)),
        ('GeneticAlgorithm', GeneticTOP(start_point, end_point, clients, m, L, max_runtime, debug)),
        ('AntColony', AntColonyTOP(start_point, end_point, clients, m, L, max_runtime, debug))
    ]
    
    folders = create_experiment_structure(instance_name, [name for name, _ in algorithms])
    results = []
    
    for name, algorithm in algorithms:
        print(f"\nRunning {name} on instance {instance_name}")
        start_time = time.time()
        
        solution = algorithm.evolve() if isinstance(algorithm, GeneticTOP) else algorithm.solve()
        execution_time = time.time() - start_time
        
        stats_df = algorithm.get_stats()
        stats_df.to_csv(f'{folders["stats"]}/{name.lower()}/raw_stats.csv', index=False)
        
        plot_algorithm_stats(stats_df, name, f'{folders["visualizations"]}/{name.lower()}')
        visualize_solution(solution, start_point, end_point, clients,
                         f'{folders["visualizations"]}/{name.lower()}/solution.png')
        
        metrics = calculate_solution_metrics(solution, clients, L, m, execution_time, algorithm.best_fitness)
        metrics.update({
            'Algorithm': name,
            'Instance': instance_name,
            'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        results.append(metrics)
        pd.DataFrame([metrics]).to_csv(f'{folders["raw_data"]}/{name.lower()}_results.csv', index=False)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{folders["summary"]}/experiment_summary.csv', index=False)
    return folders['root'], results_df

def run_benchmark(instance_sets, run_symmetric=True, run_non_symmetric=True, debug=False):
    all_results = []
    
    for set_type, size_dict in instance_sets.items():
        # Skip if set type doesn't match flags
        if (set_type == "symmetric" and not run_symmetric) or \
           (set_type == "non_symmetric" and not run_non_symmetric):
            continue
            
        for size, files in size_dict.items():
            print(f"\nRunning {set_type} set ({size} instances):")
            for instance_file in files:
                print(f"\nProcessing instance: {instance_file}")
                experiment_dir, results = run_experiment(instance_file, debug)
                results['Set_Type'] = set_type
                results['Size_Category'] = size
                all_results.append(results)
                print(f"Experiment results saved in: {experiment_dir}")
    
    if not all_results:
        print("No instances were run. Check your flags!")
        return None, None
        
    final_results = pd.concat(all_results)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    benchmark_dir = f'benchmark_results_{timestamp}'
    os.makedirs(benchmark_dir, exist_ok=True)
    final_results.to_csv(f'{benchmark_dir}/complete_benchmark_summary.csv', index=False)
    return benchmark_dir, final_results


if __name__ == "__main__":
    instance_sets = {
        "symmetric": {
            "small": ["Set_64_234/p6.2.d.txt", "Set_64_234/p6.2.i.txt", "Set_64_234/p6.2.n.txt"],
            "medium": ["Set_64_234/p6.3.d.txt", "Set_64_234/p6.3.i.txt", "Set_64_234/p6.3.n.txt"],
            "large": ["Set_64_234/p6.4.d.txt", "Set_64_234/p6.4.i.txt", "Set_64_234/p6.4.n.txt"]
        },
        "non_symmetric": {
            "small": ["Set_100_234/p4.2.a.txt", "Set_100_234/p4.2.j.txt", "Set_100_234/p4.2.t.txt"],
            "medium": ["Set_100_234/p4.3.a.txt", "Set_100_234/p4.3.j.txt", "Set_100_234/p4.3.t.txt"],
            "large": ["Set_100_234/p4.4.a.txt", "Set_100_234/p4.4.j.txt", "Set_100_234/p4.4.t.txt"]
        }
    }
    
    # Set flags for what to run
    RUN_SYMMETRIC = True       # Set to False to skip symmetric instances
    RUN_NON_SYMMETRIC = False  # Set to False to skip non-symmetric instances
    DEBUG = True              # Set to False to disable debug output
    
    print(f"\nRunning benchmark with settings:")
    print(f"Symmetric instances: {'Yes' if RUN_SYMMETRIC else 'No'}")
    print(f"Non-symmetric instances: {'Yes' if RUN_NON_SYMMETRIC else 'No'}")
    print(f"Debug mode: {'On' if DEBUG else 'Off'}")
    
    benchmark_dir, results = run_benchmark(
        instance_sets,
        run_symmetric=RUN_SYMMETRIC,
        run_non_symmetric=RUN_NON_SYMMETRIC,
        debug=DEBUG
    )
    
    if benchmark_dir:
        print(f"\nComplete benchmark results saved in: {benchmark_dir}")