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
        self.id = id; self.x = x; self.y = y; self.profit = profit
        
    def __repr__(self):
        return f"Client({self.id}, x={self.x}, y={self.y}, profit={self.profit})"
    
class TOPHeuristic:
    def __init__(self, start_point, end_point, clients, m, L, max_runtime=300, debug=False):
        self.start_point = start_point; self.end_point = end_point; self.clients = clients
        self.m = m; self.L = L; self.debug = debug; self.max_runtime = max_runtime
        # Match computation budget with other algorithms
        self.max_iterations = min(125, len(clients) * 4)  # Match GA/ACO iterations
        self.solutions_per_iter = min(150, len(clients) * 2)  # Match population/colony sizes
        self.distances = self._precompute_distances()
        self.best_solution = None; self.best_fitness = float('-inf')
        self.stats_data = {'iteration': [], 'best_fitness': [], 'avg_fitness': [], 'diversity': []}

    def _precompute_distances(self):
        distances = {}; all_points = [self.start_point] + self.clients + [self.end_point]
        for i, p1 in enumerate(all_points):
            distances[p1.id] = {}
            for p2 in all_points:
                if p1 != p2: distances[p1.id][p2.id] = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
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

    def _construct_route(self, available_clients, construction_type='balanced'):
        route = [self.start_point]; current = self.start_point; current_time = 0
        remaining = list(available_clients)
        center_x = sum(c.x for c in self.clients) / len(self.clients)
        center_y = sum(c.y for c in self.clients) / len(self.clients)
        max_profit = max(c.profit for c in self.clients)
        
        while remaining:
            candidates = []
            for client in remaining[:min(10, len(remaining))]:
                time_to_client = self.distances[current.id][client.id]
                time_to_end = self.distances[client.id][self.end_point.id]
                if current_time + time_to_client + time_to_end <= self.L:
                    dist = self.distances[current.id][client.id]
                    profit = client.profit
                    centrality = 1 - (((client.x - center_x)**2 + (client.y - center_y)**2)**0.5 / 
                                    max(abs(c.x - center_x) for c in self.clients))
                    
                    if construction_type == 'balanced':
                        score = (profit/max_profit * 0.4 + (1/dist) * 0.4 + centrality * 0.2)
                    elif construction_type == 'greedy':
                        score = profit/dist
                    elif construction_type == 'random':
                        score = random.random()
                    candidates.append((client, score))
            
            if not candidates: break
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected = candidates[0][0] if random.random() < 0.7 else random.choice(candidates[:3])[0]
            route.append(selected)
            current_time += self.distances[current.id][selected.id]
            current = selected
            remaining.remove(selected)
        
        route.append(self.end_point)
        return route if len(route) > 2 else None

    def _construct_solution(self, construction_type='balanced'):
        solution = []; used_clients = set()
        available_clients = set(self.clients)
        for _ in range(self.m):
            remaining = available_clients - used_clients
            if not remaining: break
            route = self._construct_route(remaining, construction_type)
            if route:
                solution.append(route)
                used_clients.update(set(route[1:-1]))
        return solution

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
        construction_types = ['balanced', 'greedy', 'random']
        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.max_runtime: break
            iteration_solutions = []
            iteration_qualities = []
            
            for _ in range(self.solutions_per_iter):
                construction_type = random.choice(construction_types)
                solution = self._construct_solution(construction_type)
                quality = self._calculate_solution_quality(solution)
                iteration_solutions.append(solution)
                iteration_qualities.append(quality)
                
                if quality > self.best_fitness:
                    self.best_fitness = quality
                    self.best_solution = copy.deepcopy(solution)
            
            avg_fitness = sum(iteration_qualities) / len(iteration_qualities)
            diversity = self._calculate_diversity(iteration_solutions)
            
            self.stats_data['iteration'].append(iteration)
            self.stats_data['best_fitness'].append(self.best_fitness)
            self.stats_data['avg_fitness'].append(avg_fitness)
            self.stats_data['diversity'].append(diversity)
            
            if self.debug and iteration % 10 == 0:
                print(f"\nIteration {iteration}:")
                print(f"  Best Fitness = {self.best_fitness:.2f}")
                print(f"  Average Fitness = {avg_fitness:.2f}")
                print(f"  Diversity = {diversity:.3f}")
        
        return self.best_solution

    def get_stats(self):
        return pd.DataFrame(self.stats_data)

class AntColonyTOP:
    def __init__(self, start_point, end_point, clients, m, L, max_runtime=300, debug=False):
        self.start_point = start_point; self.end_point = end_point; self.clients = clients
        self.m = m; self.L = L; self.debug = debug; self.max_runtime = max_runtime
        n = len(clients)
        self.max_iterations = min(125, n * 4)
        self.n_ants = min(150, n * 2)  # Match GA population
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic importance
        self.rho = 0.1    # Evaporation rate
        self.q0_initial = 0.7  # Initial exploitation rate
        self.q0_final = 0.9    # Final exploitation rate
        self.local_search_freq = max(10, n // 10)
        self.max_stagnation = max(40, n // 2)
        self.tau_max = 2.0; self.tau_min = 0.01
        self.distances = self._precompute_distances()
        self.pheromone = self._initialize_pheromone()
        self.eta = self._initialize_heuristic()
        self.best_solution = None; self.best_fitness = float('-inf')
        self.stats_data = {'iteration': [], 'best_fitness': [], 'avg_fitness': [], 'diversity': [], 
                          'pheromone_avg': [], 'exploitation_rate': []}

    def _precompute_distances(self):
        distances = {}; all_points = [self.start_point] + self.clients + [self.end_point]
        for i, p1 in enumerate(all_points):
            distances[p1.id] = {}
            for p2 in all_points:
                if p1 != p2: distances[p1.id][p2.id] = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
        return distances

    def _initialize_pheromone(self):
        all_points = [self.start_point] + self.clients + [self.end_point]
        return {i.id: {j.id: self.tau_max for j in all_points if i != j} for i in all_points}

    def _initialize_heuristic(self):
        eta = {}; all_points = [self.start_point] + self.clients + [self.end_point]
        center_x = sum(c.x for c in self.clients) / len(self.clients)
        center_y = sum(c.y for c in self.clients) / len(self.clients)
        max_profit = max(c.profit for c in self.clients)
        max_dist = max(abs(c.x - center_x) for c in self.clients)
        for i in all_points:
            eta[i.id] = {}
            for j in all_points:
                if i != j:
                    dist = self.distances[i.id][j.id]
                    if j in self.clients:
                        profit_factor = j.profit / max_profit
                        centrality = 1 - (((j.x - center_x)**2 + (j.y - center_y)**2)**0.5 / max_dist)
                        eta[i.id][j.id] = (0.4 * profit_factor + 0.4 / dist + 0.2 * centrality)
                    else:
                        eta[i.id][j.id] = 1.0 / dist
        return eta

    def _select_next_client(self, current, available, current_time, q0):
        if not available: return self.end_point
        feasible = []
        for client in available:
            time_to_client = self.distances[current.id][client.id]
            time_to_end = self.distances[client.id][self.end_point.id]
            if current_time + time_to_client + time_to_end <= self.L:
                tau = self.pheromone[current.id][client.id]
                eta = self.eta[current.id][client.id]
                score = (tau ** self.alpha) * (eta ** self.beta)
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

    def _construct_solution(self, q0):
        solution = []; used_clients = set()
        for _ in range(self.m):
            remaining = set(self.clients) - used_clients
            if not remaining: break
            route = [self.start_point]; current = self.start_point; current_time = 0
            while remaining:
                next_client = self._select_next_client(current, remaining, current_time, q0)
                if next_client == self.end_point: break
                route.append(next_client)
                current_time += self.distances[current.id][next_client.id]
                current = next_client; remaining.remove(next_client)
            route.append(self.end_point)
            if len(route) > 2:
                solution.append(route)
                used_clients.update(set(route[1:-1]))
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
                total_profit += route_profit; total_distance += route_distance
        if total_distance > self.L * self.m: return 0
        coverage = len(used_clients) / len(self.clients)
        efficiency = 1 - (total_distance / (self.L * self.m))
        return total_profit * (1 + 0.15 * coverage + 0.1 * efficiency)

    def _update_pheromone(self, solutions, qualities):
        # Evaporation
        for i in self.pheromone:
            for j in self.pheromone[i]:
                self.pheromone[i][j] *= (1 - self.rho)
        # Deposit
        for solution, quality in zip(solutions, qualities):
            deposit = quality / 100
            for route in solution:
                for i in range(len(route)-1):
                    self.pheromone[route[i].id][route[i+1].id] += self.rho * deposit
                    self.pheromone[route[i+1].id][route[i].id] = self.pheromone[route[i].id][route[i+1].id]
        # Bounds check
        for i in self.pheromone:
            for j in self.pheromone[i]:
                self.pheromone[i][j] = min(self.tau_max, max(self.tau_min, self.pheromone[i][j]))

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
        stagnation = 0
        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.max_runtime: break
            q0 = self.q0_initial + (self.q0_final - self.q0_initial) * (iteration / self.max_iterations)
            solutions = []; qualities = []
            for ant in range(self.n_ants):
                solution = self._construct_solution(q0)
                if iteration % self.local_search_freq == 0:
                    solution = self._local_search(solution)
                quality = self._calculate_solution_quality(solution)
                solutions.append(solution); qualities.append(quality)
                if quality > self.best_fitness:
                    self.best_fitness = quality
                    self.best_solution = copy.deepcopy(solution)
                    stagnation = 0
                else: stagnation += 1
            self._update_pheromone(solutions, qualities)
            self._update_stats(iteration, qualities, solutions, q0)
            if stagnation > self.max_stagnation and iteration > self.max_iterations // 4: break
        return self.best_solution

    def _local_search(self, solution):
        if not solution: return solution
        max_local_time = self.max_runtime * 0.1
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time <= max_local_time:
            improved = False
            for route_idx, route in enumerate(solution):
                if len(route) <= 4: continue
                best_length = sum(self.distances[route[i].id][route[i+1].id] for i in range(len(route)-1))
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

    def _update_stats(self, iteration, qualities, solutions, q0):
        avg_fitness = sum(qualities) / len(qualities)
        diversity = self._calculate_diversity(solutions)
        pheromone_values = [v for d in self.pheromone.values() for v in d.values()]
        pheromone_avg = sum(pheromone_values) / len(pheromone_values)
        self.stats_data['iteration'].append(iteration)
        self.stats_data['best_fitness'].append(self.best_fitness)
        self.stats_data['avg_fitness'].append(avg_fitness)
        self.stats_data['diversity'].append(diversity)
        self.stats_data['pheromone_avg'].append(pheromone_avg)
        self.stats_data['exploitation_rate'].append(q0)
        if self.debug and iteration % 10 == 0:
            print(f"\nIteration {iteration}:")
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
        self.max_iterations = min(125, n * 4)  # Match GA/ACO iterations
        self.iterations_per_temp = min(150, n * 2)  # Match population/colony sizes
        self.initial_temp = 100.0; self.final_temp = 1.0
        self.alpha = pow(self.final_temp/self.initial_temp, 1.0/self.max_iterations)
        self.local_search_freq = max(10, n // 10)
        self.distances = self._precompute_distances()
        self.best_solution = None; self.best_fitness = float('-inf')
        self.stats_data = {'iteration': [], 'best_fitness': [], 'avg_fitness': [], 
                          'current_temp': [], 'acceptance_rate': [], 'diversity': []}

    def _precompute_distances(self):
        distances = {}; all_points = [self.start_point] + self.clients + [self.end_point]
        for i, p1 in enumerate(all_points):
            distances[p1.id] = {}
            for p2 in all_points:
                if p1 != p2: distances[p1.id][p2.id] = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
        return distances

    def _create_initial_solution(self):
        solution = []; available_clients = set(self.clients)
        center_x = sum(c.x for c in self.clients) / len(self.clients)
        center_y = sum(c.y for c in self.clients) / len(self.clients)
        max_profit = max(c.profit for c in self.clients)
        
        for _ in range(self.m):
            if not available_clients: break
            route = [self.start_point]; current = self.start_point; current_time = 0
            while available_clients:
                candidates = []
                for client in list(available_clients)[:10]:
                    time_to_client = self.distances[current.id][client.id]
                    time_to_end = self.distances[client.id][self.end_point.id]
                    if current_time + time_to_client + time_to_end <= self.L:
                        dist = self.distances[current.id][client.id]
                        profit_factor = client.profit / max_profit
                        centrality = 1 - (((client.x - center_x)**2 + (client.y - center_y)**2)**0.5 / 
                                        max(abs(c.x - center_x) for c in self.clients))
                        score = profit_factor * 0.4 + (1/dist) * 0.4 + centrality * 0.2
                        candidates.append((client, score))
                if not candidates: break
                candidates.sort(key=lambda x: x[1], reverse=True)
                selected = candidates[0][0] if random.random() < 0.7 else random.choice(candidates[:3])[0]
                route.append(selected); current_time += self.distances[current.id][selected.id]
                current = selected; available_clients.remove(selected)
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
                total_profit += route_profit; total_distance += route_distance
        if total_distance > self.L * self.m: return 0
        coverage = len(used_clients) / len(self.clients)
        efficiency = 1 - (total_distance / (self.L * self.m))
        return total_profit * (1 + 0.15 * coverage + 0.1 * efficiency)

    def _generate_neighbor(self, solution, temperature):
        if not solution: return solution
        neighbor = copy.deepcopy(solution)
        moves = [(self._swap_move, 0.3), (self._reverse_move, 0.3), 
                (self._insert_move, 0.2), (self._relocate_move, 0.2)]
        
        # Temperature-based move selection
        if temperature > self.initial_temp * 0.7:
            # Higher temp: favor more disruptive moves
            moves[0] = (self._swap_move, 0.4)  # More swaps
            moves[3] = (self._relocate_move, 0.3)  # More relocations
        elif temperature < self.final_temp * 3:
            # Lower temp: favor local improvements
            moves[1] = (self._reverse_move, 0.4)  # More reversals
            moves[2] = (self._insert_move, 0.3)  # More insertions
            
        move_func = random.choices([m[0] for m in moves], weights=[m[1] for m in moves])[0]
        return move_func(neighbor)

    def _swap_move(self, solution):
        if len(solution) >= 2:
            route1, route2 = random.sample(solution, 2)
            if len(route1) > 3 and len(route2) > 3:
                pos1 = random.randrange(1, len(route1) - 1)
                pos2 = random.randrange(1, len(route2) - 1)
                route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
        return solution

    def _reverse_move(self, solution):
        if solution:
            route = random.choice(solution)
            if len(route) > 4:
                i = random.randrange(1, len(route) - 2)
                j = random.randrange(i + 1, len(route) - 1)
                route[i:j+1] = reversed(route[i:j+1])
        return solution

    def _insert_move(self, solution):
        if solution:
            route = random.choice(solution)
            if len(route) > 3:
                pos1 = random.randrange(1, len(route) - 1)
                pos2 = random.randrange(1, len(route) - 1)
                client = route.pop(pos1)
                route.insert(pos2, client)
        return solution

    def _relocate_move(self, solution):
        if len(solution) >= 2:
            source_route = random.choice(solution)
            target_route = random.choice([r for r in solution if r != source_route])
            if len(source_route) > 3:
                pos = random.randrange(1, len(source_route) - 1)
                client = source_route.pop(pos)
                insert_pos = random.randrange(1, len(target_route))
                target_route.insert(insert_pos, client)
        return solution

    def _calculate_diversity(self, solutions):
        if not solutions or len(solutions) < 2: return 0
        clients_sets = [set(c.id for route in sol for c in route[1:-1]) for sol in solutions]
        diversity = 0; comparisons = 0
        for i in range(len(clients_sets)):
            for j in range(i+1, len(clients_sets)):
                if clients_sets[i] or clients_sets[j]:
                    hamming = len(clients_sets[i].symmetric_difference(clients_sets[j]))
                    diversity += hamming / max(len(clients_sets[i]), len(clients_sets[j]))
                    comparisons += 1
        return diversity / max(1, comparisons)

    def solve(self):
        start_time = time.time()
        current_solution = self._create_initial_solution()
        current_quality = self._calculate_solution_quality(current_solution)
        self.best_solution = copy.deepcopy(current_solution)
        self.best_fitness = current_quality
        
        temp = self.initial_temp
        accepted_moves = 0; total_moves = 0
        iteration_solutions = []  # Track solutions for diversity
        
        for iteration in range(self.max_iterations):
            if time.time() - start_time > self.max_runtime: break
            iteration_qualities = []
            iteration_solutions = [current_solution]  # Reset for new iteration
            
            for step in range(self.iterations_per_temp):
                neighbor = self._generate_neighbor(current_solution, temp)
                if iteration % self.local_search_freq == 0:
                    neighbor = self._local_search(neighbor)
                    
                quality = self._calculate_solution_quality(neighbor)
                iteration_qualities.append(quality)
                iteration_solutions.append(neighbor)
                
                delta = quality - current_quality
                if delta > 0 or random.random() < math.exp(delta / temp):
                    current_solution = copy.deepcopy(neighbor)
                    current_quality = quality
                    accepted_moves += 1
                    if quality > self.best_fitness:
                        self.best_solution = copy.deepcopy(neighbor)
                        self.best_fitness = quality
                total_moves += 1
            
            self._update_stats(iteration, iteration_qualities, iteration_solutions, 
                             temp, accepted_moves/max(1, total_moves))
            temp *= self.alpha
            
        return self.best_solution

    def _local_search(self, solution):
        if not solution: return solution
        max_local_time = self.max_runtime * 0.1
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time <= max_local_time:
            improved = False
            for route_idx, route in enumerate(solution):
                if len(route) <= 4: continue
                best_length = sum(self.distances[route[i].id][route[i+1].id] for i in range(len(route)-1))
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

    def _update_stats(self, iteration, qualities, solutions, temp, acceptance_rate):
        avg_fitness = sum(qualities) / len(qualities)
        diversity = self._calculate_diversity(solutions)
        self.stats_data['iteration'].append(iteration)
        self.stats_data['best_fitness'].append(self.best_fitness)
        self.stats_data['avg_fitness'].append(avg_fitness)
        self.stats_data['current_temp'].append(temp)
        self.stats_data['acceptance_rate'].append(acceptance_rate)
        self.stats_data['diversity'].append(diversity)
        
        if self.debug and iteration % 10 == 0:
            print(f"\nIteration {iteration}:")
            print(f"  Temperature = {temp:.2f}")
            print(f"  Best Fitness = {self.best_fitness:.2f}")
            print(f"  Average Fitness = {avg_fitness:.2f}")
            print(f"  Diversity = {diversity:.3f}")
            print(f"  Acceptance Rate = {acceptance_rate:.3f}")

    def get_stats(self):
        return pd.DataFrame(self.stats_data)

class GeneticTOP:
    def __init__(self, start_point, end_point, clients, m, L, max_runtime=300, debug=False):
        self.start_point = start_point; self.end_point = end_point; self.clients = clients
        self.m = m; self.L = L; self.debug = debug; self.max_runtime = max_runtime
        n = len(clients)
        self.population_size = min(150, n * 2)
        self.generations = min(125, n * 4)
        self.crossover_rate = 0.85
        self.mutation_rate = 0.15
        self.elite_size = max(2, self.population_size // 20)
        self.tournament_size = max(3, self.population_size // 25)
        self.local_search_freq = max(10, n // 10)
        self.max_stagnation = max(40, n // 2)
        self.distances = self._precompute_distances()
        self.best_solution = None; self.best_fitness = float('-inf')
        self.stats_data = {'iteration': [], 'best_fitness': [], 'avg_fitness': [], 
                          'diversity': [], 'mutation_rate': [], 'crossover_rate': []}

    def _precompute_distances(self):
        distances = {}; all_points = [self.start_point] + self.clients + [self.end_point]
        for i, p1 in enumerate(all_points):
            distances[p1.id] = {}
            for p2 in all_points:
                if p1 != p2: distances[p1.id][p2.id] = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
        return distances

    def _create_initial_solution(self):
        solution = []; available_clients = set(self.clients)
        center_x = sum(c.x for c in self.clients) / len(self.clients)
        center_y = sum(c.y for c in self.clients) / len(self.clients)
        max_profit = max(c.profit for c in self.clients)
        
        for _ in range(self.m):
            if not available_clients: break
            route = [self.start_point]; current = self.start_point; current_time = 0
            while available_clients:
                candidates = []
                for client in list(available_clients)[:10]:
                    time_to_client = self.distances[current.id][client.id]
                    time_to_end = self.distances[client.id][self.end_point.id]
                    if current_time + time_to_client + time_to_end <= self.L:
                        dist = self.distances[current.id][client.id]
                        profit_factor = client.profit / max_profit
                        centrality = 1 - (((client.x - center_x)**2 + (client.y - center_y)**2)**0.5 / 
                                        max(abs(c.x - center_x) for c in self.clients))
                        score = profit_factor * 0.4 + (1/dist) * 0.4 + centrality * 0.2
                        candidates.append((client, score))
                if not candidates: break
                candidates.sort(key=lambda x: x[1], reverse=True)
                selected = candidates[0][0] if random.random() < 0.7 else random.choice(candidates[:3])[0]
                route.append(selected); current_time += self.distances[current.id][selected.id]
                current = selected; available_clients.remove(selected)
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
                total_profit += route_profit; total_distance += route_distance
        if total_distance > self.L * self.m: return 0
        coverage = len(used_clients) / len(self.clients)
        efficiency = 1 - (total_distance / (self.L * self.m))
        return total_profit * (1 + 0.15 * coverage + 0.1 * efficiency)

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate: return copy.deepcopy(parent1), copy.deepcopy(parent2)
        child1, child2 = [], []; used_clients1, used_clients2 = set(), set()
        # Route-based crossover with efficiency scoring
        routes1 = [(r, self._evaluate_route(r)) for r in parent1]
        routes2 = [(r, self._evaluate_route(r)) for r in parent2]
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
        mutation_ops = [self._swap_mutation, self._reverse_mutation, self._insert_mutation]
        for _ in range(2):
            random.choice(mutation_ops)(mutated)
        return mutated

    def _swap_mutation(self, solution):
        if len(solution) >= 2:
            route1, route2 = random.sample(solution, 2)
            if len(route1) > 3 and len(route2) > 3:
                pos1 = random.randrange(1, len(route1) - 1)
                pos2 = random.randrange(1, len(route2) - 1)
                route1[pos1], route2[pos2] = route2[pos2], route1[pos1]

    def _reverse_mutation(self, solution):
        if solution:
            route = random.choice(solution)
            if len(route) > 4:
                i = random.randrange(1, len(route) - 2)
                j = random.randrange(i + 1, len(route) - 1)
                route[i:j+1] = reversed(route[i:j+1])

    def _insert_mutation(self, solution):
        if solution:
            route = random.choice(solution)
            if len(route) > 3:
                pos1 = random.randrange(1, len(route) - 1)
                pos2 = random.randrange(1, len(route) - 1)
                client = route.pop(pos1)
                route.insert(pos2, client)

    def _evaluate_route(self, route):
        if len(route) < 3: return 0
        route_time = sum(self.distances[route[i].id][route[i+1].id] for i in range(len(route)-1))
        if route_time > self.L: return 0
        return sum(c.profit for c in route[1:-1]) / route_time

    def _tournament_selection(self, population, fitnesses):
        tournament = random.sample(list(enumerate(fitnesses)), self.tournament_size)
        winner_idx = max(tournament, key=lambda x: x[1])[0]
        return copy.deepcopy(population[winner_idx])

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
        population = [self._create_initial_solution() for _ in range(self.population_size)]
        stagnation_counter = 0
        
        for generation in range(self.generations):
            if time.time() - start_time > self.max_runtime: break
            fitnesses = [self._calculate_solution_quality(solution) for solution in population]
            best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_solution = copy.deepcopy(population[best_idx])
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            self._update_stats(generation, fitnesses, population)
            if stagnation_counter > self.max_stagnation and generation > self.generations // 4: break
            
            new_population = [copy.deepcopy(population[i]) for i in sorted(
                range(len(fitnesses)), key=lambda k: fitnesses[k], reverse=True)[:self.elite_size]]
                
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

    def _local_search(self, solution):
        if not solution: return solution
        max_local_time = self.max_runtime * 0.1
        start_time = time.time()
        improved = True
        while improved and time.time() - start_time <= max_local_time:
            improved = False
            for route_idx, route in enumerate(solution):
                if len(route) <= 4: continue
                best_length = sum(self.distances[route[i].id][route[i+1].id] for i in range(len(route)-1))
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

    def _update_stats(self, generation, fitnesses, population):
        avg_fitness = sum(fitnesses) / len(fitnesses)
        diversity = self._calculate_diversity(population)
        self.stats_data['iteration'].append(generation)
        self.stats_data['best_fitness'].append(self.best_fitness)
        self.stats_data['avg_fitness'].append(avg_fitness)
        self.stats_data['diversity'].append(diversity)
        self.stats_data['mutation_rate'].append(self.mutation_rate)
        self.stats_data['crossover_rate'].append(self.crossover_rate)
        
        if self.debug and generation % 10 == 0:
            print(f"\nGeneration {generation}:")
            print(f"  Best Fitness = {self.best_fitness:.2f}")
            print(f"  Average Fitness = {avg_fitness:.2f}")
            print(f"  Diversity = {diversity:.3f}")

    def get_stats(self):
        return pd.DataFrame(self.stats_data)
    

def lire_instance_chao(nom_fichier):
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
    if not points: raise ValueError("No valid points found in file")
    return points[0], points[-1], points[1:-1], m, L

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
    for path in folders.values(): os.makedirs(path, exist_ok=True)
    for algo in algorithms:
        os.makedirs(f'{folders["visualizations"]}/{algo.lower()}', exist_ok=True)
        os.makedirs(f'{folders["stats"]}/{algo.lower()}', exist_ok=True)
    return folders

def visualize_solution(solution, start_point, end_point, clients, filename):
    plt.figure(figsize=(12, 8))
    visited_clients = {c.id for route in solution for c in route[1:-1]}
    unvisited = [c for c in clients if c.id not in visited_clients]
    if unvisited:
        plt.scatter([c.x for c in unvisited], [c.y for c in unvisited], c='gray', alpha=0.5, label='Unvisited')
    plt.scatter(start_point.x, start_point.y, c='green', marker='s', s=100, label='Start')
    plt.scatter(end_point.x, end_point.y, c='red', marker='s', s=100, label='End')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(solution)))
    for i, (route, color) in enumerate(zip(solution, colors)):
        route_x = [c.x for c in route]; route_y = [c.y for c in route]
        plt.plot(route_x, route_y, c=color, linewidth=2)
        if len(route) > 2:
            plt.scatter(route_x[1:-1], route_y[1:-1], c=[color], s=100, label=f'Route {i+1}')
    plt.title('TOP Solution Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_algorithm_stats(stats_data, algorithm_name, output_dir):
    plt.style.use('seaborn-v0_8')
    plot_params = {'figsize': (12, 6), 'grid': True, 'alpha': 0.7, 'linewidth': 2}
    # Fitness Evolution
    plt.figure(figsize=plot_params['figsize'])
    plt.plot(stats_data['iteration'], stats_data['best_fitness'], label='Best', linewidth=2)
    if 'avg_fitness' in stats_data.columns:
        plt.plot(stats_data['iteration'], stats_data['avg_fitness'], label='Average', alpha=0.7)
    plt.title(f'{algorithm_name} Fitness Evolution')
    plt.xlabel('Iteration'); plt.ylabel('Fitness')
    plt.legend(); plt.grid(True)
    plt.savefig(f'{output_dir}/fitness_evolution.png', bbox_inches='tight', dpi=300)
    plt.close()
    # Diversity Plot
    if 'diversity' in stats_data.columns:
        plt.figure(figsize=plot_params['figsize'])
        plt.plot(stats_data['iteration'], stats_data['diversity'], color='purple', linewidth=2)
        plt.title(f'{algorithm_name} Population Diversity')
        plt.xlabel('Iteration'); plt.ylabel('Diversity')
        plt.grid(True)
        plt.savefig(f'{output_dir}/diversity.png', bbox_inches='tight', dpi=300)
        plt.close()

def calculate_solution_metrics(solution, clients, L, m, execution_time, best_fitness):
    total_profit = 0; total_distance = 0; used_clients = set(); route_times = []
    routes_exceeding_L = 0
    for route in solution:
        route_profit = 0; route_time = 0; route_clients = set()
        for i in range(len(route) - 1):
            dist = ((route[i].x - route[i+1].x)**2 + (route[i].y - route[i+1].y)**2)**0.5
            route_time += dist
            if i > 0 and route[i].id not in used_clients:
                route_profit += route[i].profit
                route_clients.add(route[i].id)
        route_times.append(route_time)
        if route_time <= L:
            total_profit += route_profit
            total_distance += route_time
            used_clients.update(route_clients)
        else: routes_exceeding_L += 1
    return {
        'Execution_Time': execution_time,
        'Best_Fitness': best_fitness,
        'Total_Profit': total_profit,
        'Unique_Clients_Visited': len(used_clients),
        'Coverage_Ratio': len(used_clients) / len(clients),
        'Number_of_Routes': len(solution),
        'Routes_Exceeding_TimeLimit': routes_exceeding_L,
        'Avg_Route_Time': statistics.mean(route_times) if route_times else 0,
        'Route_Time_Std': statistics.stdev(route_times) if len(route_times) > 1 else 0,
        'Time_Efficiency': 1 - (total_distance / (L * m)) if L > 0 else 0,
        'Constraints_Respected': routes_exceeding_L == 0 and len(solution) <= m
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
        metrics.update({'Algorithm': name, 'Instance': instance_name,
                       'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')})
        results.append(metrics)
        pd.DataFrame([metrics]).to_csv(f'{folders["raw_data"]}/{name.lower()}_results.csv', index=False)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{folders["summary"]}/experiment_summary.csv', index=False)
    return folders['root'], results_df

def run_benchmark(instance_sets, run_symmetric=True, run_non_symmetric=True, debug=False):
    all_results = []
    for set_type, size_dict in instance_sets.items():
        if (set_type == "symmetric" and not run_symmetric) or \
           (set_type == "non_symmetric" and not run_non_symmetric): continue
        for size, files in size_dict.items():
            print(f"\nRunning {set_type} set ({size} instances):")
            for instance_file in files:
                print(f"\nProcessing instance: {instance_file}")
                experiment_dir, results = run_experiment(instance_file, debug)
                results['Set_Type'] = set_type; results['Size_Category'] = size
                all_results.append(results)
                print(f"Results saved in: {experiment_dir}")
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
        "non_symmetric": {
            "small": ["Set_100_234/p4.2.a.txt", "Set_100_234/p4.3.d.txt", "Set_100_234/p4.4.e.txt"],
            "medium": ["Set_100_234/p4.2.j.txt", "Set_100_234/p4.3.j.txt", "Set_100_234/p4.4.m.txt"],
            "high": ["Set_100_234/p4.2.t.txt", "Set_100_234/p4.3.t.txt", "Set_100_234/p4.4.t.txt"]
        },
        "symmetric": {
            "small": ["Set_64_234/p6.2.d.txt", "Set_64_234/p6.3.g.txt", "Set_64_234/p6.4.j.txt"],
            "medium": ["Set_64_234/p6.2.j.txt", "Set_64_234/p6.3.i.txt", "Set_64_234/p6.4.l.txt"],
            "high": ["Set_64_234/p6.2.n.txt", "Set_64_234/p6.3.n.txt", "Set_64_234/p6.4.n.txt"]
        }
    }
    RUN_SYMMETRIC = True
    RUN_NON_SYMMETRIC = True
    DEBUG = True
    print(f"\nBenchmark settings:")
    print(f"Symmetric: {'Yes' if RUN_SYMMETRIC else 'No'}")
    print(f"Non-symmetric: {'Yes' if RUN_NON_SYMMETRIC else 'No'}")
    print(f"Debug: {'On' if DEBUG else 'Off'}")
    benchmark_dir, results = run_benchmark(instance_sets, RUN_SYMMETRIC, RUN_NON_SYMMETRIC, DEBUG)
    if benchmark_dir: print(f"\nResults saved in: {benchmark_dir}")