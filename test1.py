import math
import random
import numpy as np
from multiprocessing import Pool
import copy
import time
import pandas as pd
import tracemalloc
import statistics
import matplotlib.pyplot as plt
import itertools
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple
import os 
import glob


class Client:
    def __init__(self, id, x, y, profit):
        self.id = id
        self.x = x
        self.y = y
        self.profit = profit


class GreedyTOP:
    def __init__(self, start_point, end_point, clients, m, L, debug=False):
        self.start_point = start_point
        self.end_point = end_point
        self.clients = clients
        self.m = m
        self.L = L
        self.debug = debug
        
        # Solution tracking
        self.best_solution = None
        self.best_fitness = float('-inf')
        
        # Precompute distances for efficiency
        self.distances = self._precompute_distances()
        
        # Statistics collection (for consistency with other algorithms)
        self.stats_data = {
            'iteration': [0],
            'best_fitness': [0],
            'avg_fitness': [0],
            'diversity': [0]
        }
    
    def _precompute_distances(self):
        """Precompute distances between all points for efficiency."""
        distances = {}
        all_points = [self.start_point] + self.clients + [self.end_point]
        for i, p1 in enumerate(all_points):
            for p2 in all_points[i+1:]:
                dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                distances[(p1.id, p2.id)] = distances[(p2.id, p1.id)] = dist
        return distances
    
    def get_distance(self, client1, client2):
        """Get precomputed distance between two clients."""
        return self.distances[(client1.id, client2.id)]
    
    def _calculate_score(self, current_client, candidate_client, current_time):
        """Calculate score for a candidate client based on multiple factors."""
        distance_to_candidate = self.get_distance(current_client, candidate_client)
        distance_to_end = self.get_distance(candidate_client, self.end_point)
        
        # Skip if adding this client would exceed time limit
        if current_time + distance_to_candidate + distance_to_end > self.L:
            return float('-inf')
        
        # Score based on profit per unit time with diminishing returns for longer routes
        time_factor = 1 - (current_time / self.L)
        profit_per_distance = candidate_client.profit / (distance_to_candidate + 0.1)
        
        # Penalty for getting too close to time limit
        time_remaining = self.L - (current_time + distance_to_candidate)
        time_buffer_penalty = 1 - (0.5 * (1 - time_remaining / self.L))
        
        # Consider path to end point
        end_accessibility = 1 / (1 + distance_to_end)
        
        return profit_per_distance * time_factor * time_buffer_penalty * end_accessibility
    
    def _construct_route(self, available_clients):
        """Construct a single route using greedy approach."""
        if not available_clients:
            return [self.start_point, self.end_point]
        
        route = [self.start_point]
        current_time = 0
        current = self.start_point
        local_available = available_clients.copy()
        
        while local_available:
            best_score = float('-inf')
            best_next = None
            
            # Evaluate all remaining clients
            for client in local_available:
                score = self._calculate_score(current, client, current_time)
                if score > best_score:
                    best_score = score
                    best_next = client
            
            # If no valid next client found, end route
            if best_score == float('-inf'):
                break
            
            # Add best client to route
            route.append(best_next)
            current_time += self.get_distance(current, best_next)
            current = best_next
            local_available.remove(best_next)
        
        route.append(self.end_point)
        return route if len(route) > 2 else [self.start_point, self.end_point]
    
    def _calculate_solution_quality(self, solution):
        """Calculate solution quality considering profits and time constraints."""
        if not solution:
            return 0
        
        total_profit = sum(sum(c.profit for c in route[1:-1]) for route in solution)
        total_time = sum(sum(self.get_distance(route[i], route[i+1]) 
                        for i in range(len(route)-1)) for route in solution)
        
        if total_time > self.L:
            return 0
        
        # Quality score considers both profit and time efficiency
        time_efficiency = 1 - (total_time / (self.L * self.m))
        coverage_ratio = sum(len(route)-2 for route in solution) / len(self.clients)
        
        return total_profit * (1 + 0.1 * time_efficiency) * (1 + 0.1 * coverage_ratio)
    
    def solve(self):
        """Main solving method using greedy approach."""
        solution = []
        available_clients = set(self.clients)
        
        # Construct routes while clients remain and routes available
        for _ in range(self.m):
            if not available_clients:
                break
            
            route = self._construct_route(available_clients)
            if len(route) > 2:
                solution.append(route)
                # Remove used clients
                available_clients -= set(route[1:-1])
        
        # Calculate solution quality
        solution_quality = self._calculate_solution_quality(solution)
        
        # Update best solution if improved
        if solution_quality > self.best_fitness:
            self.best_fitness = solution_quality
            self.best_solution = solution
        
        # Update stats for consistency with other algorithms
        self.stats_data['best_fitness'] = [self.best_fitness]
        self.stats_data['avg_fitness'] = [self.best_fitness]
        
        if self.debug:
            print(f"\nGreedy Solution Quality: {self.best_fitness:.2f}")
            print(f"Total Clients Served: {sum(len(route)-2 for route in solution)}")
            print(f"Number of Routes: {len(solution)}")
        
        return self.best_solution
    
    def get_stats(self):
        """Return collected statistics as a pandas DataFrame."""
        return pd.DataFrame(self.stats_data)

######################################################################################################################################################################################################
class AntColonyTOP:
    def __init__(self, start_point, end_point, clients, m, L, debug=False):
        self.start_point = start_point
        self.end_point = end_point
        self.clients = clients
        self.m = m
        self.L = L
        self.debug = debug
        n = len(clients)
        self.n_colonies = min(3, m)
        self.n_ants_per_colony = min(30, n)
        self.max_iterations = min(500, n * 10)
        self.colony_params = [{'alpha': 1.0, 'beta': 2.0, 'rho': 0.1}, {'alpha': 2.0, 'beta': 1.0, 'rho': 0.15}, {'alpha': 0.5, 'beta': 2.5, 'rho': 0.05}]
        self.q0_initial = 0.7
        self.q0_final = 0.9
        self.local_search_freq = max(5, n // 20)
        self.max_stagnation = max(100, n // 2)
        self.tau_max = 2.0
        self.tau_min = self.tau_max * 0.01
        self.pheromone_matrices = self._initialize_pheromone_matrices()
        self.eta = self._initialize_heuristic()
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.iteration_best_solutions = []
        self.stats_data = {'iteration': [], 'best_fitness': [], 'avg_fitness': [], 'diversity': [], 'pheromone_avg': [], 'pheromone_max': [], 'pheromone_min': [], 'colony_performance': [[] for _ in range(self.n_colonies)]}

    def _initialize_pheromone_matrices(self):
        matrices = []
        all_points = [self.start_point] + self.clients + [self.end_point]
        for _ in range(self.n_colonies):
            tau = {}
            for i in all_points:
                tau[i] = {}
                for j in all_points:
                    if i != j:
                        tau[i][j] = self.tau_max
            matrices.append(tau)
        return matrices

    def _initialize_heuristic(self):
        eta = {}
        all_points = [self.start_point] + self.clients + [self.end_point]
        max_profit = max((c.profit for c in self.clients), default=1)
        center_x = statistics.mean(c.x for c in self.clients)
        center_y = statistics.mean(c.y for c in self.clients)
        max_dist = max(math.sqrt((c.x - center_x)**2 + (c.y - center_y)**2) for c in self.clients)
        for i in all_points:
            eta[i] = {}
            for j in all_points:
                if i != j:
                    dist = math.sqrt((i.x - j.x)**2 + (i.y - j.y)**2)
                    if j in self.clients:
                        profit_factor = j.profit / max_profit
                        distance_factor = 1.0 / (dist if dist > 0 else 0.1)
                        centrality = 1.0 - (math.sqrt((j.x - center_x)**2 + (j.y - center_y)**2) / max_dist)
                        eta[i][j] = (1.0 + profit_factor) * distance_factor * (1.0 + 0.2 * centrality)
                    else:
                        eta[i][j] = 1.0 / (dist if dist > 0 else 0.1)
        return eta

    def _select_next_client(self, ant_route, current, available_clients, current_time, colony_idx, iteration):
        if not available_clients:
            return self.end_point
        q0 = self.q0_initial + (self.q0_final - self.q0_initial) * (iteration / self.max_iterations)
        params = self.colony_params[colony_idx]
        feasible = []
        for client in available_clients:
            time_to_client = math.sqrt((current.x - client.x)**2 + (current.y - client.y)**2)
            time_to_end = math.sqrt((client.x - self.end_point.x)**2 + (client.y - self.end_point.y)**2)
            if current_time + time_to_client + time_to_end <= self.L:
                route_position = len(ant_route) / self.L
                score = (self.tau_max if time_to_client + time_to_end <= self.L * 0.3 else self.pheromone_matrices[colony_idx][current][client]**params['alpha'] * self.eta[current][client]**params['beta'] * (1 + 0.2 * (1 - route_position)))
                feasible.append((client, score))
        if not feasible:
            return self.end_point
        if random.random() < q0:
            return max(feasible, key=lambda x: x[1])[0]
        else:
            total = sum(score for _, score in feasible)
            if total == 0:
                return random.choice([client for client, _ in feasible])
            r = random.random() * total
            cum_prob = 0
            for client, score in feasible:
                cum_prob += score
                if cum_prob >= r:
                    return client
            return feasible[-1][0]

    def _construct_route(self, colony_idx, iteration, used_clients=None):
        route = [self.start_point]
        current = self.start_point
        current_time = 0
        if used_clients is None:
            used_clients = set()
        available = set(self.clients) - used_clients
        while available:
            next_client = self._select_next_client(route, current, available, current_time, colony_idx, iteration)
            if next_client == self.end_point:
                break
            route.append(next_client)
            time_to_next = math.sqrt((current.x - next_client.x)**2 + (current.y - next_client.y)**2)
            current_time += time_to_next
            current = next_client
            available.remove(next_client)
            self.pheromone_matrices[colony_idx][route[-2]][route[-1]] *= (1 - self.colony_params[colony_idx]['rho'])
        route.append(self.end_point)
        return route

    def _construct_solution(self, colony_idx, iteration):
        solution = []
        used_clients = set()
        for _ in range(self.m):
            if not (set(self.clients) - used_clients):
                break
            route = self._construct_route(colony_idx, iteration, used_clients)
            if len(route) > 2:
                solution.append(route)
                used_clients.update(set(route[1:-1]))
        return solution

    def _local_search(self, solution):
        if not solution:
            return solution
        max_local_time = 120
        start_time = time.time()
        max_local_iterations = 100
        local_iter = 0
        improved = True
        while improved and local_iter < max_local_iterations:
            if time.time() - start_time > max_local_time:
                if self.debug:
                    print("\nLocal search stopped due to time limit")
                break
            local_iter += 1
            improved = False
            
            # First phase: Systematic 2-opt for best routes
            for route_idx, route in enumerate(solution):
                if len(route) <= 4:
                    continue
                route_profit = sum(c.profit for c in route[1:-1])
                if route_profit < max(sum(c.profit for c in r[1:-1]) for r in solution) * 0.7:
                    continue  # Skip low profit routes for systematic search
                
                best_length = sum(math.sqrt((route[i].x - route[i+1].x)**2 + (route[i].y - route[i+1].y)**2) for i in range(len(route)-1))
                for i in range(1, len(route)-2):
                    if time.time() - start_time > max_local_time:
                        break
                    for j in range(i+1, min(i+5, len(route)-1)):  # Limited window size
                        new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                        new_length = sum(math.sqrt((new_route[k].x - new_route[k+1].x)**2 + (new_route[k].y - new_route[k+1].y)**2) for k in range(len(new_route)-1))
                        if new_length < best_length and new_length <= self.L:
                            solution[route_idx] = new_route
                            improved = True
                            break
                    if improved:
                        break
                        
            # Second phase: Random exchange between profitable routes
            if not improved:
                max_attempts = 20
                routes_profits = [(i, sum(c.profit for c in route[1:-1])) for i, route in enumerate(solution)]
                routes_profits.sort(key=lambda x: x[1], reverse=True)
                top_routes = routes_profits[:max(2, len(solution)//2)]
                
                for idx1, _ in top_routes:
                    if improved:
                        break
                    route1 = solution[idx1]
                    if len(route1) <= 3:
                        continue
                        
                    for idx2, _ in top_routes:
                        if idx1 == idx2:
                            continue
                        route2 = solution[idx2]
                        if len(route2) <= 3:
                            continue
                            
                        attempts = 0
                        while attempts < max_attempts and not improved:
                            attempts += 1
                            # Try to exchange the most profitable clients
                            clients1 = [(pos, c.profit) for pos, c in enumerate(route1[1:-1], 1)]
                            clients2 = [(pos, c.profit) for pos, c in enumerate(route2[1:-1], 1)]
                            clients1.sort(key=lambda x: x[1], reverse=True)
                            clients2.sort(key=lambda x: x[1], reverse=True)
                            
                            for (pos1, _) in clients1[:3]:  # Try top 3 profitable positions
                                for (pos2, _) in clients2[:3]:
                                    new_route1 = route1.copy()
                                    new_route2 = route2.copy()
                                    new_route1[pos1], new_route2[pos2] = new_route2[pos2], new_route1[pos1]
                                    
                                    length1 = sum(math.sqrt((new_route1[k].x - new_route1[k+1].x)**2 + (new_route1[k].y - new_route1[k+1].y)**2) for k in range(len(new_route1)-1))
                                    length2 = sum(math.sqrt((new_route2[k].x - new_route2[k+1].x)**2 + (new_route2[k].y - new_route2[k+1].y)**2) for k in range(len(new_route2)-1))
                                    
                                    if length1 <= self.L and length2 <= self.L:
                                        old_profit = sum(c.profit for c in route1[1:-1]) + sum(c.profit for c in route2[1:-1])
                                        new_profit = sum(c.profit for c in new_route1[1:-1]) + sum(c.profit for c in new_route2[1:-1])
                                        
                                        if new_profit >= old_profit * 0.95:  # Accept if not much worse
                                            solution[idx1] = new_route1
                                            solution[idx2] = new_route2
                                            improved = True
                                            break
                                if improved:
                                    break
                            if improved:
                                break
        return solution

    def _calculate_solution_quality(self, solution):
        if not solution:
            return 0
        total_profit = 0
        total_distance = 0
        used_clients = set()
        route_loads = []
        
        for route in solution:
            route_profit = 0
            route_distance = 0
            route_clients = set()
            
            for i in range(len(route) - 1):
                route_distance += math.sqrt((route[i].x - route[i+1].x)**2 + (route[i].y - route[i+1].y)**2)
                if i > 0:
                    client = route[i]
                    if client.id not in used_clients:
                        route_profit += client.profit
                        route_clients.add(client.id)
                        
            if route_distance <= self.L:
                total_profit += route_profit
                total_distance += route_distance
                used_clients.update(route_clients)
                route_loads.append(len(route_clients))
                
        if not route_loads or total_distance > self.L * self.m:
            return 0
            
        coverage_ratio = len(used_clients) / len(self.clients)
        time_efficiency = 1 - (total_distance / (self.L * self.m))
        
        # Handle load balance calculation for different numbers of routes
        if len(route_loads) < 2:
            load_balance = 1.0  # Perfect balance when only one route
        else:
            try:
                load_balance = 1 - (statistics.stdev(route_loads) / (max(route_loads) if route_loads else 1))
            except statistics.StatisticsError:
                load_balance = 1.0  # Fallback if stdev calculation fails
                
        quality = (total_profit * 
                (1 + 0.2 * coverage_ratio) * 
                (1 + 0.1 * time_efficiency) * 
                (1 + 0.1 * load_balance))
                
        return quality

    def _update_pheromone(self, colony_idx, iteration):
        params = self.colony_params[colony_idx]
        decay = 1 - params['rho']
        for i in self.pheromone_matrices[colony_idx]:
            for j in self.pheromone_matrices[colony_idx][i]:
                self.pheromone_matrices[colony_idx][i][j] *= decay
        if self.iteration_best_solutions:
            best_solution = self.iteration_best_solutions[-1]
            for route in best_solution:
                for i in range(len(route)-1):
                    deposit = 1.0 / (1.0 + len(route))
                    self.pheromone_matrices[colony_idx][route[i]][route[i+1]] += params['rho'] * deposit
                    self.pheromone_matrices[colony_idx][route[i+1]][route[i]] = self.pheromone_matrices[colony_idx][route[i]][route[i+1]]
        for i in self.pheromone_matrices[colony_idx]:
            for j in self.pheromone_matrices[colony_idx][i]:
                self.pheromone_matrices[colony_idx][i][j] = min(self.tau_max, max(self.tau_min, self.pheromone_matrices[colony_idx][i][j]))

    def _diversification(self, solution):
        if not solution:
            return solution
        for route in solution:
            if len(route) > 4 and random.random() < 0.3:
                i = random.randrange(1, len(route)-1)
                candidates = [c for c in self.clients if c not in route]
                if candidates:
                    route[i] = random.choice(candidates)
        return solution

    def _calculate_diversity(self, solutions):
        if not solutions:
            return 0
        diversity = 0
        comparisons = 0
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
        stagnation_counter = 0
        colony_solutions = [[] for _ in range(self.n_colonies)]
        colony_qualities = [[] for _ in range(self.n_colonies)]
        for iteration in range(self.max_iterations):
            iteration_solutions = []
            iteration_qualities = []
            for colony_idx in range(self.n_colonies):
                colony_best_solution = None
                colony_best_quality = float('-inf')
                for ant in range(self.n_ants_per_colony):
                    solution = self._construct_solution(colony_idx, iteration)
                    if iteration % self.local_search_freq == 0:
                        solution = self._local_search(solution)
                    if random.random() < 0.1:
                        solution = self._diversification(solution)
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
            avg_fitness = statistics.mean(iteration_qualities)
            diversity = self._calculate_diversity(iteration_solutions)
            pheromone_values = [v for matrix in self.pheromone_matrices for d in matrix.values() for v in d.values()]
            self.stats_data['iteration'].append(iteration)
            self.stats_data['best_fitness'].append(self.best_fitness)
            self.stats_data['avg_fitness'].append(avg_fitness)
            self.stats_data['diversity'].append(diversity)
            self.stats_data['pheromone_avg'].append(statistics.mean(pheromone_values))
            self.stats_data['pheromone_max'].append(max(pheromone_values))
            self.stats_data['pheromone_min'].append(min(pheromone_values))
            for colony_idx in range(self.n_colonies):
                self.stats_data['colony_performance'][colony_idx].append(colony_qualities[colony_idx][-1])
            if self.debug and iteration % 10 == 0:
                print(f"\nIteration {iteration}:")
                print(f"  Best Fitness = {self.best_fitness:.2f}")
                print(f"  Average Fitness = {avg_fitness:.2f}")
                print(f"  Diversity = {diversity:.3f}")
                print(f"  Colony performances = {[qualities[-1] for qualities in colony_qualities]}")
            if stagnation_counter > self.max_stagnation and iteration > self.max_iterations // 4:
                if self.debug:
                    print(f"\nEarly stopping at iteration {iteration}")
                    print(f"No improvement for {stagnation_counter} iterations")
                break
        return self.best_solution

    def get_stats(self):
        # First, find the minimum length across all arrays
        min_length = min(
            len(self.stats_data['iteration']),
            len(self.stats_data['best_fitness']),
            len(self.stats_data['avg_fitness']),
            len(self.stats_data['diversity']),
            len(self.stats_data['pheromone_avg']),
            len(self.stats_data['pheromone_max']),
            len(self.stats_data['pheromone_min'])
        )
        
        # Truncate all arrays to the minimum length
        stats_dict = {
            'iteration': self.stats_data['iteration'][:min_length],
            'best_fitness': self.stats_data['best_fitness'][:min_length],
            'avg_fitness': self.stats_data['avg_fitness'][:min_length],
            'diversity': self.stats_data['diversity'][:min_length],
            'pheromone_avg': self.stats_data['pheromone_avg'][:min_length],
            'pheromone_max': self.stats_data['pheromone_max'][:min_length],
            'pheromone_min': self.stats_data['pheromone_min'][:min_length]
        }
        
        # Add colony performance data
        for i in range(self.n_colonies):
            colony_data = self.stats_data['colony_performance'][i][:min_length]
            # Pad with NaN if necessary
            if len(colony_data) < min_length:
                colony_data = colony_data + [float('nan')] * (min_length - len(colony_data))
            stats_dict[f'colony_{i}_performance'] = colony_data
        
        # Create DataFrame with synchronized lengths
        stats_df = pd.DataFrame(stats_dict)
        return stats_df




def lire_instance_chao(nom_fichier):
    with open(nom_fichier, "r") as f:
        lines = f.readlines()
        
    # Read m and tmax from second and third line
    m = int(lines[1].split()[1])  # gets P value
    L = float(lines[2].split()[1])  # gets Tmax value
    
    # Read points (skip first 3 header lines)
    points = []
    for i, line in enumerate(lines[3:], 0):  # start index at 0
        x, y, score = map(float, line.split())
        points.append(Client(i, x, y, score))  # i is the ID
    
    return points[0], points[-1], points[1:-1], m, L  # start, end, clients, m, L

def visualize_solution(solution, start_point, end_point, clients, filename):
    plt.figure(figsize=(12, 8))
    
    # Create sets of visited and unvisited clients
    visited_clients = {c.id for route in solution for c in route[1:-1]}
    unvisited_clients = [c for c in clients if c.id not in visited_clients]
    
    # Plot unvisited clients
    if unvisited_clients:
        plt.scatter([c.x for c in unvisited_clients], 
                   [c.y for c in unvisited_clients], 
                   c='gray', alpha=0.5, label='Unvisited Clients')
    
    # Plot start and end points
    plt.scatter(start_point.x, start_point.y, c='green', marker='s', s=100, label='Start')
    plt.scatter(end_point.x, end_point.y, c='red', marker='s', s=100, label='End')
    
    # Plot routes and their visited clients
    colors = plt.cm.rainbow(np.linspace(0, 1, len(solution)))
    for i, (route, color) in enumerate(zip(solution, colors)):
        route_x = [c.x for c in route]
        route_y = [c.y for c in route]
        
        # Plot route path
        plt.plot(route_x, route_y, c=color, linewidth=2)
        
        # Plot visited clients for this route
        if len(route) > 2:
            plt.scatter(route_x[1:-1], route_y[1:-1], 
                       c=[color], s=100, 
                       label=f'Route {i+1} ({len(route)-2} clients)')
    
    plt.title('TOP Solution Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_algorithm_stats(stats_data, algorithm_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Fitness Evolution
    plt.figure(figsize=(12, 6))
    plt.plot(stats_data['iteration'], stats_data['best_fitness'], 
             label='Best Fitness', linewidth=2)
    plt.plot(stats_data['iteration'], stats_data['avg_fitness'], 
             label='Average Fitness', linewidth=2, alpha=0.7)
    plt.title(f'{algorithm_name} Fitness Evolution', pad=20)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{algorithm_name.lower()}_fitness.png')
    plt.close()
    
    # Diversity Evolution
    plt.figure(figsize=(12, 6))
    plt.plot(stats_data['iteration'], stats_data['diversity'], 
             color='purple', linewidth=2)
    plt.title(f'{algorithm_name} Population Diversity', pad=20)
    plt.xlabel('Iteration')
    plt.ylabel('Diversity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{algorithm_name.lower()}_diversity.png')
    plt.close()
    
    # Algorithm-specific plots
    if 'mutation_rate' in stats_data:  # Genetic Algorithm
        plt.figure(figsize=(12, 6))
        plt.plot(stats_data['iteration'], stats_data['mutation_rate'], 
                 color='orange', linewidth=2)
        plt.title(f'{algorithm_name} Mutation Rate Evolution', pad=20)
        plt.xlabel('Iteration')
        plt.ylabel('Mutation Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{algorithm_name.lower()}_mutation_rate.png')
        plt.close()
    
    elif 'pheromone_avg' in stats_data:  # Ant Colony
        plt.figure(figsize=(12, 6))
        plt.plot(stats_data['iteration'], stats_data['pheromone_max'], 
                 label='Max', linewidth=2)
        plt.plot(stats_data['iteration'], stats_data['pheromone_avg'], 
                 label='Average', linewidth=2)
        plt.plot(stats_data['iteration'], stats_data['pheromone_min'], 
                 label='Min', linewidth=2)
        plt.title(f'{algorithm_name} Pheromone Levels Evolution', pad=20)
        plt.xlabel('Iteration')
        plt.ylabel('Pheromone Level')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{algorithm_name.lower()}_pheromone.png')
        plt.close()


def main(instance_file, debug=False):
    # Read instance
    start_point, end_point, clients, m, L = lire_instance_chao(instance_file)

    # Extract instance size from filename
    instance_name = f"{instance_file.split('/')[-1].replace('.txt', '')}_{m}_{L}"
    
    # Create output directory with instance size
    output_dir = f'results_{instance_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run algorithms and collect results
    algorithms = [
        ('GreedyTOP', GreedyTOP(start_point, end_point, clients, m, L, debug=debug)),
        ('AntColonyTOP', AntColonyTOP(start_point, end_point, clients, m, L, debug=debug))
        # ('SimulatedAnnealingTOP', SimulatedAnnealingTOP(start_point, end_point, clients, m, L, debug=debug)),
        # ('GeneticTOP', GeneticTOP(start_point, end_point, clients, m, L, debug=debug))
    ]
    
    results = []
    for name, algorithm in algorithms:
        print(f"\nRunning {name}...")
        
        start_time = time.time()
        # if isinstance(algorithm, GeneticTOP):
        #     solution = algorithm.evolve()
        # else:
        #     solution = algorithm.solve()
        solution = algorithm.solve()
        execution_time = time.time() - start_time
        
        # Get algorithm-specific statistics
        stats_df = algorithm.get_stats()
        
        # Filter for every 10th iteration
        stats_df = stats_df[stats_df['iteration'] % 10 == 0].copy()
        
        # Add instance information
        stats_df['Instance_Size'] = L
        stats_df['Instance_File'] = instance_file
        stats_df['Timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to algorithm-specific CSV
        stats_file = f'{name.lower()}_stats.csv'
        
        if os.path.exists(stats_file):
            stats_df.to_csv(stats_file, mode='a', header=False, index=False)
        else:
            stats_df.to_csv(stats_file, index=False)
        
        # Client uniqueness check
        visited_clients = set()
        duplicate_clients = set()
        clients_per_route = []
        
        for route in solution:
            route_clients = set()
            for client in route[1:-1]:  # Exclude start and end points
                if client.id in visited_clients:
                    duplicate_clients.add(client.id)
                visited_clients.add(client.id)
                route_clients.add(client.id)
            clients_per_route.append(len(route_clients))
        
        # Calculate client-related metrics
        total_unique_clients = len(visited_clients)
        total_duplicate_clients = len(duplicate_clients)
        clients_visited_multiple_times = list(duplicate_clients)
        avg_clients_per_route = statistics.mean(clients_per_route) if clients_per_route else 0
        max_clients_per_route = max(clients_per_route) if clients_per_route else 0
        min_clients_per_route = min(clients_per_route) if clients_per_route else 0
        
        # Calculate original metrics
        total_profit = sum(sum(c.profit for c in route[1:-1]) for route in solution)
        total_clients = sum(len(route)-2 for route in solution)  # This might include duplicates
        total_distance = sum(sum(math.sqrt((route[i].x - route[i+1].x)**2 + 
                                         (route[i].y - route[i+1].y)**2)
                              for i in range(len(route)-1))
                           for route in solution)
        
        # Calculate route times
        route_times = []
        route_profits = []
        routes_exceeding_L = 0
        max_route_time = 0
        min_route_time = float('inf')
        
        for route in solution:
            route_time = sum(math.sqrt((route[i].x - route[i+1].x)**2 + 
                                     (route[i].y - route[i+1].y)**2)
                           for i in range(len(route)-1))
            route_profit = sum(c.profit for c in route[1:-1])
            
            route_times.append(route_time)
            route_profits.append(route_profit)
            
            if route_time > L:
                routes_exceeding_L += 1
            
            max_route_time = max(max_route_time, route_time)
            min_route_time = min(min_route_time, route_time) if route_time > 0 else min_route_time
        
        # Calculate route time statistics
        avg_route_time = statistics.mean(route_times) if route_times else 0
        route_time_std = statistics.stdev(route_times) if len(route_times) > 1 else 0
        
        # Visualize solution and stats
        visualize_solution(solution, start_point, end_point, clients,
                         f'{output_dir}/{name.lower()}_solution.png')
        plot_algorithm_stats(algorithm.get_stats(), name, output_dir)
        
        # Store results for overall history
        results.append({
            'Instance_Size': L,
            'Algorithm': name,
            'Execution_Time': execution_time,
            'Final_Fitness': algorithm.best_fitness,
            'Total_Profit': total_profit,
            'Total_Available_Clients': len(clients),
            'Total_Unique_Clients_Visited': total_unique_clients,
            'Total_Duplicate_Clients': total_duplicate_clients,
            'Uniqueness_Constraint_Respected': total_duplicate_clients == 0,
            'Client_Coverage_Ratio': total_unique_clients / len(clients),
            'Avg_Clients_Per_Route': avg_clients_per_route,
            'Max_Clients_Per_Route': max_clients_per_route,
            'Min_Clients_Per_Route': min_clients_per_route,
            'Routes': len(solution),
            'Time_Limit_L': L,
            'Max_Route_Time': max_route_time,
            'Min_Route_Time': min_route_time if min_route_time != float('inf') else 0,
            'Avg_Route_Time': avg_route_time,
            'Route_Time_Std': route_time_std,
            'Routes_Exceeding_L': routes_exceeding_L,
            'Time_Constraint_Respected': routes_exceeding_L == 0,
            'Route_Time_Usage': avg_route_time / L if L > 0 else 0,
            'Time_Efficiency': 1 - (avg_route_time / L) if L > 0 else 0,
            'All_Constraints_Respected': (routes_exceeding_L == 0 and 
                            total_duplicate_clients == 0 and 
                            len(solution) <= m),
            'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'Instance_File': instance_file
        })
        
        print(f"Completed {name}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Final fitness: {algorithm.best_fitness:.2f}")
        print(f"Total profit: {total_profit}")
        print(f"Unique clients served: {total_unique_clients}/{len(clients)}")
        print(f"Duplicate clients: {total_duplicate_clients}")
        print(f"Number of routes: {len(solution)}")
        print(f"Time constraint (L): {L}")
        print(f"Maximum route time: {max_route_time:.2f}")
        print(f"Average route time: {avg_route_time:.2f}")
        print(f"Routes exceeding time limit: {routes_exceeding_L}")
        print(f"All constraints respected: {routes_exceeding_L == 0 and total_duplicate_clients == 0}")
        
    # Append results to main history CSV
    results_df = pd.DataFrame(results)
    history_file = 'algorithm_history.csv'
    
    if os.path.exists(history_file):
        results_df.to_csv(history_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(history_file, index=False)
    
    print(results_df)
    print(f"\nResults appended to {history_file}")
    print("Individual algorithm statistics saved to respective CSV files")

def main_for_all_instances(folder_pattern, debug=False):
        instance_files = glob.glob(folder_pattern)
        for instance_file in instance_files:
            main(instance_file, debug)

if __name__ == "__main__":
     non_symmetric_files = {
         "small": [
             "Set_100_234/p4.2.a.txt",
             "Set_100_234/p4.2.j.txt",
             "Set_100_234/p4.2.t.txt"
         ],
         "medium": [
             "Set_100_234/p4.3.a.txt",
             "Set_100_234/p4.3.j.txt",
             "Set_100_234/p4.3.t.txt"
         ],
         "big": [
             "Set_100_234/p4.4.a.txt",
             "Set_100_234/p4.4.j.txt",
             "Set_100_234/p4.4.t.txt"
         ]
     }
     # Symmetric test set
     symmetric_files = {
         "small": [
             "Set_64_234/p6.2.d.txt",
             "Set_64_234/p6.2.i.txt",
             "Set_64_234/p6.2.n.txt"
         ],
         "medium": [
             "Set_64_234/p6.3.d.txt",
             "Set_64_234/p6.3.i.txt",
             "Set_64_234/p6.3.n.txt"
         ],
         "high": [
             "Set_64_234/p6.4.d.txt",
             "Set_64_234/p6.4.i.txt",
             "Set_64_234/p6.4.n.txt"
         ]
     }
     # Run main for all instance files
    #  for size, files in non_symmetric_files.items():
    #      print(f"\nRunning non-symmetric test set ({size} time constraint instances):")
    #      for instance_file in files:
    #          main(instance_file, debug=True)
     for size, files in symmetric_files.items():
         print(f"\nRunning symmetric test set ({size} time constraint instances):")
         for instance_file in files:
             main(instance_file, debug=True)


