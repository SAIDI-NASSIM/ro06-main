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
    
###################################################################################################################################################
class SimulatedAnnealingTOP:
    def __init__(self, start_point, end_point, clients, m, L, debug=False):
        self.start_point = start_point
        self.end_point = end_point
        self.clients = clients
        self.m = m
        self.L = L
        self.debug = debug
        
        n = len(clients)
        # Temperature parameters
        self.initial_temp = 100.0
        self.final_temp = 0.01
        self.cooling_rate = 0.97
        self.iterations_per_temp = min(100, n * 2)
        self.max_iterations = min(500, n * 10)
        
        # Solution tracking
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.current_solution = None
        self.current_fitness = float('-inf')
        
        # Precompute distances for efficiency
        self.distances = self._precompute_distances()
        
        # Statistics collection
        self.stats_data = {
            'iteration': [],
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'temperature': [],
            'acceptance_rate': [],
            'iterations_without_improvement': []
        }
        
        # Performance tracking
        self.iterations_without_improvement = 0
        self.max_stagnation = max(100, n // 2)
        self.accepted_moves = 0
        self.total_moves = 0
        
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
    
    def _create_initial_solution(self):
        """Create initial solution using greedy approach with randomization."""
        solution = []
        available_clients = set(self.clients)
        
        for _ in range(self.m):
            if not available_clients:
                break
                
            route = [self.start_point]
            current_time = 0
            
            while available_clients:
                candidates = []
                for client in available_clients:
                    new_time = (current_time + 
                              self.get_distance(route[-1], client) + 
                              self.get_distance(client, self.end_point))
                    if new_time <= self.L:
                        score = client.profit / (self.get_distance(route[-1], client) + 0.1)
                        candidates.append((client, score))
                
                if not candidates:
                    break
                
                # Select client with probability proportional to score
                total_score = sum(score for _, score in candidates)
                if total_score <= 0:
                    break
                    
                r = random.random() * total_score
                cumsum = 0
                selected = None
                
                for client, score in candidates:
                    cumsum += score
                    if cumsum >= r:
                        selected = client
                        break
                
                if not selected:
                    selected = candidates[-1][0]
                
                route.append(selected)
                available_clients.remove(selected)
                current_time += self.get_distance(route[-2], selected)
            
            route.append(self.end_point)
            if len(route) > 2:
                solution.append(route)
        
        return solution

    def _calculate_solution_quality(self, solution):
        """Calculate solution quality with penalties and bonuses."""
        if not solution:
            return 0
        
        total_profit = sum(sum(c.profit for c in route[1:-1]) for route in solution)
        
        # Time constraint penalty
        total_time = sum(sum(self.get_distance(route[i], route[i+1]) 
                        for i in range(len(route)-1)) for route in solution)
        time_penalty = 0
        if total_time > self.L:
            excess = total_time - self.L
            time_penalty = (excess * 2) + (excess ** 2)
        
        # Route balance penalty
        route_profits = [sum(c.profit for c in route[1:-1]) for route in solution]
        if route_profits:
            profit_std = statistics.stdev(route_profits) if len(route_profits) > 1 else 0
            balance_penalty = profit_std * 0.2
        else:
            balance_penalty = 0
        
        # Coverage bonus
        total_clients = sum(len(route)-2 for route in solution)
        coverage_ratio = total_clients / len(self.clients)
        coverage_bonus = total_profit * coverage_ratio * 0.1
        
        return max(0, total_profit - time_penalty - balance_penalty + coverage_bonus)

    def _generate_neighbor(self, solution):
        """Generate neighboring solution that maintains client uniqueness."""
        neighbor = copy.deepcopy(solution)
        if not neighbor:
            return neighbor
                
        # Select random operation
        operation = random.choice(['swap', 'insert', 'reverse', 'exchange'])
        
        if operation == 'swap':
            # Swap two random clients within a route (maintains uniqueness)
            route_idx = random.randrange(len(neighbor))
            route = neighbor[route_idx]
            if len(route) > 3:  # Need at least 2 clients to swap
                i, j = random.sample(range(1, len(route)-1), 2)
                route[i], route[j] = route[j], route[i]
        
        elif operation == 'insert':
            # Move a client to a new position (maintains uniqueness as just moving)
            if len(neighbor) > 1:
                route1_idx = random.randrange(len(neighbor))
                route1 = neighbor[route1_idx]
                if len(route1) > 3:  # Need at least one client to move
                    client_idx = random.randrange(1, len(route1)-1)
                    client = route1.pop(client_idx)
                    
                    route2_idx = random.randrange(len(neighbor))
                    route2 = neighbor[route2_idx]
                    insert_pos = random.randrange(1, len(route2))
                    route2.insert(insert_pos, client)
                    
                    # Remove empty routes
                    if len(route1) <= 2:
                        neighbor.remove(route1)
        
        elif operation == 'reverse':
            # Reverse a segment within a route (maintains uniqueness)
            route_idx = random.randrange(len(neighbor))
            route = neighbor[route_idx]
            if len(route) > 4:
                i = random.randrange(1, len(route)-2)
                j = random.randrange(i+1, len(route)-1)
                route[i:j+1] = reversed(route[i:j+1])
        
        else:  # exchange
            # Exchange single clients between routes (maintains uniqueness)
            if len(neighbor) > 1:
                i, j = random.sample(range(len(neighbor)), 2)
                route1, route2 = neighbor[i], neighbor[j]
                if len(route1) > 3 and len(route2) > 3:
                    pos1 = random.randrange(1, len(route1)-1)
                    pos2 = random.randrange(1, len(route2)-1)
                    route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
        
        return neighbor
    
    def _calculate_acceptance_probability(self, current_fitness, new_fitness, temperature):
        """Calculate probability of accepting worse solution."""
        if new_fitness > current_fitness:
            return 1.0
        return math.exp((new_fitness - current_fitness) / temperature)
    
    def _calculate_diversity_metric(self, solution1, solution2):
        """Calculate diversity between two solutions."""
        if not solution1 or not solution2:
            return 1.0
            
        clients1 = set(c.id for route in solution1 for c in route[1:-1])
        clients2 = set(c.id for route in solution2 for c in route[1:-1])
        
        if not clients1 and not clients2:
            return 0.0
            
        intersection = len(clients1.intersection(clients2))
        union = len(clients1.union(clients2))
        
        return 1.0 - (intersection / union if union > 0 else 0)
    
    def solve(self):
        """Main simulated annealing algorithm."""
        # Initialize solution
        self.current_solution = self._create_initial_solution()
        self.current_fitness = self._calculate_solution_quality(self.current_solution)
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_fitness = self.current_fitness
        
        temperature = self.initial_temp
        iteration = 0
        
        while temperature > self.final_temp and iteration < self.max_iterations:
            accepted_at_temp = 0
            
            for _ in range(self.iterations_per_temp):
                # Generate and evaluate neighbor
                neighbor = self._generate_neighbor(self.current_solution)
                neighbor_fitness = self._calculate_solution_quality(neighbor)
                
                # Calculate acceptance probability
                acceptance_prob = self._calculate_acceptance_probability(
                    self.current_fitness, neighbor_fitness, temperature)
                
                self.total_moves += 1
                
                # Accept or reject neighbor
                if random.random() < acceptance_prob:
                    self.current_solution = neighbor
                    self.current_fitness = neighbor_fitness
                    accepted_at_temp += 1
                    self.accepted_moves += 1
                    
                    # Update best solution
                    if neighbor_fitness > self.best_fitness:
                        self.best_solution = copy.deepcopy(neighbor)
                        self.best_fitness = neighbor_fitness
                        self.iterations_without_improvement = 0
                    else:
                        self.iterations_without_improvement += 1
                
                # Collect statistics
                self.stats_data['iteration'].append(iteration)
                self.stats_data['best_fitness'].append(self.best_fitness)
                self.stats_data['avg_fitness'].append(self.current_fitness)
                self.stats_data['temperature'].append(temperature)
                self.stats_data['acceptance_rate'].append(self.accepted_moves / max(1, self.total_moves))
                self.stats_data['iterations_without_improvement'].append(self.iterations_without_improvement)
                
                if self.current_solution and self.best_solution:
                    diversity = self._calculate_diversity_metric(
                        self.current_solution, self.best_solution)
                    self.stats_data['diversity'].append(diversity)
                else:
                    self.stats_data['diversity'].append(0)
                
                # Debug output
                if self.debug and iteration % 10 == 0:
                    print(f"\nIteration {iteration}:")
                    print(f"  Temperature: {temperature:.2f}")
                    print(f"  Best Fitness: {self.best_fitness:.2f}")
                    print(f"  Current Fitness: {self.current_fitness:.2f}")
                    print(f"  Acceptance Rate: {self.accepted_moves/max(1, self.total_moves):.3f}")
                
                iteration += 1
                
                # Early stopping check
                if (self.iterations_without_improvement > self.max_stagnation and
                    iteration > self.max_iterations // 4):
                    if self.debug:
                        print(f"\nEarly stopping at iteration {iteration}")
                        print(f"No improvement for {self.iterations_without_improvement} iterations")
                    return self.best_solution
            
            # Cool down
            temperature *= self.cooling_rate
        
        return self.best_solution
    
    def get_stats(self):
        """Return collected statistics as a pandas DataFrame."""
        return pd.DataFrame(self.stats_data)
#######################################################################################################################################################
class AntColonyTOP:
    def __init__(self, start_point, end_point, clients, m, L, debug=False):
        self.start_point = start_point
        self.end_point = end_point
        self.clients = clients
        self.m = m
        self.L = L
        self.debug = debug
        n = len(clients)
        self.n_ants = min(50, n * 2)
        self.max_iterations = min(500, n * 10)
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.1
        self.q0 = 0.7
        self.local_search_freq = 20
        self.max_stagnation = max(100, n // 2)
        self.tau_max = 1.0
        self.tau_min = self.tau_max * 0.01
        self.tau = self._initialize_pheromones()
        self.eta = self._initialize_heuristic()
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.stats_data = {
            'iteration': [], 'best_fitness': [], 'avg_fitness': [],
            'diversity': [], 'pheromone_avg': [], 'pheromone_max': [],
            'pheromone_min': []
        }

    def _initialize_pheromones(self):
        tau = {}
        all_points = [self.start_point] + self.clients + [self.end_point]
        for i in all_points:
            tau[i] = {}
            for j in all_points:
                if i != j:
                    tau[i][j] = self.tau_max
        return tau

    def _initialize_heuristic(self):
        eta = {}
        all_points = [self.start_point] + self.clients + [self.end_point]
        max_profit = max((c.profit for c in self.clients), default=1)
        for i in all_points:
            eta[i] = {}
            for j in all_points:
                if i != j:
                    dist = math.sqrt((i.x - j.x)**2 + (i.y - j.y)**2)
                    profit = j.profit/max_profit if j in self.clients else 0
                    eta[i][j] = (1.0 + profit) / (dist if dist > 0 else 0.1)
        return eta

    def _select_next_client(self, ant_route, current, available_clients, current_time):
        if not available_clients:
            return self.end_point
        feasible = []
        for client in available_clients:
            time_to_client = math.sqrt((current.x - client.x)**2 + (current.y - client.y)**2)
            time_to_end = math.sqrt((client.x - self.end_point.x)**2 + (client.y - self.end_point.y)**2)
            if current_time + time_to_client + time_to_end <= self.L:
                feasible.append(client)
        if not feasible:
            return self.end_point
        if random.random() < self.q0:
            max_val = float('-inf')
            chosen = None
            for client in feasible:
                val = self.tau[current][client]**self.alpha * self.eta[current][client]**self.beta
                if val > max_val:
                    max_val = val
                    chosen = client
            return chosen or self.end_point
        total = sum(self.tau[current][client]**self.alpha * self.eta[current][client]**self.beta for client in feasible)
        if total == 0:
            return random.choice(feasible)
        r = random.random()
        cum_prob = 0
        for client in feasible:
            prob = (self.tau[current][client]**self.alpha * self.eta[current][client]**self.beta) / total
            cum_prob += prob
            if cum_prob >= r:
                return client
        return feasible[-1]

    def _construct_route(self, ant_id, used_clients=None):
        route = [self.start_point]
        current = self.start_point
        current_time = 0
        if used_clients is None:
            used_clients = set()
        available = set(self.clients) - used_clients  # Only consider unused clients
        while available:
            next_client = self._select_next_client(route, current, available, current_time)
            if next_client == self.end_point:
                break
            route.append(next_client)
            time_to_next = math.sqrt((current.x - next_client.x)**2 + (current.y - next_client.y)**2)
            current_time += time_to_next
            current = next_client
            available.remove(next_client)
            self.tau[route[-2]][route[-1]] = (1 - self.rho) * self.tau[route[-2]][route[-1]] + self.rho * self.tau_min
        route.append(self.end_point)
        return route

    def _construct_solution(self):
        solution = []
        used_clients = set()
        for _ in range(self.m):
            if not (set(self.clients) - used_clients):  # If no more available clients
                break
            route = self._construct_route(len(solution), used_clients)
            if len(route) > 2:
                solution.append(route)
                used_clients.update(set(route[1:-1]))  # Add newly visited clients to used set
        return solution

    def _calculate_solution_quality(self, solution):
        if not solution:
            return 0
        total_profit = sum(sum(c.profit for c in route[1:-1]) for route in solution)
        total_distance = sum(sum(math.sqrt((route[i].x - route[i+1].x)**2 + (route[i].y - route[i+1].y)**2)
                               for i in range(len(route)-1)) for route in solution)
        if total_distance > self.L:
            return 0
        quality = total_profit * (1 - total_distance/(self.m * self.L))
        total_clients = sum(len(route)-2 for route in solution)
        coverage_bonus = total_clients / len(self.clients)
        return quality * (1 + coverage_bonus)

    def _update_pheromones(self, solutions, qualities):
        for i in self.tau:
            for j in self.tau[i]:
                self.tau[i][j] *= (1 - self.rho)
        best_idx = np.argmax(qualities)
        best_solution = solutions[best_idx]
        best_quality = qualities[best_idx]
        for route in best_solution:
            for i in range(len(route)-1):
                deposit = 1.0 / (1.0 + len(route))
                self.tau[route[i]][route[i+1]] += self.rho * deposit * best_quality
                self.tau[route[i+1]][route[i]] = self.tau[route[i]][route[i+1]]
        for i in self.tau:
            for j in self.tau[i]:
                self.tau[i][j] = min(self.tau_max, max(self.tau_min, self.tau[i][j]))

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
        for iteration in range(self.max_iterations):
            solutions = []
            qualities = []
            for ant in range(self.n_ants):
                solution = self._construct_solution()
                if iteration % self.local_search_freq == 0 and len(solution) > 0:
                    solution = self._local_search(solution)
                quality = self._calculate_solution_quality(solution)
                solutions.append(solution)
                qualities.append(quality)
                if quality > self.best_fitness:
                    self.best_fitness = quality
                    self.best_solution = copy.deepcopy(solution)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            avg_fitness = statistics.mean(qualities)
            diversity = self._calculate_diversity(solutions)
            pheromone_values = [v for d in self.tau.values() for v in d.values()]
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
            self._update_pheromones(solutions, qualities)
            if stagnation_counter > self.max_stagnation and iteration > self.max_iterations // 4:
                if self.debug:
                    print(f"\nEarly stopping at iteration {iteration}")
                    print(f"No improvement for {stagnation_counter} iterations")
                break
        return self.best_solution

    def _local_search(self, solution):
        if not solution:
            return solution
        improved = True
        while improved:
            improved = False
            for route_idx, route in enumerate(solution):
                if len(route) <= 4:
                    continue
                best_length = sum(math.sqrt((route[i].x - route[i+1].x)**2 + (route[i].y - route[i+1].y)**2)
                                for i in range(len(route)-1))
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                        new_length = sum(math.sqrt((new_route[k].x - new_route[k+1].x)**2 + 
                                                 (new_route[k].y - new_route[k+1].y)**2)
                                       for k in range(len(new_route)-1))
                        if new_length < best_length and new_length <= self.L:
                            solution[route_idx] = new_route
                            improved = True
                            best_length = new_length
        return solution

    def get_stats(self):
        return pd.DataFrame(self.stats_data)


########################################################################################################################################################################

class GeneticTOP:
    def __init__(self, start_point, end_point, clients, m, L, debug=False):
        self.start_point = start_point
        self.end_point = end_point
        self.clients = clients
        self.m = m  # Number of vehicles
        self.L = L  # Time limit per route
        self.debug = debug
        
        n = len(clients)
        # Population parameters
        self.population_size = min(200, n * 4)
        self.generations = min(300, n * 10)
        
        # Genetic parameters
        self.crossover_rate = 0.85
        self.mutation_rate = 0.3
        self.elite_size = max(2, self.population_size // 20)
        self.tournament_size = max(3, self.population_size // 20)
        
        # Solution tracking
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.generations_without_improvement = 0
        self.max_stagnation = max(50, n // 2)
        
        # Precompute distances
        self.distances = self._precompute_distances()
        
        # Statistics collection
        self.stats_data = {
            'iteration': [],
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'mutation_rate': [],
            'crossover_rate': [],
            'tournament_size': [],
            'generations_without_improvement': []
        }

    def _precompute_distances(self):
        """Precompute distances between all points."""
        distances = {}
        all_points = [self.start_point] + self.clients + [self.end_point]
        for i, p1 in enumerate(all_points):
            for p2 in all_points[i+1:]:
                dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                distances[(p1.id, p2.id)] = distances[(p2.id, p1.id)] = dist
        return distances

    def get_distance(self, client1, client2):
        """Get precomputed distance between two points."""
        return self.distances[(client1.id, client2.id)]

    def _create_initial_solution(self):
        """Create an initial solution using a clustered approach."""
        # Initialize empty solution with m routes
        solution = [[] for _ in range(self.m)]
        available_clients = set(self.clients)
        
        # For each route
        for route_idx in range(self.m):
            if not available_clients:
                break
                
            current_route = [self.start_point]
            current_time = 0
            current = self.start_point
            
            # Try to add clients to current route
            while available_clients:
                # Find feasible clients
                feasible = []
                for client in available_clients:
                    time_to_client = self.get_distance(current, client)
                    time_to_end = self.get_distance(client, self.end_point)
                    total_time = current_time + time_to_client + time_to_end
                    
                    if total_time <= self.L:
                        score = client.profit / (time_to_client + 0.1)
                        feasible.append((client, score))
                
                if not feasible:
                    break
                    
                # Select client probabilistically based on score
                total_score = sum(score for _, score in feasible)
                if total_score <= 0:
                    break
                    
                r = random.random() * total_score
                cumsum = 0
                selected = None
                
                for client, score in feasible:
                    cumsum += score
                    if cumsum >= r:
                        selected = client
                        break
                
                if not selected:
                    selected = feasible[-1][0]
                
                # Add selected client to route
                current_route.append(selected)
                current_time += self.get_distance(current, selected)
                current = selected
                available_clients.remove(selected)
            
            # Complete route
            if len(current_route) > 1:  # Only add routes with clients
                current_route.append(self.end_point)
                solution[route_idx] = current_route
        
        # Remove empty routes
        solution = [route for route in solution if len(route) > 2]
        return solution

    def _evaluate_solution(self, solution):
        """Evaluate solution quality with penalties."""
        if not solution:
            return 0
        
        total_profit = 0
        total_time = 0
        used_clients = set()
        
        for route in solution:
            # Skip invalid routes
            if len(route) < 2:
                continue
                
            route_time = 0
            route_profit = 0
            route_clients = set()
            
            # Calculate route metrics
            for i in range(len(route) - 1):
                route_time += self.get_distance(route[i], route[i + 1])
                if i > 0:  # Skip start point
                    client = route[i]
                    route_profit += client.profit
                    route_clients.add(client.id)
            
            # Apply penalties
            if route_time > self.L:
                continue  # Invalid route
            
            # Check for duplicates
            if not route_clients & used_clients:  # No overlap with used clients
                total_profit += route_profit
                total_time += route_time
                used_clients.update(route_clients)
        
        # Calculate final score with bonuses
        if total_time > self.L * self.m:
            return 0
        
        coverage_bonus = len(used_clients) / len(self.clients)
        time_efficiency = 1 - (total_time / (self.L * self.m))
        
        return total_profit * (1 + 0.1 * coverage_bonus) * (1 + 0.1 * time_efficiency)

    def _crossover(self, parent1, parent2):
        """Route-based crossover operator."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1, child2 = [], []
        used_clients1, used_clients2 = set(), set()
        
        # Randomly select routes to inherit
        routes1 = random.sample(parent1, len(parent1) // 2)
        routes2 = random.sample(parent2, len(parent2) // 2)
        
        # Add selected routes to children if they don't create conflicts
        for route in routes1:
            route_clients = set(c.id for c in route[1:-1])
            if not (route_clients & used_clients1):
                child1.append(copy.deepcopy(route))
                used_clients1.update(route_clients)
        
        for route in routes2:
            route_clients = set(c.id for c in route[1:-1])
            if not (route_clients & used_clients2):
                child2.append(copy.deepcopy(route))
                used_clients2.update(route_clients)
        
        # Add remaining feasible routes from opposite parent
        remaining_routes1 = [r for r in parent2 if not any(c.id in used_clients1 for c in r[1:-1])]
        remaining_routes2 = [r for r in parent1 if not any(c.id in used_clients2 for c in r[1:-1])]
        
        while len(child1) < self.m and remaining_routes1:
            route = remaining_routes1.pop(random.randrange(len(remaining_routes1)))
            child1.append(copy.deepcopy(route))
            used_clients1.update(c.id for c in route[1:-1])
        
        while len(child2) < self.m and remaining_routes2:
            route = remaining_routes2.pop(random.randrange(len(remaining_routes2)))
            child2.append(copy.deepcopy(route))
            used_clients2.update(c.id for c in route[1:-1])
        
        return child1, child2

    def _mutation(self, solution):
        """Apply multiple mutation operators."""
        if random.random() > self.mutation_rate or not solution:
            return solution
        
        mutated = copy.deepcopy(solution)
        mutation_type = random.choice(['swap', 'reverse', 'relocate'])
        
        if mutation_type == 'swap':
            # Swap clients between routes
            if len(mutated) >= 2:
                route1_idx, route2_idx = random.sample(range(len(mutated)), 2)
                route1, route2 = mutated[route1_idx], mutated[route2_idx]
                
                if len(route1) > 3 and len(route2) > 3:
                    pos1 = random.randrange(1, len(route1) - 1)
                    pos2 = random.randrange(1, len(route2) - 1)
                    route1[pos1], route2[pos2] = route2[pos2], route1[pos1]
        
        elif mutation_type == 'reverse':
            # Reverse segment within route
            if mutated:
                route = random.choice(mutated)
                if len(route) > 4:
                    i = random.randrange(1, len(route) - 2)
                    j = random.randrange(i + 1, len(route) - 1)
                    route[i:j+1] = reversed(route[i:j+1])
        
        else:  # relocate
            # Move client to different position
            if len(mutated) >= 1:
                route_idx = random.randrange(len(mutated))
                route = mutated[route_idx]
                
                if len(route) > 3:
                    client_pos = random.randrange(1, len(route) - 1)
                    client = route.pop(client_pos)
                    
                    # Choose new route and position
                    new_route_idx = random.randrange(len(mutated))
                    new_route = mutated[new_route_idx]
                    new_pos = random.randrange(1, len(new_route))
                    new_route.insert(new_pos, client)
        
        return mutated

    def _tournament_selection(self, population, fitnesses):
        """Tournament selection with diversity consideration."""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx]

    def evolve(self):
        """Main evolutionary process."""
        # Initialize population
        population = [self._create_initial_solution() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # Evaluate population
            fitnesses = [self._evaluate_solution(solution) for solution in population]
            
            # Update best solution
            max_fitness_idx = fitnesses.index(max(fitnesses))
            if fitnesses[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_fitness_idx]
                self.best_solution = copy.deepcopy(population[max_fitness_idx])
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1
            
            # Collect statistics
            self.stats_data['iteration'].append(generation)
            self.stats_data['best_fitness'].append(self.best_fitness)
            self.stats_data['avg_fitness'].append(sum(fitnesses) / len(fitnesses))
            self.stats_data['diversity'].append(len(set(tuple(str(s)) for s in population)) / self.population_size)
            self.stats_data['mutation_rate'].append(self.mutation_rate)
            self.stats_data['crossover_rate'].append(self.crossover_rate)
            self.stats_data['tournament_size'].append(self.tournament_size)
            self.stats_data['generations_without_improvement'].append(self.generations_without_improvement)
            
            # Debug output
            if self.debug and generation % 10 == 0:
                print(f"Generation {generation}:")
                print(f"  Best Fitness = {self.best_fitness:.2f}")
                print(f"  Average Fitness = {sum(fitnesses) / len(fitnesses):.2f}")
                print(f"  Diversity = {self.stats_data['diversity'][-1]:.3f}")
            
            # Check for early stopping
            if (self.generations_without_improvement > self.max_stagnation and 
                generation > self.generations // 4):
                if self.debug:
                    print(f"\nEarly stopping at generation {generation}")
                    print(f"No improvement for {self.generations_without_improvement} generations")
                break
            
            # Create new population
            new_population = []
            
            # Elitism
            sorted_indices = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k], reverse=True)
            for i in range(self.elite_size):
                new_population.append(copy.deepcopy(population[sorted_indices[i]]))
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population
        
        return self.best_solution

    def get_stats(self):
        """Return collected statistics as a pandas DataFrame."""
        return pd.DataFrame(self.stats_data)



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
        ('SimulatedAnnealingTOP', SimulatedAnnealingTOP(start_point, end_point, clients, m, L, debug=debug)),
        ('AntColonyTOP', AntColonyTOP(start_point, end_point, clients, m, L, debug=debug)),
        ('GeneticTOP', GeneticTOP(start_point, end_point, clients, m, L, debug=debug))
    ]
    
    results = []
    for name, algorithm in algorithms:
        print(f"\nRunning {name}...")
        
        start_time = time.time()
        if isinstance(algorithm, GeneticTOP):
            solution = algorithm.evolve()
        else:
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
     for size, files in non_symmetric_files.items():
         print(f"\nRunning non-symmetric test set ({size} time constraint instances):")
         for instance_file in files:
             main(instance_file, debug=True)
     for size, files in symmetric_files.items():
         print(f"\nRunning symmetric test set ({size} time constraint instances):")
         for instance_file in files:
             main(instance_file, debug=True)


