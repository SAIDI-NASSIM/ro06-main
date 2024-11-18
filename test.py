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


class Client:
    def __init__(self, id, x, y, profit):
        self.id = id
        self.x = x
        self.y = y
        self.profit = profit

def distance(client1, client2):
    return math.sqrt((client1.x - client2.x) ** 2 + (client1.y - client2.y) ** 2)

def temps_total(route):
    return sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1))

def profit_total(route):
    return sum(client.profit for client in route[1:-1])

def profit_incremental(route, client, position):
    if position == 0 or position == len(route):
        return -float("inf")
    cout_additionnel = (distance(route[position - 1], client) + distance(client, route[position]) - distance(route[position - 1], route[position]))
    return client.profit - cout_additionnel

class AntColonyTOP:
    def __init__(self, start_point, end_point, clients, m: int, L: float, debug=False):
        self.start_point = start_point
        self.end_point = end_point
        self.clients = clients
        self.m = m  # number of routes
        self.L = L  # time limit
        self.debug = debug
        
        # Problem size dependent parameters
        n = len(clients)
        self.n_ants = min(50, n * 2)  # number of ants scales with problem size
        self.max_iterations = min(200, n * 5)
        
        # ACO parameters
        self.alpha = 1.0  # pheromone importance
        self.beta = 2.0   # heuristic information importance
        self.rho = 0.1    # evaporation rate
        self.Q = 100.0    # pheromone deposit factor
        
        # Initialize pheromone matrix
        self.tau = {}
        for i in [self.start_point] + clients + [self.end_point]:
            self.tau[i] = {}
            for j in [self.start_point] + clients + [self.end_point]:
                if i != j:
                    self.tau[i][j] = 1.0

        # Initialize heuristic information (eta)
        self.eta = {}
        for i in [self.start_point] + clients + [self.end_point]:
            self.eta[i] = {}
            for j in [self.start_point] + clients + [self.end_point]:
                if i != j:
                    dist = distance(i, j)
                    # Combine distance and profit in heuristic
                    profit = j.profit if j != self.start_point and j != self.end_point else 0
                    self.eta[i][j] = (1.0 + profit) / dist if dist > 0 else 1.0

        # Dynamic parameters
        self.local_search_freq = max(1, self.max_iterations // 10)
        self.diversification_factor = 0.0

        # Statistics tracking
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.iterations_without_improvement = 0
        self.max_stagnation = 20
        
        # Statistics collection
        self.stats_data = {
            'iteration': [],
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'pheromone_avg': [],
            'pheromone_max': [],
            'pheromone_min': [],
            'alpha': [],
            'beta': [],
            'rho': []
        }

    def is_valid_route(self, route):
        if not route or len(route) < 2:
            return False
        if route[0] != self.start_point or route[-1] != self.end_point:
            return False
        if len(set(c.id for c in route[1:-1])) != len(route[1:-1]):
            return False
        total_time = sum(distance(route[i], route[i+1]) for i in range(len(route)-1))
        return total_time <= self.L

    def _select_next_client(self, ant_route, current, available_clients):
        if not available_clients:
            return self.end_point

        # Calculate remaining time
        current_time = sum(distance(ant_route[i], ant_route[i+1]) 
                         for i in range(len(ant_route)-1))
        
        # Filter feasible clients
        feasible = []
        for client in available_clients:
            time_to_client = distance(current, client)
            time_to_end = distance(client, self.end_point)
            temp_route = ant_route + [client, self.end_point]
            if (current_time + time_to_client + time_to_end <= self.L and 
                len(set(c.id for c in temp_route[1:-1])) == len(temp_route[1:-1])):
                feasible.append(client)
        
        if not feasible:
            return self.end_point

        # Calculate selection probabilities
        total = 0.0
        probabilities = {}
        
        for client in feasible:
            # Include diversification factor in probability calculation
            prob = ((self.tau[current][client] + self.diversification_factor) ** self.alpha * 
                   self.eta[current][client] ** self.beta)
            probabilities[client] = prob
            total += prob

        if total == 0.0:
            return random.choice(feasible)

        # Roulette wheel selection
        r = random.random() * total
        curr_sum = 0.0
        for client in feasible:
            curr_sum += probabilities[client]
            if curr_sum >= r:
                return client
        
        return feasible[-1]

    def _construct_route(self, available_clients):
        route = [self.start_point]
        current = self.start_point
        local_available = available_clients.copy()
        
        while local_available:
            next_client = self._select_next_client(route, current, local_available)
            if next_client != self.end_point:
                test_route = route + [next_client, self.end_point]
                if not self.is_valid_route(test_route):
                    next_client = self.end_point
            if next_client == self.end_point:
                break
            route.append(next_client)
            current = next_client
            local_available.remove(next_client)
        
        route.append(self.end_point)
        return route

    def _construct_solution(self):
        available_clients = set(self.clients)
        solution = []
        
        for _ in range(self.m):
            if not available_clients:
                break
            route = self._construct_route(available_clients)
            if len(route) > 2:  # Only add non-empty routes
                solution.append(route)
                available_clients -= set(route[1:-1])
        
        return solution

    def _local_search(self, solution):
        improved = True
        while improved:
            improved = False
            
            # 2-opt improvement for each route
            for i, route in enumerate(solution):
                if len(route) <= 4:  # Need at least 2 clients for 2-opt
                    continue
                    
                for j in range(1, len(route)-2):
                    for k in range(j+1, len(route)-1):
                        new_route = route[:j] + list(reversed(route[j:k+1])) + route[k+1:]
                        if self.is_valid_route(new_route):
                            if (temps_total(new_route) <= self.L and 
                                profit_total(new_route) > profit_total(route)):
                                solution[i] = new_route
                                improved = True
            
            # Inter-route client exchange
            for i in range(len(solution)):
                for j in range(i+1, len(solution)):
                    for pos1 in range(1, len(solution[i])-1):
                        for pos2 in range(1, len(solution[j])-1):
                            new_route1 = (solution[i][:pos1] + 
                                        [solution[j][pos2]] + 
                                        solution[i][pos1+1:])
                            new_route2 = (solution[j][:pos2] + 
                                        [solution[i][pos1]] + 
                                        solution[j][pos2+1:])
                            
                            if (self.is_valid_route(new_route1) and
                                self.is_valid_route(new_route2)):
                                old_profit = profit_total(solution[i]) + profit_total(solution[j])
                                new_profit = profit_total(new_route1) + profit_total(new_route2)
                                if new_profit > old_profit:
                                    solution[i] = new_route1
                                    solution[j] = new_route2
                                    improved = True
        
        return solution

    def _update_pheromones(self, solutions, solution_qualities):
        # Evaporation
        for i in self.tau:
            for j in self.tau[i]:
                self.tau[i][j] *= (1 - self.rho)

        # Deposit new pheromones
        for solution, quality in zip(solutions, solution_qualities):
            deposit = self.Q * quality
            for route in solution:
                for i in range(len(route)-1):
                    self.tau[route[i]][route[i+1]] += deposit
                    self.tau[route[i+1]][route[i]] += deposit  # Symmetric deposit

    def _calculate_solution_quality(self, solution):
        if not solution:
            return 0
        
        total_profit = sum(profit_total(route) for route in solution)
        total_time = sum(temps_total(route) for route in solution)
        
        # Penalize solutions that violate time constraints
        if any(temps_total(route) > self.L for route in solution):
            return 0
        
        # Calculate coverage ratio
        served_clients = len({c.id for route in solution for c in route[1:-1]})
        coverage_ratio = served_clients / len(self.clients)
        
        # Calculate route balance factor
        route_lengths = [len(route) - 2 for route in solution]
        length_variance = np.var(route_lengths) if route_lengths else 0
        balance_factor = 1 + 0.2 * (1 / (1 + length_variance))
        
        # Combine factors
        quality = (total_profit * 
                  (1 + 0.3 * coverage_ratio) * 
                  balance_factor * 
                  (1 - total_time / (self.m * self.L)))
        
        return quality

    def _calculate_diversity(self, solutions):
        if not solutions:
            return 0
        
        diversity_sum = 0
        comparisons = 0
        
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                sol1_clients = {c.id for route in solutions[i] for c in route[1:-1]}
                sol2_clients = {c.id for route in solutions[j] for c in route[1:-1]}
                
                if sol1_clients or sol2_clients:
                    jaccard = len(sol1_clients & sol2_clients) / len(sol1_clients | sol2_clients)
                    diversity_sum += 1 - jaccard
                    comparisons += 1
        
        return diversity_sum / max(1, comparisons)


    def solve(self):
        # Initialize elite solutions archive
        elite_solutions = []
        elite_size = 3
        stagnation_counter = 0
        last_improvement = 0
        
        # Adaptive parameters
        min_alpha = 0.5
        max_alpha = 3.0
        min_beta = 1.0
        max_beta = 4.0
        
        # Initialize pheromone bounds
        tau_max = 5.0
        tau_min = 0.01
        
        for iteration in range(self.max_iterations):
            solutions = []
            qualities = []
            
            # Adjust parameters based on stagnation
            if stagnation_counter > 10:
                # Increase exploration
                self.alpha = max(min_alpha, self.alpha * 0.9)
                self.beta = min(max_beta, self.beta * 1.1)
                self.rho = min(0.3, self.rho * 1.1)  # Increase evaporation
                # Reset pheromone levels partially
                if stagnation_counter % 20 == 0:
                    for i in self.tau:
                        for j in self.tau[i]:
                            self.tau[i][j] = max(tau_min, self.tau[i][j] * 0.5)
            else:
                # Favor exploitation
                self.alpha = min(max_alpha, self.alpha * 1.05)
                self.beta = max(min_beta, self.beta * 0.95)
                self.rho = max(0.1, self.rho * 0.95)

            # Construct solutions with varying randomness
            for ant in range(self.n_ants):
                # Adjust randomness based on ant index
                exploration_factor = 1.0 - (ant / self.n_ants)
                temp_alpha = self.alpha * (1 + exploration_factor * 0.5)
                temp_beta = self.beta * (1 - exploration_factor * 0.3)
                
                self.alpha, temp_alpha = temp_alpha, self.alpha  # Temporarily modify parameters
                self.beta, temp_beta = temp_beta, self.beta
                
                solution = self._construct_solution()
                
                # Restore original parameters
                self.alpha, self.beta = temp_alpha, temp_beta
                
                # Apply immediate local search to promising solutions
                if ant < self.n_ants // 4:  # Apply to top 25% of ants
                    solution = self._local_search(solution)
                
                quality = self._calculate_solution_quality(solution)
                solutions.append(solution)
                qualities.append(quality)
                
                # Update best solution
                if quality > self.best_fitness:
                    self.best_fitness = quality
                    self.best_solution = copy.deepcopy(solution)
                    last_improvement = iteration
                    stagnation_counter = 0
                    
                    # Update elite solutions
                    elite_solutions.append((solution, quality))
                    elite_solutions.sort(key=lambda x: x[1], reverse=True)
                    elite_solutions = elite_solutions[:elite_size]

            # Collect statistics
            avg_quality = statistics.mean(qualities)
            pheromone_values = [v for i in self.tau.values() for v in i.values()]

            self.stats_data['iteration'].append(iteration)
            self.stats_data['best_fitness'].append(self.best_fitness)
            self.stats_data['avg_fitness'].append(avg_quality)
            self.stats_data['diversity'].append(statistics.stdev(pheromone_values) if pheromone_values else 0)
            self.stats_data['pheromone_avg'].append(statistics.mean(pheromone_values))
            self.stats_data['pheromone_max'].append(max(pheromone_values))
            self.stats_data['pheromone_min'].append(min(pheromone_values))
            self.stats_data['alpha'].append(self.alpha)
            self.stats_data['beta'].append(self.beta)
            self.stats_data['rho'].append(self.rho)
            
            if self.debug and iteration % 10 == 0:
                print(f"\nIteration {iteration}:")
                print(f"  Best Fitness: {self.best_fitness:.2f}")
                print(f"  Average Fitness: {avg_quality:.2f}")
                print(f"  Pheromone Diversity: {self.stats_data['diversity'][-1]:.3f}")
                print(f"  Parameters - Alpha: {self.alpha:.3f}, Beta: {self.beta:.3f}, Rho: {self.rho:.3f}")
                print(f"  Avg Pheromone: {self.stats_data['pheromone_avg'][-1]:.3f}")
                
            # Update pheromones with elite reinforcement
            self._update_pheromones(solutions, qualities)
            
            # Additional pheromone update from elite solutions
            if elite_solutions:
                elite_deposit = self.Q * 2  # Stronger influence from elite solutions
                for elite_sol, elite_qual in elite_solutions:
                    for route in elite_sol:
                        for i in range(len(route)-1):
                            self.tau[route[i]][route[i+1]] += elite_deposit
                            self.tau[route[i+1]][route[i]] += elite_deposit
            
            # Enforce pheromone bounds
            for i in self.tau:
                for j in self.tau[i]:
                    self.tau[i][j] = min(tau_max, max(tau_min, self.tau[i][j]))
            
            # Dynamic convergence criteria
            stagnation_counter += 1
            if stagnation_counter >= 30 and iteration > 50:  # Minimum iterations before early stop
                if self.best_fitness >= 0.95 * max(qualities):  # Within 5% of current best
                    if self.debug:
                        print(f"Converged at iteration {iteration}")
                    break
            
            # Restart mechanism
            if iteration - last_improvement > 40:  # No improvement for 40 iterations
                if self.debug:
                    print(f"Restarting at iteration {iteration}")
                # Preserve best solution but reset pheromones partially
                for i in self.tau:
                    for j in self.tau[i]:
                        self.tau[i][j] = max(tau_min, min(tau_max, 
                                        self.tau[i][j] * 0.5 + random.random() * 0.5))
                last_improvement = iteration  # Reset counter
                stagnation_counter = 0
        
        if self.debug:
            self._print_final_stats()
            
        return self.best_solution

    def _print_final_stats(self):
        """Print final statistics about the optimization process."""
        print("\n" + "="*50)
        print("Final Statistics:")
        print("="*50)
        print(f"Total Iterations: {len(self.stats_data['iteration'])}")
        print(f"Best Solution Fitness: {self.best_fitness:.2f}")
        print(f"Final Diversity: {self.stats_data['diversity'][-1]:.3f}")
        
        if self.best_solution:
            total_profit = sum(sum(c.profit for c in route[1:-1]) 
                            for route in self.best_solution)
            total_clients = sum(len(route)-2 for route in self.best_solution)
            total_time = sum(sum(distance(route[i], route[i+1])
                            for i in range(len(route)-1))
                        for route in self.best_solution)
            
            print(f"\nBest Solution Details:")
            print(f"Total Profit: {total_profit}")
            print(f"Total Clients Served: {total_clients}")
            print(f"Average Clients per Route: {total_clients/len(self.best_solution):.2f}")
            print(f"Total Time: {total_time:.2f}")
            
            print("\nParameter Evolution:")
            print(f"Final Alpha: {self.stats_data['alpha'][-1]:.3f}")
            print(f"Final Beta: {self.stats_data['beta'][-1]:.3f}")
            print(f"Final Rho: {self.stats_data['rho'][-1]:.3f}")
            print(f"Final Average Pheromone: {self.stats_data['pheromone_avg'][-1]:.3f}")

    def get_stats(self):
        """Return collected statistics as a pandas DataFrame."""
        return pd.DataFrame(self.stats_data)


class AntColonyTOP:
    def __init__(self, start_point, end_point, clients, m: int, L: float, debug=False):
        self.start_point = start_point
        self.end_point = end_point
        self.clients = clients
        self.m = m  # number of routes
        self.L = L  # time limit
        self.debug = debug
        
        # Problem size dependent parameters
        n = len(clients)
        self.n_ants = min(50, n * 2)  # number of ants scales with problem size
        self.max_iterations = min(200, n * 5)
        
        # ACO parameters
        self.alpha = 1.0  # pheromone importance
        self.beta = 2.0   # heuristic information importance
        self.rho = 0.1    # evaporation rate
        self.Q = 100.0    # pheromone deposit factor
        
        # Initialize pheromone matrix and heuristic information
        self.tau = {}
        self.eta = {}
        all_points = [self.start_point] + self.clients + [self.end_point]
        for i in all_points:
            self.tau[i] = {}
            self.eta[i] = {}
            for j in all_points:
                if i != j:
                    self.tau[i][j] = 1.0  # Initial pheromone
                    # Heuristic combining distance and profit
                    dist = math.sqrt((i.x - j.x)**2 + (i.y - j.y)**2)
                    profit = j.profit if j != self.start_point and j != self.end_point else 0
                    self.eta[i][j] = (1.0 + profit) / (dist if dist > 0 else 0.1)

        # Statistics tracking
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.stats_data = {
            'iteration': [],
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'pheromone_avg': [],
            'pheromone_max': [],
            'pheromone_min': []
        }

    def _select_next_client(self, current, available_clients, current_time, route):
        if not available_clients:
            return self.end_point

        # Calculate probabilities for available clients
        probabilities = []
        feasible_clients = []
        
        for client in available_clients:
            time_to_client = math.sqrt((current.x - client.x)**2 + (current.y - client.y)**2)
            time_to_end = math.sqrt((client.x - self.end_point.x)**2 + 
                                  (client.y - self.end_point.y)**2)
            
            if current_time + time_to_client + time_to_end <= self.L:
                feasible_clients.append(client)
                prob = (self.tau[current][client]**self.alpha * 
                       self.eta[current][client]**self.beta)
                probabilities.append(prob)
        
        if not feasible_clients:
            return self.end_point

        # Convert to numpy array for efficient operations
        probabilities = np.array(probabilities)
        total = probabilities.sum()
        
        if total == 0:
            return random.choice(feasible_clients)
        
        # Select next client using roulette wheel selection
        probabilities = probabilities / total
        return np.random.choice(feasible_clients, p=probabilities)

    def _construct_route(self, available_clients):
        route = [self.start_point]
        current = self.start_point
        current_time = 0.0
        local_available = available_clients.copy()

        while local_available:
            next_client = self._select_next_client(current, local_available, current_time, route)
            if next_client == self.end_point:
                break
            
            time_to_next = math.sqrt((current.x - next_client.x)**2 + 
                                   (current.y - next_client.y)**2)
            current_time += time_to_next
            route.append(next_client)
            current = next_client
            local_available.remove(next_client)

        route.append(self.end_point)
        return route

    def _construct_solution(self):
        available_clients = set(self.clients)
        solution = []

        for _ in range(self.m):
            if not available_clients:
                break
            route = self._construct_route(available_clients)
            if len(route) > 2:  # Only add routes with clients
                solution.append(route)
                available_clients -= set(route[1:-1])

        return solution

    def _calculate_solution_quality(self, solution):
        if not solution:
            return 0
        
        total_profit = 0
        total_time = 0
        used_clients = set()

        for route in solution:
            # Check time constraint
            route_time = sum(math.sqrt((route[i].x - route[i+1].x)**2 + 
                                     (route[i].y - route[i+1].y)**2)
                           for i in range(len(route)-1))
            
            if route_time > self.L:
                return 0
            
            # Calculate route profit and check for duplicates
            route_profit = sum(client.profit for client in route[1:-1])
            route_clients = {client.id for client in route[1:-1]}
            
            if route_clients & used_clients:  # Check for duplicate clients
                return 0
            
            used_clients.update(route_clients)
            total_profit += route_profit
            total_time += route_time

        # Quality score includes profit and time efficiency
        quality = (total_profit * 
                  (1 - total_time/(self.m * self.L)) * 
                  (len(used_clients) / len(self.clients)))
        
        return quality

    def _update_pheromones(self, solutions, qualities):
        # Evaporation
        for i in self.tau:
            for j in self.tau[i]:
                self.tau[i][j] *= (1 - self.rho)

        # Deposit new pheromones
        for solution, quality in zip(solutions, qualities):
            if quality > 0:  # Only deposit for valid solutions
                deposit = self.Q * quality
                for route in solution:
                    for i in range(len(route)-1):
                        self.tau[route[i]][route[i+1]] += deposit
                        # Symmetric update
                        self.tau[route[i+1]][route[i]] = self.tau[route[i]][route[i+1]]

    def _calculate_diversity(self, solutions):
        if not solutions:
            return 0
        
        diversity_sum = 0
        comparisons = 0
        
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                sol1_clients = {c.id for route in solutions[i] for c in route[1:-1]}
                sol2_clients = {c.id for route in solutions[j] for c in route[1:-1]}
                
                if sol1_clients or sol2_clients:
                    jaccard = len(sol1_clients & sol2_clients) / len(sol1_clients | sol2_clients)
                    diversity_sum += 1 - jaccard
                    comparisons += 1
        
        return diversity_sum / max(1, comparisons)

    def solve(self):
        start_time = time.time()
        stagnation_counter = 0
        
        for iteration in range(self.max_iterations):
            # Construct solutions
            solutions = []
            qualities = []
            
            for _ in range(self.n_ants):
                solution = self._construct_solution()
                quality = self._calculate_solution_quality(solution)
                solutions.append(solution)
                qualities.append(quality)
                
                # Update best solution
                if quality > self.best_fitness:
                    self.best_fitness = quality
                    self.best_solution = copy.deepcopy(solution)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            # Update pheromones
            self._update_pheromones(solutions, qualities)
            
            # Collect statistics
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
            
            # Debug printing
            if self.debug and iteration % 10 == 0:
                print(f"\nIteration {iteration}:")
                print(f"  Best Fitness: {self.best_fitness:.2f}")
                print(f"  Average Fitness: {avg_fitness:.2f}")
                print(f"  Diversity: {diversity:.3f}")
            
            # Early stopping check
            if stagnation_counter >= 50:  # Increased from 30 to 50
                if self.debug:
                    print(f"\nEarly stopping at iteration {iteration}: No improvement for {stagnation_counter} iterations")
                break

        return self.best_solution

    def get_stats(self):
        """Return collected statistics as a pandas DataFrame."""
        return pd.DataFrame(self.stats_data)
class GeneticTOP:
    def __init__(self, start_point, end_point, clients, m, L, debug=False):
        self.start_point = start_point
        self.end_point = end_point
        self.clients = clients
        self.m = m
        self.L = L
        self.debug = debug
        
        n = len(clients)
        
        # Dynamic population size and generation parameters
        self.population_size = min(200, n * 4)
        self.generations = min(300, n * 10)
        
        # Adaptive parameters
        self.crossover_rate = 0.85
        self.mutation_rate = 0.3
        self.elite_size = max(2, self.population_size // 25)
        
        # Tournament selection parameters
        self.tournament_size = max(3, self.population_size // 20)
        
        # Diversity maintenance parameters
        self.diversity_threshold = 0.2
        self.similarity_penalty = 0.15
        
        # Solution tracking
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.generations_without_improvement = 0
        self.max_stagnation = max(50, n // 2)
        
        # Precompute distances for efficiency
        self.distances = self._precompute_distances()
        
        # Population diversity metrics
        self.population_entropy = 0
        
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
        distances = {}
        all_points = [self.start_point] + self.clients + [self.end_point]
        for i, p1 in enumerate(all_points):
            for p2 in all_points[i+1:]:
                dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                distances[(p1.id, p2.id)] = distances[(p2.id, p1.id)] = dist
        return distances

    def get_distance(self, client1, client2):
        return self.distances[(client1.id, client2.id)]

    def _calculate_route_diversity(self, route1, route2):
        if not route1 or not route2:
            return 1.0
        clients1 = set(c.id for c in route1[1:-1])
        clients2 = set(c.id for c in route2[1:-1])
        union = len(clients1.union(clients2))
        if union == 0:
            return 1.0
        return 1.0 - len(clients1.intersection(clients2)) / union

    def _calculate_population_diversity(self, population):
        if not population:
            return 0.0
        total_diversity = 0
        comparisons = 0
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                solution_diversity = 0
                valid_routes = 0
                for route1 in population[i]:
                    for route2 in population[j]:
                        diversity = self._calculate_route_diversity(route1, route2)
                        solution_diversity += diversity
                        valid_routes += 1
                if valid_routes > 0:
                    total_diversity += solution_diversity / valid_routes
                    comparisons += 1
        return total_diversity / max(1, comparisons)

    def _create_initial_solution(self):
        solution = []
        available_clients = set(self.clients)
        max_attempts = 3  # Allow multiple attempts per route
        
        for _ in range(self.m):
            if not available_clients:
                break
                
            best_route = None
            best_profit = -1
            
            for _ in range(max_attempts):
                route = self._create_route(available_clients)
                if route and len(route) > 2:
                    profit = sum(c.profit for c in route[1:-1])
                    if profit > best_profit:
                        best_route = route
                        best_profit = profit
            
            if best_route and len(best_route) > 2:
                solution.append(best_route)
                available_clients -= set(best_route[1:-1])
        
        return solution

    def _create_route(self, available_clients):
        if not available_clients:
            return [self.start_point, self.end_point]
            
        route = [self.start_point]
        current_time = 0
        local_clients = list(available_clients)
        
        # Randomize client selection with probability bias towards profit/distance ratio
        while local_clients:
            feasible_clients = []
            for client in local_clients:
                new_time = (current_time + 
                           self.get_distance(route[-1], client) + 
                           self.get_distance(client, self.end_point))
                if new_time <= self.L:
                    score = client.profit / (self.get_distance(route[-1], client) + 0.1)
                    feasible_clients.append((client, score))
                    
            if not feasible_clients:
                break
                
            # Randomized selection with bias towards better scores
            total_score = sum(score for _, score in feasible_clients)
            if total_score <= 0:
                break
                
            r = random.random() * total_score
            cumsum = 0
            selected_client = None
            
            for client, score in feasible_clients:
                cumsum += score
                if cumsum >= r:
                    selected_client = client
                    break
            
            if not selected_client:
                selected_client = feasible_clients[-1][0]
                
            route.append(selected_client)
            local_clients.remove(selected_client)
            current_time += self.get_distance(route[-2], selected_client)
            
        route.append(self.end_point)
        return route if len(route) > 2 else [self.start_point, self.end_point]

    def _adaptive_fitness(self, solution):
        if not solution:
            return 0
        
        total_profit = sum(sum(c.profit for c in route[1:-1]) for route in solution)
        
        # Calcul du temps total et pénalité graduelle
        total_time = sum(sum(self.get_distance(route[i], route[i+1]) 
                        for i in range(len(route)-1)) for route in solution)
        time_penalty = 0
        if total_time > self.L:
            excess = total_time - self.L
            time_penalty = (excess * 5) + (excess ** 2)
        
        # Pénalité pour distribution non uniforme des profits
        route_profits = [sum(c.profit for c in route[1:-1]) for route in solution]
        profit_std = statistics.stdev(route_profits) if len(route_profits) > 1 else 0
        distribution_penalty = profit_std * 0.5
        
        # Bonus pour utilisation efficace du temps
        time_efficiency = 1 - (total_time / (self.L * 1.2))
        efficiency_bonus = max(0, time_efficiency * total_profit * 0.1)
        
        # Score final
        fitness = (total_profit 
                - time_penalty 
                - distribution_penalty 
                + efficiency_bonus)
        
        return max(0, fitness)
    def crossover(self, parent1, parent2):
        if not parent1 or not parent2 or random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
            
        child1, child2 = [], []
        used_clients1, used_clients2 = set(), set()
        
        # Determine crossover points adaptively
        max_routes = max(len(parent1), len(parent2))
        crossover_points = sorted(random.sample(range(max_routes), 
                                min(2, max_routes)))
        
        if len(crossover_points) < 2:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
            
        start, end = crossover_points
        
        # Preserve route segments between crossover points
        for i in range(start, end):
            if i < len(parent1):
                new_route = self._extract_valid_clients(parent1[i], used_clients1)
                if len(new_route) > 2:
                    child1.append(new_route)
                    used_clients1.update(c.id for c in new_route[1:-1])
                    
            if i < len(parent2):
                new_route = self._extract_valid_clients(parent2[i], used_clients2)
                if len(new_route) > 2:
                    child2.append(new_route)
                    used_clients2.update(c.id for c in new_route[1:-1])
        
        # Complete children with remaining feasible routes
        remaining_routes1 = (parent2[:start] + parent2[end:] if start < len(parent2) else [])
        remaining_routes2 = (parent1[:start] + parent1[end:] if start < len(parent1) else [])
        
        for route in remaining_routes1:
            new_route = self._extract_valid_clients(route, used_clients1)
            if len(new_route) > 2:
                child1.append(new_route)
                used_clients1.update(c.id for c in new_route[1:-1])
                
        for route in remaining_routes2:
            new_route = self._extract_valid_clients(route, used_clients2)
            if len(new_route) > 2:
                child2.append(new_route)
                used_clients2.update(c.id for c in new_route[1:-1])
        
        return child1, child2

    def mutation(self, solution):
        if not solution or random.random() > self.mutation_rate:
            return solution
            
        mutated = copy.deepcopy(solution)
        route = mutated[0]  # Pour m=1
        
        # Sélectionner plusieurs clients à réorganiser
        internal_points = route[1:-1]
        if len(internal_points) <= 2:
            return mutated
            
        # Choisir entre 2 et 4 points à permuter
        k = random.randint(2, min(4, len(internal_points)))
        positions = random.sample(range(len(internal_points)), k)
        
        # Permuter les points sélectionnés
        for i, j in itertools.combinations(positions, 2):
            internal_points[i], internal_points[j] = internal_points[j], internal_points[i]
        
        new_route = [route[0]] + internal_points + [route[-1]]
        
        # Vérifier si la nouvelle route est valide
        if self.is_valid_route(new_route):
            mutated[0] = new_route
            return mutated
        return solution  # Retourner solution originale si mutation invalide

    def _adjust_parameters(self):
        """Adjust genetic parameters based on population diversity."""
        if self.population_entropy < 0.3:  # Plus strict threshold
            self.mutation_rate = min(0.8, self.mutation_rate * 1.2)  # More aggressive increase
            self.crossover_rate = max(0.6, self.crossover_rate * 0.95)
        elif self.population_entropy > 0.7:  # High diversity
            self.mutation_rate = max(0.2, self.mutation_rate * 0.9)
            self.crossover_rate = min(0.95, self.crossover_rate * 1.05)
        
        # Adjust based on improvement stagnation
        if self.generations_without_improvement > self.max_stagnation // 2:
            self.mutation_rate = min(0.9, self.mutation_rate * 1.3)
            self.tournament_size = max(2, self.tournament_size - 1)
        else:
            self.tournament_size = min(self.population_size // 10, self.tournament_size + 1)

    def evolve(self):
            """Genetic algorithm's main evolution method with statistics tracking."""
            population = [self._create_initial_solution() 
                        for _ in range(self.population_size)]
            
            # Initialize statistics
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
            
            for generation in range(self.generations):
                # Calculate population entropy for diversity metric
                self.population_entropy = self._calculate_population_diversity(population)
                
                # Adjust parameters based on diversity
                self._adjust_parameters()
                
                # Evaluate population
                population_with_fitness = [(p, self._adaptive_fitness(p)) for p in population]
                population_with_fitness.sort(key=lambda x: x[1], reverse=True)
                
                current_best = population_with_fitness[0]
                
                # Update best solution if improved
                if current_best[1] > self.best_fitness:
                    self.best_fitness = current_best[1]
                    self.best_solution = copy.deepcopy(current_best[0])
                    self.generations_without_improvement = 0
                else:
                    self.generations_without_improvement += 1
                
                # Collect statistics
                self.stats_data['iteration'].append(generation)
                self.stats_data['best_fitness'].append(self.best_fitness)
                self.stats_data['avg_fitness'].append(
                    statistics.mean([f for _, f in population_with_fitness]))
                self.stats_data['diversity'].append(self.population_entropy)
                self.stats_data['mutation_rate'].append(self.mutation_rate)
                self.stats_data['crossover_rate'].append(self.crossover_rate)
                self.stats_data['tournament_size'].append(self.tournament_size)
                self.stats_data['generations_without_improvement'].append(
                    self.generations_without_improvement)
                
                # Debug printing if enabled
                if self.debug and generation % 10 == 0:
                    print(f"Generation {generation}:")
                    print(f"  Best Fitness = {self.best_fitness:.2f}")
                    print(f"  Diversity = {self.population_entropy:.3f}")
                    print(f"  Mutation Rate = {self.mutation_rate:.3f}")
                    print(f"  Crossover Rate = {self.crossover_rate:.3f}")
                    print(f"  Tournament Size = {self.tournament_size}")
                
                # Early stopping check
                if (self.generations_without_improvement > self.max_stagnation and
                    generation > self.generations // 4):
                    if self.debug:
                        print(f"\nEarly stopping at generation {generation}")
                        print(f"No improvement for {self.generations_without_improvement} generations")
                    break
                
                # Create new population
                new_population = []
                
                # Elitism
                for solution, fitness in population_with_fitness:
                    if len(new_population) >= self.elite_size:
                        break
                    is_diverse = True
                    for added in new_population:
                        if self._calculate_route_diversity(solution[0], added[0]) < 0.3:
                            is_diverse = False
                            break
                    if is_diverse:
                        new_population.append(copy.deepcopy(solution))
                
                # Generate rest of population
                while len(new_population) < self.population_size:
                    parent1 = self._diverse_tournament_select(population_with_fitness)
                    parent2 = self._diverse_tournament_select(population_with_fitness, exclude=parent1)
                    
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutation(child1)
                    child2 = self.mutation(child2)
                    
                    for child in [child1, child2]:
                        if len(new_population) < self.population_size and child:
                            new_population.append(child)
                
                population = new_population
            
            return self.best_solution
    def _diverse_tournament_select(self, population_with_fitness, exclude=None):
        """Tournament selection that considers both fitness and diversity."""
        tournament_size = min(self.tournament_size, len(population_with_fitness))
        tournament = random.sample(population_with_fitness, tournament_size)
        
        if exclude:
            # Add diversity score to tournament candidates
            tournament_with_diversity = []
            for solution, fitness in tournament:
                diversity = self._calculate_route_diversity(solution[0], exclude[0])
                # Combine fitness and diversity into score
                score = fitness * (1 + diversity)
                tournament_with_diversity.append((solution, score))
            
            # Select based on combined score
            return max(tournament_with_diversity, key=lambda x: x[1])[0]
        
        # If no exclusion, select based on fitness
        return max(tournament, key=lambda x: x[1])[0]
    def _extract_valid_clients(self, route, used_clients):
        """Extract valid clients from a route that haven't been used."""
        if not route or len(route) <= 2:
            return [self.start_point, self.end_point]
            
        new_route = [self.start_point]
        current_time = 0
        
        for client in route[1:-1]:
            if client.id not in used_clients:
                new_time = (current_time + 
                           self.get_distance(new_route[-1], client) + 
                           self.get_distance(client, self.end_point))
                if new_time <= self.L:
                    new_route.append(client)
                    current_time += self.get_distance(new_route[-2], client)
        
        new_route.append(self.end_point)
        return new_route

    def is_valid_route(self, route):
        """Check if a route is valid according to time constraints."""
        if not route or len(route) < 2:
            return False
            
        total_time = sum(self.get_distance(route[i], route[i+1]) 
                        for i in range(len(route)-1))
        return total_time <= self.L
    
    def _print_iteration_stats(self, generation: int) -> None:
        if generation % 10 == 0:
            print(f"\nIteration {generation}:")
            print(f"  Best Fitness: {self.best_fitness:.2f}")
            
            current_fitnesses = [self._adaptive_fitness(solution) for solution in self.population]
            print(f"  Average Fitness: {statistics.mean(current_fitnesses):.2f}")
            print(f"  Population Diversity: {self.population_entropy:.3f}")
            print(f"  Parameters - Mutation: {self.mutation_rate:.3f}, "
                f"Crossover: {self.crossover_rate:.3f}, "
                f"Tournament: {self.tournament_size}")

    def _print_final_stats(self) -> None:
        print("\n" + "="*50)
        print("Final Statistics:")
        print("="*50)
        print(f"Total Iterations: {len(self.diversity_history)}")
        print(f"Best Solution Fitness: {self.best_fitness:.2f}")
        print(f"Final Diversity: {self.population_entropy:.3f}")
        
        if self.best_solution:
            total_profit = sum(sum(c.profit for c in route[1:-1]) 
                            for route in self.best_solution)
            total_clients = sum(len(route)-2 for route in self.best_solution)
            total_time = sum(sum(self.get_distance(route[i], route[i+1])
                            for i in range(len(route)-1))
                        for route in self.best_solution)
            
            print(f"\nBest Solution Details:")
            print(f"Total Profit: {total_profit}")
            print(f"Total Clients Served: {total_clients}")
            print(f"Average Clients per Route: {total_clients/len(self.best_solution):.2f}")
            print(f"Total Time: {total_time:.2f}")
            
            print("\nRoute Details:")
            for i, route in enumerate(self.best_solution):
                route_profit = sum(c.profit for c in route[1:-1])
                route_time = sum(self.get_distance(route[i], route[i+1])
                            for i in range(len(route)-1))
                print(f"\nRoute {i+1}:")
                print(f"  Clients: {len(route)-2}")
                print(f"  Profit: {route_profit}")
                print(f"  Time: {route_time:.2f}")
    def get_stats(self):
        """Return collected statistics as a pandas DataFrame."""
        return pd.DataFrame(self.stats_data)



def lire_instance_chao(nom_fichier):
    with open(nom_fichier, "r") as f:
        lignes = f.readlines()
    L, m = map(float, lignes[0].split())
    points = []
    for i, ligne in enumerate(lignes[1:], 1):
        x, y, profit = map(float, ligne.split())
        points.append(Client(i-1, x, y, profit))
    return points[0], points[1], points[2:], int(m), L

def visualize_solution(solution, start_point, end_point, clients, filename):
    """Simple visualization of solution routes."""
    plt.figure(figsize=(10, 6))
    
    # Plot start and end points
    plt.scatter(start_point.x, start_point.y, c='green', marker='s', s=100, label='Start')
    plt.scatter(end_point.x, end_point.y, c='red', marker='s', s=100, label='End')
    
    # Plot routes with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(solution)))
    for i, (route, color) in enumerate(zip(solution, colors)):
        route_x = [c.x for c in route]
        route_y = [c.y for c in route]
        plt.plot(route_x, route_y, c=color, linewidth=2, label=f'Route {i+1}')
    
    plt.title('TOP Solution Routes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_algorithm_stats(stats_data, algorithm_name, output_dir):
    """Plot various statistics for algorithm performance."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Fitness Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(stats_data['iteration'], stats_data['best_fitness'], label='Best Fitness')
    plt.plot(stats_data['iteration'], stats_data['avg_fitness'], label='Average Fitness')
    plt.title(f'{algorithm_name} Fitness Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/{algorithm_name.lower()}_fitness.png')
    plt.close()
    
    # Diversity Evolution
    plt.figure(figsize=(10, 6))
    plt.plot(stats_data['iteration'], stats_data['diversity'])
    plt.title(f'{algorithm_name} Population Diversity')
    plt.xlabel('Iteration')
    plt.ylabel('Diversity')
    plt.grid(True)
    plt.savefig(f'{output_dir}/{algorithm_name.lower()}_diversity.png')
    plt.close()
    
    if 'mutation_rate' in stats_data:  # For Genetic Algorithm
        plt.figure(figsize=(10, 6))
        plt.plot(stats_data['iteration'], stats_data['mutation_rate'])
        plt.title(f'{algorithm_name} Mutation Rate Evolution')
        plt.xlabel('Iteration')
        plt.ylabel('Mutation Rate')
        plt.grid(True)
        plt.savefig(f'{output_dir}/{algorithm_name.lower()}_mutation_rate.png')
        plt.close()

def main(instance_file, debug=False):
    # Read instance
    start_point, end_point, clients, m, L = lire_instance_chao(instance_file)
    
    # Create output directory for plots
    output_dir = 'algorithm_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run algorithms and collect results
    algorithms = [
        
        ('AntColonyTOP', AntColonyTOP(start_point, end_point, clients, m, L, debug=debug)),
        ('GeneticTOP', GeneticTOP(start_point, end_point, clients, m, L, debug=debug))
    ]
    
    results = []
    for name, algorithm in algorithms:
        print(f"\nRunning {name}...")
        
        # Run algorithm
        start_time = time.time()
        if isinstance(algorithm, GeneticTOP):
            solution = algorithm.evolve()
        else:
            solution = algorithm.solve()
        execution_time = time.time() - start_time
        
        # Calculate solution quality
        total_profit = sum(sum(c.profit for c in route[1:-1]) for route in solution)
        total_clients = sum(len(route)-2 for route in solution)
        
        # Visualize solution
        visualize_solution(solution, start_point, end_point, clients,
                         f'{output_dir}/{name.lower()}_solution.png')
        
        # Store basic results
        results.append({
            'Algorithm': name,
            'Execution Time': execution_time,
            'Final Fitness': algorithm.best_fitness,
            'Total Profit': total_profit,
            'Total Clients': total_clients,
            'Routes': len(solution)
        })
        
        print(f"Completed {name}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Final fitness: {algorithm.best_fitness:.2f}")
        print(f"Total profit: {total_profit}")
        print(f"Total clients served: {total_clients}")
        print(f"Number of routes: {len(solution)}")
    
    # Save summary results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/summary_results.csv', index=False)
    print("\nResults saved in 'algorithm_results' directory")

if __name__ == "__main__":
    instance_file = "set_66_1/set_66_1_050.txt"
    main(instance_file, debug=True)  # Set debug=True if you want to see progress prints
    
