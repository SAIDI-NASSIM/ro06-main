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

    def _construct_route(self, ant_id):
        route = [self.start_point]
        current = self.start_point
        current_time = 0
        available = set(self.clients)
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
        available_clients = set(self.clients)
        for _ in range(self.m):
            if not available_clients:
                break
            route = self._construct_route(len(solution))
            if len(route) > 2:
                solution.append(route)
                available_clients -= set(route[1:-1])
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
    # Extract instance size from filename
    instance_size = instance_file.split('_')[-1].replace('.txt', '')
    
    # Read instance
    start_point, end_point, clients, m, L = lire_instance_chao(instance_file)
    
    # Create output directory with instance size
    output_dir = f'algorithm_results_{instance_size}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run algorithms and collect results
    algorithms = [
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
        
        # Calculate solution metrics
        total_profit = sum(sum(c.profit for c in route[1:-1]) for route in solution)
        total_clients = sum(len(route)-2 for route in solution)
        total_distance = sum(sum(math.sqrt((route[i].x - route[i+1].x)**2 + 
                                         (route[i].y - route[i+1].y)**2)
                              for i in range(len(route)-1))
                           for route in solution)
        
        # Visualize solution and stats
        visualize_solution(solution, start_point, end_point, clients,
                         f'{output_dir}/{name.lower()}_solution.png')
        plot_algorithm_stats(algorithm.get_stats(), name, output_dir)
        
        # Store results
        results.append({
            'Instance_Size': instance_size,
            'Algorithm': name,
            'Execution_Time': execution_time,
            'Final_Fitness': algorithm.best_fitness,
            'Total_Profit': total_profit,
            'Total_Clients': total_clients,
            'Total_Distance': total_distance,
            'Routes': len(solution),
            'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        print(f"Completed {name}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Final fitness: {algorithm.best_fitness:.2f}")
        print(f"Total profit: {total_profit}")
        print(f"Total clients served: {total_clients}")
        print(f"Number of routes: {len(solution)}")
    
    # Append results to history CSV
    results_df = pd.DataFrame(results)
    history_file = 'algorithm_history.csv'
    
    if os.path.exists(history_file):
        results_df.to_csv(history_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(history_file, index=False)
    
    print(results_df)
    print(f"\nResults appended to {history_file}")

if __name__ == "__main__":
    instance_file = "set_66_1/set_66_1_100.txt"
    main(instance_file, debug=True)
