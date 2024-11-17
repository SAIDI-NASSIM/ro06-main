import math
import random
import numpy as np
from multiprocessing import Pool
import copy
import time
import pandas as pd
from typing import List, Tuple
import tracemalloc
from memory_profiler import profile

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
    cout_additionnel = (
        distance(route[position - 1], client)
        + distance(client, route[position])
        - distance(route[position - 1], route[position])
    )
    return client.profit - cout_additionnel

def heuristique_constructive_TOP(clients, depot, m, L):
    routes = []
    clients_non_visites = clients[:]

    for _ in range(m):
        route = [depot, depot]
        temps_restant = L

        while clients_non_visites:
            meilleur_client = max(
                clients_non_visites,
                key=lambda c: profit_incremental(route, c, len(route) - 1),
            )
            increment = profit_incremental(route, meilleur_client, len(route) - 1)

            if increment > 0 and temps_restant >= distance(
                route[-2], meilleur_client
            ) + distance(meilleur_client, depot):
                route.insert(-1, meilleur_client)
                temps_restant -= (
                    distance(route[-3], meilleur_client)
                    + distance(meilleur_client, route[-1])
                    - distance(route[-3], route[-1])
                )
                clients_non_visites.remove(meilleur_client)
            else:
                break

        if len(route) > 2:
            routes.append(route)

    return routes

def recherche_locale_TOP(routes, L, max_iter=50):
    a_ameliore = True
    iteration = 0

    while a_ameliore and iteration < max_iter:
        a_ameliore = False
        iteration += 1

        for route in routes:
            if appliquer_2opt(route, L):
                a_ameliore = True
            if appliquer_or_opt(route, L):
                a_ameliore = True

        if appliquer_string_cross(routes, L):
            a_ameliore = True
        if appliquer_string_exchange(routes, L):
            a_ameliore = True
        if appliquer_string_relocation(routes, L):
            a_ameliore = True

    return routes

def appliquer_2opt(route, L):
    a_ameliore = False
    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            nouvelle_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
            if temps_total(nouvelle_route) <= L and profit_total(nouvelle_route) > profit_total(route):
                route[:] = nouvelle_route
                a_ameliore = True
                return a_ameliore
    return a_ameliore

def appliquer_or_opt(route, L):
    a_ameliore = False
    for taille_sous_seq in range(1, 4):
        for i in range(1, len(route) - taille_sous_seq):
            sous_seq = route[i:i + taille_sous_seq]
            for j in range(1, len(route) - taille_sous_seq + 1):
                if i != j:
                    nouvelle_route = route[:i] + route[i + taille_sous_seq:j] + sous_seq + route[j:]
                    if temps_total(nouvelle_route) <= L and profit_total(nouvelle_route) > profit_total(route):
                        route[:] = nouvelle_route
                        a_ameliore = True
                        return a_ameliore
    return a_ameliore

def appliquer_string_cross(routes, L):
    a_ameliore = False
    for i, route1 in enumerate(routes):
        for j, route2 in enumerate(routes[i + 1:], i + 1):
            for a in range(1, len(route1) - 1):
                for b in range(1, len(route2) - 1):
                    nouvelle_route1 = route1[:a] + route2[b:]
                    nouvelle_route2 = route2[:b] + route1[a:]
                    if (temps_total(nouvelle_route1) <= L
                            and temps_total(nouvelle_route2) <= L
                            and profit_total(nouvelle_route1) + profit_total(nouvelle_route2)
                            > profit_total(route1) + profit_total(route2)):
                        routes[i], routes[j] = nouvelle_route1, nouvelle_route2
                        a_ameliore = True
                        return a_ameliore
    return a_ameliore

def appliquer_string_exchange(routes, L):
    a_ameliore = False
    for i, route1 in enumerate(routes):
        for j, route2 in enumerate(routes[i + 1:], i + 1):
            for a in range(1, len(route1) - 1):
                for b in range(1, len(route2) - 1):
                    nouvelle_route1 = route1[:a] + route2[b:b + 1] + route1[a + 1:]
                    nouvelle_route2 = route2[:b] + route1[a:a + 1] + route2[b + 1:]
                    if (temps_total(nouvelle_route1) <= L
                            and temps_total(nouvelle_route2) <= L
                            and profit_total(nouvelle_route1) + profit_total(nouvelle_route2)
                            > profit_total(route1) + profit_total(route2)):
                        routes[i], routes[j] = nouvelle_route1, nouvelle_route2
                        a_ameliore = True
                        return a_ameliore
    return a_ameliore

def appliquer_string_relocation(routes, L):
    a_ameliore = False
    for i, route1 in enumerate(routes):
        for j, route2 in enumerate(routes):
            if i != j:
                for a in range(1, len(route1) - 1):
                    for b in range(1, len(route2)):
                        nouvelle_route1 = route1[:a] + route1[a + 1:]
                        nouvelle_route2 = route2[:b] + [route1[a]] + route2[b:]
                        if (temps_total(nouvelle_route1) <= L
                                and temps_total(nouvelle_route2) <= L
                                and profit_total(nouvelle_route1) + profit_total(nouvelle_route2)
                                > profit_total(route1) + profit_total(route2)):
                            routes[i], routes[j] = nouvelle_route1, nouvelle_route2
                            a_ameliore = True
                            return a_ameliore
    return a_ameliore

def creer_tour_geant(clients, depot):
    tour_geant = [depot]
    clients_non_visites = clients[:]
    while clients_non_visites:
        prochain_client = max(
            clients_non_visites,
            key=lambda c: c.profit / distance(tour_geant[-1], c)
        )
        tour_geant.append(prochain_client)
        clients_non_visites.remove(prochain_client)
    tour_geant.append(depot)
    return tour_geant

def diviser_tour(tour_geant, m, L):
    n = len(tour_geant)
    F = [0] * n
    P = [0] * n

    for i in range(1, n):
        F[i] = float("-inf")
        temps = 0
        profit = 0
        j = i - 1
        while j >= 0 and temps <= L:
            temps += distance(tour_geant[j], tour_geant[j + 1])
            if j > 0:
                profit += tour_geant[j].profit
            if temps <= L:
                if F[j] + profit > F[i]:
                    F[i] = F[j] + profit
                    P[i] = j
            j -= 1

    routes = []
    i = n - 1
    while i > 0:
        route = tour_geant[P[i]:i + 1]
        routes.append([tour_geant[0]] + route + [tour_geant[0]])
        i = P[i]

    return routes[::-1][:m]

def beasley_adapte_TOP(clients, depot, m, L):
    tour_geant = creer_tour_geant(clients, depot)
    routes = diviser_tour(tour_geant, m, L)
    return routes

class GeneticTOP:
    def __init__(self, depot, clients, m: int, L: float):
        self.depot = depot
        self.clients = clients
        self.m = m
        self.L = L
        n_clients = len(clients)
        self.population_size = min(n_clients * 20, 400)
        self.generations = min(n_clients * 15, 300)
        self.base_mutation_rate = max(0.1, 2.0 / n_clients)
        self.elite_size = max(10, self.population_size // 10)
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.diversity_threshold = 0.3
        self.local_search_freq = 5
    def _is_route_valid(self, route):
        if len(route) < 2:  # Route should at least have depot start and end
            return False
        if route[0] != self.depot or route[-1] != self.depot:  # Should start and end at depot
            return False
        # Check for duplicates
        if len(set(client.id for client in route[1:-1])) != len(route[1:-1]):
            return False
        # Check time constraint
        if temps_total(route) > self.L:
            return False
        return True
    def _adaptive_params(self, diversity):
        mutation_rate = self.base_mutation_rate * (1 + (1 - diversity) * 2)
        crossover_rate = 0.8 * (1 + diversity)
        return mutation_rate, crossover_rate
    def _calculate_diversity(self, population):
        if not population:
            return 0
        total_clients = set()
        for solution in population:
            for route in solution:
                total_clients.update(c.id for c in route[1:-1])
        unique_arrangements = len(set(tuple(tuple(c.id for c in route[1:-1]) for route in sol) for sol in population))
        return unique_arrangements / len(population)
    def fitness(self, solution):
        if not solution:
            return float('-inf')
        total_profit = 0
        total_distance = 0
        total_clients = 0
        for route in solution:
            if not self._is_route_valid(route):
                return float('-inf')
            route_profit = profit_total(route)
            route_distance = temps_total(route)
            if route_distance > 0:
                total_profit += route_profit * (1 + 0.1 * (self.L - route_distance) / self.L)
            total_distance += route_distance
            total_clients += len(route) - 2
        diversity_bonus = 0.05 * total_profit * (total_clients / len(self.clients))
        return total_profit + diversity_bonus - 0.01 * total_distance
    def _create_greedy_solution(self):
        solution = []
        available_clients = self.clients.copy()
        random.shuffle(available_clients)
        for _ in range(self.m):
            if not available_clients:
                break
            route = [self.depot]
            current_time = 0
            while available_clients:
                candidates = [(c, c.profit / (distance(route[-1], c) + distance(c, self.depot))) for c in available_clients[:min(5, len(available_clients))]]
                best_client = max(candidates, key=lambda x: x[1])[0]
                new_time = current_time + distance(route[-1], best_client) + distance(best_client, self.depot)
                if new_time <= self.L:
                    route.append(best_client)
                    current_time = new_time - distance(best_client, self.depot)
                    available_clients.remove(best_client)
                else:
                    break
            if len(route) > 1:
                route.append(self.depot)
                solution.append(route)
        return solution
    def _local_search(self, solution):
        if not solution:
            return solution
        improved = True
        while improved:
            improved = False
            for i, route in enumerate(solution):
                if len(route) <= 3:
                    continue
                for j in range(1, len(route)-2):
                    for k in range(j+1, len(route)-1):
                        new_route = route[:j] + route[j:k+1][::-1] + route[k+1:]
                        if temps_total(new_route) <= self.L:
                            old_fitness = self.fitness([route])
                            new_fitness = self.fitness([new_route])
                            if new_fitness > old_fitness:
                                solution[i] = new_route
                                improved = True
        return solution
    def crossover(self, parent1, parent2):
        if not parent1 or not parent2:
            return parent1, parent2
        child1, child2 = [], []
        used_clients1, used_clients2 = set(), set()
        min_length = min(len(parent1), len(parent2))
        if min_length <= 2:
            return parent1, parent2
        crossover_point = random.randint(1, min_length-1)
        child1.extend(copy.deepcopy(parent1[:crossover_point]))
        child2.extend(copy.deepcopy(parent2[:crossover_point]))
        used_clients1.update(c.id for route in child1 for c in route[1:-1])
        used_clients2.update(c.id for route in child2 for c in route[1:-1])
        remaining_routes1 = [route for route in parent2[crossover_point:] if all(c.id not in used_clients1 for c in route[1:-1])]
        remaining_routes2 = [route for route in parent1[crossover_point:] if all(c.id not in used_clients2 for c in route[1:-1])]
        child1.extend(copy.deepcopy(remaining_routes1))
        child2.extend(copy.deepcopy(remaining_routes2))
        return child1, child2
    def mutation(self, solution, mutation_rate):
        if random.random() > mutation_rate or not solution:
            return solution
        mutated = copy.deepcopy(solution)
        mutation_type = random.choice(['swap', 'insert', 'reverse', 'merge_split', 'relocate'])
        if mutation_type == 'swap' and len(mutated) >= 2:
            route_idx1, route_idx2 = random.sample(range(len(mutated)), 2)
            if len(mutated[route_idx1]) > 2 and len(mutated[route_idx2]) > 2:
                pos1 = random.randint(1, len(mutated[route_idx1])-2)
                pos2 = random.randint(1, len(mutated[route_idx2])-2)
                mutated[route_idx1][pos1], mutated[route_idx2][pos2] = mutated[route_idx2][pos2], mutated[route_idx1][pos1]
        elif mutation_type == 'relocate' and len(mutated) >= 2:
            if random.random() < 0.5:
                route_idx = random.randint(0, len(mutated)-1)
                if len(mutated[route_idx]) > 3:
                    pos = random.randint(1, len(mutated[route_idx])-2)
                    client = mutated[route_idx].pop(pos)
                    new_route = [self.depot, client, self.depot]
                    if temps_total(new_route) <= self.L:
                        mutated.append(new_route)
        return self._local_search(mutated) if random.random() < 0.1 else mutated
    def _solution_similarity(self, solution1, solution2):
        if not solution1 or not solution2:
            return 0
        clients1 = set(c.id for route in solution1 for c in route[1:-1])
        clients2 = set(c.id for route in solution2 for c in route[1:-1])
        common_clients = clients1.intersection(clients2)
        return len(common_clients) / (len(clients1) + len(clients2) - len(common_clients))
    def _calculate_solution_diversity(self, solution, population):
        if not solution or not population:
            return 0
        return sum(self._solution_similarity(solution, other) for other in population) / len(population)
    def _create_random_solution(self):
        solution = []
        available_clients = self.clients.copy()
        random.shuffle(available_clients)
        for _ in range(self.m):
            if not available_clients:
                break
            route = [self.depot]
            current_time = 0
            while available_clients:
                client = random.choice(available_clients)
                new_time = current_time + distance(route[-1], client) + distance(client, self.depot)
                if new_time <= self.L:
                    route.append(client)
                    current_time = new_time - distance(client, self.depot)
                    available_clients.remove(client)
                else:
                    break
            if len(route) > 1:
                route.append(self.depot)
                solution.append(route)
        return solution
               
    def evolve(self):
            # Initialize population with more diversity
            population = []
            for _ in range(self.population_size):
                if random.random() < 0.7:  # 70% greedy solutions
                    solution = self._create_greedy_solution()
                else:  # 30% random solutions for diversity
                    solution = self._create_random_solution()
                population.append(solution)
            
            generations_without_improvement = 0
            reset_count = 0
            max_resets = 3
            best_fitness_history = []
            
            for generation in range(self.generations):
                diversity = self._calculate_diversity(population)
                mutation_rate, crossover_rate = self._adaptive_params(diversity)
                
                # Dynamic mutation rate based on stagnation
                if generations_without_improvement > 15:
                    mutation_rate = min(0.9, mutation_rate * (1 + generations_without_improvement/50))
                
                # Evaluate population with penalties for similar solutions
                fitness_values = []
                for sol in population:
                    base_fitness = self.fitness(sol)
                    # Add diversity penalty
                    similar_solutions = sum(1 for other_sol in population[:10] 
                                        if self._solution_similarity(sol, other_sol) > 0.8)
                    diversity_penalty = 0.05 * base_fitness * (similar_solutions / 10)
                    fitness_values.append(base_fitness - diversity_penalty)
                
                # Sort population by penalized fitness
                population = [x for _, x in sorted(zip(fitness_values, population), 
                                                key=lambda pair: pair[0], reverse=True)]
                
                # Stronger elitism - keep best 20% solutions
                elite_size = self.population_size // 5
                new_population = population[:elite_size]
                
                # Tournament selection with diversity consideration
                while len(new_population) < self.population_size:
                    tournament_size = 5
                    tournament = random.sample(population, tournament_size)
                    tournament_diversity = [self._calculate_solution_diversity(sol, population) 
                                        for sol in tournament]
                    # Select based on both fitness and diversity
                    parent1 = max(zip(tournament, tournament_diversity), 
                                key=lambda x: self.fitness(x[0]) * (1 + 0.2 * x[1]))[0]
                    parent2 = max(zip(tournament, tournament_diversity), 
                                key=lambda x: self.fitness(x[0]) * (1 + 0.2 * x[1]))[0]
                    
                    if random.random() < crossover_rate:
                        child1, child2 = self.crossover(parent1, parent2)
                        child1 = self.mutation(child1, mutation_rate)
                        child2 = self.mutation(child2, mutation_rate)
                        new_population.extend([child1, child2])
                
                # Periodic intensification through local search
                if generation % self.local_search_freq == 0:
                    for i in range(elite_size):
                        new_population[i] = self._local_search(new_population[i])
                
                population = new_population[:self.population_size]  # Ensure fixed size
                
                current_best = max(population, key=self.fitness)
                current_fitness = self.fitness(current_best)
                best_fitness_history.append(current_fitness)
                
                # Update best solution if improved
                if current_fitness > self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_solution = copy.deepcopy(current_best)
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Population reset with memory
                if generations_without_improvement >= 30:
                    reset_count += 1
                    if reset_count >= max_resets:
                        # Check for significant improvement trend
                        if len(best_fitness_history) > 50:
                            recent_improvement = (max(best_fitness_history[-50:]) - 
                                            min(best_fitness_history[-50:])) / min(best_fitness_history[-50:])
                            if recent_improvement < 0.01:  # Less than 1% improvement
                                print(f"Early stopping at generation {generation}: Stagnated improvement")
                                break
                    
                    print(f"Resetting population at generation {generation} (Reset #{reset_count})")
                    
                    # Keep top 10% solutions
                    preserved_solutions = population[:self.population_size//10]
                    population = []
                    
                    # Add preserved solutions
                    population.extend(preserved_solutions)
                    
                    # Add modified versions of best solutions
                    for solution in preserved_solutions:
                        for _ in range(3):
                            modified = self.mutation(copy.deepcopy(solution), mutation_rate * 2)
                            population.append(modified)
                    
                    # Fill rest with new diverse solutions
                    while len(population) < self.population_size:
                        if random.random() < 0.6:
                            solution = self._create_greedy_solution()
                        else:
                            solution = self._create_random_solution()
                        population.append(solution)
                    
                    generations_without_improvement = 0
                
                if generation % 10 == 0:
                    print(f"Generation {generation}: Best Fitness = {self.best_fitness}")
            
            return self.best_solution

class ParallelGeneticTOP(GeneticTOP):
    def __init__(self, *args, n_processes=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_processes = n_processes
        
    def parallel_fitness(self, solutions):
        with Pool(self.n_processes) as pool:
            return pool.map(self.fitness, solutions)
            
    def evolve(self) -> List[List[Client]]:
        population = self.create_initial_population()
        
        for generation in range(self.generations):
            # Sélection avec fitness parallèle
            fitness_values = self.parallel_fitness(population)
            population_with_fitness = list(zip(population, fitness_values))
            sorted_population = sorted(population_with_fitness, 
                                    key=lambda x: x[1], 
                                    reverse=True)
            
            new_population = [sol for sol, _ in sorted_population[:self.elite_size]]
            
            while len(new_population) < self.population_size:
                tournament = random.sample(population_with_fitness, 3)
                winner = max(tournament, key=lambda x: x[1])[0]
                new_population.append(copy.deepcopy(winner))
            
            # Crossover et mutation
            offspring = []
            for i in range(0, len(new_population)-1, 2):
                child1, child2 = self.crossover(new_population[i], 
                                              new_population[i+1])
                offspring.extend([child1, child2])
            
            # Mutation parallèle
            with Pool(self.n_processes) as pool:
                offspring = pool.map(self.mutation, offspring)
            
            # Mise à jour de la population
            population = offspring
            
            # Mise à jour de la meilleure solution
            current_best = max(population, key=self.fitness)
            current_fitness = self.fitness(current_best)
            
            if current_fitness > self.best_fitness:
                self.best_fitness = current_fitness
                self.best_solution = copy.deepcopy(current_best)
                
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_fitness}")
                
        return self.best_solution
    
def lire_instance_chao(nom_fichier):
    with open(nom_fichier, "r") as f:
        lignes = f.readlines()
    n, m = map(int, lignes[0].split())
    clients = []
    depot = None
    for i, ligne in enumerate(lignes[1:]):
        x, y, profit = map(float, ligne.split())
        if i == 0:
            depot = Client(0, x, y, 0)
        else:
            clients.append(Client(i, x, y, profit))
    L = 100  # Default time limit
    return depot, clients, m, L

def visualize_solution(solution, depot, clients, filename):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Plot all clients
    client_x = [c.x for c in clients]
    client_y = [c.y for c in clients]
    plt.scatter(client_x, client_y, c='gray', alpha=0.5, label='Unvisited Clients')
    
    # Plot depot
    plt.scatter(depot.x, depot.y, c='black', marker='s', s=100, label='Depot')
    
    # Plot routes with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(solution)))
    for route, color in zip(solution, colors):
        route_x = [c.x for c in route]
        route_y = [c.y for c in route]
        plt.plot(route_x, route_y, c=color, linewidth=2)
        plt.scatter(route_x[1:-1], route_y[1:-1], c=[color], s=100, label=f'Route {len(route)-2} clients')
    
    plt.title('TOP Solution Visualization')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def compare_algorithms(depot, clients, m, L):
    results = []
    algorithms = [
        ('Basic Heuristic', lambda: heuristique_constructive_TOP(clients, depot, m, L)),
        ('Beasley Adapted', lambda: beasley_adapte_TOP(clients, depot, m, L)),
        ('Genetic Algorithm', lambda: GeneticTOP(depot, clients, m, L).evolve())
        # ,('Parallel Genetic', lambda: ParallelGeneticTOP(depot, clients, m, L).evolve()) # doesnt work for now
    ]
    
    for name, algo in algorithms:
        # Memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        # Run algorithm
        solution = algo()
        
        # Collect metrics
        execution_time = time.time() - start_time
        memory_current, memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate profit
        total_profit = sum(profit_total(route) for route in solution)
        
        # Visualize solution
        visualize_solution(solution, depot, clients, f'solution_{name.lower().replace(" ", "_")}.png')
        
        results.append({
            'Algorithm': name,
            'Execution Time (s)': execution_time,
            'Memory Usage (MB)': memory_peak / (1024 * 1024),
            'Total Profit': total_profit,
            'Number of Routes': len(solution),
            'Total Clients Served': sum(len(route) - 2 for route in solution)
        })
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    results_df.to_csv('algorithm_comparison.csv', index=False)
    
    return results_df

def main():
    # Read instance
    nom_fichier = "set_64_1/set_64_1_15.txt"
    depot, clients, m, L = lire_instance_chao(nom_fichier)
    
    # Compare algorithms and get results
    results_df = compare_algorithms(depot, clients, m, L)
    
    # Display results
    print("\nAlgorithm Comparison Results:")
    print(results_df.to_string(index=False))
    
    # Visualize comparison
    import matplotlib.pyplot as plt
    
    # Plot execution time vs profit
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Execution Time (s)'], results_df['Total Profit'])
    for i, label in enumerate(results_df['Algorithm']):
        plt.annotate(label, (results_df['Execution Time (s)'].iloc[i], 
                           results_df['Total Profit'].iloc[i]))
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Total Profit')
    plt.title('Algorithm Performance Comparison')
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()