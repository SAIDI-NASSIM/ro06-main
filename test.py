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
    def __init__(self, depot, clients, m: int, L: float, 
                 population_size: int = 100,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 elite_size: int = 10):
        self.depot = depot
        self.clients = clients
        self.m = m
        self.L = L
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.best_solution = None
        self.best_fitness = float('-inf')
        
    def create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            available_clients = self.clients.copy()
            solution = []
            
            for _ in range(self.m):
                if not available_clients:
                    break
                    
                route = [self.depot]
                current_time = 0
                
                while available_clients:
                    client = random.choice(available_clients)
                    new_time = (current_time + 
                              distance(route[-1], client) + 
                              distance(client, self.depot))
                    
                    if new_time <= self.L:
                        route.append(client)
                        current_time = new_time
                        available_clients.remove(client)
                    else:
                        break
                        
                if len(route) > 1:
                    route.append(self.depot)
                    solution.append(route)
                    
            population.append(solution)
        return population

    def fitness(self, solution):
        total_profit = 0
        for route in solution:
            if not self._is_route_valid(route):
                return float('-inf')
            total_profit += profit_total(route)
        return total_profit
    
    def _is_route_valid(self, route):
        if not route or route[0] != self.depot or route[-1] != self.depot:
            return False
        return temps_total(route) <= self.L

    def selection(self, population):
        tournament_size = 3
        selected = []
        
        sorted_population = sorted(population, 
                                 key=lambda x: self.fitness(x), 
                                 reverse=True)
        selected.extend(copy.deepcopy(sorted_population[:self.elite_size]))
        
        while len(selected) < self.population_size:
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda x: self.fitness(x))
            selected.append(copy.deepcopy(winner))
            
        return selected

    def crossover(self, parent1, parent2):
        if not parent1 or not parent2:
            return parent1, parent2
            
        child1 = []
        child2 = []
        
        for route in parent1:
            if random.random() < 0.5:
                child1.append(copy.deepcopy(route))
                
        for route in parent2:
            if random.random() < 0.5:
                child2.append(copy.deepcopy(route))
                
        used_clients1 = {client.id for route in child1 
                        for client in route[1:-1]}
        used_clients2 = {client.id for route in child2 
                        for client in route[1:-1]}
        
        available_clients1 = [client for client in self.clients 
                            if client.id not in used_clients1]
        available_clients2 = [client for client in self.clients 
                            if client.id not in used_clients2]
        
        while available_clients1 and len(child1) < self.m:
            route = self._create_random_route(available_clients1)
            if route:
                child1.append(route)
                used_clients = {client.id for client in route[1:-1]}
                available_clients1 = [client for client in available_clients1 
                                    if client.id not in used_clients]
                
        while available_clients2 and len(child2) < self.m:
            route = self._create_random_route(available_clients2)
            if route:
                child2.append(route)
                used_clients = {client.id for client in route[1:-1]}
                available_clients2 = [client for client in available_clients2 
                                    if client.id not in used_clients]
                
        return child1, child2

    def _create_random_route(self, available_clients):
        if not available_clients:
            return None
            
        route = [self.depot]
        current_time = 0
        clients_copy = available_clients.copy()
        
        while clients_copy:
            client = random.choice(clients_copy)
            new_time = (current_time + 
                       distance(route[-1], client) + 
                       distance(client, self.depot))
            
            if new_time <= self.L:
                route.append(client)
                current_time = new_time
                clients_copy.remove(client)
            else:
                break
                
        if len(route) > 1:
            route.append(self.depot)
            return route
        return None
    
    def mutation(self, solution: List[List[Client]]) -> List[List[Client]]:
            if random.random() > self.mutation_rate:
                return solution
                
            mutated = copy.deepcopy(solution)
            
            # Choisir aléatoirement une opération de mutation
            mutation_ops = ['swap', 'insert', 'reverse']
            operation = random.choice(mutation_ops)
            
            if operation == 'swap':
                # Échanger deux clients aléatoires
                if len(mutated) >= 2:
                    route_idx1 = random.randint(0, len(mutated)-1)
                    route_idx2 = random.randint(0, len(mutated)-1)
                    
                    if (len(mutated[route_idx1]) > 2 and 
                        len(mutated[route_idx2]) > 2):
                        pos1 = random.randint(1, len(mutated[route_idx1])-2)
                        pos2 = random.randint(1, len(mutated[route_idx2])-2)
                        
                        (mutated[route_idx1][pos1], 
                        mutated[route_idx2][pos2]) = (mutated[route_idx2][pos2], 
                                                    mutated[route_idx1][pos1])
                        
            elif operation == 'insert':
                # Déplacer un client vers une nouvelle position
                if mutated:
                    route_idx = random.randint(0, len(mutated)-1)
                    if len(mutated[route_idx]) > 3:
                        pos1 = random.randint(1, len(mutated[route_idx])-2)
                        pos2 = random.randint(1, len(mutated[route_idx])-2)
                        client = mutated[route_idx].pop(pos1)
                        mutated[route_idx].insert(pos2, client)
                        
            else:  # reverse
                # Inverser une sous-séquence dans une route
                if mutated:
                    route_idx = random.randint(0, len(mutated)-1)
                    if len(mutated[route_idx]) > 3:
                        pos1 = random.randint(1, len(mutated[route_idx])-3)
                        pos2 = random.randint(pos1+1, len(mutated[route_idx])-2)
                        mutated[route_idx][pos1:pos2+1] = (
                            mutated[route_idx][pos1:pos2+1][::-1]
                        )
                        
            return mutated

    def evolve(self) -> List[List[Client]]:
        population = self.create_initial_population()
        
        for generation in range(self.generations):
            # Sélection
            new_population = self.selection(population)
            
            # Crossover
            offspring = []
            for i in range(0, len(new_population)-1, 2):
                child1, child2 = self.crossover(new_population[i], 
                                              new_population[i+1])
                offspring.extend([child1, child2])
            
            # Mutation
            offspring = [self.mutation(sol) for sol in offspring]
            
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
    nom_fichier = "set_66_1/set_66_1_005.txt"
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