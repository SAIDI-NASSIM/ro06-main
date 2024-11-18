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
    cout_additionnel = (distance(route[position - 1], client) + distance(client, route[position]) - distance(route[position - 1], route[position]))
    return client.profit - cout_additionnel

class GeneticTOP:
    def __init__(self, start_point, end_point, clients, m: int, L: float):
        self.start_point = start_point
        self.end_point = end_point
        self.clients = clients
        self.m = m
        self.L = L
        n = len(clients)
        self.population_size = min(100, n * 4)
        self.tournament_size = max(3, self.population_size // 20)
        self.elite_size = max(2, self.population_size // 10)
        self.generations = min(150, n * 8)
        self.mutation_rate = min(0.3, 1.0 / n)
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.convergence_generations = 15
        self.convergence_threshold = 0.001
        self.max_distance = max(distance(c1, c2) for c1 in clients for c2 in clients)
        self.total_possible_profit = sum(c.profit for c in clients)

    def _route_fitness(self, route):
        if len(route) < 3 or route[0] != self.start_point or route[-1] != self.end_point:
            return 0
        time = sum(distance(route[i], route[i+1]) for i in range(len(route)-1))
        if time > self.L:
            return 0
        profit = sum(c.profit for c in route[1:-1])
        time_ratio = time / self.L
        efficiency_bonus = 1 + 0.2 * (time_ratio if time_ratio <= 0.95 else 2 - time_ratio)
        profit_per_time = profit / time if time > 0 else 0
        density_bonus = 1 + 0.1 * (profit_per_time / (self.total_possible_profit / self.L))
        return profit * efficiency_bonus * density_bonus

    def fitness(self, solution):
        if not solution:
            return 0
        total_profit = sum(self._route_fitness(route) for route in solution)
        served_clients = len({c.id for route in solution for c in route[1:-1]})
        coverage_ratio = served_clients / len(self.clients)
        coverage_bonus = 1 + 0.3 * coverage_ratio
        route_lengths = [len(route) - 2 for route in solution]
        length_variance = np.var(route_lengths) if route_lengths else 0
        balance_bonus = 1 + 0.2 * (1 / (1 + length_variance))
        return total_profit * coverage_bonus * balance_bonus

    def _create_initial_route(self):
        available = set(self.clients)
        route = [self.start_point]
        current_time = 0
        while available and current_time < self.L:
            candidates = [(c, c.profit/distance(route[-1], c)) for c in available]
            if not candidates:
                break
            client = max(candidates, key=lambda x: x[1])[0]
            new_time = current_time + distance(route[-1], client) + distance(client, self.end_point)
            if new_time <= self.L:
                route.append(client)
                available.remove(client)
                current_time = new_time - distance(client, self.end_point)
            else:
                break
        route.append(self.end_point)
        return route if len(route) > 2 else None

    def _create_initial_solution(self):
        solution = []
        for _ in range(self.m):
            route = self._create_initial_route()
            if route:
                solution.append(route)
        return solution

    def crossover(self, parent1, parent2):
        if not parent1 or not parent2:
            return parent1, parent2
        child1, child2 = [], []
        used1, used2 = set(), set()
        for i in range(min(len(parent1), len(parent2))):
            route1, route2 = parent1[i], parent2[i]
            if random.random() < 0.5:
                route1, route2 = route2, route1
            new_route1 = [self.start_point] + [c for c in route1[1:-1] if c.id not in used1] + [self.end_point]
            new_route2 = [self.start_point] + [c for c in route2[1:-1] if c.id not in used2] + [self.end_point]
            used1.update(c.id for c in new_route1[1:-1])
            used2.update(c.id for c in new_route2[1:-1])
            if len(new_route1) > 2:
                child1.append(new_route1)
            if len(new_route2) > 2:
                child2.append(new_route2)
        return child1, child2

    def mutation(self, solution):
        if not solution or random.random() > self.mutation_rate:
            return solution
        mutated = copy.deepcopy(solution)
        mutation_type = random.random()
        if mutation_type < 0.4 and len(mutated) >= 2:
            i, j = random.sample(range(len(mutated)), 2)
            if len(mutated[i]) > 2 and len(mutated[j]) > 2:
                pos1 = random.randint(1, len(mutated[i])-2)
                pos2 = random.randint(1, len(mutated[j])-2)
                mutated[i][pos1], mutated[j][pos2] = mutated[j][pos2], mutated[i][pos1]
        elif mutation_type < 0.7:
            i = random.randint(0, len(mutated)-1)
            if len(mutated[i]) > 3:
                pos1, pos2 = sorted(random.sample(range(1, len(mutated[i])-1), 2))
                mutated[i][pos1:pos2] = reversed(mutated[i][pos1:pos2])
        else:
            i = random.randint(0, len(mutated)-1)
            if len(mutated[i]) > 3:
                pos = random.randint(1, len(mutated[i])-2)
                client = mutated[i].pop(pos)
                for j, route in enumerate(mutated):
                    if j != i and temps_total(route[:-1] + [client, route[-1]]) <= self.L:
                        route.insert(-1, client)
                        break
        return mutated

    def evolve(self):
        population = [self._create_initial_solution() for _ in range(self.population_size)]
        best_fitness_counter = 0
        for generation in range(self.generations):
            population.sort(key=self.fitness, reverse=True)
            current_best_fitness = self.fitness(population[0])
            if current_best_fitness > self.best_fitness * (1 + self.convergence_threshold):
                self.best_fitness = current_best_fitness
                self.best_solution = copy.deepcopy(population[0])
                best_fitness_counter = 0
            else:
                best_fitness_counter += 1
            if best_fitness_counter >= self.convergence_generations:
                print(f"Early stopping at generation {generation}")
                break
            new_population = population[:self.elite_size]
            while len(new_population) < self.population_size:
                tournament = random.sample(population, self.tournament_size)
                parent1 = max(tournament, key=self.fitness)
                tournament = random.sample(population, self.tournament_size)
                parent2 = max(tournament, key=self.fitness)
                child1, child2 = self.crossover(parent1, parent2)
                child1, child2 = self.mutation(child1), self.mutation(child2)
                if child1:
                    new_population.append(child1)
                if child2 and len(new_population) < self.population_size:
                    new_population.append(child2)
            population = new_population[:self.population_size]
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_fitness}")
        return self.best_solution

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
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    client_x = [c.x for c in clients]
    client_y = [c.y for c in clients]
    plt.scatter(client_x, client_y, c='gray', alpha=0.5, label='Unvisited Clients')
    plt.scatter(start_point.x, start_point.y, c='green', marker='s', s=100, label='Start')
    plt.scatter(end_point.x, end_point.y, c='red', marker='s', s=100, label='End')
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

def compare_algorithms(start_point, end_point, clients, m, L):
    results = []
    algorithms = [
        ('Genetic Algorithm', lambda: GeneticTOP(start_point, end_point, clients, m, L).evolve())
    ]
    for name, algo in algorithms:
        tracemalloc.start()
        start_time = time.time()
        solution = algo()
        execution_time = time.time() - start_time
        memory_current, memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_profit = sum(profit_total(route) for route in solution)
        visualize_solution(solution, start_point, end_point, clients, f'solution_{name.lower().replace(" ", "_")}.png')
        results.append({
            'Algorithm': name,
            'Execution Time (s)': execution_time,
            'Memory Usage (MB)': memory_peak / (1024 * 1024),
            'Total Profit': total_profit,
            'Number of Routes': len(solution),
            'Total Clients Served': sum(len(route) - 2 for route in solution)
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv('algorithm_comparison.csv', index=False)
    return results_df

def main():
    nom_fichier = "set_66_1/set_66_1_110.txt"
    start_point, end_point, clients, m, L = lire_instance_chao(nom_fichier)
    results_df = compare_algorithms(start_point, end_point, clients, m, L)
    print("\nAlgorithm Comparison Results:")
    print(results_df.to_string(index=False))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Execution Time (s)'], results_df['Total Profit'])
    for i, label in enumerate(results_df['Algorithm']):
        plt.annotate(label, (results_df['Execution Time (s)'].iloc[i], results_df['Total Profit'].iloc[i]))
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Total Profit')
    plt.title('Algorithm Performance Comparison')
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()