import math
import random


class Client:
    def __init__(self, id, x, y, profit):
        self.id = id
        self.x = x
        self.y = y
        self.profit = profit


def distance(client1, client2):
    return math.sqrt((client1.x - client2.x) ** 2 + (client1.y - client2.y) ** 2)


def temps_total(route):
    temps = sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1))
    return temps


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
        print(f"Iteration {iteration}")
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
            nouvelle_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
            if temps_total(nouvelle_route) <= L and profit_total(
                nouvelle_route
            ) > profit_total(route):
                route[:] = nouvelle_route
                a_ameliore = True
                return a_ameliore
    return a_ameliore


def appliquer_or_opt(route, L):
    a_ameliore = False
    for taille_sous_seq in range(1, 4):
        for i in range(1, len(route) - taille_sous_seq):
            sous_seq = route[i : i + taille_sous_seq]
            for j in range(1, len(route) - taille_sous_seq + 1):
                if i != j:
                    nouvelle_route = (
                        route[:i]
                        + route[i + taille_sous_seq : j]
                        + sous_seq
                        + route[j:]
                    )
                    if temps_total(nouvelle_route) <= L and profit_total(
                        nouvelle_route
                    ) > profit_total(route):
                        route[:] = nouvelle_route
                        a_ameliore = True
                        return a_ameliore
    return a_ameliore


def appliquer_string_cross(routes, L):
    a_ameliore = False
    for i, route1 in enumerate(routes):
        for j, route2 in enumerate(routes[i + 1 :], i + 1):
            for a in range(1, len(route1) - 1):
                for b in range(1, len(route2) - 1):
                    nouvelle_route1 = route1[:a] + route2[b:]
                    nouvelle_route2 = route2[:b] + route1[a:]
                    if (
                        temps_total(nouvelle_route1) <= L
                        and temps_total(nouvelle_route2) <= L
                        and profit_total(nouvelle_route1)
                        + profit_total(nouvelle_route2)
                        > profit_total(route1) + profit_total(route2)
                    ):
                        routes[i], routes[j] = nouvelle_route1, nouvelle_route2
                        a_ameliore = True
                        return a_ameliore
    return a_ameliore


def appliquer_string_exchange(routes, L):
    a_ameliore = False
    for i, route1 in enumerate(routes):
        for j, route2 in enumerate(routes[i + 1 :], i + 1):
            for a in range(1, len(route1) - 1):
                for b in range(1, len(route2) - 1):
                    nouvelle_route1 = route1[:a] + route2[b : b + 1] + route1[a + 1 :]
                    nouvelle_route2 = route2[:b] + route1[a : a + 1] + route2[b + 1 :]
                    if (
                        temps_total(nouvelle_route1) <= L
                        and temps_total(nouvelle_route2) <= L
                        and profit_total(nouvelle_route1)
                        + profit_total(nouvelle_route2)
                        > profit_total(route1) + profit_total(route2)
                    ):
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
                        nouvelle_route1 = route1[:a] + route1[a + 1 :]
                        nouvelle_route2 = route2[:b] + [route1[a]] + route2[b:]
                        if (
                            temps_total(nouvelle_route1) <= L
                            and temps_total(nouvelle_route2) <= L
                            and profit_total(nouvelle_route1)
                            + profit_total(nouvelle_route2)
                            > profit_total(route1) + profit_total(route2)
                        ):
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
            key=lambda c: c.profit / distance(tour_geant[-1], c),
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

    # Reconstruire les routes
    routes = []
    i = n - 1
    while i > 0:
        route = tour_geant[P[i] : i + 1]
        routes.append([tour_geant[0]] + route + [tour_geant[0]])
        i = P[i]

    return routes[::-1][:m]


def beasley_adapte_TOP(clients, depot, m, L):
    tour_geant = creer_tour_geant(clients, depot)
    routes = diviser_tour(tour_geant, m, L)
    return routes


def lire_instance_chao(nom_fichier):
    with open(nom_fichier, "r") as f:
        lignes = f.readlines()

    # Lire les paramètres de l'instance
    n, m = map(int, lignes[0].split())

    # Lire les coordonnées et profits des clients
    clients = []
    depot = None
    for i, ligne in enumerate(lignes[1:]):
        x, y, profit = map(float, ligne.split())
        if i == 0:  # Le premier nœud est considéré comme le dépôt
            depot = Client(0, x, y, 0)
        else:
            clients.append(Client(i, x, y, profit))

    # La limite de temps L n'est pas dans le fichier, donc on définit une valeur par défaut
    L = 100  # Vous pouvez ajuster cette valeur si nécessaire

    return depot, clients, m, L


def resoudre_TOP(clients, depot, m, L):
    solution_initiale = heuristique_constructive_TOP(clients, depot, m, L)

    # solution_locale = recherche_locale_TOP(solution_initiale, L)

    solution_beasley = beasley_adapte_TOP(clients, depot, m, L)

    meilleure_solution = max(
        [solution_initiale, solution_beasley],
        key=lambda x: sum(profit_total(route) for route in x if len(route) > 2),
    )

    # Retourner uniquement les routes non vides
    return [route for route in meilleure_solution if len(route) > 2]


# Exemple d'utilisation :
nom_fichier = "set_66_1\\set_66_1_005.txt"
depot, clients, m, L = lire_instance_chao(nom_fichier)

# Maintenant vous pouvez utiliser ces informations pour résoudre le TOP
meilleure_solution = resoudre_TOP(clients, depot, m, L)

# Imprimer les résultats
profit_total = sum(profit_total(route) for route in meilleure_solution)
print(f"Profit total : {profit_total}")
for i, route in enumerate(meilleure_solution):
    print(f"Route {i+1} : {' -> '.join(str(client.id) for client in route)}")
