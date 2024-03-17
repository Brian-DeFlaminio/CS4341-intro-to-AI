import random
import itertools

def calculate_total_distance(tour, distances):
    """Calculate the total distance of a tour."""
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distances[tour[i]][tour[i + 1]]
    total_distance += distances[tour[-1]][tour[0]]  # Return to the starting city
    return total_distance

def generate_random_tour(num_cities):
    """Generate a random tour."""
    return random.sample(range(num_cities), num_cities)

def crossover(parent1, parent2):
    """Perform crossover (ordered crossover) to create offspring."""
    start = random.randint(0, len(parent1) - 1)
    end = random.randint(start, len(parent1))
    child = [-1] * len(parent1)
    child[start:end] = parent1[start:end]
    remaining_cities = [city for city in parent2 if city not in child]
    index = 0
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = remaining_cities[index]
            index += 1
    return child

def genetic_algorithm_tsp(distances, population_size=50, generations=1000):
    """Solve TSP using a genetic algorithm."""
    num_cities = len(distances)
    population = [generate_random_tour(num_cities) for _ in range(population_size)]
    
    for generation in range(generations):
        population.sort(key=lambda x: calculate_total_distance(x, distances))
        new_population = []
        # Elitism: Keep the top 10% of the population
        elite_size = int(0.1 * population_size)
        new_population.extend(population[:elite_size])
        
        # Crossover and mutation to create offspring
        for _ in range(population_size - elite_size):
            parent1, parent2 = random.sample(population[:elite_size], 2)
            child = crossover(parent1, parent2)
            # Mutation (swap two cities)
            if random.random() < 0.2:
                swap_indices = random.sample(range(num_cities), 2)
                child[swap_indices[0]], child[swap_indices[1]] = child[swap_indices[1]], child[swap_indices[0]]
            new_population.append(child)
        
        population = new_population
    
    # Select the best tour from the final population
    best_tour = min(population, key=lambda x: calculate_total_distance(x, distances))
    best_distance = calculate_total_distance(best_tour, distances)
    return best_tour, best_distance

def hill_climbing_tsp(distances):
    """Solve TSP using hill climbing with the MST heuristic."""
    num_cities = len(distances)
    initial_tour = list(range(num_cities))
    current_tour = initial_tour
    current_cost = calculate_total_distance(current_tour, distances)
    
    while True:
        neighbors = []
        # Generate neighbors by swapping cities in the current tour
        for i, j in itertools.combinations(range(num_cities), 2):
            neighbor_tour = current_tour[:]
            neighbor_tour[i], neighbor_tour[j] = neighbor_tour[j], neighbor_tour[i]
            neighbors.append(neighbor_tour)
        
        # Find the neighbor with the lowest total distance
        best_neighbor = min(neighbors, key=lambda x: calculate_total_distance(x, distances))
        best_neighbor_cost = calculate_total_distance(best_neighbor, distances)
        
        # If the best neighbor's cost is not better, stop the search
        if best_neighbor_cost >= current_cost:
            break
        
        # Update current tour and cost with the best neighbor
        current_tour = best_neighbor
        current_cost = best_neighbor_cost
    
    return current_tour, current_cost

# city layout
distances = [
 [0, 1, 2, 3],
 [1, 0, 4, 5],
 [2, 4, 0, 6],
 [3, 5, 6, 0]
]



# Solve TSP using Genetic Algorithm
genetic_tour, genetic_cost = genetic_algorithm_tsp(distances)
print("Genetic Algorithm - Optimal Tour:", genetic_tour)
print("Genetic Algorithm - Optimal Cost:", genetic_cost)

# Solve TSP using Hill Climbing with MST Heuristic
hill_climbing_tour, hill_climbing_cost = hill_climbing_tsp(distances)
print("Hill Climbing - Optimal Tour:", hill_climbing_tour)
print("Hill Climbing - Optimal Cost:", hill_climbing_cost)
