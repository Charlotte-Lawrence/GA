import numpy as np
import matplotlib.pyplot as plt
import random

# Initialize population using permuation encoding
def initialize_population(pop_size, num_cities):
    initialPopulation = [np.random.permutation(num_cities) for _ in range(pop_size)]
    return initialPopulation

# Fitness function
# A common fitness function for Travelling Salesman problem is fitness = distance
def calculate_distance(route, dist_matrix):
    # Initialize the total distance
    total_distance = 0
    
    # Loop through each city in the route
    for i in range(len(route) - 1):
        # Get the index of the current city and the next city
        current_city = route[i]
        next_city = route[i + 1]
        
        # Add the distance between the current city and the next city to the total distance
        total_distance += dist_matrix[current_city, next_city]
    
    # Add the distance from the last city back to the starting city
    last_city = route[-1]
    starting_city = route[0]
    total_distance += dist_matrix[last_city, starting_city]
    
    return total_distance

# Selection
def tournament_selection(population, fitnesses, k=3):
    selected = []
    population_size = len(population)
    
    for _ in range(population_size):
        # Randomly select k individuals from the population with replacement
        tournament_indices = random.choices(range(population_size), k=k)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        
        # Find the index of the individual with the highest fitness score
        best_individual_index = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
        
        # Get the selected individual from the population
        selected_individual = population[best_individual_index]

        # Add the selected individual to the list of selected individuals
        selected.append(selected_individual)
    
    return selected


def ordered_crossover(parent1, parent2):
    # Randomly select a segment from parent1
    start, end = sorted(random.sample(range(len(parent1)), 2))
    
    # Initialize the child with the segment from parent1
    child = parent1[start:end]

    # Fill in the remaining positions with cities from parent2
    p2_notInChild = [city for city in parent2 if city not in child]
    child = np.concatenate((child, p2_notInChild))
    return child

# Mutation
def swap_mutation(route, mutation_rate):
    for i in range(len(route)):
        # Generate random num for each city in route
        # Mutuation occurs if it is less than the rate
        if random.random() < mutation_rate:
            # Select a random city to swap with
            j = random.randint(0, len(route) - 1)

            route[i], route[j] = route[j], route[i]

    return route


def main(num_cities, pop_size, gen, stop):
    num_cities = int(num_cities)

    population_size = int(pop_size)
    num_generations = int(gen)
    mutation_rate = 0.01

    # Generate random cities
    cities = np.random.rand(num_cities, 2)

    # Calculate distance matrix using Euclidean distance
    # For N cities, D is N by N matrix
    # Where D[i][j] is the distance between two cities i, j
    dist_matrix = np.sqrt(((cities[:, np.newaxis] - cities[np.newaxis, :]) ** 2).sum(axis=2))

    population = initialize_population(population_size, num_cities)

    # Initialise best route to none and highest value of distance
    best_route = None
    best_distance = float('inf')
    
    max_unchanged_generations = int(num_generations * 0.7)
    unchanged_generations = 0

    fitness_over_generations = []  # List to store average fitness values over generations
    min_fitness_over_generations = []  # List to store minimum fitness values over generations
    max_fitness_over_generations = []  # List to store maximum fitness values over generations

    # Loop for specified generations
    for generation in range(num_generations):
        # Get fitness of population
        fitnesses = [calculate_distance(chromosome, dist_matrix) for chromosome in population]

        # Get best chromosome in generation by the minimum fitness
        best_gen_chromosome = population[np.argmin(fitnesses)]
        best_gen_distance = min(fitnesses)
        

        if best_gen_distance < best_distance:
            best_distance = best_gen_distance
            best_route = best_gen_chromosome
            unchanged_generations = 0
        else:
            unchanged_generations += 1

        if (unchanged_generations >= max_unchanged_generations) and (stop == "Y"):
            print(f"Stopping Criteria reached: fitness has not improved for {max_unchanged_generations} generations.")    
            break

        print(f"Generation {generation}: Best Distance = {best_distance}")

        # Store average fitness in this generation
        fitness_over_generations.append(np.mean(fitnesses))
        min_fitness_over_generations.append(np.min(fitnesses))
        max_fitness_over_generations.append(np.max(fitnesses))
    
        # Select a population
        selected_population = tournament_selection(population, fitnesses)
        next_population = []

        # Perform crossover and mutation
        for i in range(0, population_size, 2):
            # Grab parents
            parent1 = selected_population[i]
            parent2 = selected_population[i + 1]

            # Generate child by performing ordered crossover
            child1 = ordered_crossover(parent1, parent2)
            child2 = ordered_crossover(parent2, parent1)

            next_population.append(swap_mutation(child1, mutation_rate))
            next_population.append(swap_mutation(child2, mutation_rate))
        
        population = next_population

    # Plot the best route with different colors for each segment
    plt.figure(figsize=(10, 6))
    plt.scatter(cities[:, 0], cities[:, 1], color='red')


    for i, city_index in enumerate(best_route):
        plt.text(cities[city_index, 0], cities[city_index, 1], "City Number: " + str(city_index) + "\nPosition: " + str(i + 1))
    
   # Determine the maximum and minimum segment lengths
    max_length = max([np.linalg.norm(cities[best_route[i]] - cities[best_route[i+1]]) for i in range(len(best_route) - 1)])
    min_length = min([np.linalg.norm(cities[best_route[i]] - cities[best_route[i+1]]) for i in range(len(best_route) - 1)])

    # Plot each segment of the route with a color indicating segment length
    for i in range(len(best_route) - 1):
        segment_coords = cities[best_route[i:i+2]]
        segment_length = np.linalg.norm(segment_coords[1] - segment_coords[0])
        normalized_length = (segment_length - min_length) / (max_length - min_length)  # Normalize segment length between 0 and 1

        color = plt.cm.jet(normalized_length)  # Get color based on normalized length

        plt.plot(segment_coords[:, 0], segment_coords[:, 1], color=color, alpha=0.7)
        
    # Connect the last city back to the starting city
    plt.plot([cities[best_route[-1], 0], cities[best_route[0], 0]], [cities[best_route[-1], 1], cities[best_route[0], 1]], 'b-', alpha=0.5)
    
    plt.title(f'Best Route Found: {best_distance:.2f}')
    plt.show()

    # Display fitness over generations
    plt.plot(range(len(fitness_over_generations)), fitness_over_generations, label='Average Fitness')
    plt.plot(range(len(fitness_over_generations)), min_fitness_over_generations, label='Min Fitness')
    plt.plot(range(len(fitness_over_generations)), max_fitness_over_generations, label='Max Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(input("Enter number of cities: "),input("Enter population size: "), input("Enter num of generations: "), input("Stopping criteria(Y/N): "))