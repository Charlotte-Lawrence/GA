# Travelling Salesman Problem with Genetic Algorithm #

## Problem ## 

The Travelling Salesman Problem is a common problem in computing describing a situation where we need to find an optimal solution given:

- A list of cities with distances between each pair
- You can only visit a city exactly once
- You return to the start city

An example solution: [Reference](https://en.wikipedia.org/wiki/Travelling_salesman_problem)

![image](https://github.com/Charlotte-Lawrence/Travelling-Salesman-GA/assets/122492109/e0b8c7a0-c28f-4faa-afec-be48819e1271)

These problems are often represented as a graph with weights. The nodes of the graph represent the cities and the edges of the graph represent the distance between that city pair.

![image](https://github.com/Charlotte-Lawrence/Travelling-Salesman-GA/assets/122492109/d8dba199-8d61-45b9-a56f-44b89ca94d40)

## Genetic Algorithm ## 

A Genetic Algorithm is a heuristic search algorithm based on evolution and natural selection and are commonly used to solve optimisation or search problems.

A typical GA involves:
1. **Initialisation**: The population is initialised, each individual representing a potential solution.
2. **Evaluation**: A fitness function is used to measure the fitness of the population. A typical fitness function for the Travelling Salesman Problem (TSP) is the total distance of the route between cities with the aim to minimise this distance.
3. **Selection**: Individuals are selected from the population based on fitness, with a better fitness giving an individual a higher chance of being selected.
4. **Crossover & Mutation**: The selected individuals are paired as two parents to perform crossover and mutation to generate new children.
5. **Termination**: The algorithm performs these previous steps until a condition is met. Generally, it will run until a predefined number of generations but there are also cases where you may want to stop when the population has converged to the same fitness values.

For the case of the **Travelling Salesman Problem** we can utilise a **Genetic Algorithm** to discover potential paths to generate an optimal solution:

- Chromosomes (individuals) are initialised using permutation encoding - An encoding method used for problems where order matters such as the order of the cities to be visited in.
- Tournament Selection - An individual is selected by randomly selecting _k_ individuals from the population. The fitness of these _k_ individuals are evaluated, selected the best individual (a lower fitness value = lower distance which is the aim)
- Ordered Crossover - A subset of cities are selected from a parent to be placed in a new child. The rest of the cities are filled from a second parent in the population.
- Swap Mutation - Two positions in the chromosome are selected at random, and their places are swapped. This is to introduce diversity, but it only occurs with a low _mutation rate_ such as 0.1 to 0.01.
- The algorithm terminates after a set amount of generations, with additional options to terminate when the fitness values are unchanged for a long period of time across generations.

### Implementation ###

The implementation in python utilises a Genetic Algorithm designed to solve the Travelling Salesman Problem.

You can:
- Select _n_ cities
- Population Size
- Number of Generations
- Optional Stopping criteria (Y/N): The program will stop when the fitness is unchanged for **70% of the generations**.

These parameters will need to be adjusted based on the number of cities to generate an optimal result.

#### Example Results ####

Cities: 15
Population Size: 200
Generations: 100
Stopping: Y

Resulting Path:

![image](https://github.com/Charlotte-Lawrence/Travelling-Salesman-GA/assets/122492109/5db78465-6809-4ed7-9903-215129c48353)

Fitness Evolution over Generations: (Where a lower fitness value is preffered as it indicates distance of the route)

![image](https://github.com/Charlotte-Lawrence/Travelling-Salesman-GA/assets/122492109/f4019874-1832-4d44-aac6-5d30f62eda5b)


