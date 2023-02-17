import numpy as np
import random
from numpy.fft import fft2, ifft2

def gerchberg_saxton_genetic_algorithm(measured_intensity, 
                                       target_image, n_generations, 
                                       population_size, mutation_rate):
    # Define the chromosome size as the number of pixels in the image
    chromosome_size = measured_intensity.size

    # Initialize the population with random phase values
    population = np.exp(2j * np.pi * np.random.rand(population_size, chromosome_size))

    for generation in range(n_generations):
        # Evaluate fitness of each chromosome in the population
        fitness = evaluate_fitness(population, measured_intensity, target_image)

        # Select parents for reproduction
        parents = select_parents(population, fitness)

        # Reproduce to create new offspring population
        offspring = reproduce(parents, population_size)

        # Mutate offspring
        mutated_offspring = mutate(offspring, mutation_rate)

        # Evaluate fitness of offspring population
        offspring_fitness = evaluate_fitness(mutated_offspring, measured_intensity, target_image)

        # Select the best chromosomes from the parent and offspring populations
        population = select_best(population, fitness, mutated_offspring, 
                                 offspring_fitness, population_size)

    # Return the best chromosome (phase) in the final population
    best_chromosome = select_best(population, fitness, mutated_offspring, offspring_fitness, 1)[0]
    
    # Apply Fourier transform to best chromosome to obtain amplitude and phase components
    amplitude = np.sqrt(measured_intensity) * np.exp(1j * np.angle(fft2(best_chromosome)))
    
    # Apply inverse Fourier transform to obtain updated image
    updated_image = ifft2(amplitude)

    # Return the final updated image
    return updated_image.real

def evaluate_fitness(population, measured_intensity, target_image):
    # Calculate fitness as the inverse mean squared error between the current image and target image
    fitness = np.zeros(len(population))
    for i, chromosome in enumerate(population):
        amplitude = np.sqrt(measured_intensity) * np.exp(1j * np.angle(fft2(chromosome)))
        reconstructed_image = ifft2(amplitude).real
        fitness[i] = 1.0 / np.mean((reconstructed_image - target_image) ** 2)
    return fitness

def select_parents(population, fitness):
    # Use tournament selection to select parents for reproduction
    n_parents = len(population) // 2
    parents = np.zeros((n_parents, 2, population.shape[1]), dtype=np.complex)
    for i in range(n_parents):
        tournament_size = 4
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        tournament_population = population[tournament_indices]
        parents[i] = tournament_population[np.argsort(tournament_fitness)[-2:]]
    return parents

def reproduce(parents, population_size):
    # Use uniform crossover to create new offspring population
    offspring = np.zeros((population_size, parents.shape[2]), dtype=np.complex)
    for i in range(population_size):
        parent_indices = np.random.choice(len(parents), size=2, replace=False)
        parent1, parent2 = parents[parent_indices]
        offspring[i] = parent1 * (np.random.rand(*parent1.shape) < 0.5) + parent2 * (np.random.rand(*parent2.shape) < 0.5)
    return offspring

def mutate(offspring, mutation_rate):
    # Randomly mutate some chromosomes
    mutated_offspring = np.copy(offspring)
    for i in range(len(offspring)):
        for j in range(mutated_offspring.shape[1]):
            if random.random() < mutation_rate:
                mutated_offspring[i, j] = np.exp(2j * np.pi * np.random.rand())
    return mutated_offspring

def select_best(population, fitness, offspring, offspring_fitness, n_best):
    # Combine parent and offspring populations and select the best chromosomes
    all_population = np.concatenate((population, offspring), axis=0)
    all_fitness = np.concatenate((fitness, offspring_fitness), axis=0)
    best_indices = np.argsort(all_fitness)[-n_best:]
    return all_population[best_indices]

# Example usage:
updated_image = gerchberg_saxton_genetic_algorithm(measured_intensity, target_image, n_generations=100, population_size=100, mutation_rate=0.01)
