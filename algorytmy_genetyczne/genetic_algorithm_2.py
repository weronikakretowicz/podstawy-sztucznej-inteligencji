import random


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_probability, chromosome_size, crossover_probability):
        self.chromosome_size = chromosome_size
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.population = []
        self.fitness_scores = []

    def generate_population(self):
        for i in range(self.population_size):
            individual = ''.join(random.choice('01') for _ in range(self.chromosome_size))
            self.population.append(individual)

    def decode_chromosome(self, chromosome):
        a = int(chromosome[:4], 2)
        b = int(chromosome[4:], 2)

        return a, b

    def generate_fitness_scores(self):
        self.fitness_scores.clear()

        for i in range(self.population_size):
            a, b = self.decode_chromosome(self.population[i])
            self.fitness_scores.append(1/(abs(2 * a ** 2 + b - 33)+1))

    def select_parent(self, random_value, probabilities_of_selection):
        cumulative_probability = 0
        selected_parent_index = 0

        for i, probability in enumerate(probabilities_of_selection):
            cumulative_probability += probability
            if random_value <= cumulative_probability:
                selected_parent_index = i
                break

        return selected_parent_index

    # metoda ruletki
    def select_parents(self):
        probabilities_of_selection = []
        selected_parents = []

        for i in range(self.population_size):
            res = self.fitness_scores[i] / sum(self.fitness_scores)
            probabilities_of_selection.append(res)

        random_value = random.uniform(0, 1)
        selected_parents.append(self.population[self.select_parent(random_value, probabilities_of_selection)])
        random_value = random.uniform(0, 1)
        selected_parents.append(self.population[self.select_parent(random_value, probabilities_of_selection)])

        return selected_parents

    def crossover(self, parent1, parent2):
        if random.uniform(0, 1) < self.crossover_probability:
            crossover_point = round(random.uniform(1, self.population_size - 1))
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]

            return child1, child2

        return parent1, parent2

    def mutate(self, chromosome):
        if random.uniform(0, 1) < self.mutation_probability:
            index_to_flip = round(random.uniform(0, self.chromosome_size - 1))
            flipped_chromosome = chromosome[:index_to_flip] + (
                '1' if chromosome[index_to_flip] == '0' else '0') + chromosome[
                                                                    index_to_flip + 1:]
            return flipped_chromosome

        return chromosome

    def find_best_individual(self):
        for i in range(self.population_size):
            a, b = self.decode_chromosome(self.population[i])
            if 2 * a ** 2 + b == 33:
                return i

        return -1

    def run_genetic_algorithm(self):
        self.generate_population()
        self.generate_fitness_scores()
        iterations = 0

        while self.find_best_individual() == -1:
            iterations += 1
            new_population = []

            while len(new_population) != self.population_size:
                parent1, parent2 = self.select_parents()
                children = self.crossover(parent1, parent2)
                child1 = self.mutate(children[0])
                child2 = self.mutate(children[1])
                new_population.append(child1)
                new_population.append(child2)

            self.population = new_population
            self.generate_fitness_scores()

        best_individual = self.find_best_individual()
        decoded_best_individual = self.decode_chromosome(self.population[best_individual])
        print(iterations)

        return decoded_best_individual


if __name__ == "__main__":
    genetic_algorithm = GeneticAlgorithm(10, 0.1, 8, 0.5)
    best_solution = genetic_algorithm.run_genetic_algorithm()
    print("Najlepsze rozwiązanie (a, b):", best_solution)
