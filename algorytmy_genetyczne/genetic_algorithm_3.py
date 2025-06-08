import random


class GeneticKnapsack:
    def __init__(self, population_size, mutation_probability, chromosome_size, elite_threshold, items):
        self.chromosome_size = chromosome_size
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.elite_threshold = elite_threshold
        self.population = []
        self.fitness_scores = []
        self.items = items

    def generate_population(self):
        for i in range(self.population_size):
            individual = ''.join(random.choice('01') for _ in range(self.chromosome_size))
            self.population.append(individual)

    # def decode_chromosome(self, chromosome):
    #     return int(chromosome[:], 2)

    def generate_fitness_scores(self):
        self.fitness_scores.clear()

        for j in range(self.population_size):
            chromosome = (self.population[j])
            total_weight = sum(int(chromosome[i]) * int(self.items[i][0]) for i in range(self.chromosome_size))
            total_value = sum(int(chromosome[i]) * int(self.items[i][1]) for i in range(self.chromosome_size))

            if total_weight <= 35:
                self.fitness_scores.append(total_value)
            else:
                self.fitness_scores.append(0)

    def select_parent(self, random_value, probabilities_of_selection):
        cumulative_probability = 0
        selected_parent_index = 0

        for i, probability in enumerate(probabilities_of_selection):
            cumulative_probability += probability
            if random_value <= cumulative_probability:
                selected_parent_index = i
                break

        return selected_parent_index

    def count_probabilities_for_selection(self):
        probabilities_of_selection = []

        for i in range(self.population_size):
            res = self.fitness_scores[i] / sum(self.fitness_scores)
            probabilities_of_selection.append(res)

        return probabilities_of_selection

    # metoda ruletki
    def select_parents(self):
        selected_parents = []
        probabilities_of_selection = self.count_probabilities_for_selection()

        random_value = random.uniform(0, 1)
        selected_parents.append(self.population[self.select_parent(random_value, probabilities_of_selection)])
        random_value = random.uniform(0, 1)
        selected_parents.append(self.population[self.select_parent(random_value, probabilities_of_selection)])

        return selected_parents

    def crossover(self, parent1, parent2):
        crossover_point = round(random.uniform(1, self.population_size - 1))
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    # def mutate(self, chromosome):
    #     if random.uniform(0, 1) < self.mutation_probability:
    #         index_to_flip = round(random.uniform(0, self.chromosome_size - 1))
    #         flipped_chromosome = chromosome[:index_to_flip] + (
    #             '1' if chromosome[index_to_flip] == '0' else '0') + chromosome[
    #                                                                 index_to_flip + 1:]
    #         return flipped_chromosome
    #
    #     return chromosome
    def mutate(self, chromosome):
        for i in range(self.chromosome_size):
            if random.uniform(0, 1) < self.mutation_probability:
                index_to_flip = i
                flipped_chromosome = chromosome[:index_to_flip] + (
                    '1' if chromosome[index_to_flip] == '0' else '0') + chromosome[
                                                                        index_to_flip + 1:]
                chromosome = flipped_chromosome

        return chromosome

    def find_best_solution(self):
        max_value = max(self.fitness_scores)
        if max_value != 2222:
            return -1

        max_val_index = self.fitness_scores.index(max_value)

        return max_val_index

    def run_genetic_algorithm(self):
        self.generate_population()
        self.generate_fitness_scores()
        iterations = 0

        while self.find_best_solution() == -1:
            iterations += 1
            new_population = []

            probabilities_of_selection = self.count_probabilities_for_selection()
            for i in range(round(self.population_size * self.elite_threshold)):
                random_value = random.uniform(0, 1)
                new_population.append(self.population[self.select_parent(random_value, probabilities_of_selection)])

            while len(new_population) != self.population_size:
                parent1, parent2 = self.select_parents()
                children = self.crossover(parent1, parent2)
                child1 = self.mutate(children[0])
                child2 = self.mutate(children[1])
                new_population.append(child1)
                new_population.append(child2)

            self.population = new_population
            self.generate_fitness_scores()

        best_solution_index = self.find_best_solution()
        print(iterations)

        return self.population[best_solution_index]


if __name__ == "__main__":
    items = [
        (3, 266),
        (13, 442),
        (10, 671),
        (9, 526),
        (7, 388),
        (1, 245),
        (8, 210),
        (8, 145),
        (2, 126),
        (9, 322)
    ]

    genetic_algorithm = GeneticKnapsack(8, 0.05, 10, 0.25, items)
    best_solution = genetic_algorithm.run_genetic_algorithm()
    print("Najlepsze rozwiązanie:", best_solution, "\nWartość:",
          genetic_algorithm.fitness_scores[genetic_algorithm.population.index(best_solution)], "\nWaga:",
          sum(int(best_solution[i]) * genetic_algorithm.items[i][0] for i in range(len(best_solution))))

