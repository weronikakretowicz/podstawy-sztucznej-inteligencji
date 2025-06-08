import random

class Genetic_Algorithm:
    def __init__(self, population_size, mutation_probability, chromosome_size):
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.chromosome_size = chromosome_size
        self.population = []
        self.fitness_scores = []

    def generate_fitness_scores(self):
        self.fitness_scores.clear()

        for i in range(self.population_size):
            self.fitness_scores.append(self.population[i].count("1"))

    def generate_population(self):
        for i in range(self.population_size):
            individual = ''.join(random.choice('01') for _ in range(self.chromosome_size))
            self.population.append(individual)

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
        selected_parents.append(self.select_parent(random_value, probabilities_of_selection))
        random_value = random.uniform(0, 1)
        selected_parents.append(self.select_parent(random_value, probabilities_of_selection))

        return selected_parents

    def cross_over(self, parents):
        children = []
        cross_point = round(random.uniform(1, self.population_size - 1))

        first_child = (parents[0])[:cross_point] + parents[1][cross_point:]
        children.append(first_child)
        second_child = (parents[1])[:cross_point] + parents[0][cross_point:]
        children.append(second_child)

        return children

    def mutate(self, parent):
        if random.uniform(0, 1) < self.mutation_probability:
            index_to_flip = round(random.uniform(0, self.population_size - 1))
            flipped_parent = parent[:index_to_flip] + ('1' if parent[index_to_flip] == '0' else '0') + parent[
                                                                                                       index_to_flip + 1:]
            return flipped_parent

        return parent

    def is_ideal_chromosome(self):
        for i in range(self.population_size):
            if self.population[i].count("1") == self.chromosome_size:
                return 1

        return 0

    def remove_the_weakest_individual(self):
        min_val = min(self.fitness_scores)
        min_val_index = self.fitness_scores.index(min_val)
        self.population.pop(min_val_index)
        self.fitness_scores.pop(min_val_index)

    def run_genetic_algorithm(self):
        self.generate_population()
        self.generate_fitness_scores()
        iterations = 0

        while self.is_ideal_chromosome() != 1:
            parents_indices = self.select_parents()
            parents = [self.population[index] for index in parents_indices]
            children = self.cross_over(parents)
            child0 = self.mutate(children[0])
            child1 = self.mutate(children[1])

            self.remove_the_weakest_individual()
            self.remove_the_weakest_individual()
            self.population.append(child0)
            self.population.append(child1)

            self.generate_fitness_scores()
            iterations += 1

        print(iterations)


def main():
    genetic_algorithm = Genetic_Algorithm(10, 0.6, 10)
    genetic_algorithm.run_genetic_algorithm()


if __name__ == "__main__":
    main()
