import math
import random


class SalesmanProblem:
    def __init__(self, population_size, elite_threshold, mutation_probability, path_size, cities):
        self.path_size = path_size
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.elite_threshold = elite_threshold
        self.population = []
        self.fitness_scores = []
        self.cities = cities

    def generate_population(self):
        while len(self.population) != self.population_size:
            chromosome = random.sample(cities, self.path_size)
            if len(self.population) > 0 and chromosome in self.population:
                continue
            else:
                self.population.append(chromosome)

    def calculate_distance(self, path):
        total_distance = 0

        for i in range(self.path_size - 1):
            city1 = path[i]
            city2 = path[i + 1]
            distance = math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
            total_distance += distance

        first_city = path[0]
        last_city = path[-1]
        total_distance += math.sqrt((first_city[0] - last_city[0]) ** 2 + (first_city[1] - last_city[1]) ** 2)

        return total_distance

    def generate_fitness_scores(self):
        self.fitness_scores.clear()

        for path in self.population:
            distance = self.calculate_distance(path)
            self.fitness_scores.append(distance)

    def select_parent(self, random_value, probabilities_of_selection):
        cumulative_probability = 0
        selected_parent_index = 0

        for i, probability in enumerate(probabilities_of_selection):
            cumulative_probability += probability
            if random_value <= cumulative_probability:
                selected_parent_index = i
                break

        return selected_parent_index

    def select_the_best_paths(self):
        copy_of_fitness_scores = self.fitness_scores.copy()
        best_res = []

        for i in range(round(self.population_size * self.elite_threshold)):
            best = min(copy_of_fitness_scores)
            best_index = copy_of_fitness_scores.index(best)
            copy_of_fitness_scores.pop(best_index)
            best_res.append(self.population[best_index])

        return best_res

    def count_probabilities_for_selection(self):
        probabilities_of_selection = []

        for i in range(len(self.fitness_scores)):
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

    def have_crossovers_duplicates(self, chromosome1, chromosome2):
        seen = []
        for city in chromosome1:
            if city in seen:
                return True
            seen.append(city)

        seen.clear()
        for city in chromosome2:
            if city in seen:
                return True
            seen.append(city)

        return False

    def crossover(self, parent1, parent2):
        crossover_point = round(random.uniform(1, self.path_size - 1))
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        while self.have_crossovers_duplicates(child1, child2):
            parent1, parent2 = self.select_parents()
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    def mutate(self, path):
        if random.uniform(0, 1) < self.mutation_probability:
            city1, city2 = random.sample(path, 2)
            index1 = path.index(city1)
            index2 = path.index(city2)
            path[index1], path[index2] = path[index2], path[index1]

        return path

    def find_best_solution(self):
        min_value = min(self.fitness_scores)
        if min_value != 869:
            return -1
        min_val_index = self.fitness_scores.index(min_value)

        return min_val_index

    def run_genetic_algorithm(self):
        self.generate_population()
        self.generate_fitness_scores()
        iterations = 0

        while self.find_best_solution() == -1:
            iterations += 1
            new_population = []

            best_paths = self.select_the_best_paths()
            for i in range(round(self.population_size * self.elite_threshold)):
                new_population.append(best_paths[i])

            while len(new_population) != self.population_size:
                parent1, parent2 = self.select_parents()
                children = self.crossover(parent1, parent2)
                child1 = self.mutate(children[0])
                child2 = self.mutate(children[1])
                new_population.append(child1)
                new_population.append(child2)

            self.population = new_population
            self.generate_fitness_scores()
            if iterations % 1000 == 0:
                print(min(self.fitness_scores))

        best_solution_index = self.find_best_solution()
        print(iterations)

        return self.population[best_solution_index]


if __name__ == "__main__":
    cities = [
        (119, 38), (37, 38), (197, 55), (85, 165), (12, 50),
        (100, 53), (81, 142), (121, 137), (85, 145), (80, 197),
        (91, 176), (106, 55), (123, 57), (40, 81), (78, 125),
        (190, 46), (187, 40), (37, 107), (17, 11), (67, 56),
        (78, 133), (87, 23), (184, 197), (111, 12), (66, 178)
    ]

    genetic_algorithm = SalesmanProblem(100, 0.2, 0.01, 25, cities)
    best_solution = genetic_algorithm.run_genetic_algorithm()
    print("Najlepsze rozwiązanie:", best_solution,
          "\nZ odległością:", genetic_algorithm.calculate_distance(best_solution))
