import random
import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt

# Klasa kojom je predstavljen cvor (operacija ili vrednost)
class Node:
    def __init__(self, operation=None, left=None, right=None, value=None):
        self.operation = operation
        self.left = left
        self.right = right
        self.value = value
    # funkcija kojom se evaluira vrednost iz tekuceg cvora
    def evaluate(self, x):
        if self.operation is None:
            return self.value * x if self.value is not None else x
        elif self.operation == "+":
            return self.left.evaluate(x) + self.right.evaluate(x)
        elif self.operation == "-":
            return self.left.evaluate(x) - self.right.evaluate(x)
        elif self.operation == "*":
            return self.left.evaluate(x) * self.right.evaluate(x)
        elif self.operation == "/":
            return self.left.evaluate(x) / self.right.evaluate(x)
        elif self.operation == "cos+":
            return cos(self.left.evaluate(x) + self.right.evaluate(x))
        elif self.operation == "cos-":
            return cos(self.left.evaluate(x) - self.right.evaluate(x))
        elif self.operation == "sin+":
            return sin(self.left.evaluate(x) + self.right.evaluate(x))
        elif self.operation == "sin-":
            return sin(self.left.evaluate(x) - self.right.evaluate(x))
        elif self.operation == "cos*":
            return cos(self.left.evaluate(x) * self.right.evaluate(x))
        elif self.operation == "cos/":
            return cos(self.left.evaluate(x) / self.right.evaluate(x))
        elif self.operation == "sin*":
            return sin(self.left.evaluate(x) * self.right.evaluate(x))
        elif self.operation == "sin/":
            return sin(self.left.evaluate(x) / self.right.evaluate(x))
        else:
            raise ValueError("Nepodrzana operacija")
    # reprezentacija u stringu
    def __repr__(self):
        if self.operation is None:
            return str(self.value)
        else:
            return f"({self.left.__repr__()}{self.operation}{self.right.__repr__()})"

# funkcija za generisanje izraza na osnovu tokena
def generate_expression(depth):
    if depth == 0:
        return Node(value=random.uniform(-100, 100))
    else:
        op = random.choice(["+", "-", "*", "/", "cos+",
                            "sin+"])
        left = generate_expression(depth - 1)
        right = generate_expression(depth - 1)
        return Node(operation=op, left=left, right=right)

# funkcija za generisanje populacije
def generate_population(size, depth):
    return [generate_expression(depth) for _ in range(size)]

# funkcija za merenje poklapanja po MSE metrici za gresku
def fitness(individual, X, y):
    predictions = [individual.evaluate(x) for x in X]
    for i in range(len(predictions)):
        if predictions[i] == None:
            predictions[i] = 0.0
    
    mse = ((predictions - y) ** 2).mean()
    return 1 / (mse + 1e-9)  # Dodajemo ovu malu konstantu kako bismo izbegli deljenje sa nulom

# funkcija koja uzima 'hromozome' od roditelja i pravi nove jedinke od njih
def crossover(parent1, parent2):
    child1_left = parent1.left
    child1_right = parent1.right
    child2_left = parent2.left
    child2_right = parent2.right
    return Node(left=child1_left, right=child2_right), \
           Node(left=child2_left, right=child1_right)

# funkcija koja vrsi mutaciju nad genima u hromozomu
def mutate(individual):
    if random.random() < 0.4:  # Sansa mutacije
        individual.left.operation = random.choice(["+", "-", "*", "/", "cos+", "sin+", "cos*"])
    else:
        individual.right.value = random.uniform(-100, 100)
    return individual

# funkcija koja vrsi evoluciju populacije
def evolve(population, X, y, generations=1000):
    for _ in range(generations):
        # izaberi roditelje
        parents = sorted(random.sample(population, 2), key=lambda x: fitness(x, X, y), reverse=True)[:]

        # napravi novu jedinku kao kompoziciju roditelja
        offspring = crossover(*parents)

        # mutiraj gene novonastale dece
        offspring = [mutate(ind) for ind in offspring]

        # zameniti najgore potomke novonastalim potomcima
        population.sort(key=lambda x: fitness(x, X, y), reverse=True)
        population[-len(offspring):] = offspring

    return population

# main funkcija
if __name__ == "__main__":
    X = np.random.rand(100, 1)
    y = X + np.random.randn(100, 1) 
    population_size = 500
    depth = 6
    population = generate_population(population_size, depth)
    evolved_population = evolve(population, X, y)
    best_individual = max(evolved_population, key=lambda x: fitness(x, X, y))

    X_values = X
    Y_values = [best_individual.evaluate(x) for x in X]
    plt.scatter(X.squeeze(), y.squeeze())
    plt.plot(X_values, Y_values, 'r')
    plt.xlabel('X osa')
    plt.ylabel('Y osa')
    plt.title('Plot generisanih podataka')
    plt.show()
