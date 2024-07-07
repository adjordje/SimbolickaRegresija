import random
import numpy as np
from math import cos, sin
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
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
            if self.value is None:
                return x
            return self.value * x
        elif self.operation == "+":
            return self.left.evaluate(x) + self.right.evaluate(x)
        elif self.operation == "-":
            return self.left.evaluate(x) - self.right.evaluate(x)
        elif self.operation == "*":
            return self.left.evaluate(x) * self.right.evaluate(x)
        elif self.operation == "/":
            b = self.right.evaluate(x)
            if b == 0:
                b = 0.000001
            return self.left.evaluate(x) / b
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
        elif self.operation == "l^":
            return self.left.evaluate(x)**2 + self.right.evaluate(x)
        elif self.operation == "r^":
            return self.left.evaluate(x) + self.right.evaluate(x)**3
        elif self.operation == "X":
            if self.value is None:
                return x
            else:
                return x
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
        return Node(value=random.uniform(-1000, 1000))
    else:
        op = random.choice(["+", "*", "l^", "r^", "-", "/", "cos+", "sin+", "X"])
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
    if random.random() < 0.8:  # Sansa mutacije
        individual.left = generate_expression(depth=8) 
    else:
        individual.right.value = random.uniform(-100000, 100000)
    return individual

# funkcija koja vrsi evoluciju populacije
def evolve(population, X, y, generations=400):
    for _ in tqdm(range(generations)):
        # izaberi roditelje
        parents = sorted(random.sample(population, 2), key=lambda x: fitness(x, X, y))[:]


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

    # funkcija je (x-5)^3
    # nsample = 400
    # sig = 0.2
    # x = np.linspace(-50, 50, nsample)
    # X = np.column_stack((x/5, 10*np.sin(x), (x-5)**3, np.ones(nsample)))
    # beta = [0.01, 1, 0.001, 5.]
    # y_true = np.dot(X, beta)
    # y = y_true + sig * np.random.normal(size=nsample)
    # df = pd.DataFrame()
    # df['x']=x
    # df['y']=y


    # point_color = '#852a5b'
    # edgecolorsA = '#ffa600'
    # edgecolorsB = '#73fffa'

    # fig, (ax1, ax2) = plt.subplots(1,  2, figsize=(10,  5))

    # ax1.scatter(df['x'], df['y'], c=point_color, edgecolors=edgecolorsA, alpha=1.0)
    # ax2.scatter(df['x'], df['y'], c=point_color, edgecolors=edgecolorsA, alpha=1.0)
    # plt.xlim(-30, 30)
    # plt.ylim(-30,  30)
    # plt.show()

    

    X = np.linspace(-50, 50, 100)
    y = (X-5) ** 3 

    population_size = 100
    depth = 5
    population = generate_population(population_size, depth)
    evolved_population = evolve(population, X, y)
    best_individual = max(evolved_population, key=lambda x: fitness(x, X, y))
    print(best_individual)

    X_values = X
    Y_values = [best_individual.evaluate(x) for x in X]
    plt.scatter(X.squeeze(), y.squeeze())
    plt.plot(X_values, Y_values, 'r')
    plt.xlabel('X osa')
    plt.ylabel('Y osa')
    plt.title('Plot generisanih podataka')
    plt.show()
