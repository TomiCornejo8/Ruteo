import random
import numpy as np

def writeFile(string):
     with open('result.txt','a', encoding='utf-8') as f: f.write(f"{string}\n")

def iterarEOO(maxIter, it, population, bestSolution):
    condition = it<3 or it==maxIter
    population = np.array(population)
    bestSolution = np.array(bestSolution)

    it = maxIter - it + 1
    n = population.__len__()

    for i in range(n):
        # if condition: writeFile(f"ind {i} dim {dim} iter {it}")
        L = random.uniform(3, 5)
        T = ( ( (L-5)/(5-3) ) * 10 ) - 5
        E = ( (it-1)/(n-1) ) - 0.5 if it > 1 else (1/(n-1)) - 0.5
        C = ( ( (L-3)/(5-3) ) * 2 ) + 0.6        
        r = random.uniform(0, 1)

        if condition:
            writeFile(f"ind {i+1} iter {maxIter-it+1}")
            writeFile(f"L = random(3,5) = {L:.4f}")
            writeFile(f"T = ( ( (L-5)/(5-3) ) * 10 ) - 5 = ( ( ({L:.4f}-5)/(5-3) ) * 10 ) - 5 = {T:.4f}")
            writeFile(f"E = ( (it-1)/(n-1) ) - 0.5 = ( ({it}-1)/({n}-1) ) - 0.5 = {E:.4f}")
            writeFile(f"C = ( ( (L-3)/(5-3) ) * 2 ) + 0.6 = ( ( ({L:.4f}-3)/(5-3) ) * 2 ) + 0.6 = {C:.4f}")
            writeFile(f"r = random(0,1) = {r:.4f}")
            writeFile(f"")
        
        Y = np.zeros(population.shape[1])
        for j in range(population.shape[1]):
            if condition: writeFile(f"ind {i+1} dim {j+1} iter {maxIter-it+1}")
            Y[j] = T + E + L * r * (bestSolution[j] - population[i,j])
            if condition: writeFile(f"Y[{j+1}] = T + E + L * r * (best[{j+1}] - X[{i+1},{j+1}]) = {T:.4f} + {E:.4f} + {L:.4f} * {r:.4f} * ({bestSolution[j]:.4f} - {population[i,j]:.4f}) = {Y[j]:.4f}")
            aux = population[i,j]
            population[i,j] *= C
            if condition: writeFile(f"X[{i+1},{j+1}] = X[{i+1},{j+1}] * C = {aux:.4f} * {C:.4f} = {population[i,j]:.4f}")
            aux = population[i,j]
            population[i,j] += Y[j]
            if condition: writeFile(f"X[{i+1},{j+1}] = X[{i+1},{j+1}] + Y[{j+1}] = {aux:.4f} + {Y[j]:.4f} = {population[i,j]:.4f}")
            if condition: writeFile(f"")

    return population