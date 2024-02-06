from random import *
from math import *
import numpy as np
import math

# condition = it<3 or it==maxIter
# if condition: writeFile(f"")
def writeFile(string):
     with open('result.txt','a', encoding='utf-8') as f: f.write(f"{string}\n")

# ========= PARAMETROS =========
M = 2.5  # Peso en Kg del Gannet
VEL = 1.5  # Velocidad en el agua en m/s del Gannet
C = 0.2  # Determina si se ejejuta movimiento levy o ajustes de trayectoria
FACTOR_FASE = 0.5  # Determina que fase exploracion o explotacion usa el algoritmo en la iteracion

BETA = 1.5

# ========= FUNCIONES =========
def v(x,condition):
    if x > math.pi:
        if condition:
            writeFile(f"x > pi -> {x:.4f} > {math.pi:.4f} -> v() = (1 / pi) * x - 1 = (1 / {pi:.4f}) * {x:.4f} - 1 = {((1 / pi) * x - 1):.4f}")
        return ((1 / pi) * x - 1)
    if condition:
        writeFile(f"x <= pi -> {x:.4f} <= {math.pi:.4f} -> v() = -(1 / pi) * x + 1 = -(1 / {pi:.4f}) * {x:.4f} + 1 = {(-(1 / pi) * x + 1):.4f}")
    return (-(1 / pi) * x + 1)


def levy(condition):
    mu = uniform(0, 1)
    v = uniform(0, 1)
    gamma1 = math.gamma(1 + BETA)
    gamma2 = math.gamma((1+BETA)/2)
    seno = math.sin(math.pi * BETA / 2)
    expo = 1 / BETA
    sigma = ((gamma1 * seno) / (gamma2 * BETA * 2**((BETA-1)/2))) ** expo
    resultado = 0.01 * ((mu * sigma) / (abs(v) ** expo))
    if condition:
        writeFile(f"mu = random() = {mu:.4f}")
        writeFile(f"v = random() = {v:.4f}")
        writeFile(f"sigma = ((gamma(1+beta) * sin(pi*beta/2)) / (gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta) = (({gamma1:.4f} * {seno:.4f}) / ({gamma2:.4f} * {BETA} * 2^(({BETA}-1)/2)))^(1/{BETA}) = {sigma:.4f}")
        writeFile(f"p = levy() = 0.01 * ((mu * sigma) / (|v|^(1/beta))) = 0.01 * (({mu:.4f} * {sigma:.4f}) / ({abs(v):.4f}^{expo:.4f})) = {resultado:.4f}")
    return resultado


def getRandomPopulation(population):
    index_random = randint(0, len(population) - 1)
    return population[index_random]


def getAveragePopulation(population, dim):
    Xm = []
    for j in range(dim):
        sum = 0
        for i in range(len(population)):
            sum += population[i][j]
        Xm.append(sum / len(population))
    return Xm


# ========= ALGORITMO =========

# Busca mejor solucion
def iterarGOA(maxIter, it, dim, population, bestSolution, fitness, function, typeProblem):
    condition = it<3 or it==maxIter

    t = 1 - (it / maxIter)
    r = uniform(0, 1)
    MX = population.copy()
    # Calculos constantes por iteracion

    Xr = getRandomPopulation(population)
    Xm = getAveragePopulation(population, dim)

    if condition:
        writeFile(f"iter {it}")
        writeFile(f"t = 1 - (it / maxIter) = 1 - ({it} / {maxIter}) = {t:.4f}")
        writeFile(f"r = random() = {r:.4f}")
        writeFile(f"Xr = randomPopulation() = {np.array(Xr)}")
        writeFile(f"Xm = averagePopulation() = {np.array(Xm)}")

    # ========= Exploracion =========
    if r > FACTOR_FASE:
        if condition: 
            writeFile(f"r > {FACTOR_FASE} -> {r:.4f} > {FACTOR_FASE}")
            writeFile("")
        for i in range(len(population)):
            q = uniform(0, 1)
            if condition: 
                writeFile(f"ind {i+1} iter {it}")
                writeFile(f"q = random() = {q:.4f}")
                writeFile("")
            if q >= 0.5:
                for j in range(dim):
                    r2 = uniform(0, 1)
                    r4 = uniform(0, 1)

                    a = 2 * cos(2 * pi * r2) * t
                    A = (2 * r4 - 1) * a

                    u1 = uniform(-a, a)
                    u2 = A * (population[i][j] - Xr[j])

                    # Ecuacion 7a
                    MX[i][j] = population[i][j] + u1 + u2
                    if condition:
                        writeFile(f"ind {i+1} dim {j+1} iter {it}")
                        writeFile(f"q >= 0.5 -> {q:.4f} >= 0.5")
                        writeFile(f"r2 = random() = {r2:.4f}")
                        writeFile(f"r4 = random() = {r4:.4f}")
                        writeFile(f"a = 2 * cos(2 * pi * r2) * t = 2 * cos(2 * {pi:.4f} * {r2:.4f}) * {t:.4f} = {a:.4f}")
                        writeFile(f"A = (2 * r4 - 1) * a = (2 * {r4:.4f} - 1) * {a:.4f} = {A:.4f}")
                        writeFile(f"u1 = random(-a, a) = random({-a:.4f}, {a:.4f}) = {u1:.4f}")
                        writeFile(f"u2 = A * (X[i][j] - Xr[j]) = A * (X[{i+1}][{j+1}] - Xr[{j+1}]) = {A:.4f} * ({population[i][j]:.4f} - {Xr[j]:.4f}) = {u2:.4f}")
                        writeFile(f"MX[{i}][{j}] = X[i][j] + u1 + u2 = X[{i+1}][{j+1}] + u1 + u2 = {population[i][j]:.4f} + {u1:.4f} + {u2:.4f} = {MX[i][j]:.4f}")
                        writeFile("")
            else:
                for j in range(dim):
                    r3 = uniform(0, 1)
                    r5 = uniform(0, 1)

                    if condition:
                        writeFile(f"ind {i+1} dim {j+1} iter {it}")
                        writeFile(f"q < 0.5 -> {q:.4f} < 0.5")
                        writeFile(f"r3 = random() = {r3:.4f}")
                        writeFile(f"r5 = random() = {r5:.4f}")
                        writeFile(f"v(2 * pi * r3) = v({(2 * pi * r3):.4f}):")
                    aux = v(2 * pi * r3,condition) 
                    b = 2 * aux * t
                    B = (2 * r5 - 1) * b

                    v1 = uniform(-b, b)
                    v2 = B * (population[i][j] - Xm[j])
                    # Ecuacion 7b
                    MX[i][j] = population[i][j] + v1 + v2
                    if condition:
                        writeFile(f"b = 2 * v({(2 * pi * r3):.4f}) * t = 2 * {aux:.4f} * {t:.4f} = {b:.4f}")
                        writeFile(f"B = (2 * r5 - 1) * b = (2 * {r5:.4f} - 1) * {b:.4f} = {B:.4f}")
                        writeFile(f"v1 = random(-b, b) = random({-b:.4f}, {b:.4f}) = {v1:.4f}")
                        writeFile(f"v2 = B * (X[i][j] - Xm[j]) = B * (X[{i+1}][{j+1}] - Xm[{j+1}]) = {B:.4f} * ({population[i][j]:.4f} - {Xm[j]:.4f}) = {v2:.4f}")
                        writeFile(f"MX[{i+1}][{j+1}] = X[i][j] + v1 + v2 = X[{i+1}][{j+1}] + v1 + v2 = {population[i][j]:.4f} + {v1:.4f} + {v2:.4f} = {MX[i][j]:.4f}")
                        writeFile("")


    # ========= Explotacion =========
    else:
        t2 = 1 + (it / maxIter)
        if condition:
            writeFile(f"r < {FACTOR_FASE} -> {r:.4f} < {FACTOR_FASE}\n") 
            writeFile(f"t2 = 1 + (it / maxIter) = 1 + ({it} / {maxIter}) = {t2:.4f}")
            writeFile("")

        for i in range(len(population)):
            r6 = uniform(0, 1)
            l = 0.2 + (2 - 0.2) * r6
            R = (M * VEL**2) / l
            capturability = 1 / (R * t2)
            if condition:
                writeFile(f"ind {i+1} iter {it}")
                writeFile(f"r6 = random() = {r6:.4f}")
                writeFile(f"l = 0.2 + (2 - 0.2) * r6 = 0.2 + (2 - 0.2) * {r6:.4f} = {l:.4f}")
                writeFile(f"R = (M * VEL**2) / l = ({M} * {VEL}**2) / {l:.4f} = {R:.4f}")
                writeFile(f"capturability = 1 / (R * t2) = 1 / ({R:.4f} * {t2:.4f}) = {capturability:.4f}")
            # Caso ajustes exitosos
            if capturability >= C:
                if condition: 
                    writeFile(f"capturability >= C -> {capturability:.4f} >= {C}")
                    writeFile("")
                for j in range(dim):
                    delta = capturability * abs(population[i][j] - bestSolution[j])
                    # Ecuacion 17a
                    MX[i][j] = t * delta * (population[i][j] - bestSolution[j]) + population[i][j]
                    if condition:
                        writeFile(f"ind {i+1} dim {j+1} iter {it}")
                        writeFile(f"delta = capturability * |X[i][j] - best[j]| = capturability * |X[{i+1}][{j+1}] - best[{j+1}]| = {capturability:.4f} * |{population[i][j]:.4f} - {bestSolution[j]:.4f}| = {delta:.4f}")
                        writeFile(f"MX[{i+1}][{j+1}] = t * delta * (X[i][j] - best[j]) + X[i][j] = t * delta * (X[{i+1}][{j+1}] - best[{j+1}]) + X[{i+1}][{j+1}] = {t:.4f} * {delta:.4f} * ({population[i][j]:.4f} - {bestSolution[j]:.4f}) + {population[i][j]:.4f} = {MX[i][j]:.4f}")
                        writeFile("")

            # Caso movimiento Levy
            else:
                if condition: 
                    writeFile(f"capturability < C -> {capturability:.4f} < {C}")
                    writeFile("")
                for j in range(dim):
                    if condition: writeFile(f"ind {i+1} dim {j+1} iter {it}")
                    p = levy(condition)
                    # Ecuacion 17b
                    MX[i][j] = bestSolution[j] - (population[i][j] - bestSolution[j]) * p * t
                    if condition:
                        writeFile(f"MX[{i+1}][{j+1}] = best[j] - (X[i][j] - best[j]) * p * t = best[{j+1}] - (X[{i+1}][{j+1}] - best[{j+1}]) * p * t = {bestSolution[j]:.4f} - ({population[i][j]:.4f} - {bestSolution[j]:.4f}) * {p:.4f} * {t:.4f} = {MX[i][j]:.4f}")
                        writeFile("")
    
    mxFitness = np.zeros(len(fitness))
    for i in range(len(MX)):
        MX[i],mxFitness[i] = function(MX[i])

    if condition: 
        writeFile(f"MX:\n{np.array(MX)}")
        writeFile("")
        writeFile(f"X:\n{np.array(population)}")
        writeFile("")
    
    for i in range(len(MX)):
        if typeProblem == 'MIN': condition1 = mxFitness[i] < fitness[i]
        elif typeProblem == 'MAX': condition1 = mxFitness[i] > fitness[i]
        if condition1: population[i] = MX[i]
    
    if condition:
        writeFile(f"X':\n{np.array(population)}")
        writeFile("")
    return np.array(population)