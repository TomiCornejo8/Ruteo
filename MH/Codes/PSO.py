import random
import numpy as np

# condition = it<3 or it==maxIter
# if condition: writeFile(f"")
def writeFile(string):
     with open('result.txt','a', encoding='utf-8') as f: f.write(f"{string}\n")

def iterarPSO(maxIter, it, dim, population, bestSolution,bestPop):
    '''
    maxIter: Máximo de iteraciones 
    it: iteración actual
    dim: Dimensión de las soluciones
    population: population actual de soluciones
    bestSolution: Mejor individuo obtenido hasta ahora
    bestPop: Mejores partículas obtenidas hasta ahora
    '''
    condition = it<3 or it==maxIter

    Vmax = 20
    wMax = 1.0
    wMin = 0.2
    c1 = 2
    c2 = 2

    vel = np.zeros((population.__len__(), dim))
    # Update the W of PSO
    w = wMax - it * ((wMax - wMin) / maxIter)

    if condition:
        writeFile(f"iter {it}")
        writeFile(f"BestX:\n{bestPop}\n")
        writeFile(f"w = wMax - it * ((wMax - wMin) / maxIter) = {wMax} - {it} * (({wMax} - {wMin}) / {maxIter}) = {w:.4f}")
        writeFile("")
    
    #For de población
    for i in range(population.__len__()):
        #For de dimensión
        for j in range(dim):
            r1 = random.random()
            r2 = random.random()
            #actualización de la velocidad de las partículas
            aux = vel[i,j]
            vel[i, j] = (
                w * vel[i, j]
                + c1 * r1 * (bestPop[i][j] - population[i][j])
                + c2 * r2 * (bestSolution[j] - population[i][j])
            )

            if condition:
                writeFile(f"ind {i+1} dim {j+1} iter {it}")
                writeFile(f"r1 = random() = {r1:.4f}")
                writeFile(f"r2 = random() = {r2:.4f}")
                writeFile(f"v[{i+1},{j+1}] = w * v[{i+1},{j+1}] + c1*r1*(BestX[{i+1},{j+1}] - X[{i+1},{j+1}]) + c2*r2*(Best[{j+1}]-X[{i+1},{j+1}]) = {w:.4f}*{aux:.4f} + {c1}*{r1:.4f}*({bestPop[i][j]:.4f}-{population[i][j]:.4f}) + {c2}*{r2:.4f}*({bestSolution[j]:.4f}-{population[i][j]:.4f}) = {vel[i, j]:.4f}")

            #Se mantiene la velocidad en sus márgenes mínimos y máximos
            if vel[i, j] > Vmax:
                if condition:
                    writeFile(f"v[{i+1},{j+1}] > vMax -> {vel[i,j]:.4f} > {Vmax} -> v[{i+1},{j+1}] = vMax = {Vmax}")
                vel[i, j] = Vmax

            if vel[i, j] < -Vmax:
                if condition:
                    writeFile(f"v[{i+1},{j+1}] < -vMax -> {vel[i,j]:.4f} < {-Vmax} -> v[{i+1},{j+1}] = -vMax = {-Vmax}")
                vel[i, j] = -Vmax
            
            #se actualiza la población utilizando las velocidades calculadas
            aux2 = population[i][j]
            population[i][j] = population[i][j] + vel[i][j]

            if condition:
                writeFile(f"X[{i+1},{j+1}] = X[{i+1},{j+1}] + v[{i+1},{j+1}] = {aux2:.4f} + {vel[i][j]:.4f} = {population[i][j]:.4f}")
                writeFile("")
    return np.array(population)