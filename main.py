import numpy as np
from MH.imports import iterarFOX,iterarEOO

np.set_printoptions(precision=4)

def writeFile(string):
     with open('result.txt','a', encoding='utf-8') as f: f.write(f"{string}\n")

mh = 'EOO'
maxIter = 100
dim = 3
nPopulation = 3

ub = 100
lb = -100

def fitnessFunction(X):
    return np.sum(X**2)

def checkSolution(X):
     cont = 0
     for x in X:
          if x>ub or x<lb: cont += 1
     return cont

def repairSolution(X):
    for i in range(len(X)):
        if X[i]>ub:
             X[i] = ub
        elif X[i]<lb:
             X[i] = lb
    return X

X = np.random.uniform(lb, ub+1, size=(nPopulation, dim))

for it in range(maxIter+1):
     condition = it<3 or it==100

     if it==0: 
          with open('result.txt','w', encoding='utf-8') as f: f.write(f"SOLUCIÓN INICIAL:\n")
     else:
          if condition: writeFile(f"ITERACIÓN {it}\n")
          if condition: writeFile(f"ECUACIONES GENERALES:")

          #  Metaheuristica
          if mh == 'FOX':
               X = iterarFOX(maxIter,it,dim,X.tolist(),best.tolist())
          elif mh == 'EOO':
               X = iterarEOO(maxIter,it,X.tolist(),best.tolist())

          if condition: writeFile(f"SOLUCIONES OBTENIDAS EN LA ITERACIÓN {it}:")
          for i in range(nPopulation):
               contInf = checkSolution(X[i])
               if condition: writeFile(f"ind {i+1}: {X[i]}, infactibles: {contInf}")
               repairSolution(X[i])
          if condition: writeFile(f"\nREPARACIÓN DE SOLUCIONES:")
     
     fitness = []
     for i in range(nPopulation):
          fitness.append(fitnessFunction(X[i]))
          if condition: writeFile(f"ind {i+1}: {X[i]} / fitness: {fitness[i]:.4f}")

     bestIndex = np.argmin(np.array(fitness))
     best = X[bestIndex]
     bestFitness = fitness[bestIndex]

     if condition: writeFile(f"\nMejor solución: ind {bestIndex+1}: {best} / fitness: {bestFitness:.4f}")
     if condition: writeFile(f"----------------------------------------------------------------------------------------")
    