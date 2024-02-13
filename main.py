import numpy as np
import time
from MH.imports import iterarFOX,iterarEOO,iterarRSA,iterarGOA,iterarPSO,iterarHBA,iterarTDO

np.set_printoptions(precision=4)

# condition = it<3 or it==maxIter
# if condition: writeFile(f"")
def writeFile(string):
     with open('result.txt','a', encoding='utf-8') as f: f.write(f"{string}\n")

# CONFIGURACIÓN
mh = 'TDO'
N = 2
dim = 2
maxIter = 100
ub = 100
lb = -100

def fitnessFunction(X):
    return np.sum(X**2)

def checkSolution(X):
     cont = 0
     for x in X:
          if x>ub or x<lb: cont += 1
     return cont

def repairSolution(x):
    for j in range(len(x)): np.clip(x[j],lb,ub)
    return x

def fo(x):
     x = repairSolution(x)
     return x,fitnessFunction(np.array(x))

start = time.time()
print(f"Ejecutando {mh}")

X = np.random.uniform(lb, ub+1, size=(N, dim))

bestX = np.copy(X) 

for it in range(maxIter+1):
     condition = it<3 or it==maxIter

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
          elif mh == 'RSA':
               X = iterarRSA(maxIter,it,dim,X.tolist(),best.tolist(),lb,ub)
          elif mh == 'GOA':
               X = iterarGOA(maxIter,it,dim,X.tolist(),best.tolist(),fitness,fo,'MIN')
          elif mh == 'PSO':
               X = iterarPSO(maxIter, it, dim, X.tolist(),best.tolist(),bestX)
          elif mh == 'HBA':
               X = iterarHBA(maxIter, it, dim, X.tolist(),best.tolist(), fitness, fo,'MIN')
          elif mh == 'TDO':
               X = iterarTDO(maxIter,it,dim,X.tolist(),fitness,fo,'MIN')

          if condition: writeFile(f"SOLUCIONES OBTENIDAS EN LA ITERACIÓN {it}:")
          for i in range(N):
               contInf = checkSolution(X[i])
               if condition: writeFile(f"ind {i+1}: {X[i]}, infactibles: {contInf}")
               repairSolution(X[i])
          if condition: writeFile(f"\nREPARACIÓN DE SOLUCIONES:")
     
     fitness = []
     for i in range(N):
          fitness.append(fitnessFunction(X[i]))
          if condition: writeFile(f"ind {i+1}: {X[i]} / fitness: {fitness[i]:.4f}")
          if mh == 'PSO':
               if fitness[i] < fitnessFunction(bestX): bestX[i] = np.copy(X[i])          

     bestIndex = np.argmin(np.array(fitness))
     best = X[bestIndex]
     bestFitness = fitness[bestIndex]

     if condition: writeFile(f"\nMejor solución: ind {bestIndex+1}: {best} / fitness: {bestFitness:.4f}")
     if condition: writeFile(f"----------------------------------------------------------------------------------------")

print(f"{mh} termino de ejecutarse en: {(time.time()-start):.4f} (s)")