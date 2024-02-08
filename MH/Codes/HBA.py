import random
import numpy as np
import math

# condition = it<3 or it==maxIter
# if condition: writeFile(f"")
def writeFile(string):
     with open('result.txt','a', encoding='utf-8') as f: f.write(f"{string}\n")


def iterarHBA(maxIter, it, dim, population, bestSolution, fitness, function,typeProblem):
  condition = it<3 or it==maxIter

  C = 2
  beta = 6
  epsilon = 0.00000000000000022204
  pi = math.pi
  
  N = population.__len__()
  Xnew = np.zeros([N,dim])
  
  alpha = C * math.exp(-it/maxIter)

  if condition:
    writeFile(f"iter {it}")
    writeFile(f"alpha = C * exp(-it/maxIter) = {C} * exp({-it}/{maxIter}) = {alpha}")
    writeFile("")
  
  for i in range(N):    
    r6 = random.uniform(0,1)
    if condition: 
      writeFile(f"ind {i+1} iter {it}")
      writeFile(f"r6 = random() = {r6:.4f}")
    if r6 <= 0.5:
      if condition: writeFile(f"r6 <= 0.5 -> {r6:.4f} <= 0.5 -> F = 1")
      F = 1
    else:
      if condition: writeFile(f"r6 > 0.5 -> {r6:.4f} > 0.5 -> F = -1")
      F = -1

    r=random.uniform(0,1)
    if condition: writeFile(f"r = random() = {r:.4f}")
    if r < 0.5:
      r2 = random.uniform(0,1)
      r3 = random.uniform(0,1)
      r4 = random.uniform(0,1)
      r5 = random.uniform(0,1)
      if condition:
        writeFile(f"r < 0.5 -> {r:.4f} < 0.5")
        writeFile(f"r2 = random() = {r2:.4f}")
        writeFile(f"r3 = random() = {r3:.4f}")
        writeFile(f"r4 = random() = {r4:.4f}")
        writeFile(f"r5 = random() = {r5:.4f}")
        writeFile("")
    else:
      r7 = random.uniform(0,1)
      if condition:
        writeFile(f"r >= 0.5 -> {r:.4f} >= 0.5")
        writeFile(f"r7 = random() = {r7:.4f}")
        writeFile("")
    
    for j in range(dim):   
      di = bestSolution[j] - population[i][j]
      
      if condition:
        writeFile(f"ind {i+1} dim {j+1} iter {it}")
        writeFile(f"di = Best[{j+1}] - X[{i+1},{j+1}] = {bestSolution[j]:.4f} - {population[i][j]:.4f} = {di:.4f}")
      
      if r < 0.5:
        if condition: writeFile(f"r < 0.5 -> {r:.4f} < 0.5")

        if i != N-1:
          S = np.power((population[i][j] - population[i+1][j]) ,2)
          if condition: writeFile(f"S = (X[{i+1}][{j+1}] - X[{i+2}][{j+1}])^2 = ({population[i][j]:.4f} - {population[i+1][j]:.4f})^2 = {S:.4f}")
        else:
          S = np.power((population[i][j] - population[0][j]) ,2)
          if condition: writeFile(f"S = (X[{i+1}][{j+1}] - X[0][{j+1}])^2 = ({population[i][j]:.4f} - {population[0][j]:.4f})^2 = {S:.4f}")
        
        I = r2 * S / (4 * pi * np.power(di + epsilon, 2))

        Xnew[i][j] = (bestSolution[j] + 
                      F * beta * I * bestSolution[j] + 
                      F * r3 * alpha * di * 
                      np.abs( math.cos(2 * pi * r4) * (1 - math.cos(2 * pi * r5)) )
                      )
        if condition:
          writeFile(f"I = r2 * S / (4 * pi * (di + epsilon)^2) = {r2:.4f} * {S:.4f} / (4 * {pi:.4f} * ({di:.4f} + {epsilon})^2) = {I:.4f}")
          writeFile(f"""Xnew[{i+1}][{j+1}] = (Best[{j+1}] +
  F * beta * I * Best[{j+1}] +
  F * r3 * alpha * di *
  |cos(2 * pi * r4) * (1 - cos( 2 * pi * r5))|) =
  ({bestSolution[j]:.4f} +
  {F} * {beta} * {I:.4f} * {bestSolution[j]:.4f} +
  {F} * {r3:.4f} * {alpha:.4f} * {di:.4f} *
  |cos(2 * {pi:.4f} * {r4:.4f}) * (1 - cos(2 * {pi:.4f} * {r5:.4f}))|) =
  {Xnew[i][j]:.4f}""")
          writeFile("")
      else:
        Xnew[i][j] = bestSolution[j]+F*r7*alpha*di
        if condition:
          writeFile(f"Xnew[{i+1}][{j+1}] = Best[{j+1}]+F*r7*alpha*di = {bestSolution[j]:.4f}+{F}*{r7:.4f}*{alpha:.4f}*{di:.4f} = {Xnew[i][j]:.4f}")
          writeFile("")
    
    if condition:
      writeFile(f"X:\n{np.array(population)}")
      writeFile("")
      writeFile(f"Xnew:\n{np.array(Xnew)}")
      writeFile("")
    Xnew[i], newFitness = function(Xnew[i])
    if typeProblem == 'MIN': condition1 = newFitness < fitness[i]
    elif typeProblem == 'MAX': condition1 = newFitness > fitness[i]
    if condition1: population[i] = Xnew[i]

    if condition:
      writeFile(f"Result X:\n{np.array(population)}")
      writeFile("")

  return np.array(population)