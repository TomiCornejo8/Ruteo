import numpy as np
import random
import math

# condition = it<3 or it==maxIter
# if condition: writeFile(f"")
def writeFile(string):
     with open('result.txt','a', encoding='utf-8') as f: f.write(f"{string}\n")

def levyFunction(condition):
    lambd = 1.5
    s = 0.01
    
    w = random.uniform(0,1)
    k = random.uniform(0,1)

    gamma1 = math.gamma(lambd+1)
    sin = math.sin(math.pi*lambd)
    gamma2 = math.gamma((lambd+1)/2)
    sigma = (gamma1 * sin)/(gamma2 * lambd * (2**((lambd-1)/2)))

    result = s*((w*sigma)/abs(pow(k,(1/lambd))))

    if condition:
        writeFile(f"w = random() = {w:.4f}")
        writeFile(f"k = random() = {k:.4f}")
        writeFile(f"σ = (Γ(λ+1) * sin(π*λ)) / (Γ((λ+1)/2) * λ * 2^((λ-1)/2)) = ({gamma1:.4f} * {sin:.4f})/({gamma2:.4f} * {lambd} * (2^(({lambd}-1)/2))) = {sigma:.4f}")
        writeFile(f"levy = levy() = s * ((w*σ) / |k^(1/λ)|) = {s} * (({w:.4f}*{sigma:.4f}) / |{k:.4f}^(1/{lambd:.4f})|) = {result:.4f}")
    return result

def iterarSHO(maxIter, it, dim, population,bestSolution, function, typeProblem):
    condition = it<3 or it==maxIter
    population = np.array(population)
    bestSolution = np.array(bestSolution)
    N = population.shape[0]
    
    u = 0.05
    v = 0.05
    l = 0.05

    beta = np.random.randn(N, dim)
    r1 = np.random.randn(dim)

    if condition:
        writeFile(f"iter {it}")
        writeFile(f"β = randn(N,dim) = randn({N},{dim}) =\n{beta}")
        writeFile(f"r₁ = randn(dim) = randn({dim}) = {r1}")
        writeFile("")

    for i in range(N):
        for j in range(dim):
            if condition: writeFile(f"ind {i+1} dim {j+1} iter {it}")
            if(r1[j] > 0):
                if condition: writeFile(f"r₁[{j+1}] > 0 -> {r1[j]:.4f} > 0")
                levy = levyFunction(condition)
                theta = random.uniform(0,2*math.pi)   
                p = u * np.exp(theta * v)
                x = p * math.cos(theta)
                y = p * math.sin(theta)
                z = p * theta
                aux = population[i,j]
                population[i,j] = (population[i,j] +
                                   levy * (bestSolution[j] - population[i,j]) *
                                   x * y * z +
                                   bestSolution[j])
                if condition:
                    writeFile(f"r₁[{j+1}] > 0 -> {r1[j]:.4f} > 0")
                    writeFile(f"θ = random(0,2*π)")
                    writeFile(f"ρ = u * exp(θ*v) = {u} * exp({theta:.4f} * {v}) = {p:.4f}")
                    writeFile(f"x = ρ * cos(θ) = {p:.4f} * cos({theta:.4f}) = {x:.4f}")
                    writeFile(f"y = ρ * sin(θ) = {p:.4f} * sin({theta:.4f}) = {y:.4f}")
                    writeFile(f"z = ρ * θ = {p:.4f} * {theta:.4f} = {z:.4f}")
                    writeFile(f"X[{i+1},{j+1}] = X[{i+1},{j+1}] + levy * (Best[{j+1}] - X[{i+1},{j+1}]) * x * y * z + Best[{j+1}] = {aux:.4f} + {levy:.4f} * ({bestSolution[j]:.4f} - {aux:.4f}) * {x:.4f} * {y:.4f} * {z:.4f} + {bestSolution[j]:.4f} = {population[i,j]}")
            else:
                rand = random.uniform(0,1)
                aux = population[i,j]
                population[i,j] = (population[i,j] +
                                   rand * l * beta[i,j] *
                                   (population[i,j] - beta[i,j] * bestSolution[j]))
                if condition:
                    writeFile(f"r₁[{j+1}] ≤ 0 -> {r1[j]:.4f} ≤ 0")
                    writeFile(f"X[{i+1},{j+1}] = X[{i+1},{j+1}] + random() * l * β[{i+1},{j+1}] * (X[{i+1},{j+1}] - β[{i+1},{j+1}] * Best[{j+1}]) = {aux:.4f} + {rand:.4f} * {l} * {beta[i,j]:.4f} * ({aux:.4f} - {beta[i,j]:.4f} * {bestSolution[j]:.4f}) = {population[i,j]:.4f}")
            if condition: writeFile("")
        population[i],_ = function(population[i])
        
    alpha = (1-it/maxIter)**((2*it)/maxIter)
    if condition:
        writeFile(f"iter {it}")
        writeFile(f"α = (1-it/maxIter)^((2*it)/maxIter) = (1-{it}/{maxIter})^((2*{it})/{maxIter}) = {alpha:.4f}")
        writeFile("")
    fitness = np.zeros(N)
    for i in range (N):
        for j in range (dim):
            r2 = random.uniform(0,1)
            rand = random.uniform(0,1)
            if condition:
                writeFile(f"ind {i+1} dim {j+1} iter {it}")
                writeFile(f"r₂ = random() = {r2:.4f}")
            if(r2>0.1):
               aux = population[i,j]
               population[i,j] = alpha * (bestSolution[j] - rand*population[i,j])
               if condition:
                   writeFile(f"r₂ > 0.1 -> {r2:.4f} > 0.1")
                   writeFile(f"X[{i+1},{j+1}] = α * (Best[{j+1}] - random() * X[{i+1},{j+1}]) = {alpha:.4f} * ({bestSolution[j]:.4f} - {rand:.4f} * {aux:.4f}) = {population[i,j]:.4f}")
            else:
               aux = population[i,j]
               population[i,j] = ((1-alpha) * (population[i,j] -
                                    rand * bestSolution[j]) +
                                    alpha * population[i,j])
               if condition:
                   writeFile(f"r₂ ≤ 0.1 -> {r2:.4f} ≤ 0.1")
                   writeFile(f"X[{i+1},{j+1}] = (1-α) * (X[{i+1},{j+1}] - random() * Best[{j+1}]) + α * X[{i+1},{j+1}] = (1-{alpha:.4f}) * ({aux:.4f} - {rand:.4f} * {bestSolution[j]:.4f}) + {alpha:.4f} * {aux:.4f} = {population[i,j]:.4f}")
            if condition: writeFile("")
        
        population[i],fitness[i] = function(population[i])
        
    if typeProblem == 'MIN': sortIndex = np.argsort(fitness)
    elif typeProblem == 'MAX': sortIndex = np.argsort(fitness)[::-1]

    if condition:
        writeFile(f"iter ")    

    father = population[sortIndex[:N // 2]]
    mother = population[sortIndex[N // 2:]]

    if condition:
        writeFile(f"iter {it}")
        writeFile(f"f(X) = {fitness}")
        writeFile(f"SortFitness = {fitness[sortIndex]}")
        writeFile(f"father = SortedFitness[:N // 2] = SortedFitness[:{N} // 2] = {father}")
        writeFile(f"mother = SortedFitness[N // 2:] = SortedFitness[{N} // 2:] = {mother}")
        writeFile("")
        
    offspring = np.zeros((N //2 , dim))
    fitnessOffspring = np.zeros(N // 2)
    for k in range(N // 2):
            r3 = np.random.rand()
            if condition:
                writeFile(f"ind {k+1} iter {it}")
                writeFile(f"r₃ = random() = {r3:.4f}")
                writeFile("")

            for j in range(dim):
                offspring[k,j] = r3 * father[k,j] + (1 - r3) * mother[k,j]
                if condition:
                    writeFile(f"ind {k+1} dim {j+1} iter {it}")
                    writeFile(f"offspring[{k+1},{j+1}] = r₃ * father[{k+1},{j+1}] + (1 - r₃) * mother[{k+1},{j+1}] = {r3:.4f} * {father[k,j]:.4f} + (1 - {r3:.4f}) * {mother[k,j]:.4f} = {offspring[k,j]:.4f}")
                    writeFile("")

            offspring[k],fitnessOffspring[k] = function(offspring[k])

    newFitness = np.concatenate((fitness, fitnessOffspring))
    newPopulation = np.concatenate((population, offspring))
        
    if typeProblem == 'MIN': sortIndex = np.argsort(newFitness)
    elif typeProblem == 'MAX': sortIndex = np.argsort(newFitness)[::-1]

    if condition:
        writeFile(f"iter {it}")
        writeFile(f"X =\n{population}")
        writeFile(f"f(X) = {fitness}")
        writeFile(f"offspring =\n{offspring}")
        writeFile(f"f(offspring) = {fitnessOffspring}")
        writeFile(f"sort = sort(concatenate(f(X),f(offspring))) = {newFitness[sortIndex]}")
        writeFile(f"sort[:N] = sort[:{N}] = {newFitness[sortIndex[:N]]}")
        writeFile(f"population =\n{population}")
        writeFile("")

    population = newPopulation[sortIndex[:N]]
    
    return population