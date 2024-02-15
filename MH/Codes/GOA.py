import numpy as np
import random
import math

# condition = it<3 or it==maxIter
# if condition: writeFile(f"")
def writeFile(string):
     with open('result.txt','a', encoding='utf-8') as f: f.write(f"{string}\n")

def V(x,condition):
    if x <= math.pi:
         result =  (-(1 / math.pi) * x + 1) # (0,π)
         if condition: writeFile(f"x ≤ π → {x:.4f} ≤ {math.pi:.4f} → V(x) = -(1/π) * x + 1 = -(1/{math.pi:.4f}) * {x:.4f} + 1 = {result:.4f}")
         return result
    elif x > math.pi:
         result = ((1 / math.pi) * x - 1) # (π,2π)
         if condition: writeFile(f"x > π → {x:.4f} > {math.pi:.4f} → V(x) = (1/π) * x - 1 = (1/{math.pi:.4f}) * {x:.4f} - 1 = {result:.4f}")
         return result

def levy(condition):
    beta = 1.5
    
    mu = random.uniform(0, 1)
    v = random.uniform(0, 1)

    gamma1 = math.gamma(1 + beta)
    gamma2 = math.gamma((1+beta)/2)
    seno = math.sin(math.pi * beta / 2)
    expo = 1 / beta
    
    sigma = ((gamma1 * seno) / (gamma2 * beta * 2**((beta-1)/2))) ** expo

    result = 0.01 * ((mu * sigma) / (abs(v) ** expo))

    if condition:
         writeFile(f"μ = random() = {mu:.4f}")
         writeFile(f"v = random() = {v:.4f}")
         writeFile(f"σ = ((Γ(1+β)*sin((π*β)/2)) / (Γ((1+β)/2)*β*2^((β-1)/2)))^(1/β) = ((Γ(1+{beta})*sin(({math.pi:.4f}*{beta})/2)) / (Γ((1+{beta})/2)*{beta}*2^(({beta}-1)/2)))^(1/{beta}) = {sigma:.4f}")
         writeFile(f"P = levy() = 0.01 * ( (μ * σ) / (|v|^(1/β)) ) = 0.01 * ( ({mu:.4f} * {sigma:.4f}) / (|{v:.4f}|^(1/{beta})) ) = {result:.4f}")
        
    return result

def iterarGOA(maxIter, it, dim, population, bestSolution, fitness, function, typeProblem):
    condition = it<3 or it==maxIter

    population = np.array(population)

    m = 2.5  # Peso en Kg del Gannet
    vel = 1.5  # Velocidad en el agua en m/s del Gannet
    c = 0.2  # Determina si se ejejuta movimiento levy o ajustes de trayectoria
    
    MX = population.copy()
    t = 1 - (it / maxIter)
    t2 = 1 + (it / maxIter)
    randomIndex = random.randint(0, len(population) - 1)
    Xr = population[randomIndex]
    Xm = [np.mean(population[:, j]) for j in range(dim)]

    if condition:
         writeFile(f"iter {it}")
         writeFile(f"MX = X =\n{MX}")
         writeFile(f"t = 1 - (it / maxIter) = 1 - ({it} / {maxIter}) = {t:.4f}")
         writeFile(f"t₂ = 1 + (it / maxIter) = 1 + ({it} / {maxIter}) = {t2:.4f}")
         writeFile(f"randomIndex = randint(0,N-1) = {randomIndex}")
         writeFile(f"Xr = X[randomIndex] = {Xr}")
         writeFile(f"Xm = [mean(X[:,j]) for j in dim] = [mean(X[:,j]) for j in {dim}] = {np.array(Xm)}")
         writeFile("")

    # ========= Exploracion =========
    for i in range(len(population)):
        r = random.uniform(0, 1)
        if condition:
             writeFile(f"ind {i+1} iter {it}")
             writeFile(f"r = random() = {r:.4f}")
        if r > 0.5:
                q = random.uniform(0, 1)
                if condition:
                     writeFile(f"r > 0.5 → {r:.4f} > 0.5")
                     writeFile(f"q = random() = {q:.4f}")
                if q >= 0.5:
                    if condition:
                         writeFile(f"q ≥ 0.5 → {q:.4f} ≥ 0.5")
                         writeFile("")
                    for j in range(dim):
                        r2 = random.uniform(0, 1)
                        r4 = random.uniform(0, 1)

                        a = 2 * math.cos(2 * math.pi * r2) * t
                        A = (2 * r4 - 1) * a

                        u1 = random.uniform(-a, a)
                        u2 = A * (population[i,j] - Xr[j])

                        # Ecuacion 7a
                        MX[i][j] = population[i,j] + u1 + u2
                        
                        if condition:
                             writeFile(f"ind {i+1} dim {j+1} iter {it}")
                             writeFile(f"r₂ = random() = {r2:.4f}")
                             writeFile(f"r₄ = random() = {r4:.4f}")
                             writeFile(f"a = 2 * cos(2 * π * r₂) * t = 2 * cos(2 * {math.pi:.4f} * {r2:.4f}) * {t:.4f} = {a:.4f}")
                             writeFile(f"A = (2 * r₄ - 1) * a = (2 * {r4:.4f} - 1) * {a:.4f} = {A:.4f}")
                             writeFile(f"u₁ = random(-a, a) = random({-a:.4f}, {a:.4f}) = {u1:.4f}")
                             writeFile(f"u₂ = A * (X[{i+1},{j+1}] - Xr[{j+1}]) = {A:.4f} * ({population[i,j]:.4f} - {Xr[j]:.4f}) = {u2:.4f}")
                             writeFile(f"MX[{i+1},{j+1}] = X[{i+1},{j+1}] + u₁ + u₂ = {population[i,j]:.4f} + {u1:.4f} + {u2:.4f} = {MX[i][j]:.4f}")
                             writeFile("")

                else:
                    if condition:
                         writeFile(f"q < 0.5 → {q:.4f} < 0.5")
                         writeFile("")
                    for j in range(dim):
                        r3 = random.uniform(0, 1)
                        r5 = random.uniform(0, 1)

                        if condition:
                             writeFile(f"ind {i+1} dim {j+1} iter {it}")
                             writeFile(f"r₃ = random() = {r3:.4f}")
                             writeFile(f"r₅ = random() = {r5:.4f}")
                             writeFile(f"V(2 * π * r₃) = V(2 * {math.pi:.4f} * {r3:.4f}) = V({2 * math.pi * r3})")
                        
                        Vx = V(2 * math.pi * r3, condition)
                        b = 2 * Vx * t
                        B = (2 * r5 - 1) * b

                        v1 = random.uniform(-b, b)
                        v2 = B * (population[i,j] - Xm[j])
                        # Ecuacion 7b
                        MX[i][j] = population[i,j] + v1 + v2

                        if condition:
                             writeFile(f"b = 2 * V(x) * t = 2 * {Vx:.4f} * {t:.4f} = {b:.4f}")
                             writeFile(f"B = (2 * r₅ - 1) * b = (2 * {r5:.4f} - 1) * {b:.4f} = {B:.4f}")
                             writeFile(f"v₁ = random(-b, b) = random({-b:.4f}, {b:.4f}) = {v1:.4f}")
                             writeFile(f"v₂ = B * (X[{i+1},{j+1}] - Xm[{j+1}]) = {B:.4f} * ({population[i,j]:.4f} - {Xm[j]:.4f}) = {v2:.4f}")
                             writeFile(f"MX[{i+1},{j+1}] = X[{i+1},{j+1}] + v₁ + v₂ = {population[i,j]:.4f} + {v1:.4f} + {v2:.4f} = {MX[i][j]:.4f}")
                             writeFile("")

        # ========= Explotacion =========
        else:
                r6 = random.uniform(0, 1)
                L = 0.2 + (2 - 0.2) * r6
                R = (m * vel**2) / L
                capturability = 1 / (R * t2)

                if condition:
                     writeFile(f"r ≤ 0.5 → {r:.4f} ≤ 0.5")
                     writeFile(f"r₆ = random() = {r6:.4f}")
                     writeFile(f"L = 0.2 + (2 - 0.2) * r₆ = 0.2 + (2 - 0.2) * {r6:.4f} = {L:.4f}")
                     writeFile(f"R = (m * vel^2) / L = ({m:.4f} * {vel:.4f}^2) / {L:.4f} = {R:.4f}")
                     writeFile(f"capturability = 1 / (R * t₂) = 1 / ({R:.4f} * {t2:.4f}) = {capturability:.4f}")
                
                # Caso ajustes exitosos
                if capturability >= c:
                    if condition:
                         writeFile(f"capturability ≥ c → {capturability:.4f} ≥ {c}")
                         writeFile("")
                    for j in range(dim):
                        delta = capturability * abs(population[i,j] - bestSolution[j])
                        # Ecuacion 17a
                        MX[i][j] = t * delta * (population[i,j] - bestSolution[j]) + population[i,j]

                        if condition:
                             writeFile(f"ind {i+1} dim {j+1} iter {it}")
                             writeFile(f"δ = capturability * |X[{i+1},{j+1}] - Best[{j+1}]| = {capturability:.4f} * |{population[i,j]:.4f} - {bestSolution[j]:.4f}| = {delta:.4f}")
                             writeFile(f"MX[{i+1},{j+1}] = t * δ * (X[{i+1},{j+1}] - Best[{j+1}]) + X[{i+1},{j+1}] = {t:.4f} * {delta:.4f} * ({population[i,j]:.4f} - {bestSolution[j]:.4f}) + {population[i,j]:.4f} = {MX[i][j]:.4f}")
                             writeFile("")

                # Caso movimiento Levy
                else:
                    if condition:
                         writeFile(f"capturability < c → {capturability:.4f} < {c}")
                         writeFile("")
                    for j in range(dim):
                        if condition:
                             writeFile(f"ind {i+1} dim {j+1} iter {it}")
                        P = levy(condition)
                        # Ecuacion 17b
                        MX[i][j] = bestSolution[j] - (population[i,j] - bestSolution[j]) * P * t
                        if condition:
                             writeFile(f"MX[{i+1},{j+1}] = Best[{j+1}] - (X[{i+1},{j+1}] - Best[{j+1}]) * P * t = {bestSolution[j]:.4f} - ({population[i,j]:.4f} - {bestSolution[j]:.4f}) * {P:.4f} * {t:.4f} = {MX[i][j]:.4f}")
                             writeFile("")
        
        MX[i],mxFitness = function(MX[i])
        if typeProblem == 'MIN': condition1 = mxFitness < fitness[i]
        elif typeProblem == 'MAX': condition1 = mxFitness > fitness[i]
        if condition:
                  writeFile(f"ind {i+1} iter {it}")
                  writeFile(f"X[{i+1}] = {population[i]} → f(X[{i+1}]) = {fitness[i]:.4f}")
                  writeFile(f"MX[{i+1}] = {MX[i]} → f(MX[{i+1}]) = {mxFitness:.4f}")
        if condition1:
             if condition:
                  writeFile(f"f(MX[{i+1}]) < f(X[{i+1}]) → {mxFitness:.4f} < {fitness[i]:.4f} → X[{i+1}] = MX[{i+1}]")
             population[i] = MX[i]
        else:
             if condition:
                  writeFile(f"f(MX[{i+1}]) ≥ f(X[{i+1}]) → {mxFitness:.4f} ≥ {fitness[i]:.4f} → X[{i+1}] se mantiene")
        if condition: writeFile("")

    return population