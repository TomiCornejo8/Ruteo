import numpy as np

def writeFile(string):
     with open('result.txt','a', encoding='utf-8') as f: f.write(f"{string}\n")

def iterarFOX(maxIter,it,dim, population, bestSolution):
    condition = it<3 or it==100
    population = np.array(population)
    bestSolution = np.array(bestSolution)
    
    c1 = 0.18  # range of c1 is [0, 0.18]
    c2 = 0.82  # range of c2 is [0.19, 1]
    
    MinT = 0
    Jump = 0

    a = 2 * (1 - (it / maxIter))
    if condition: writeFile(f"a = 2 * (1 - (it / maxIter)) = 2 * (1 - ({it} / {maxIter})) = {a}")

    for i in range(population.shape[0]):
        r = np.random.rand()
        p = np.random.rand()

        if condition: writeFile(f"r = {r:.4f}")
        if condition: writeFile(f"")
        if r >= 0.5:    
            Time_S_T = np.random.rand(dim)
            Sp_S = bestSolution / Time_S_T
            Dist_S_T = Sp_S * Time_S_T
            Dist_Fox_Prey = 0.5 * Dist_S_T
            tt = np.sum(Time_S_T) / dim
            t = tt / 2
            Jump = 0.5 * 9.81 * t**2

            if condition: writeFile(f"tt = sum(Time_S_T / dim = sum({Time_S_T}) / {dim} = {tt:.4f}")
            if condition: writeFile(f"t = tt / 2 = {tt:.4f} / 2 = {t:.4f}")
            if condition: writeFile(f"Jump = 0.5 * 9.81 * t**2 = 0.5 * 9.81 * {t:.4f}**2= {Jump:.4f}")

            for j in range(population.shape[1]):
                if condition: writeFile(f"ind {i+1} dim {j+1} iter {it}")
                if condition: writeFile(f"Time_S_T = {Time_S_T[j]:.4f}")
                if condition: writeFile(f"BestSolution = {bestSolution[j]:.4f} = Dist_S_T = {Dist_S_T[j]:.4f}")
                if condition: writeFile(f"Dist_Fox_Prey = 0.5 * Dist_S_T = 0.5 * {Dist_S_T[j]:.4f} ={Dist_Fox_Prey[j]:.4f}")
                if condition: writeFile(f"p = {p:.4f}")

                if p > 0.18:
                    population[i, j] = Dist_Fox_Prey[j] * Jump * c1
                    if condition: writeFile(f"X[{i+1},{j+1}] = Dist_Fox_Prey * Jump * c1 = {Dist_Fox_Prey[j]:.4f} * {Jump:.4f} * {c1:.4f} = {population[i,j]}")
                elif p <= 0.18:
                    population[i, j] = Dist_Fox_Prey[j] * Jump * c2
                    if condition: writeFile(f"X[{i+1},{j+1}] = Dist_Fox_Prey * Jump * c2 = {Dist_Fox_Prey[j]:.4f} * {Jump:.4f} * {c2:.4f} = {population[i,j]}")
                if condition: writeFile(f"")
            
            if MinT > tt: 
                MinT = tt
                if condition: writeFile(f"MinT = {MinT:.4f}")
                if condition: writeFile(f"\n")
        elif r < 0.5:
            for j in range(population.shape[1]):
                if condition: writeFile(f"ind {i+1} dim {j+1} iter {it}")
                randT = np.random.randn(dim)
                population[i,j] = bestSolution[j] + randT[j] * (MinT * a)
                if condition: writeFile(f"X[{i+1},{j+1}] = bestSolution + randT * MinT * a = {bestSolution[j]:.4f} + {randT[j]:.4f} * {MinT:.4f} * {a:.4f} = {population[i,j]:.4f}")
                if condition: writeFile(f"")
    return population