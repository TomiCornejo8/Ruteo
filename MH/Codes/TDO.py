import numpy as np
import random

# condition = it<3 or it==maxIter
# if condition: writeFile(f"")
def writeFile(string):
     with open('result.txt','a', encoding='utf-8') as f: f.write(f"{string}\n")

def iterarTDO(maxIter, it, dim, population, fitness, function, typeProblem):
    condition = it<3 or it==maxIter
    N = len(population)
    population = np.array(population)

    for i in range(N):
        r = random.uniform(0.0, 1.0)

        # Se escoge un demonio de Tasmania aleatorio de entre la población
        k = np.random.choice(np.delete(np.arange(N), i))
        CPi = population[k]
        xNew = np.copy(population[i])

        if condition:
            writeFile(f"ind {i+1} iter {it}")
            writeFile(f"arange = arange(N) = arange({N}) = {np.arange(N)+1}")
            writeFile(f"arange = delete(arange,i) = delete(arange,{i+1}) = {np.delete(np.arange(N)+1,i)}")
            writeFile(f"k = randomChoice(arange) = {k+1}")
            writeFile(f"CPi = X[k] = X[{k+1}] = {CPi}")
            writeFile(f"xNew = X[i] = X[{i+1}]:{xNew}")
            writeFile("")

        if typeProblem == 'MIN': condition1 = function(CPi)[1] < fitness[i]
        elif typeProblem == 'MAX': condition1 = function(CPi)[1] > fitness[i]
        for j in range(dim):
            if condition: writeFile(f"ind {i+1} dim {j+1} iter {it}")
            # Definimos si este se debe alejar o acercar, en función de su evaluación según la FO
            # Si el demonio escogido evaluado resulta ser mejor, se acerca
            if condition1:
                I = random.randint(1, 2)
                rand = random.uniform(0.0, 1.0)
                xNew[j] = population[i][j] + rand * (CPi[j] - I * population[i][j])
                if condition:
                    writeFile(f"f(CPi) < f(X[{i+1}]) -> {function(CPi)[1]:.4f} < {fitness[i]:.4f}")
                    writeFile(f"I = randint(1,2) = {I}")
                    writeFile(f"xNew[{j+1}] = X[{i+1},{j+1}] + random() * (CPi[{j+1}] - I * X[{i+1},{j+1}]) = {population[i][j]:.4f} + {rand:.4f} * ({CPi[j]:.4f} - {I} * {population[i][j]:.4f}) = {xNew[j]:.4f}")
                    writeFile("")
            # Si el demonio escogido evaluado resulta ser peor, se aleja
            else:
                rand = random.uniform(0.0, 1.0)
                xNew[j] = population[i][j] + rand * (population[i][j] - CPi[j])
                if condition:
                    writeFile(f"xNew[{j+1}] = X[{i+1},{j+1}] + random() * (X[{i+1},{j+1}] - CPi[{j+1}]) = {population[i][j]:.4f} + {rand:.4f} * ({population[i][j]:.4f} - {CPi[j]:.4f}) = {xNew[j]:.4f}")
                    writeFile("")
        
        xNew,fitnessNew = function(xNew)
        if typeProblem == 'MIN': condition2 = fitnessNew < fitness[i]
        elif typeProblem == 'MAX': condition2 = fitnessNew > fitness[i]
        if condition2: population[i] = np.copy(xNew)
        if condition:
            writeFile(f"ind {i+1} iter {it}")
            writeFile(f"f(xNew) < f(X[{i+1}]) -> {fitnessNew:.4f} < {fitness[i]:.4f} -> X[{i+1}] = xNew = {xNew}")

        if r >= 0.5:
            # EXPLOTACIÓN
            R = 0.01 * (1 - (it / maxIter))

            if condition:
                writeFile(f"r = random() = {r}")
                writeFile(f"r >= 0.5 -> {r:.4f} >= 0.5")
                writeFile(f"R = 0.01 * (1 - (it / maxIter)) = 0.01 * (1 - ({it} / {maxIter})) = {R:.4f}")
                writeFile("")

            # Se realiza la búsqueda local (Nueva posición)
            for j in range(dim):
                rand = random.uniform(0.0, 1.0)
                aux = xNew[j]
                xNew[j] = population[i][j] + (2 * rand - 1) * R * xNew[j]
                if condition:
                    writeFile(f"ind {i+1} dim {j+1} iter {it}")
                    writeFile(f"xNew[{j+1}] = X[{i+1},{j+1}] + (2 * random() - 1) * R * xNew[{j+1}] = {population[i][j]:.4f} + (2 * {rand:.4f} - 1) * {R:.4f} * {aux:.4f} = {xNew[j]:.4f}")
                    writeFile("")

            xNew,fitnessNew = function(xNew)
            if typeProblem == 'MIN': condition3 = fitnessNew < fitness[i]
            elif typeProblem == 'MAX': condition3 = fitnessNew > fitness[i]
            if condition3: population[i] = np.copy(xNew)
            if condition:
                writeFile(f"f(xNew) < f(X[{i+1}]) -> {fitnessNew:.4f} < {fitness[i]:.4f} -> X[{i+1}] = xNew = {xNew}")
                writeFile("")
        else:
            if condition: writeFile("")

    return population