import random
import numpy as np

# condition = it<3 or it==100
# if condition: writeFile(f"")
def writeFile(string):
     with open('result.txt','a', encoding='utf-8') as f: f.write(f"{string}\n")

def iterarRSA(maxIter, it, dim, population, bestSolution,LB,UB):
    '''
    maxIter: Máximo de iteraciones 
    it: iteración actual
    dim: Dimensión de las soluciones
    population: population actual de soluciones
    bestSolution: Mejor individuo obtenido hasta ahora
    LB: Margen inferior
    UB: Margen superior
    '''
    condition = it<3 or it==100
    #PARAM
    alfa = 0.1
    beta = 0.1
    #Small value epsilon
    eps = 1e-10
    
    #Actualización de valor ES
    r3 = random.randint(-1, 1) #r3 denotes to a random integer number between −1 and 1, pag4
    ES = 2*r3*(1-(1/maxIter))
    if condition:
        writeFile(f"iter {it}")
        writeFile(f"r3 = random(-1,1) = {r3}")
        writeFile(f"ES = 2*r3*(1-(1/maxIter)) = 2*{r3}*(1-(1/{maxIter})) = {ES:.4f}")
        writeFile("")

    #Pob size
    N = population.__len__()

    #For de población
    for i in range(N):
        #For de dimensión
        for j in range(dim):
            #Actualización de valores de la metaheurística
            r2 = random.randint(0, N-1)
            R =  (bestSolution[j] - population[r2][j])/(bestSolution[j]+eps)
            P = alfa + (population[i][j]-np.mean(population[i])) / (UB-LB+eps)
            Eta = bestSolution[j]*P
            rand = random.random()
            ##Ecc de movimiento##

            if condition:
                writeFile(f"ind {i+1} dim {j+1} iter {it}")
                writeFile("")
                writeFile("r2 = random(0,N-1)")
                writeFile("R = (Best[j] - X[r2][j])/(Best[j]+eps)")
                writeFile("P = alfa + (X[i][j]-mean(X[i])) / (UB-LB+eps)")
                writeFile("Eta = Best[j] * P")
                writeFile("")
                writeFile(f"r2 = random(0,{N}-1) = {r2}")
                writeFile(f"R = = (Best[{j+1}] - X[{r2}][{j+1}])/(Best[{j+1}]+eps) = ({bestSolution[j]:.4f} - {population[r2][j]:.4f})/({bestSolution[j]:.4f}+{eps:.4f}) = {R:.4f}")
                writeFile(f"P = alfa + (X[{i+1}][{j+1}]-mean(X[{i+1}])) / (UB-LB+eps) = {alfa} + ({population[i][j]:.4f} - {np.mean(population[i]):.4f}) / ({UB}-{LB}+{eps:.4f}) = {P:.4f}")
                writeFile(f"Eta = Best[{j+1}] * P = {bestSolution[j]:.4f} * {P:.4f} = {Eta:.4f}")
                writeFile(f"rand = random() = {rand:.4f}")
                writeFile("")

            #ec1
            if(it<maxIter/4):
                population[i][j] = bestSolution[j] - Eta*beta - R*rand
                if condition:
                    writeFile(f"it < maxIter/4 -> {it} < {(maxIter/4):.4f}")
                    writeFile("X[i][j] = Best[j] - Eta * beta - R * rand")
                    writeFile(f"X[{i+1}][{j+1}] = Best[{j+1}] - Eta*beta - R*rand = {bestSolution[j]:.4f} - {Eta:.4f} * {beta} - {R:.4f} * {rand:.4f} = {population[i][j]}")
            #ec2
            elif(it<(2*maxIter)/4 and it>=maxIter/4):
                r1 = random.randint(0, N-1)
                population[i][j] = bestSolution[j] * population[r1][j] * ES * rand
                if condition:
                    writeFile(f"it < (2*maxIter)/4 and it >= maxIter/4 -> {it} < {((2*maxIter)/4):.4f} and {it} >= {(maxIter/4):.4f}")
                    writeFile("r1 = random(0,N-1)")
                    writeFile("X[i][j] = Best[j] * X[r1][j] * ES * rand")
                    writeFile(f"r1 = rand(0,{N}-1) = {r1}")
                    writeFile(f"X[{i+1}][{j+1}] = Best[{j+1}] * X[{r1}][{j+1}] * ES * rand = {bestSolution[j]:.4f} * {population[r1][j]:.4f} * {ES:.4f} * {rand:.4f} = {population[i][j]:.4f}")
            #ec3
            elif(it<(maxIter*3)/4 and it>=(2*it)/4):
                population[i][j] = bestSolution[j] * P * rand
                if condition:
                    writeFile(f"it < (maxIter*3)/4 and it >= (2*it)/4 -> {it} < {((maxIter*3)/4):.4f} and {it} >= {((2*it)/4):.4f}")
                    writeFile("X[i][j] = Best[j] * P * rand")
                    writeFile(f"X[{i+1}][{j+1}] = Best[{j+1}] * P * rand = {bestSolution[j]:.4f} * {P:.4f} * {rand:.4f}")
            #ec4
            else:
                population[i][j] = bestSolution[j] - Eta*eps - R*rand
                if condition:
                    writeFile("else")
                    writeFile("X[i][j] = Best[j] - Eta * eps - R * rand")
                    writeFile(f"X[{i+1}][{j+1}] = Best[{j+1}] - Eta * eps - R * rand = {bestSolution[j]:.4f} - {Eta:.4f} * {eps:.4f} - {R:.4f} * {rand:.4f} = {population[i][j]:.4f}")
            if condition: writeFile("")
        #Fin for dim

    return np.array(population)