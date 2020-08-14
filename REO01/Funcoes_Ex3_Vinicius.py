# EXERCÍCIO 03:
# Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e a variância amostral um vetor qualquer,
# baseada em um loop (for).

import numpy as np
np.set_printoptions(precision=2)#Determina duas casas depois da vírgula

def media (vetor):
    soma = 0
    it = 0#iterador
    for vi in vetor:

        soma += vi # somador = somador + vi
        it+=1 # it = it+1
    mean = soma/it
    return mean# essa função retorna a média  do nosso vetor

def variancia_amostral (vetor):
    soma = 0
    it = 0
    sdq = 0
    for vi in vetor:

        soma += vi  # soma = soma + j
        it += 1  # it = it+1
        sdq += vi**2 #sdq = sdq + j**2

    var = (sdq - ((soma**2/it)))/(it-1)
    return var
