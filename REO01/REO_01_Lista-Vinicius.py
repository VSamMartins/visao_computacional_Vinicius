########################################################################################################################
# DATA: 07/08/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# ALUNO: Vinicius Samuel Martins
# E-MAIL: viniciusmartins93@outlook.com
# GITHUB: VSamMartins
########################################################################################################################
# REO 01 - LISTA DE EXERCÍCIOS
#Importação
import numpy as np
np.set_printoptions(precision=2)#Determina duas casas depois da vírgula
np.set_printoptions(suppress=True)
#'''
# EXERCÍCIO 01:
# a) Declare os valores 43.5,150.30,17,28,35,79,20,99.07,15 como um array numpy.
print('Lista 1-a')
print('-'*100)

Vetor= ([43.5,150.30,17,28,35,79,20,99.07,15])
print("Vetor", Vetor)
print("Tipo")
print(type(Vetor))
print('-='*50)

Vetor1a= np.array(Vetor)# Transformação em lista array para obter a média e variâncias
print("Vetor1a", Vetor1a)
print("Tipo")
print(type(Vetor1a))
print('-='*50)

# b) Obtenha as informações de dimensão, média, máximo, mínimo e variância deste vetor.
print('Lista 1-b')
print('-'*100)

Comp = len(Vetor1a)# Para realizar a contagem da dimensão do vetor1a
print ("Dimensão: "+str(Comp))

#Outra forma de se obter a dimensão do vetor1a
Comp2 = np.shape(Vetor1a)
print ("Dimensão (função np. shape): "+str(Comp2))
print('-'*100)
media = np.mean(Vetor1a)
max = np.max(Vetor1a)
min = np.min(Vetor1a)
var = np.var(Vetor1a)
print ("Dimensão: "+str(Comp))
print ("Média: "+str(media))
print ("Máximo: "+str(max))
print ("Mínimo: "+str(min))
print ("Variância: "+str(var))
print('-='*50)


# c) Obtenha um novo vetor em que cada elemento é dado pelo quadrado da diferença entre cada elemento do vetor declarado
# na letra a e o valor da média deste.
print('Lista 1-c')
print('-'*100)

Novo_vetor = (Vetor1a - media)
print("Novo_vetor: " +str (Novo_vetor ** 2))# Novo vetor criado da diferença de cada valor individual do vetor menos a
# a média ao quadrado
print('-='*50)

# d) Obtenha um novo vetor que contenha todos os valores superiores a 30.

print('Lista 1-d')
print('-'*100)

Pos = np.where(Vetor1a>=30)# Indicar a posição dos valores acima de 30 do vetor1a.
Novo_vetor30 = Vetor1a[Pos[0]]# Capturar estes valores indicados na posição do objeto "Pos" dentro do vetor1a.
print("Novo_vetor>30: " +str(Novo_vetor30))
print('-='*50)

# e) Identifique quais as posições do vetor original possuem valores superiores a 30

print('Lista 1-e')
print('-'*100)
Pos = np. where(Vetor1a>30)
print("Posição (Novo_vetor>30): " +str (Pos[0]))# Para acessar somente as posições da saída do np. where
print('-='*50)

# f) Apresente um vetor que contenha os valores da primeira, quinta e última posição.

print('Lista 1-f')
print('-'*100)

#Valores_do_vetor1a = Vetor1a[0:]#Outra maneira de acessar os valores do vetor1a da posição 0(43,5) até o final do vetor(15)
#print("Valores do vetor1a:" +str(Valores_do_vetor1a))
print("Vetor total (Vetor1a):",(Vetor1a))
posvetor_f = Vetor1a[[0,5,8]]# Obtendo um vetor_f com valores do vetor1a na posição primeira, quinta e última.
print("vetor_f pos1_5_9:", posvetor_f)
print('-='*50)

# g) Crie uma estrutura de repetição usando o for para apresentar cada valor e a sua respectiva posição durante as iterações
print('Lista 1-g')
print('-'*100)

for i in range(Comp):
    posição = i + 1
    print("Posição: " + str(posição) + ("| Valor: " + str(Vetor1a[i])))
print('-=' * 50)

# h) Crie uma estruta de repetição usando o for para fazer a soma dos quadrados de cada valor do vetor.

print('Lista 1-h')
print('-'*100)

v2= np.zeros(Comp)
somador = 0
for i in range(Comp):
    v2[i] = (Vetor1a[i]) ** 2
    somador += v2[i]
print("Vetor^2: " + str(v2))
print("Soma de quadrados: " + str(somador))
print('-=' * 50)

# i) Crie uma estrutura de repetição usando o while para apresentar todos os valores do vetor

print('Lista 1-i')
print('-'*100)
Comp = 0
while Vetor1a[Comp] != 10:
    print(Vetor1a[Comp])
    Comp = Comp + 1
    if Comp == (len(Vetor1a)):
        print('Posição igual a: ' + str(Comp) + ' - A condição estabelecida retornou true, vamos sair do while')
        break
print('-=' * 50)

# j) Crie um sequência de valores com mesmo tamanho do vetor original e que inicie em 1 e o passo seja também 1.

print('Lista 1-j')
print('-'*100)

novo_vetorj = np.arange(1, Comp + 1, 1)
print('Arange:Sequência de 1 até 9 (passo: 1)')
print(novo_vetorj)
print('-=' * 50)

# h) Concatene o vetor da letra a com o vetor da letra j.

print('Lista 1-h')
print('-'*100)
cont_vetor = np.concatenate((Vetor1a, novo_vetorj))
print('Concatenando o Vetor')
print(cont_vetor )
print('-=' * 50)
print('-' * 100)

########################################################################################################################
########################################################################################################################
########################################################################################################################

# Exercício 02
# a) Declare a matriz abaixo com a biblioteca numpy.
# 1 3 22
# 2 8 18
# 3 4 22
# 4 1 23
# 5 2 52
# 6 2 18
# 7 2 25

print('Exercicio 2-a')
print('-' * 100)

matriz = np.array([[1, 3, 22], [2, 8, 18], [3, 4, 22], [4, 1, 23], [5, 2, 52], [6, 2, 18], [7, 2, 25]])
print("Matriz: ", matriz)
print('-=' * 50)
#matriz1 = np.array([[1, 3, 22], [2, 8, 18], [3, 4, 22], [4, 1, 23], [5, 2, 52], [6, 2, 18], [7, 2, 25]])
#print("Matriz: " + str(matriz1))# Duas maneiras de visualizar no console
#print('-=' * 50)

# b) Obtenha o número de linhas e de colunas desta matriz

print('Exercicio 2-b')
print('-' * 100)
nl, nc = np.shape(matriz)# separando duas variáveis pelo uso da vírgula e começando sempre com o número de linhas e depois colunas
print('Número de linhas: ', nl)
print('Número de colunas: ', nc)
print('-=' * 50)

# c) Obtenha as médias das colunas 2 e 3.

print('Exercicio 2-c')
print('-' * 100)
media_coluna2 = np.mean(matriz[:, 1])
media_coluna3 = np.mean(matriz[:, 2])
print('Média coluna 2', str(media_coluna2))
print('Média coluna 3', str(media_coluna3))
print('-=' * 50)

# d) Obtenha as médias das linhas considerando somente as colunas 2 e 3.

print('Exercicio 2-d')
print('-' * 100)
submatriz_lin = matriz[0:, 1:]#começa apartir da linha 1 com os valores apartir da coluna 2 ([início:fim:incremento])
medias_linhas = np.mean(submatriz_lin, axis=1)# Colocar axis=1 Para linha e axis=0 para coluna
print('Média Linhas: ', str(medias_linhas))
print('-=' * 50)

# e) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade inferior a 5.

print('Exercicio 2-e')
print('-' * 100)
submatriz_sev = np.squeeze(np.asarray(matriz[:, 1])) < 5#Gera um array com os valores que são true e false com a restrição (matriz[:, 1])) < 5
print(' Genótipos severidade a doença inferior a 5', submatriz_sev)
print('-=' * 50)

# f) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de peso de 100 grãos superior ou igual a 22.

print('Exercicio 2-f')
print('-' * 100)
submatriz_sev = np.squeeze(np.asarray(matriz[:, 2])) >=22
print(' Genótipos- peso de 100 >=22', submatriz_sev)
print('-=' * 50)

# g) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade igual ou inferior a 3 e peso de 100
# grãos igual ou superior a 22.

print('Exercicio 2-f')
print('-' * 100)
submatriz_sev = np.squeeze(np.asarray(matriz[:, 2])) >=22
print(' Genótipos- peso de 100 >=22', submatriz_sev)
print('-=' * 50)


# h) Crie uma estrutura de repetição com uso do for (loop) para apresentar na tela cada uma das posições da matriz e o seu
#  respectivo valor. Utilize um iterador para mostrar ao usuário quantas vezes está sendo repetido.
#  Apresente a seguinte mensagem a cada iteração "Na linha X e na coluna Y ocorre o valor: Z".
#  Nesta estrutura crie uma lista que armazene os genótipos com peso de 100 grãos igual ou superior a 25

print('Exercicio 2-h')
print('-' * 100)
Contador = 0
Genotipos = []
for i in np.arange(0, nl, 1):
    if matriz[i, 2] >= 25:
        Genotipos.append(matriz[i, 0])
    for j in np.arange(0, nc, 1):
        Contador += 1
        print('Iteração: ' + str(Contador))
        # No numpy a linha e coluna se inicia por 0, se somarmos +1 no indexador facilita à visualização,pois se inicia na linha 1 e não na linha 0 e coluna 0.
        print(
            'Na linha ' + str(i + 1) + ' e coluna ' + str(j + 1) + ' ocorre o valor: ' + str(matriz[int(i), int(j)]))
        print('-' * 100)
print("Lista de genótipos com peso de 100 grãos igual ou superior a 25:")
print(Genotipos)
print('-=' * 50)


########################################################################################################################
########################################################################################################################
########################################################################################################################

# EXERCÍCIO 03:
# a) Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e a variância amostral um vetor qualquer,
# baseada em um loop (for).

print('Exercicio 3-a')
print('-' * 100)
from Funcoes_Ex3_Vinicius import media, variancia_amostral

print('Exemplo aplicação da função:')
vet = np.array([2, 4, 6, 8, 10, 20, 30, 40, 50])#vetor=vet
print('Vetor: ' + str(vet) + ' Média: ' + str(media(vet)) + ' Variância amostral: ' + str(variancia_amostral(vet)))
print('-=' * 50)

# b) Simule três arrays com a biblioteca numpy de 10, 100, e 1000 valores e com distribuição normal com média 100 e 
#variância 2500. Pesquise na documentação do numpy por funções de simulação.

print('Exercicio 3-b')
print('-' * 100)
med, sigma = 100, 50 #Como o sigma é o desvio padrão (variancia= sigma**2).Então temos sigma =50
vetor1 = np.random.normal(med, sigma, 10)  # np.random.normal (media, desvio padrao, tamanho)
vetor2 = np.random.normal(med, sigma, 100)
vetor3 = np.random.normal(med, sigma, 1000)

print('Vetor 1 - 10 valores' , vetor1)
print('-'*100)
print('Vetor 2 - 100 valores' , vetor2)
print('-'*100)
print('Vetor 3 - 1000 valores' , vetor3)
print('-='*50)

# c) Utilize a função criada na letra a para obter as médias e variâncias dos vetores simulados na letra b.

print('Exercicio 3-c')
print('-' * 100)
from Funcoes_Ex3_Vinicius import media, variancia_amostral

print('Obtendo média e variância pela aplicação da função em vetores simulados na letra (a):')
print(' Média Vetor 1: ' + str(media(vetor1)) + ' Variância amostral Vetor 1: ' + str(variancia_amostral(vetor1)))
print('-' * 100)
print(' Média Vetor 2: ' + str(media(vetor2)) + ' Variância amostral Vetor 2: ' + str(variancia_amostral(vetor2)))
print('-' * 100)
print(' Média Vetor 3: ' + str(media(vetor3)) + ' Variância amostral Vetor 3: ' + str(variancia_amostral(vetor3)))
print('-=' * 50)


# d) Crie histogramas com a biblioteca matplotlib dos vetores simulados com valores de 10, 100, 1000 e 100000.

print('Exercicio 3-d')
print('-' * 100)
med, sigma = 100, 50 #Como o sigma é o desvio padrão (variancia= sigma**2).Então temos sigma =50
vetor1 = np.random.normal(med, sigma, 10)# np.random.normal (media, desvio padrao, tamanho)
vetor2 = np.random.normal(med, sigma, 100)
vetor3 = np.random.normal(med, sigma, 1000)
vetor4 = np.random.normal(med, sigma, 100000)

from matplotlib import pyplot as plt
plt.style.use('seaborn-muted')
count, bins, ignored = plt.hist(vetor1, 30, density=True)
plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
         np.exp(- (bins - med) ** 2 / (2 * sigma ** 2)), linewidth=2, color='r')
plt.title('Vetor 1 - Valor 10')
plt.show()
count, bins, ignored = plt.hist(vetor2, 100, density=True)
plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
         np.exp(- (bins - med) ** 2 / (2 * sigma ** 2)), linewidth=2, color='r')
plt.title('Vetor 2 - Valor 100')
plt.show()
count, bins, ignored = plt.hist(vetor3, 100, density=True)
plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
         np.exp(- (bins - med) ** 2 / (2 * sigma ** 2)), linewidth=2, color='r')
plt.title('Vetor 3 - Valor 1000')
plt.show()
count, bins, ignored = plt.hist(vetor4, 100, density=True)
plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
         np.exp(- (bins - med) ** 2 / (2 * sigma ** 2)), linewidth=2, color='r')
plt.title('Vetor 4 - Valor 100000')
plt.show()

########################################################################################################################
########################################################################################################################
########################################################################################################################
# EXERCÍCIO 04:
# a) O arquivo dados.txt contem a avaliação de genótipos (primeira coluna) em repetições (segunda coluna) quanto a quatro
# variáveis (terceira coluna em diante). Portanto, carregue o arquivo dados.txt com a biblioteca numpy, apresente os dados e obtenha as informações
# de dimensão desta matriz.
print('Exercicio 4-a')
print('-' * 100)

dados = np.loadtxt('dados.txt')
print('Dados', str(dados))

# Dimensão da Matriz (nl,nc)
nl, nc = np.shape(dados)
print('Número de linhas: ' + str(nl))
print('Número de colunas: ' + str(nc))
print('-=' * 50)

# b) Pesquise sobre as funções np.unique e np.where da biblioteca numpy
'''
help (np.unique)
help (np.where)
'''
# c) Obtenha de forma automática os genótipos e quantas repetições foram avaliadas

print('Exercicio 4-c')
print('-' * 100)
print('Genótipos: ')
Genotipos = np.unique(dados[0:30, 0:1], axis=0)
nlg, ncg = np.shape(Genotipos)
print('Número de linhas: ' + str(nl))
print('Número de colunas: ' + str(nc))
print(np.unique(dados[0:30, 0:1], axis=0))  # dados[0:30,0:1] ([início:fim:incremento])
print('Número de repetições: ')
print(np.unique(dados[0:30, 1:2], axis=0))
print('-=' * 50)


# d) Apresente uma matriz contendo somente as colunas 1, 2 e 4.

print('Exercicio 4-d')
print('-' * 100)

print('Matriz coluna 1, 2 e 4')
Matrizsub_col = dados[:, [0, 1, 3]]
print(Matrizsub_col)
print('-=' * 50)
# e) Obtenha uma matriz que contenha o máximo, o mínimo, a média e a variância de cada genótipo para a variavel da coluna 4.
#Salve esta matriz em bloco de notas.
print('Exercicio 4-e')
print('-' * 100)
minimos = np.zeros((nlg, 1))
maximos = np.zeros((nlg, 1))
medias = np.zeros((nlg, 1))
vars = np.zeros((nlg, 1))
it = 0
for i in np.arange(0, nl, 3):  # Realiza a leitura das 30 linhas do vetor original de acordo com o numero de repetições
    minimos[it, 0] = np.min(Matrizsub_col[i:i + 3, 2], axis=0)
    maximos[it, 0] = np.max(Matrizsub_col[i:i + 3, 2], axis=0)
    medias[it, 0] = np.mean(Matrizsub_col[i:i + 3, 2], axis=0)
    vars[it, 0] = np.var(Matrizsub_col[i:i + 3, 2], axis=0)
    it += 1  # incrementa + 1 no it

print('Matriz de parâmetros dos genótipos')
matriz_concat = np.concatenate((Genotipos, minimos, maximos, medias, vars), axis=1)
print(matriz_concat)
# Função np.savetxt para salvar um arquivo no caminho da pasta
import os
np.savetxt('matriz_ex4-e.txt', matriz_concat, delimiter=' ',newline=os.linesep, fmt = '%i %2.2f %2.2f %2.2f %2.4f')
print('-=' * 50)

# f) Obtenha os genótipos que possuem média (médias das repetições) igual ou superior a 500 da matriz gerada na letra anterior.
print('Exercicio 4-f')
print('-' * 100)
dadosmedia_vare = np.loadtxt('matriz_ex4-e.txt')
gen_letraf = np.squeeze(np.asarray(dadosmedia_vare[:, 3])) >= 500  #gera um array boleano com valores da coluna 4 (média)>=500
print("Média de genótipos maior ou igual a 500:")
print(dadosmedia_vare[gen_letraf, 0])
print('-=' * 50)

#g) Apresente os seguintes graficos:
#    - Médias dos genótipos para cada variável. Utilizar o comando plt.subplot para mostrar mais de um grafico por figura
#    - Disperão 2D da médias dos genótipos (Utilizar as três primeiras variáveis). No eixo X uma variável e no eixo Y outra.
print('Exercicio 4-g')
print('-' * 100)

print('Gráficos de médias e dispersão 2D')
dados = np.loadtxt('dados.txt')
media1 = np.zeros((nlg, 1))
media2 = np.zeros((nlg, 1))
media3 = np.zeros((nlg, 1))
media4 = np.zeros((nlg, 1))
media5 = np.zeros((nlg, 1))
it = 0
for me in np.arange(0, 30, 3):  # percorre as 30 linhas do vetor original de acordo com o numero de repetições
    media1[it, 0] = np.mean(dados[me:me + 3, 2], axis=0)
    media2[it, 0] = np.mean(dados[me:me + 3, 3], axis=0)
    media3[it, 0] = np.mean(dados[me:me + 3, 4], axis=0)
    media4[it, 0] = np.mean(dados[me:me + 3, 5], axis=0)
    media5[it, 0] = np.mean(dados[me:me + 3, 6], axis=0)
    it += 1  # incrementa + 1 no it

MEDIA_dos_dados = np.concatenate((Genotipos, media1, media2, media3, media4, media5),
                              axis=1)  # matriz de medias dos genotipos para as 5 variáveis
nl, nc = np.shape(MEDIA_dos_dados)
# Gráfico de barras das médias
plt.style.use('ggplot')
plt.figure('Gráfico de Médias')
plt.subplot(2, 3, 1)  # a figura tem 2linhas, 3 colunas, e esse grafico vai ocupar a posição 1
plt.bar(MEDIA_dos_dados[:, 0], MEDIA_dos_dados[:, 1])
plt.title('Variável 1')
plt.xticks(MEDIA_dos_dados[:, 0])

plt.subplot(2, 3, 2)
plt.bar(MEDIA_dos_dados[:, 0], MEDIA_dos_dados[:, 2])
plt.title('Variável 2')
plt.xticks(MEDIA_dos_dados[:, 0])

plt.subplot(2, 3, 3)
plt.bar(MEDIA_dos_dados[:, 0], MEDIA_dos_dados[:, 3])
plt.title('Variável 3')
plt.xticks(MEDIA_dos_dados[:, 0])

plt.subplot(2, 3, 4)
plt.bar(MEDIA_dos_dados[:, 0], MEDIA_dos_dados[:, 4])
plt.title('Variável 4')
plt.xticks(MEDIA_dos_dados[:, 0])

plt.subplot(2, 3, 5)
plt.bar(MEDIA_dos_dados[:, 0], MEDIA_dos_dados[:, 5])
plt.title('Variável 5')
plt.xticks(MEDIA_dos_dados[:, 0])
plt.show()

# Gráfico de disperssão

plt.style.use('ggplot')
fig = plt.figure('Gráfico de disperão 2D das três primeiras variaveis')
plt.subplot(2, 2, 1)
cores = ['black', 'blue', 'red', 'green', 'yellow', 'pink', 'cyan', 'orange', 'darkviolet', 'slategray']

for ij in np.arange(0, nl, 1):
    plt.scatter(MEDIA_dos_dados[ij, 1], MEDIA_dos_dados[ij, 2], s=50, alpha=0.8, label=MEDIA_dos_dados[ij, 0], c=cores[ij])

plt.xlabel('Var 1')
plt.ylabel('Var 2')
plt.subplot(2, 2, 2)
for ij in np.arange(0, nl, 1):
    plt.scatter(MEDIA_dos_dados[ij, 2], MEDIA_dos_dados[ij, 3], s=50, alpha=0.8, label=MEDIA_dos_dados[ij, 0], c=cores[ij])

plt.xlabel('Var 2')
plt.ylabel('Var 3')
plt.subplot(2, 2, 3)
for ij in np.arange(0, nl, 1):
    plt.scatter(MEDIA_dos_dados[ij, 1], MEDIA_dos_dados[ij, 3], s=50, alpha=0.8, label=MEDIA_dos_dados[ij, 0], c=cores[ij])

plt.xlabel('Var 1')
plt.ylabel('Var 3')
plt.legend(bbox_to_anchor=(2.08, 0.7), title='Genótipos', borderaxespad=0., ncol=5)
plt.show()
print('-=' * 50)
########################################################################################################################
########################################################################################################################
########################################################################################################################
