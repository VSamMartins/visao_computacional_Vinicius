########################################################################################################################
# DATA: 20/08/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# ALUNO: Vinicius Samuel Martins
# E-MAIL: viniciusmartins93@outlook.com
# GITHUB: VSamMartins
########################################################################################################################
###Importar os pacotes a ser utilizados pela função 'import'
import numpy as np
import cv2
from matplotlib import pyplot as plt

#REO02- Lista de Exercícios

#EXERCÍCIO 01:
#Selecione uma imagem a ser utilizada no trabalho prático e realize os seguintes processos utilizando o pacote OPENCV do Python:
#a) Apresente a imagem e as informações de número de linhas e colunas; número de canais e número total de pixels;

###Leitura da imagem:
print('Exer.01-a')

nome_arquivo= ("arroz.png")
img_arroz = cv2.imread("arroz.png",1)### A função cv2.imread permite carregar a imagem colorida em BGR, o código 1 permite colorir a imagem. O codigo 0 não permite colorir a imagem (cinza).
print (img_arroz)
obj_arroz_RGB = cv2.cvtColor(img_arroz,cv2.COLOR_BGR2RGB)# A função cv2.cvtColor converte a imagem de BGR pra RGB
# (Espaço de cores composto por 3 cores vermelho, verde e azul)
#Outra maneira de inverter a imagem BGR EM RGB é utilizando as funções cv2.split e cv2.merge.

#Representar a imagem através do matplotlib
plt.figure("Arroz")
plt.imshow (obj_arroz_RGB)

#plt.xticks([]) # Eliminar o eixo X
#plt.yticks([]) # Eliminar o eixo Y
plt.title("Grãos de arroz")
plt.show()

#Informações do número de linhas e colunas; número de canais e número total de pixels:
print('Descrição da imagem:')
lin,col, canais = np.shape(obj_arroz_RGB)
print("Tipo: ",obj_arroz_RGB.dtype)
print("Dimensão: "+str(lin)+'x'+str(col))
print('Largura: '+ str(col))
print('Altura: '+ str(lin))
print('Canais: '+ str(canais))### 3 matrizes sobrepostas (Vermelho, verde e azul)
print('Número total de pixels: '+ str(lin*col))
print('-='*50)

#b) Faça um recorte da imagem para obter somente a área de interesse. Utilize esta imagem para a solução das próximas
# alternativas;
print('Exer.01-b')
img_recorte = obj_arroz_RGB[50:350,100:350] ###Da linha 50 a 350 e da coluna 100 até a 350.
plt.figure("Grãos de Arroz")
plt.imshow (img_recorte)
#plt.xticks([]) # Eliminar o eixo X
#plt.yticks([]) # Eliminar o eixo Y
plt.title("Grãos de arroz-Recorte")
plt.show() ###Atráves deste comando podemos visualizar a imagem gerada.

#Informações do número de linhas e colunas; número de canais e número total de pixels:
print('Descrição da imagem Recorte:')
lin,col, canais = np.shape(img_recorte )
print("Tipo: ",img_recorte.dtype)
print("Dimensão: "+str(lin)+'x'+str(col))
print('Largura: '+ str(col))
print('Altura: '+ str(lin))
print('Canais: '+ str(canais))### 3 matrizes sobrepostas (Vermelho, verde e azul)
print('Número total de pixels: '+ str(lin*col))

# Salvar imagem
cv2.imwrite('Arroz_recorte.png',img_recorte)
print('-='*50)

#c) Converta a imagem colorida para uma de escala de cinza (intensidade) e a apresente utilizando
# os mapas de cores “Escala de Cinza” e “JET”;
print('Exer.01-c')
imagem_cinza = cv2.cvtColor(img_recorte,cv2.COLOR_RGB2GRAY)### Utilizando a imagem recorte e convertendo em RGB cinza
plt.figure('Imagens')
plt.subplot(1,3,3)# A imagem apresenta uma linha com 3 colunas e na posição 3 a imagem cinza
plt.imshow(imagem_cinza, cmap="gray") # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
plt.title("Escala de Cinza")
plt.colorbar(orientation = 'horizontal')

plt.subplot(1,3,2)
plt.imshow(imagem_cinza,cmap="jet") # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
plt.title('JET')
plt.colorbar(orientation = 'horizontal')
plt.subplot(1,3,1)
plt.imshow(img_recorte) # Utilizando somente a imagem original
plt.title("Imagem recorte original")
plt.colorbar(orientation = 'horizontal')
plt.show()

#d) Apresente a imagem em escala de cinza e o seu respectivo histograma; Relacione o histograma e a imagem.

hist_cinza = cv2.calcHist([imagem_cinza],[0],None,[256],[0,256])
plt.figure('Histograma')
plt.imshow(imagem_cinza,cmap="gray") # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
plt.title("Escala de Cinza")
plt.colorbar(orientation = 'horizontal')
plt.show()

plt.plot(hist_cinza,color = 'black')
plt.title("Histograma Imagem cinza")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()

#e) Utilizando a imagem em escala de cinza (intensidade) realize a segmentação da imagem de modo a remover o fundo da imagem utilizando um limiar manual e o limiar obtido pela técnica
#de Otsu. Nesta questão apresente o histograma com marcação dos limiares utilizados, a imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação. Explique
#os resultados.

# Limiarização Manual (THRESHOLD)
limiar_cinza = 130 # O melhorista determina.
(L, img_limiar) = cv2.threshold(imagem_cinza,limiar_cinza,255,cv2.THRESH_BINARY)
(L, img_limiar_inv) = cv2.threshold(imagem_cinza,limiar_cinza,255,cv2.THRESH_BINARY_INV)### Invertendo os valores abaixo so limiar recebe 255
# e abaixo recebe 0
plt.figure('Thresholding - MANUAL')
plt.subplot(2,3,1)
plt.imshow(img_recorte)
plt.title('RGB')



plt.subplot(2,3,2)
plt.imshow(imagem_cinza,cmap='gray')
plt.title('Escala de Cinza')




plt.subplot(2,3,3)
plt.plot(hist_cinza,color = 'black')
plt.axvline(x=limiar_cinza,color = 'r')###Indicando o ponto no x no valor do limiar que é 130
plt.title("Histograma - Cinza")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")




plt.subplot(2,3,4)
plt.imshow(img_limiar,cmap='gray')###Imagem binária que vai obter após a limiarização em que vão apresentar
# os pontos acima do limiar branco e abaixo preto.
plt.title('Binário - L: ' + str(limiar_cinza))


plt.subplot(2,3,5)
plt.imshow(img_limiar_inv,cmap='gray')### Ocorre a inversão das cores da limiarização ( acima do limiar preto e abaixo branco)
plt.title('Binário Invertido: L: ' + str(limiar_cinza))
plt.show()

# Limiarização - Thresholding - OTSU
###Permite identificar o melhor limiar

(L,img_otsu) = cv2.threshold(imagem_cinza,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)### O primeiro parametro "L" é p limiar autômatico
##calculado pela função
###Pela adição + (adição de outra função)cv2.THRESH_OTSU : Irá calcular o melhor limiar
plt.figure('Thresholding - OTSU')
plt.subplot(2,2,1)
plt.imshow(img_recorte)
plt.xticks([])
plt.yticks([])
plt.title('RGB')

plt.subplot(2,2,3)
plt.plot (hist_cinza,color = 'black')
plt.axvline(x=L,color = 'r')
plt.title("Histograma - Cinza")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,2,4)
plt.imshow(img_otsu,cmap='gray')
plt.title('OTSU - L: ' + str(L))
plt.xticks([])
plt.yticks([])
plt.show()

#Segmentação
img_segmentada_manual = cv2.bitwise_and(img_recorte,img_recorte,mask=img_limiar)
img_segmentada = cv2.bitwise_and(img_recorte,img_recorte,mask=img_otsu)
cv2.imwrite('arroz_segmentada.png',img_segmentada_manual)
cv2.imwrite('arroz_segmentada_otsu.png',img_segmentada)

plt.figure('Segmentação')
plt.subplot(2,2,1)
plt.imshow(img_recorte)
plt.title('RGB')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(imagem_cinza,cmap="gray")
plt.title("Escala de Cinza")
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(img_segmentada_manual)
plt.title('MANUAL')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(img_segmentada_manual)
plt.title('OTSU')
plt.xticks([])
plt.yticks([])
plt.show()

#f) Apresente uma figura contento a imagem selecionada nos sistemas RGB, Lab, HSV e YCrCb.
img_lab = cv2.cvtColor(img_recorte,cv2.COLOR_RGB2LAB)
img_hsv = cv2.cvtColor(img_recorte,cv2.COLOR_RGB2HSV)
img_ycrcb = cv2.cvtColor(img_recorte,cv2.COLOR_RGB2YCR_CB)
plt.figure('Segmentação')
plt.subplot(2,2,1)
plt.imshow(img_recorte)
plt.title('RGB')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(img_hsv)
plt.title("HSV")
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(img_lab)
plt.title('LAB')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(img_ycrcb)
plt.title('YCrCb')
plt.xticks([])
plt.yticks([])
plt.show()


#g) Apresente uma figura para cada um dos sistemas de cores (RGB, HSV, Lab e YCrCb) contendo a imagem de
# cada um dos canais e seus respectivos histogramas.

#RGB
hist_r = cv2.calcHist([img_recorte],[0],None,[256],[0,256])
hist_g = cv2.calcHist([img_recorte],[1],None,[256],[0,256])
hist_b = cv2.calcHist([img_recorte],[2],None,[256],[0,256])

plt.figure('IMAGEM 1 G')
plt.subplot(3,3,2)
plt.imshow(img_recorte)
plt.title("RGB")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,4)
plt.imshow(img_recorte[:,:,0],cmap='gray')
plt.title("R")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,7)
plt.plot(hist_r,color = 'r')
plt.title("Histograma - R")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,5)
plt.imshow(img_recorte[:,:,1],cmap='gray')
plt.title("G")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,8)
plt.plot(hist_g,color = 'g')
plt.title("Histograma - G")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,6)
plt.imshow(img_recorte[:,:,2],cmap='gray')
plt.title("B")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,9)
plt.plot(hist_b,color = 'blue')
plt.title("Histograma - B")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()

#LAB
hist_L = cv2.calcHist([img_lab],[0],None,[256],[0,256])
hist_a = cv2.calcHist([img_lab],[1],None,[256],[0,256])
hist_b = cv2.calcHist([img_lab],[2],None,[256],[0,256])

plt.figure('IMAGEM 2 G')
plt.subplot(3,3,2)
plt.imshow(img_lab)
plt.title("Lab")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,4)
plt.imshow(img_lab[:,:,0],cmap='gray')
plt.title("L")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,7)
plt.plot(hist_L)
plt.title("Histograma - L")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,5)
plt.imshow(img_lab[:,:,1],cmap='gray')
plt.title("a")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,8)
plt.plot(hist_g)
plt.title("Histograma - a")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,6)
plt.imshow(img_lab[:,:,2],cmap='gray')
plt.title("b")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,9)
plt.plot(hist_b)
plt.title("Histograma - b")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()

#HSV

hist_H = cv2.calcHist([img_hsv],[0],None,[256],[0,256])
hist_S = cv2.calcHist([img_hsv],[1],None,[256],[0,256])
hist_V = cv2.calcHist([img_hsv],[2],None,[256],[0,256])

plt.figure('IMAGEM 3 G')
plt.subplot(3,3,2)
plt.imshow(img_hsv)
plt.title("HSV")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,4)
plt.imshow(img_hsv[:,:,0],cmap='gray')
plt.title("Hue")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,7)
plt.plot(hist_H)
plt.title("Histograma - Hue")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,5)
plt.imshow(img_hsv[:,:,1],cmap='gray')
plt.title("Saturation")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,8)
plt.plot(hist_S)
plt.title("Histograma - Saturation")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,6)
plt.imshow(img_hsv[:,:,2],cmap='gray')
plt.title("Value")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,9)
plt.plot(hist_V)
plt.title("Histograma - Value")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()

#YCrCb
hist_Y = cv2.calcHist([img_ycrcb],[0],None,[256],[0,256])
hist_Cr = cv2.calcHist([img_ycrcb],[1],None,[256],[0,256])
hist_Cb = cv2.calcHist([img_ycrcb],[2],None,[256],[0,256])

plt.figure('IMAGEM 4 G')
plt.subplot(3,3,2)
plt.imshow(img_ycrcb)
plt.title("YCrCb")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,4)
plt.imshow(img_ycrcb[:,:,0],cmap='gray')
plt.title("Y")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,7)
plt.plot(hist_Y)
plt.title("Histograma - Y")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,5)
plt.imshow(img_ycrcb[:,:,1],cmap='gray')
plt.title("Cr")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,8)
plt.plot(hist_Cr)
plt.title("Histograma - Cr")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,6)
plt.imshow(img_ycrcb[:,:,2],cmap='gray')
plt.title("Cb")
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,9)
plt.plot(hist_Cb)
plt.title("Histograma - Cb")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()

#h) Encontre o sistema de cor e o respectivo canal que propicie melhor segmentação da imagem
#de modo a remover o fundo da imagem utilizando limiar manual e limiar obtido pela técnica
#de Otsu. Nesta questão apresente o histograma com marcação dos limiares utilizados, a
#imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação. Explique
#os resultados e sua escolha pelo sistema de cor e canal utilizado na segmentação. Nesta
#questão apresente a imagem limiarizada (binarizada) e a imagem colorida final obtida da
#segmentação.

r,g,b = cv2.split(img_recorte)
hist_r = cv2.calcHist([r],[0], None, [256],[0,256])
# Limiarização - Thresholding
limiar_cinza = 140
(Lm, img_limiar_manual) = cv2.threshold(r,limiar_cinza,255,cv2.THRESH_BINARY)
(L, img_limiar) = cv2.threshold(r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_segmentada_r = cv2.bitwise_and(img_recorte,img_recorte,mask=img_limiar)
img_segmentada_rmanual = cv2.bitwise_and(img_recorte,img_recorte,mask=img_limiar_manual)

plt.figure('Thresholding')
plt.subplot(2,2,1)
plt.imshow(img_recorte)
plt.title('RGB')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(r,cmap='gray')
plt.title('RGB - r')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.plot(hist_r,color = 'black')
plt.axvline(x=L,color = 'r')
plt.title("Histograma - r")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(2,2,4)
plt.imshow(img_limiar,cmap='gray')
plt.title('Limiar: ' + str(L))
plt.xticks([])
plt.yticks([])

plt.show()

plt.figure('Imagemsegmentada')
plt.subplot(1,2,1)
plt.imshow(img_segmentada_r)
plt.title('Imagem segmentada (OTSU)')
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(img_segmentada_rmanual)
plt.title('Imagem segmentada (MANUAL)')
plt.xticks([])
plt.yticks([])
plt.show()

#i) Obtenha o histograma de cada um dos canais da imagem em RGB, utilizando como mascara
#a imagem limiarizada (binarizada) da letra h.

hist_arroz_r = cv2.calcHist([img_segmentada_r],[0],img_limiar,[256],[0,256])
hist_arroz_g = cv2.calcHist([img_segmentada_r],[1],img_limiar,[256],[0,256])
hist_arroz_b = cv2.calcHist([img_segmentada_r],[2],img_limiar,[256],[0,256])

plt.figure('Exercicio I')
plt.subplot(3,3,2)
plt.imshow(img_segmentada_r)
plt.title('Imagem segmentada')
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,4)
plt.imshow(img_segmentada[:,:,0],cmap = 'gray')
plt.title('Segmentada - R')
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,5)
plt.imshow(img_segmentada[:,:,1],cmap = 'gray')
plt.title('Segmentada - G')
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,6)
plt.imshow(img_segmentada[:,:,2],cmap = 'gray')
plt.title('Segmentada - B')
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,7)
plt.plot(hist_arroz_r,color = 'r')
plt.title("Histograma - R")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,8)
plt.plot(hist_arroz_g,color = 'g')
plt.title("Histograma - G")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3,3,9)
plt.plot(hist_arroz_b,color = 'b')
plt.title("Histograma - B")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.show()


#j) Realize operações aritméticas na imagem em RGB de modo a realçar os aspectos de seu interesse.
# Exemplo (2*R-0.5*G). Explique a sua escolha pelas operações aritméticas. Segue abaixo algumas sugestões')

imgOPR1 = 1.7* img_recorte[:, :, 0] - 1.2* img_recorte[:, :, 1]
imgOPR = imgOPR1.astype(np.uint8) #convertendo para inteiro de 8bit
histw = cv2.calcHist([imgOPR], [0], None, [256], [0, 256])
(M, img_OTSU) = cv2.threshold(imgOPR, 0, 256, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
img_SEGM = cv2.bitwise_and(img_recorte, img_recorte, mask=img_OTSU)

# Apresentando a imagem
plt.figure('Imagem 1.j')
plt.subplot(2, 3, 1)
plt.imshow(img_recorte, cmap='gray')
plt.title('RGB')

plt.subplot(2, 3, 2)
plt.imshow(imgOPR, cmap='gray')
plt.title('1,7R - 1,2*G')

plt.subplot(2, 3, 3)
plt.plot(histw, color='black')
# plt.axline(x=M, color='black')
plt.xlim([0, 256])
plt.xlabel('Valores de pixels')
plt.xlabel('Número de pixels')

plt.subplot(2, 3, 4)
plt.imshow(img_OTSU, cmap='gray')
plt.title('Imagem binária')

plt.subplot(2, 3, 5)
plt.imshow(img_SEGM, cmap='gray')
plt.title('Img segmentada-mascara')
plt.xticks([])
plt.yticks([])
plt.show()
print('='*50)
print(' ')
