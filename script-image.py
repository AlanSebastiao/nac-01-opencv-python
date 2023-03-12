import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Lê a imagem
img = cv2.imread('./imagens/circulo.png')

# Converte as cores
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# PRIMEIRA IMAGEM
image_lower_hsv = np.array([150, 3, 3])
image_upper_hsv = np.array([200, 30, 30])

# Fazendo a máscara da imagem 1
mask_hsv = cv2.inRange(img_rgb, image_lower_hsv, image_upper_hsv)

# SEGUNDA IMAGEM
image_lower_hsv2 = np.array([75, 220, 200])
image_upper_hsv2 = np.array([83, 230, 215])

# Fazendo a máscara da imagem 2
mask_hsv2 = cv2.inRange(img_rgb, image_lower_hsv2, image_upper_hsv2)

# JUNTANDO AS DUAS IMAGENS
mask_final = cv2.bitwise_or(mask_hsv, mask_hsv2)

# Fazendo os contornos
contornos, _ = cv2.findContours(
    mask_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Convertendo as cores
mask_rgb = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2RGB)

# Copiando a mascara rgb para usar no desenho dos contornos
contornos_img = mask_rgb.copy()

# Mandando desenhar o primeiro contorno do primeiro circulo
cv2.drawContours(contornos_img, contornos, -1, [255, 0, 0], 5)

# Mandando desenhar o segundo contorno do segundo circulo
cv2.drawContours(contornos_img, contornos, -1, [7, 148, 82], 10)

# Guardando o primeiro contorno
cnt1 = contornos[0]

# Pegando o momento do primeiro contorno
M1 = cv2.moments(cnt1)

# Pegando os pontos x e y do primeiro contorno
cx1 = int(M1['m10']/M1['m00'])
cy1 = int(M1['m01']/M1['m00'])

# Guardando o segundo contorno
cnt2 = contornos[1]

# Pegando o momento do segundo contorno
M2 = cv2.moments(cnt2)

# Pegando os pontos x e y do primeiro contorno
cx2 = int(M2['m10']/M2['m00'])
cy2 = int(M2['m01']/M2['m00'])

# Fazendo a reta que liga os dois centros de massa dos circulos
cv2.line(contornos_img, (cx1, cy1), (cx2, cy2), (252, 3, 86), 5)

# Conta para tirar o coeficiente angular
suby = cy1 - cy2
subx = cx1 - cx2
tangente = suby / subx

# Convertendo a tangente para radianos
rad = math.atan(tangente)

# Convertendo o radianos para graus
degrees = round(math.degrees(rad), 2)

# Exibindo
font = cv2.FONT_HERSHEY_COMPLEX
text = degrees

cv2.putText(contornos_img, str(text), (280, 155),
            font, 1, (200, 50, 0), 2, cv2.LINE_AA)


# For para calcular o centro de massa dos dois circulos e formar a cruz
for centro in contornos:

    # Pegando o momento do circulo e pegando seus pontos x e y para o centro de massa
    M = cv2.moments(centro)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    size = 20
    color = (128, 128, 0)

    # Fazendo o a cruz no centro dos circulos
    cv2.line(contornos_img, (cx - size, cy), (cx + size, cy), color, 5)
    cv2.line(contornos_img, (cx, cy - size), (cx, cy + size), color, 5)

    # Mostrando os equivalentes ao centro de massa
    font = cv2.FONT_HERSHEY_COMPLEX
    text = cx, cy

    cv2.putText(contornos_img, str(text), (cx - 85, cy - 100),
                font, 1, (200, 50, 0), 2, cv2.LINE_AA)

# Definindo o tamanho que a figura vai aparecer quando rodar
fig = plt.figure(figsize=(20, 20))

# Mandando mostrar
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.subplot(1, 2, 2)
plt.imshow(contornos_img)
plt.show()
