import cv2
import numpy as np

video = cv2.VideoCapture("./imagens/video.mp4")

while (True):
  # Lê o video frame por frame
    rep, frame = video.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # PRIMEIRA IMAGEM HSV
    image_lower_hsv1 = np.array([54, 68, 100])
    image_upper_hsv1 = np.array([168, 255, 255])

    # Fazendo a máscara da imagem 1
    mask = cv2.inRange(hsv, image_lower_hsv1, image_upper_hsv1)

    # SEGUNDA IMAGEM HSV
    image_lower_hsv2 = np.array([30, 80, 20])
    image_upper_hsv2 = np.array([45, 255, 255])

    # Fazendo a máscara da imagem 2
    mask2 = cv2.inRange(hsv, image_lower_hsv2, image_upper_hsv2)

    contornos = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
    cv2.drawContours(frame, contornos, -1, (255, 0, 0), 5)

    contornos2 = cv2.findContours(
        mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
    cv2.drawContours(frame, contornos2, -1, (255, 0, 0), 5)

    M1 = cv2.moments(contornos[0])

    M2 = cv2.moments(contornos2[1])

    cx1 = int(M1['m10']/M1['m00'])
    cy1 = int(M1['m01']/M1['m00'])

    cx2 = int(M2['m10']/M2['m00'])
    cy2 = int(M2['m01']/M2['m00'])

    cv2.line(frame, (cx1, cy1), (cx2, cy2), (252, 3, 86), 5)

    # Mostra os resultados dos frames
    cv2.imshow('frame', frame)

    key = cv2.waitKey(20)
    if key == 27:  # sai apertando ESC
        break

    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

# Termina quando acaba o video
video.release()
cv2.destroyAllWindows()
