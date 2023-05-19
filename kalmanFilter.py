import cv2
import numpy as np

# Inicializar el filtro de Kalman
dt = 1/30.0  # Frecuencia de muestreo
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]], np.float32) * 0.03

# Definir los rangos de color del objeto rojo
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Capturar el video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Bucle principal
while True:
    # Leer un frame del video
    ret, frame = cap.read()
    if not ret:
        break 

    # Convertir a HSV y aplicar un filtro de color para obtener una máscara binaria
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Encontrar los contornos del objeto rojo en la máscara binaria
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Seleccionar el contorno más grande como el objeto a seguir
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        center = np.array([x+w/2, y+h/2], np.float32)

        # Actualizar el filtro de Kalman con la posición del objeto
        kf.predict()
        kf.correct(center)

        # Dibujar el contorno del objeto y su posición estimada por el filtro de Kalman
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #prediction = np.int32(kf.statePost[:2])
        prediction = tuple(map(int, kf.statePost[:2]))
        if len(prediction) == 2:
            print(prediction)
            cv2.circle(frame, tuple(prediction), 4, (0, 0, 255), -1)
        else:
            print("Error: prediction no es una tupla de dos elementos")


    # Mostrar el resultado
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break