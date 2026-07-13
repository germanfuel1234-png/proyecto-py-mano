import cv2  # Importa la librería OpenCV para visión por computadora

# Abre la cámara web (0 es el índice de la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Bucle principal para capturar y mostrar los frames de la cámara
while True:
    ret, frame = cap.read()  # Lee un frame de la cámara; ret=True si la lectura fue exitosa
    print("ret:", ret)  # Imprime si la lectura fue exitosa
    if not ret:
        break  # Si no se pudo leer el frame, termina el bucle
    cv2.imshow("cam", frame)  # Muestra el frame en una ventana llamada "cam"
    if cv2.waitKey(1) & 0xFF == 27:  # Espera 1 ms y verifica si se presionó ESC (código 27)
        break  # Si se presionó ESC, termina el bucle

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
