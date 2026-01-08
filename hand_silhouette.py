import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Definizione manuale delle connessioni
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_silhouette(shape, detection_result):
    # Crea un'immagine nera vuota
    height, width = shape[:2]
    silhouette = np.zeros((height, width, 3), dtype=np.uint8)

    hand_landmarks_list = detection_result.hand_landmarks

    # Se non ci sono mani, ritorna lo sfondo nero
    if not hand_landmarks_list:
        return silhouette

    # Scorre le mani rilevate
    for hand_landmarks in hand_landmarks_list:
        points = []

        # Converte e raccoglie i punti
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))

        # Disegna le linee (Ossa) - Stile Neon (Cyan)
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(silhouette, points[start_idx], points[end_idx], (255, 255, 0), 2)

        # Disegna i giunti (Articolazioni) - Stile Neon (Magenta)
        for point in points:
            cv2.circle(silhouette, point, 4, (255, 0, 255), -1)

    return silhouette

def main():
    # Configura le opzioni per il rilevatore
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Apre la webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Errore: Impossibile aprire la webcam.")
        return

    print("Premi 'ESC' per uscire.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Frame vuoto ignorato.")
            continue

        # Capovolgi l'immagine orizzontalmente
        image = cv2.flip(image, 1)

        # Converti in RGB per MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Rilevamento
        detection_result = detector.detect(mp_image)

        # Genera la silhouette
        silhouette_image = draw_silhouette(image.shape, detection_result)

        # Mostra le finestre
        cv2.imshow('Webcam Input', image)
        cv2.imshow('Hand Silhouette Digital Twin', silhouette_image)

        # Interrompi il loop se viene premuto 'ESC' (codice 27)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

