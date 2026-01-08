import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Definizione manuale delle connessioni se mp.solutions non è disponibile
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_landmarks_on_image(image_bgr, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(image_bgr)

    # Scorre le mani rilevate
    for hand_landmarks in hand_landmarks_list:
        height, width, _ = annotated_image.shape
        points = []

        # Converte i landmarks normalizzati in coordinate pixel
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
            # Disegna i punti (articolazioni)
            cv2.circle(annotated_image, (x, y), 5, (121, 22, 76), -1)

        # Disegna le linee di connessione
        for connection in HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(annotated_image, points[start_idx], points[end_idx], (250, 44, 250), 2)

    return annotated_image

def main():
    # Configura le opzioni per il rilevatore usando la nuova API Tasks
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

        # Capovolgi l'immagine orizzontalmente per un effetto specchio
        image = cv2.flip(image, 1)

        # Converti in RGB per MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Rilevamento
        detection_result = detector.detect(mp_image)

        # Disegna sull'immagine originale BGR
        annotated_image = draw_landmarks_on_image(image, detection_result)

        # Mostra l'immagine risultante
        cv2.imshow('Hand Tracking (MediaPipe Tasks)', annotated_image)

        # Interrompi il loop se viene premuto 'ESC' (codice 27)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
