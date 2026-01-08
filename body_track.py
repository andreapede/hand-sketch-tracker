import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Definizione manuale delle connessioni principali del corpo (MediaPipe Pose)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), # Occhio sinistro
    (0, 4), (4, 5), (5, 6), (6, 8), # Occhio destro
    (9, 10), # Bocca
    (11, 12), # Spalle
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), # Braccio sinistro
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), # Braccio destro
    (11, 23), (12, 24), # Tronco laterale
    (23, 24), # Bacino
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27), # Gamba sinistra
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28)  # Gamba destra
]

def draw_landmarks_on_image(image_bgr, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(image_bgr)

    # Scorre le pose rilevate (di solito 1)
    for pose_landmarks in pose_landmarks_list:
        height, width, _ = annotated_image.shape
        points = {}

        # Disegna i punti
        for idx, landmark in enumerate(pose_landmarks):
            # Ignoriamo punti con visibilità bassa o fuori schermo
            if landmark.visibility < 0.5:
                continue

            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points[idx] = (x, y)

            # Colore diverso per i punti (giallo)
            cv2.circle(annotated_image, (x, y), 4, (0, 255, 255), -1)

        # Disegna le connessioni
        for connection in POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            if start_idx in points and end_idx in points:
                # Linea verde acqua
                cv2.line(annotated_image, points[start_idx], points[end_idx], (255, 255, 0), 2)

    return annotated_image

def main():
    # Configura le opzioni per il rilevatore PoseLandmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    detector = vision.PoseLandmarker.create_from_options(options)

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

        # Disegna i landmarks
        annotated_image = draw_landmarks_on_image(image, detection_result)

        # Mostra l'immagine risultante
        cv2.imshow('Body Pose Tracker', annotated_image)

        # Interrompi il loop se viene premuto 'ESC' (codice 27)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

