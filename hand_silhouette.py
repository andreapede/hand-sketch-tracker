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
        z_values = [lm.z for lm in hand_landmarks]
        min_z = min(z_values)
        max_z = max(z_values)
        z_range = max_z - min_z if max_z != min_z else 1.0

        # Converte e raccoglie i punti (x, y, z)
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            points.append((x, y, z))

        # --- DISEGNO 3D SIMULATO ---

        # Ordiniamo le connessioni in base alla profondità media (Painter's Algorithm)
        # Disegniamo prima le connessioni più lontane (Z maggiore)
        sorted_connections = sorted(HAND_CONNECTIONS,
                                    key=lambda c: (points[c[0]][2] + points[c[1]][2])/2,
                                    reverse=True)

        for start_idx, end_idx in sorted_connections:
            if start_idx < len(points) and end_idx < len(points):
                pt1 = points[start_idx]
                pt2 = points[end_idx]

                # Calcola profondità relativa (0.0 = vicino, 1.0 = lontano)
                avg_z = (pt1[2] + pt2[2]) / 2
                rel_z = (avg_z - min_z) / z_range

                # Interpolazione Colore Linee: Ciano (Vicino) -> Blu Scuro (Lontano)
                # Vicino (rel_z=0): B=255, G=255, R=0
                # Lontano (rel_z=1): B=50, G=0, R=0
                val = 1 - rel_z # 1 è vicino
                b = int(50 + 205 * val)
                g = int(255 * val)
                r = 0

                # Spessore basato sulla profondità
                thickness = int(2 + 5 * val)

                cv2.line(silhouette, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (b, g, r), thickness)

        # Disegniamo i giunti sopra le linee, anch'essi ordinati per Z (dal lontano al vicino)
        # In MediaPipe Z negativo è vicino alla camera, ma qui usiamo il range calcolato.
        # Riordiniamo i punti per disegnarli

        # Creiamo una lista di (indice, punto) e ordiniamo
        sorted_points_indices = sorted(range(len(points)), key=lambda i: points[i][2], reverse=True)

        for i in sorted_points_indices:
            x, y, z = points[i]
            rel_z = (z - min_z) / z_range
            val = 1 - rel_z # 1 è vicino

            # Dimensione raggio dinamica
            radius = int(3 + 8 * val)

            # Colore Giunti: Magenta (Vicino) -> Viola scuro (Lontano)
            # Vicino: B=255, G=0, R=255
            # Lontano: B=50, G=0, R=50
            c_val = int(50 + 205 * val)
            color = (c_val, 0, c_val)

            cv2.circle(silhouette, (x, y), radius, color, -1)
            # Aggiungi un piccolo riflesso bianco per effetto lucido
            cv2.circle(silhouette, (x, y), max(1, radius//3), (255, 255, 255), -1)

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
