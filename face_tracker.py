import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

def draw_landmarks_on_image(image_bgr, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(image_bgr)

    # Disegna i punti per ogni volto rilevato
    for face_landmarks in face_landmarks_list:
        height, width, _ = annotated_image.shape
        for landmark in face_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)

    return annotated_image

def detect_emotion(blendshapes):
    """
    Deduce l'emozione basandosi sui coefficienti blendshapes di MediaPipe.
    Restituisce una stringa con l'emozione rilevata.
    """
    scores = {b.category_name: b.score for b in blendshapes}

    # Soglie e logica euristica semplice

    # Felicità: angoli della bocca in su
    if scores.get('mouthSmileLeft', 0) > 0.4 and scores.get('mouthSmileRight', 0) > 0.4:
        return "FELICE"

    # Rabbia: sopracciglia abbassate e contratte
    if scores.get('browDownLeft', 0) > 0.4 and scores.get('browDownRight', 0) > 0.4:
        return "ARRABBIATO"

    # Sorpresa: sopracciglia alzate e/o bocca aperta
    if scores.get('browInnerUp', 0) > 0.5 and scores.get('jawOpen', 0) > 0.1:
        return "SORPRESO"

    # Paura: occhi spalancati
    if scores.get('eyeWideLeft', 0) > 0.5 and scores.get('eyeWideRight', 0) > 0.5:
        return "SPAVENTATO"

    # Tristezza: angoli della bocca in giù e sopracciglia interne alzate
    if scores.get('mouthFrownLeft', 0) > 0.4 and scores.get('mouthFrownRight', 0) > 0.4:
        return "TRISTE"

    # Occhi chiusi (extra)
    if scores.get('eyeBlinkLeft', 0) > 0.6 and scores.get('eyeBlinkRight', 0) > 0.6:
        return "DORMIENTE"

    return "NEUTRO"

def draw_blendshapes_on_image(image_bgr, detection_result):
    face_blendshapes_list = detection_result.face_blendshapes
    annotated_image = np.copy(image_bgr)

    if not face_blendshapes_list:
        return annotated_image

    # Prendiamo il primo volto
    blendshapes = face_blendshapes_list[0]

    # Rilevamento e visualizzazione Emozione
    emotion = detect_emotion(blendshapes)
    cv2.putText(annotated_image, f"EMOZIONE: {emotion}", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Ordiniamo le espressioni per punteggio decrescente
    sorted_blendshapes = sorted(blendshapes, key=lambda x: x.score, reverse=True)

    # Mostriamo le prime 5
    y_pos = 130
    for category in sorted_blendshapes[:5]:
        text = f"{category.category_name}: {category.score:.2f}"
        cv2.putText(annotated_image, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30

    return annotated_image

def main():
    # Configura le opzioni per il rilevatore FaceLandmarker
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True, # Abilita il rilevamento delle espressioni
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    detector = vision.FaceLandmarker.create_from_options(options)

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

        # Disegna le espressioni (blendshapes)
        annotated_image = draw_blendshapes_on_image(annotated_image, detection_result)

        # Mostra l'immagine risultante
        cv2.imshow('Face Expression Tracker', annotated_image)

        # Interrompi il loop se viene premuto 'ESC' (codice 27)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

