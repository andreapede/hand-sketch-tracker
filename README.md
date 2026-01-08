# Hand Sketch & Pose Tracker

Una raccolta di strumenti basati su **Python**, **OpenCV** e **MediaPipe** per il rilevamento e la visualizzazione in tempo reale di mani, volto e corpo umano tramite webcam.

## Funzionalità

Il progetto include quattro moduli principali:

1.  **Hand Tracker** (`hand_tracker.py`): Rileva le mani e disegna lo scheletro sovrapposto al video.
2.  **Face Expression Tracker** (`face_tracker.py`): Analizza le espressioni facciali e stima l'emozione corrente (Felice, Arrabbiato, Sorpreso, ecc.).
3.  **Body Pose Tracker** (`body_track.py`): Traccia 33 punti chiave del corpo completo per l'analisi posturale.
4.  **Hand Silhouette 3D** (`hand_silhouette.py`): Un "Digital Twin" della mano che renderizza una silhouette 3D con effetti di profondità e occlusione in una finestra separata.

## Requisiti

*   Python 3.8+
*   Webcam

## Installazione

1.  Clona questo repository:
    ```bash
    git clone https://github.com/andreapede/hand-sketch-tracker.git
    cd hand-sketch-tracker
    ```

2.  Crea l'ambiente Conda e installa le dipendenze:
    ```bash
    conda env create -f environment.yml
    conda activate hand-sketch-tracker
    ```
    
    *Oppure, se preferisci usare pip manualmente:*
    ```bash
    pip install opencv-python mediapipe numpy
    ```

3.  I modelli AI necessari (`.task`) sono già inclusi nel repository per comodità:
    *   `hand_landmarker.task`
    *   `face_landmarker.task`
    *   `pose_landmarker.task`

## Utilizzo

Esegui uno degli script Python direttamente dal terminale.

### 1. Rilevamento Mani
```bash
python hand_tracker.py
```

### 2. Analisi Espressioni Facciali ed Emozioni
```bash
python face_tracker.py
```

### 3. Rilevamento Corpo Completo
```bash
python body_track.py
```

### 4. Silhouette Mano 3D (Digital Twin)
```bash
python hand_silhouette.py
```

## Note

*   Per uscire da qualsiasi programma, assicurati che la finestra sia attiva e premi il tasto **ESC**.
*   La prima esecuzione potrebbe richiedere qualche secondo per inizializzare i modelli di MediaPipe.
