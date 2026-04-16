# HoloScribe: Advanced Spatial AI & Gesture Gaming Laboratory

Developed by **Musaddik Choudhury** *Computer Science Major | 4.000 GPA | Borough of Manhattan Community College (CUNY)*

HoloScribe is a high-fidelity spatial computing interface that transforms a standard webcam into a precision input device. By integrating **Transformer-based OCR (TrOCR)** with a custom **Gesture State Machine**, this project enables seamless mid-air handwriting and interactive gesture-driven gaming.

---

## 🚀 Key Features

### 1. HoloScribe v3: AI Handwriting Scribe (`handwriting_ai_techhand.py`)
* **State Machine Architecture**: Features a robust dual-mode system (Standby and Draw Mode) with timed gesture transitions to prevent accidental input.
* **Multi-Layer Glow Ink**: Implements a triple-layered sub-sampling algorithm to render neon-glow strokes with a "hot white core" for high-end visual feedback.
* **Tech-Hand v3 HUD**: A holographic interface providing real-time skeletal pulsing, index-finger crosshairs, and a live AI history log.
* **Lexical Snapping**: Automatically fuzzy-matches recognized text against a custom `words.txt` dictionary to ensure high command accuracy.

### 2. Flappy Bird: Gesture-Driven Gaming (`flappy_bird.py`)
* **Physical Pinch Mechanics**: Uses real-time pinch edge detection to map thumb-and-index finger motions to the bird's flight.
* **Polished UX**: Features a professional particle system for feather bursts, smooth bird rotation, and parallax scrolling backgrounds.
* **High Performance**: Optimized with MediaPipe to maintain a locked 60 FPS on standard hardware.

---

## 🧠 Technical Highlights

### Jitter Mitigation: The One-Euro Filter
To eliminate the high-frequency jitter common in webcam tracking, the system utilizes a **One-Euro Filter**. This adaptive signal filter minimizes lag during rapid gestures while maintaining extreme stability for precise writing.

The filter adapts its cutoff frequency based on velocity:
$$f_{cutoff} = f_{min} + \beta \cdot |\dot{x}|$$

### Advanced Vision Pipeline
Before processing ink for OCR, HoloScribe v3 runs a sophisticated image pre-processing pipeline to maximize accuracy:
1.  **CLAHE Normalization**: Contrast Limited Adaptive Histogram Equalization is applied to normalize local contrast.
2.  **Otsu’s Binarization**: Automatically determines the optimal threshold for converting digital ink into binary format.
3.  **Letterbox Padding**: Ink is centered and padded with a 22-pixel margin into a $640 \times 128$ canvas to match the TrOCR model’s training data.
4.  **Beam Search Inference**: Uses a 5-beam search during model generation to explore multiple linguistic possibilities for higher word accuracy.

---

## 🕹 Gesture Controls

| Mode | Gesture | Action |
| :--- | :--- | :--- |
| **Standby** | ☝ Index Up (3s) | Enter **DRAW MODE** |
| **Standby** | ✌ Peace Sign | Erase Strokes |
| **Global** | ✊ Fist (3s) | Trigger AI Scan |
| **Global** | 🖐 Open Hand (5s) | Clear Canvas & Spawn Particles |
| **Draw** | ☝ Index Up | Write / Draw |
| **Draw** | ✋ Hand Down (3s) | Exit **DRAW MODE** |

---

## 🛠 Setup & Installation
1.  **Clone the Repo**:  
    `git clone https://github.com/musaddikchoudhury/AI-Hand-Tracking-Scribe.git`
2.  **Install Dependencies**:  
    `pip install -r requirements.txt`
3.  **Launch**:  
    `python handwriting_ai_techhand.py` or `python flappy_bird.py`