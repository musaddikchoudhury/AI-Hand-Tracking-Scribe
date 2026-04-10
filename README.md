# AI-Hand-Tracking-Scribe: Spatial AI & Gesture Gaming

Developed by **Musaddik Choudhury** *Computer Science Major | 4.000 GPA | BMCC (CUNY)*

An advanced computer vision laboratory that transforms a standard webcam into a high-precision spatial input device. This project features a gesture-controlled **AI Handwriting Remote** and a **Hand-Tracking Flappy Bird** game.

## 🚀 Key Features

### 1. AI Remote & Handwriting OCR (`main.py`)
* **Pinch-to-Draw:** Write in mid-air using real-time thumb-and-index pinch detection.
* **TrOCR Pipeline:** Integrated Microsoft’s `microsoft/trocr-base-handwritten` Transformer model for high-accuracy handwriting recognition.
* **Lexical Snapping:** Implements fuzzy matching against a custom dictionary (`words.txt`) to ensure commands like "VOLUME UP" or "NETFLIX" trigger perfectly.
* **Media Control:** Automatically executes system commands via `pyautogui` based on recognized text.

### 2. Flappy Hand Game (`flappy_bird.py`)
* **Gesture-Driven Physics:** Control the bird's flight using a physical pinching motion.
* **Smooth Animation:** Features adaptive bird rotation and a particle system (feather bursts) for a polished gaming experience.
* **High Performance:** Optimized with MediaPipe (Complexity 0) to maintain a locked 60 FPS.

## 🧠 Technical Highlights

### Jitter Mitigation: The One-Euro Filter
To solve the "shaky hand" problem common in webcam tracking, I implemented a **One-Euro Filter**. This adaptive signal filter minimizes lag during fast movements while providing extreme stability for precision writing.

The filter adapts its cutoff frequency based on velocity:
$$f_{cutoff} = f_{min} + \beta \cdot |\dot{x}|$$

### Vision Pre-processing
Before reaching the AI model, the digital ink is processed through a custom pipeline:
1. **Bounding Box Extraction:** Crops the canvas to the specific area of text.
2. **Binary Inversion:** Converts digital ink to black-on-white "paper" format to match the model's training data.
3. **Resizing:** Centered padding and scaling to $384 \times 96$ pixels for optimal inference.

## 🛠 Setup & Installation
1. Clone the repo: `git clone https://github.com/musaddikchoudhury/AI-Hand-Tracking-Scribe.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the remote: `python main.py`
4. Play the game: `python flappy_bird.py`
