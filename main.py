import ctypes
import difflib
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception:
    pyautogui = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- GESTURE TUNING ---
# ► NEW: hold index finger ALONE for 5 seconds to enter draw mode.
#        Hold open hand (4+ fingers) for 2 seconds to exit draw mode.
INDEX_HOLD_SECONDS   = 5.0    # seconds to hold index-only → enter draw mode
EXIT_HOLD_SECONDS    = 2.0    # seconds to hold open hand  → exit  draw mode
SCAN_HOLD_SECONDS    = 1.2    # hold fist while in draw mode → OCR scan
DRAW_COOLDOWN        = 1.0    # brief lock after entering draw mode (prevents instant stroke)
CLICK_PINCH_THRESHOLD = 0.095 # still used for keyboard mode

MAX_STROKE_THICKNESS = 28
MIN_STROKE_THICKNESS = 8
THICKNESS_SPEED_SCALE = 0.030
ERASER_RADIUS = 34

# Larger canvas → better handwriting / math recognition
TROCR_W = 512
TROCR_H = 128

WINDOW_NAME = "AI Remote Pro"

# --- THEME COLORS ---
COLOR_BG         = (16,  20,  30)
COLOR_TEXT       = (238, 243, 248)
COLOR_MUTED      = (148, 166, 184)
COLOR_PANEL      = (74,  102, 144)
COLOR_ACCENT     = (95,  220, 255)
COLOR_DRAW       = (88,  232, 168)
COLOR_ERASE      = (104, 176, 255)
COLOR_SCAN       = (108, 214, 255)
COLOR_DANGER     = (96,  126, 255)
COLOR_KEY        = (52,  64,  86)
COLOR_KEY_HOVER  = (86,  124, 178)
COLOR_KEY_PINCH  = (110, 208, 172)
COLOR_HOLD_RING  = (255, 200, 60)   # amber countdown arc

KEY_ROWS = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M","BKSP"],
    ["SPACE","CLR","ENTER"],
]

COMMAND_PHRASES = ["VOLUME UP", "PLAY", "MUTE"]
DEFAULT_WORDS = [
    "HELLO","PLAY","PAUSE","STOP","MUTE","HOME","BACK","ENTER",
    "MENU","NETFLIX","YOUTUBE","SEARCH","VOLUME","UP","DOWN",
    # math / digits
    "0","1","2","3","4","5","6","7","8","9",
    "PLUS","MINUS","TIMES","DIVIDE","EQUALS","SQUARED","CUBED",
    "SUM","TOTAL","HALF","QUARTER","PI","SQRT","ROOT","PERCENT",
]


def normalize_text(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.upper())
    return " ".join(cleaned.split())


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen, ordered = set(), []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def point_in_rect(point: Tuple[int, int], rect: Tuple[int, int, int, int]) -> bool:
    px, py = point
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def load_words() -> List[str]:
    words_path = Path(__file__).with_name("words.txt")
    loaded = []
    if words_path.exists():
        for raw_line in words_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            loaded.append(normalize_text(stripped))
    return dedupe_keep_order(COMMAND_PHRASES + DEFAULT_WORDS + loaded)


# --- SIGNAL FILTERING (One-Euro Filter) ---
class LowPassFilter:
    def __init__(self) -> None:
        self.initialized = False
        self.value = 0.0

    def reset(self) -> None:
        self.initialized = False

    def filter(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.value = value
            self.initialized = True
            return value
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.1, beta: float = 0.03, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.x_filter   = LowPassFilter()
        self.dx_filter  = LowPassFilter()
        self.prev_raw   = None
        self.last_time  = None

    @staticmethod
    def alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()
        self.prev_raw  = None
        self.last_time = None

    def filter(self, value: float, timestamp: float) -> float:
        if self.last_time is None or self.prev_raw is None:
            self.prev_raw  = value
            self.last_time = timestamp
            return self.x_filter.filter(value, 1.0)
        dt     = max(timestamp - self.last_time, 1e-3)
        raw_dx = (value - self.prev_raw) / dt
        dx     = self.dx_filter.filter(raw_dx, self.alpha(dt, self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(dx)
        filtered = self.x_filter.filter(value, self.alpha(dt, cutoff))
        self.prev_raw  = value
        self.last_time = timestamp
        return filtered


@dataclass
class HandState:
    detected:          bool  = False
    cursor:            Tuple[int, int] = (0, 0)
    raw_cursor:        Tuple[int, int] = (0, 0)
    pinch_distance:    float = 1.0
    index_up:          bool  = False
    middle_up:         bool  = False
    ring_up:           bool  = False
    pinky_up:          bool  = False
    fingers_up:        int   = 0
    is_click_pinching: bool  = False
    is_open:           bool  = False   # 4+ fingers extended (exit draw mode)
    is_fist:           bool  = False   # 0 fingers (pen lift / scan trigger)
    is_peace:          bool  = False   # index+middle only → eraser
    is_index_only:     bool  = False   # ★ NEW: only index extended → draw stroke


class HandAnalyzer:
    def __init__(self) -> None:
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.x_filter = OneEuroFilter(min_cutoff=1.0, beta=0.035)
        self.y_filter = OneEuroFilter(min_cutoff=1.0, beta=0.035)

    def close(self) -> None:
        self.hands.close()

    def reset(self) -> None:
        self.x_filter.reset()
        self.y_filter.reset()

    @staticmethod
    def normalized_distance(landmarks, idx_a: int, idx_b: int, frame_w: int, frame_h: int) -> float:
        ax, ay = landmarks[idx_a].x * frame_w, landmarks[idx_a].y * frame_h
        bx, by = landmarks[idx_b].x * frame_w, landmarks[idx_b].y * frame_h
        dist   = math.hypot(ax - bx, ay - by)
        wx, wy = landmarks[0].x * frame_w, landmarks[0].y * frame_h
        mx, my = landmarks[9].x * frame_w, landmarks[9].y * frame_h
        hand_size = max(math.hypot(wx - mx, wy - my), 1.0)
        return dist / hand_size

    def process(self, frame: np.ndarray) -> HandState:
        frame_h, frame_w = frame.shape[:2]
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            self.reset()
            return HandState()

        landmarks = result.multi_hand_landmarks[0].landmark
        now = time.perf_counter()

        raw_x    = int(np.clip(landmarks[8].x * frame_w, 0, frame_w - 1))
        raw_y    = int(np.clip(landmarks[8].y * frame_h, 0, frame_h - 1))
        smooth_x = int(np.clip(self.x_filter.filter(raw_x, now), 0, frame_w - 1))
        smooth_y = int(np.clip(self.y_filter.filter(raw_y, now), 0, frame_h - 1))

        def extended(tip: int, pip: int) -> bool:
            return landmarks[tip].y < landmarks[pip].y

        index_up  = extended(8,  6)
        middle_up = extended(12, 10)
        ring_up   = extended(16, 14)
        pinky_up  = extended(20, 18)
        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

        pinch_distance    = self.normalized_distance(landmarks, 4, 8, frame_w, frame_h)
        is_click_pinching = pinch_distance < CLICK_PINCH_THRESHOLD

        # ★ index-only: only index extended, others down (thumb ignored)
        is_index_only = index_up and not middle_up and not ring_up and not pinky_up

        return HandState(
            detected=True,
            cursor=(smooth_x, smooth_y),
            raw_cursor=(raw_x, raw_y),
            pinch_distance=pinch_distance,
            index_up=index_up,
            middle_up=middle_up,
            ring_up=ring_up,
            pinky_up=pinky_up,
            fingers_up=fingers_up,
            is_click_pinching=is_click_pinching,
            is_open=fingers_up >= 4 and not is_click_pinching,
            is_fist=fingers_up == 0 and not is_click_pinching,
            is_peace=index_up and middle_up and not ring_up and not pinky_up and not is_click_pinching,
            is_index_only=is_index_only,
        )


class DrawingEngine:
    def __init__(self) -> None:
        self.canvas       = None
        self.active_stroke: list = []

    def ensure_canvas(self, frame_w: int, frame_h: int) -> None:
        if self.canvas is None or self.canvas.shape[:2] != (frame_h, frame_w):
            self.canvas = np.zeros((frame_h, frame_w), dtype=np.uint8)
            self.active_stroke = []

    def clear(self) -> None:
        if self.canvas is not None:
            self.canvas.fill(0)
        self.active_stroke = []

    def lift(self) -> None:
        self.active_stroke = []

    def has_content(self) -> bool:
        return self.canvas is not None and cv2.countNonZero(self.canvas) > 0

    def snapshot(self) -> Optional[np.ndarray]:
        return self.canvas.copy() if self.canvas is not None else None

    def add_point(self, point: Tuple[int, int], timestamp: float) -> None:
        if self.canvas is None:
            return
        x, y = point
        if not self.active_stroke:
            t0 = float(MAX_STROKE_THICKNESS)
            self.active_stroke = [(x, y, timestamp, t0)]
            cv2.circle(self.canvas, (x, y), max(1, int(t0 / 2.5)), 255, -1, cv2.LINE_AA)
            return

        prev_x, prev_y, prev_time, prev_thick = self.active_stroke[-1]
        distance = math.hypot(x - prev_x, y - prev_y)
        if distance < 1.0:
            return

        dt    = max(timestamp - prev_time, 1e-3)
        speed = distance / dt
        thick = float(np.clip(
            MAX_STROKE_THICKNESS - speed * THICKNESS_SPEED_SCALE,
            MIN_STROKE_THICKNESS, MAX_STROKE_THICKNESS
        ))

        self._draw_variable_segment((prev_x, prev_y), (x, y), prev_thick, thick)
        self.active_stroke.append((x, y, timestamp, thick))
        if len(self.active_stroke) > 120:
            self.active_stroke = self.active_stroke[-24:]

    def _draw_variable_segment(self, start, end, start_thick, end_thick) -> None:
        dist  = max(math.hypot(end[0] - start[0], end[1] - start[1]), 1.0)
        steps = max(int(dist / 1.3), 1)
        for step in range(steps + 1):
            t  = step / steps
            ix = int(round(start[0] * (1.0 - t) + end[0] * t))
            iy = int(round(start[1] * (1.0 - t) + end[1] * t))
            r  = max(1, int(round((start_thick * (1.0 - t) + end_thick * t) / 2.0)))
            cv2.circle(self.canvas, (ix, iy), r, 255, -1, cv2.LINE_AA)

    def erase_at(self, point: Tuple[int, int], radius: int = ERASER_RADIUS) -> bool:
        if self.canvas is None:
            return False
        cv2.circle(self.canvas, point, radius, 0, -1, cv2.LINE_AA)
        self.active_stroke = []
        return True

    @staticmethod
    def _centered_ocr_image(binary_canvas: np.ndarray) -> Optional[Image.Image]:
        """
        Improved preprocessing for handwriting/math accuracy:
          1. Dilate strokes so thin lines aren't lost
          2. Add generous padding around the bounding box
          3. Scale into a larger canvas (TROCR_W x TROCR_H)
          4. Light Gaussian smooth + re-threshold for crisp edges
        """
        binary = cv2.threshold(binary_canvas, 20, 255, cv2.THRESH_BINARY)[1]

        # ── 1. Dilate to fatten thin strokes ──────────────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=2)

        coords = cv2.findNonZero(binary)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)

        # ── 2. Generous padding so ascenders/descenders aren't clipped ────────
        pad = 24
        x  = max(0, x - pad)
        y  = max(0, y - pad)
        w  = min(binary.shape[1] - x, w + 2 * pad)
        h  = min(binary.shape[0] - y, h + 2 * pad)

        crop   = binary[y : y + h, x : x + w]
        target = np.full((TROCR_H, TROCR_W), 255, dtype=np.uint8)

        # ── 3. Scale to fit the larger canvas ─────────────────────────────────
        scale  = min((TROCR_W - 40) / max(w, 1), (TROCR_H - 24) / max(h, 1))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        ox = (TROCR_W - nw) // 2
        oy = (TROCR_H - nh) // 2
        target[oy : oy + nh, ox : ox + nw] = 255 - resized   # invert: black ink on white

        # ── 4. Smooth → re-threshold for clean edges ─────────────────────────
        blurred = cv2.GaussianBlur(target, (3, 3), 0)
        _, clean = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)
        return Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB))


class OCRWorker:
    # ★ Upgraded to the large model for better handwriting + math accuracy
    MODEL_NAME = "microsoft/trocr-large-handwritten"

    def __init__(self, lexicon: List[str]) -> None:
        self.lexicon        = lexicon
        self.queue          = queue.Queue(maxsize=1)
        self.lock           = threading.Lock()
        self.pending_result = None
        self.status_text    = "READY"
        self.busy           = False
        self.processor = TrOCRProcessor.from_pretrained(self.MODEL_NAME)
        self.model     = VisionEncoderDecoderModel.from_pretrained(self.MODEL_NAME).to(DEVICE)
        self.model.eval()
        threading.Thread(target=self._worker_loop, daemon=True).start()

    def is_busy(self) -> bool:
        with self.lock:
            return self.busy

    def current_status(self) -> str:
        with self.lock:
            return self.status_text

    def consume_result(self) -> Optional[str]:
        with self.lock:
            res = self.pending_result
            self.pending_result = None
            return res

    def submit(self, canvas: np.ndarray) -> bool:
        if self.busy or self.queue.full() or cv2.countNonZero(canvas) == 0:
            return False
        with self.lock:
            self.status_text = "SCANNING..."
        self.queue.put(canvas.copy())
        return True

    def _worker_loop(self) -> None:
        while True:
            canvas = self.queue.get()
            with self.lock:
                self.busy, self.status_text = True, "SCANNING..."
            try:
                img = DrawingEngine._centered_ocr_image(canvas)
                if img:
                    pix = self.processor(images=img, return_tensors="pt").pixel_values.to(DEVICE)
                    with torch.inference_mode():
                        # ★ beam search for better accuracy
                        out = self.model.generate(
                            pix,
                            max_new_tokens=32,
                            num_beams=4,
                            early_stopping=True,
                        )
                    raw   = normalize_text(self.processor.batch_decode(out, skip_special_tokens=True)[0])
                    # ★ slightly looser cutoff so math/numbers match better
                    match = difflib.get_close_matches(raw, self.lexicon, n=1, cutoff=0.45)
                    final = match[0] if match else raw
                    with self.lock:
                        self.pending_result = self.status_text = final
                else:
                    with self.lock:
                        self.status_text = "NO TEXT"
            except Exception as e:
                with self.lock:
                    self.status_text = f"ERROR: {e}"
            finally:
                with self.lock:
                    self.busy = False
                self.queue.task_done()


class CommandExecutor:
    MEDIA_KEYS = {
        "VOLUME UP": ("volumeup",   0xAF),
        "PLAY":      ("playpause",  0xB3),
        "MUTE":      ("volumemute", 0xAD),
    }

    def __init__(self) -> None:
        self.last_exec      = {}
        self.feedback       = ""
        self.feedback_until = 0.0

    def maybe_execute(self, text: str) -> Optional[str]:
        cmd = normalize_text(text)
        if cmd in self.MEDIA_KEYS and time.time() - self.last_exec.get(cmd, 0.0) > 1.0:
            if pyautogui:
                pyautogui.press(self.MEDIA_KEYS[cmd][0])
            self.last_exec[cmd]              = time.time()
            self.feedback, self.feedback_until = f"EXECUTED {cmd}", time.time() + 2.4
            return self.feedback
        return None


class KeyboardController:
    def __init__(self, lexicon: List[str]) -> None:
        self.lexicon        = lexicon
        self.text           = ""
        self.keys           = []
        self.text_bar_rect  = (0, 0, 0, 0)
        self.keyboard_area  = (0, 0, 0, 0)
        self.frame_size     = (0, 0)

    def build(self, w, h) -> None:
        if self.frame_size != (w, h):
            margin = 16
            self.text_bar_rect = (margin, margin, w - 200, 60)
            self.keyboard_area = (margin, 140, w - 32, h - 160)
            self._rebuild_keys()
            self.frame_size = (w, h)

    def _rebuild_keys(self) -> None:
        x0, y0, kw, kh = self.keyboard_area
        row_h = kh // len(KEY_ROWS)
        self.keys = []
        for r_idx, row in enumerate(KEY_ROWS):
            r_y   = y0 + r_idx * row_h
            col_w = kw // len(row)
            for c_idx, label in enumerate(row):
                self.keys.append({
                    "label": label,
                    "rect": (x0 + c_idx * col_w + 4, r_y + 4, col_w - 8, row_h - 8),
                })

    def handle_click(self, pt) -> Tuple[str, Optional[str]]:
        for key in self.keys:
            if point_in_rect(pt, key["rect"]):
                lbl = key["label"]
                if   lbl == "BKSP":  self.text = self.text[:-1]
                elif lbl == "CLR":   self.text = ""
                elif lbl == "ENTER":
                    res = normalize_text(self.text)
                    self.text = ""
                    return "ENTER", res
                elif lbl == "SPACE": self.text += " "
                else:                self.text += lbl
                return f"KEY {lbl}", None
        return "NONE", None


class RemoteUI:
    @staticmethod
    def glass_panel(frame, rect, fill, alpha=0.24, border=COLOR_ACCENT):
        x, y, w, h = rect
        overlay = frame[y : y + h, x : x + w].copy()
        cv2.rectangle(overlay, (0, 0), (w, h), fill, -1)
        frame[y : y + h, x : x + w] = cv2.addWeighted(
            overlay, alpha, frame[y : y + h, x : x + w], 1.0 - alpha, 0
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)

    @staticmethod
    def draw_hold_arc(frame, center: Tuple[int, int], progress: float,
                      color=COLOR_HOLD_RING, radius: int = 28) -> None:
        """Amber arc around fingertip showing how far the 5-second hold has progressed."""
        if progress <= 0:
            return
        angle = int(360 * min(progress, 1.0))
        cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, color, 3, cv2.LINE_AA)

    def draw_draw_mode(self, frame, drawing, hand, ai_text: str, status: str,
                       scan_progress: float, feedback: str,
                       draw_mode_active: bool, hold_progress: float,
                       exit_progress: float) -> np.ndarray:
        output = frame.copy()
        if drawing.canvas is not None:
            output = cv2.add(output, cv2.cvtColor(drawing.canvas, cv2.COLOR_GRAY2BGR))

        mode_label = "✏  DRAW" if draw_mode_active else "STANDBY"
        self.glass_panel(output, (16, 16, 430, 160), COLOR_PANEL)
        cv2.putText(output, f"MODE: {mode_label} | AI: {ai_text}", (30, 50),
                    1, 1.5, COLOR_TEXT, 1)
        cv2.putText(output, f"STATUS: {status}", (30, 90), 1, 1.2, COLOR_DRAW, 1)

        if not draw_mode_active and hold_progress > 0:
            # ── countdown bar + label while user is holding index up ──────────
            bar_w = int(340 * hold_progress)
            cv2.rectangle(output, (30, 112), (370, 126), (40, 50, 70), -1)
            cv2.rectangle(output, (30, 112), (30 + bar_w, 126), COLOR_HOLD_RING, -1)
            secs_left = INDEX_HOLD_SECONDS * (1.0 - hold_progress)
            cv2.putText(output, f"Hold index up to enter DRAW: {secs_left:.1f}s",
                        (30, 148), 1, 1.05, COLOR_HOLD_RING, 1)
        elif draw_mode_active:
            hint = "Open hand 2s to EXIT  |  Fist = pen up  |  ✌ = erase"
            cv2.putText(output, hint, (30, 128), 1, 0.95, COLOR_MUTED, 1)

            if exit_progress > 0:
                bar_w = int(340 * exit_progress)
                cv2.rectangle(output, (30, 145), (370, 157), (40, 50, 70), -1)
                cv2.rectangle(output, (30, 145), (30 + bar_w, 157), COLOR_DANGER, -1)
                cv2.putText(output, "Exiting draw mode...", (30, 172), 1, 1.0, COLOR_DANGER, 1)

        if scan_progress > 0:
            cv2.rectangle(output, (30, 178), (30 + int(340 * scan_progress), 190),
                          COLOR_SCAN, -1)

        if feedback:
            cv2.putText(output, feedback, (30, 210), 1, 1.2, COLOR_ACCENT, 1)

        if hand.detected:
            if draw_mode_active and hand.is_index_only:
                # Bright drawing cursor
                cv2.circle(output, hand.cursor, 10, COLOR_DRAW, -1, cv2.LINE_AA)
                cv2.circle(output, hand.cursor, 12, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(output, hand.cursor, 12, COLOR_ACCENT, 2, cv2.LINE_AA)
                # Show the entry hold arc if counting down
                if not draw_mode_active:
                    self.draw_hold_arc(output, hand.cursor, hold_progress)

        return output

    def draw_keyboard_mode(self, frame, keyboard, hand, status: str, feedback: str) -> np.ndarray:
        output = frame.copy()
        self.glass_panel(output, keyboard.text_bar_rect, COLOR_PANEL)
        cv2.putText(output, keyboard.text or "Type...", (30, 55), 1, 1.8, COLOR_TEXT, 1)

        for key in keyboard.keys:
            hover = hand.detected and point_in_rect(hand.cursor, key["rect"])
            fill  = (COLOR_KEY_PINCH if hover and hand.is_click_pinching
                     else COLOR_KEY_HOVER if hover else COLOR_KEY)
            self.glass_panel(output, key["rect"], fill)
            cv2.putText(output, key["label"],
                        (key["rect"][0] + 10, key["rect"][1] + 35), 1, 1.2, COLOR_TEXT, 1)

        if hand.detected:
            cv2.circle(output, hand.cursor, 10, COLOR_ACCENT, -1)
        return output


def main():
    lexicon       = load_words()
    hand_analyzer = HandAnalyzer()
    drawing       = DrawingEngine()
    keyboard      = KeyboardController(lexicon)
    ui            = RemoteUI()
    ocr           = OCRWorker(lexicon)
    executor      = CommandExecutor()

    cap      = cv2.VideoCapture(0)
    app_mode = "DRAW"
    prev_click = False
    scan_start = 0.0

    # ── Draw-mode gesture state ───────────────────────────────────────────────
    draw_mode_active    = False
    index_hold_start    = 0.0   # when user started holding index-only (entry)
    exit_hold_start     = 0.0   # when user started holding open hand (exit)
    draw_cooldown_until = 0.0   # brief lock so the entry gesture doesn't draw

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        drawing.ensure_canvas(w, h)
        keyboard.build(w, h)

        hand = hand_analyzer.process(frame)
        now  = time.perf_counter()
        click = hand.detected and hand.is_click_pinching

        # ── Global mode toggle: pinch top-right corner ────────────────────────
        if click and not prev_click and hand.cursor[0] > w - 150 and hand.cursor[1] < 100:
            app_mode         = "KEYBOARD" if app_mode == "DRAW" else "DRAW"
            draw_mode_active = False
            index_hold_start = 0.0
            exit_hold_start  = 0.0
            drawing.lift()

        # ══════════════════════════════════════════════════════════════════════
        if app_mode == "DRAW":

            # ── ENTRY: hold index-only for INDEX_HOLD_SECONDS ─────────────────
            if not draw_mode_active:
                if hand.detected and hand.is_index_only:
                    if index_hold_start == 0.0:
                        index_hold_start = now
                    elif now - index_hold_start >= INDEX_HOLD_SECONDS:
                        draw_mode_active    = True
                        draw_cooldown_until = now + DRAW_COOLDOWN
                        index_hold_start    = 0.0
                        exit_hold_start     = 0.0
                        drawing.lift()
                else:
                    index_hold_start = 0.0

                hold_progress = (
                    (now - index_hold_start) / INDEX_HOLD_SECONDS
                    if index_hold_start > 0 else 0.0
                )
                exit_progress = 0.0

            else:   # draw_mode_active
                hold_progress = 0.0

                # ── EXIT: open hand held for EXIT_HOLD_SECONDS ────────────────
                if hand.detected and hand.is_open:
                    if exit_hold_start == 0.0:
                        exit_hold_start = now
                    elif now - exit_hold_start >= EXIT_HOLD_SECONDS:
                        draw_mode_active = False
                        exit_hold_start  = 0.0
                        drawing.lift()
                else:
                    exit_hold_start = 0.0

                exit_progress = (
                    (now - exit_hold_start) / EXIT_HOLD_SECONDS
                    if exit_hold_start > 0 else 0.0
                )

            # ── DRAWING ───────────────────────────────────────────────────────
            if hand.detected and draw_mode_active and now >= draw_cooldown_until:
                if hand.is_peace:
                    # ✌ peace sign → erase
                    drawing.erase_at(hand.cursor)
                elif hand.is_index_only:
                    # ☝ index only → draw stroke
                    drawing.add_point(hand.cursor, now)
                else:
                    # fist / other → pen up (stays in draw mode)
                    drawing.lift()

            elif not draw_mode_active:
                drawing.lift()

            # ── SCAN (OCR): fist held while in draw mode ──────────────────────
            if draw_mode_active and hand.detected and hand.is_fist:
                if scan_start == 0:
                    scan_start = time.time()
                if time.time() - scan_start > SCAN_HOLD_SECONDS:
                    ocr.submit(drawing.canvas)
                    scan_start = 0
            else:
                scan_start = 0

            scan_prog = (
                (time.time() - scan_start) / SCAN_HOLD_SECONDS
                if scan_start > 0 else 0.0
            )
            output = ui.draw_draw_mode(
                frame, drawing, hand,
                ocr.current_status(), "ACTIVE",
                scan_prog, executor.feedback,
                draw_mode_active, hold_progress, exit_progress,
            )

        else:
            # ── KEYBOARD mode ─────────────────────────────────────────────────
            if click and not prev_click:
                _, res = keyboard.handle_click(hand.cursor)
                if res:
                    executor.maybe_execute(res)
            output = ui.draw_keyboard_mode(frame, keyboard, hand, "ACTIVE", executor.feedback)

        res = ocr.consume_result()
        if res:
            executor.maybe_execute(res)

        cv2.imshow(WINDOW_NAME, output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        prev_click = click

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()