"""
╔══════════════════════════════════════════════════════════════╗
║          H O L O S C R I B E  v3  —  AI Gesture Writer      ║
╠══════════════════════════════════════════════════════════════╣
║  STANDBY MODE                                                ║
║    ☝  Index up  3 s   →  enter DRAW MODE                    ║
║    ✌  Index+Mid       →  erase strokes                      ║
║    ✊  Fist      3 s   →  AI scan                            ║
║    🖐  Open hand 5 s   →  clear canvas                       ║
║                                                              ║
║  DRAW MODE  (index up = draw, any other pose = pause)        ║
║    ☝  Index up        →  draw                               ║
║    ✊  Fist      3 s   →  AI scan                            ║
║    🖐  Open hand 5 s   →  clear canvas                       ║
║    ✋  Fingers down 3s →  exit DRAW MODE                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import torch
import threading
import mediapipe as mp
import difflib
import os
import time
import math
from collections import deque
from PIL import Image, ImageFilter, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ═══════════════════════════════════════════════════════════════
# 1. CONSTANTS & PALETTE
# ═══════════════════════════════════════════════════════════════
WIN_W, WIN_H   = 1280, 720
PANEL_L        = 215
PANEL_R        = 225
CANVAS_X       = PANEL_L
CANVAS_W       = WIN_W - PANEL_L - PANEL_R
CANVAS_H       = WIN_H - 62
CANVAS_Y       = 42

SMOOTH_ALPHA   = 0.42          # fingertip EMA smoothing
INK_THICK      = 14            # base ink line thickness
ERASE_R        = 34

DRAW_ENTER_T   = 3.0           # seconds to enter draw mode
DRAW_EXIT_T    = 3.0           # seconds to exit draw mode
FIST_HOLD_T    = 3.0
OPEN_HOLD_T    = 5.0

TROCR_W        = 640           # letterboxed input width for TrOCR
TROCR_H        = 128
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ── BGR palette ─────────────────────────────────────────────
C_PANEL   = (  8,  13,  20)
C_BORDER  = ( 28,  65,  55)
C_GRID    = ( 11,  25,  20)
CYAN      = (  0, 228, 178)
CYAN_DIM  = (  0,  85,  60)

COL_IDLE  = ( 75,  95,  85)
COL_DRAW  = (  0, 235, 175)    # cyan-green
COL_ERASE = ( 25, 115, 255)    # amber-orange
COL_SCAN  = ( 15, 195, 255)    # gold
COL_CLEAR = (  0,  90, 255)    # red-orange
COL_AI    = (160, 255,  90)    # lime

# Ink layers: outer glow → mid → bright core (BGR)
INK_L1 = (  0,  55,  42)       # deep glow
INK_L2 = (  0, 175, 135)       # mid cyan
INK_L3 = (210, 255, 248)       # hot white core

# Hand skeleton
H_DIM   = (  0,  38,  26)
H_BONE  = (  0, 145, 100)
H_TIP   = (  0, 228, 175)
H_WHITE = (195, 252, 242)

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
TIPS  = {4, 8, 12, 16, 20}
PALMS = {0, 5, 9, 13, 17}

# ═══════════════════════════════════════════════════════════════
# 2. AI MODEL
# ═══════════════════════════════════════════════════════════════
print("[HoloScribe] Loading TrOCR …")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr     = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-handwritten").to(DEVICE)
trocr.eval()
print("[HoloScribe] Ready on", DEVICE)

def load_vocab():
    if os.path.exists("words.txt"):
        with open("words.txt") as f:
            return [l.strip().upper() for l in f if l.strip()]
    return ["HELLO","WORLD","PLAY","PAUSE","STOP","VOLUME","NEXT","BACK",
            "HOME","SEARCH","YES","NO","UP","DOWN","LEFT","RIGHT","OK",
            "ON","OFF","OPEN","CLOSE","START","END","GO","WAIT"]
VOCAB = load_vocab()

# ═══════════════════════════════════════════════════════════════
# 3. MEDIAPIPE
# ═══════════════════════════════════════════════════════════════
mp_hands = mp.solutions.hands
tracker  = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# ═══════════════════════════════════════════════════════════════
# 4. RAW GESTURE CLASSIFIER
#    Returns: "IDX" | "TWO" | "FIST" | "OPEN" | "NONE"
# ═══════════════════════════════════════════════════════════════
def classify_raw(lm):
    """Classify the current hand pose into a raw gesture token."""
    def up(tip, pip): return lm[tip].y < lm[pip].y

    idx   = up(8,  6)
    mid   = up(12, 10)
    ring  = up(16, 14)
    pinky = up(20, 18)

    if idx and not mid and not ring and not pinky:
        return "IDX"
    if idx and mid and not ring and not pinky:
        return "TWO"
    if not idx and not mid and not ring and not pinky:
        return "FIST"
    if idx and mid and ring and pinky:
        return "OPEN"
    return "NONE"

# ═══════════════════════════════════════════════════════════════
# 5. SMOOTH INK RENDERER  (line-only, no circles)
# ═══════════════════════════════════════════════════════════════
def _interp_pts(p1, p2, step=4):
    """Sub-sample between two points so fast movement never leaves gaps."""
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    dist   = math.hypot(dx, dy)
    if dist < 1: return [p2]
    n = max(1, int(dist / step))
    return [(int(p1[0] + dx*i/n), int(p1[1] + dy*i/n)) for i in range(1, n+1)]

def ink_line(canvas, p1, p2):
    """Draw a smooth, multi-layer glow stroke from p1 to p2 — no circles."""
    pts = _interp_pts(p1, p2, step=3)
    all_pts = [p1] + pts
    for i in range(1, len(all_pts)):
        a, b = all_pts[i-1], all_pts[i]
        cv2.line(canvas, a, b, INK_L1, INK_THICK + 14, cv2.LINE_AA)
        cv2.line(canvas, a, b, INK_L2, INK_THICK + 4,  cv2.LINE_AA)
        cv2.line(canvas, a, b, INK_L3, max(3, INK_THICK - 6), cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════
# 6. TECH HAND RENDERER
# ═══════════════════════════════════════════════════════════════
_phase = 0.0

def draw_tech_hand(frame, pts, raw_gest, app_state, timer_frac=0.0):
    global _phase
    _phase = (_phase + 0.07) % (2 * math.pi)
    pulse  = 0.5 + 0.5 * math.sin(_phase)

    # Accent colour per state
    if app_state == "DRAW_MODE":
        acc = COL_DRAW
    else:
        acc = {"IDX": COL_DRAW, "TWO": COL_ERASE,
               "FIST": COL_SCAN, "OPEN": COL_CLEAR}.get(raw_gest, COL_IDLE)

    # ── Bones ──────────────────────────────────
    for a, b in HAND_CONN:
        col = acc if b in TIPS else H_BONE
        cv2.line(frame, pts[a], pts[b], H_DIM, 8, cv2.LINE_AA)
        cv2.line(frame, pts[a], pts[b], col,   2, cv2.LINE_AA)

    # ── Joints ─────────────────────────────────
    for i, p in enumerate(pts):
        tip  = i in TIPS
        palm = i in PALMS
        r    = 9 if tip else (7 if palm else 4)
        cv2.circle(frame, p, r+6, H_DIM,   -1, cv2.LINE_AA)
        cv2.circle(frame, p, r,   acc if tip else H_TIP, -1, cv2.LINE_AA)
        cv2.circle(frame, p, max(2, r-4), H_WHITE, -1, cv2.LINE_AA)
        if tip:
            cv2.circle(frame, p, int(r + 9 + pulse*11), acc, 1, cv2.LINE_AA)
            if timer_frac > 0:
                cv2.circle(frame, p, int(r + 22 + pulse*4), COL_SCAN, 1, cv2.LINE_AA)

    # ── Index crosshair ────────────────────────
    ix, iy = pts[8]
    cv2.circle(frame, (ix, iy), 20, acc, 1, cv2.LINE_AA)
    cv2.circle(frame, (ix, iy),  3, acc, -1, cv2.LINE_AA)
    for ax, ay, bx, by in [(-28,0,-13,0),(13,0,28,0),(0,-28,0,-13),(0,13,0,28)]:
        cv2.line(frame, (ix+ax, iy+ay), (ix+bx, iy+by), acc, 1, cv2.LINE_AA)

    # ── Erase ghost ────────────────────────────
    if raw_gest == "TWO" and app_state == "STANDBY":
        mx = (pts[8][0] + pts[12][0]) // 2
        my = (pts[8][1] + pts[12][1]) // 2
        cv2.circle(frame, (mx, my), ERASE_R+8, COL_ERASE, 1, cv2.LINE_AA)
        cv2.circle(frame, (mx, my), ERASE_R,   COL_ERASE, 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════
# 7. AI RECOGNITION  (improved preprocessing)
# ═══════════════════════════════════════════════════════════════
ai_running = False

def _letterbox(img_bgr, target_w, target_h):
    """Resize preserving aspect ratio, pad with white to target size."""
    h, w = img_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas  = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    px = (target_w - nw) // 2
    py = (target_h - nh) // 2
    canvas[py:py+nh, px:px+nw] = resized
    return canvas

def preprocess_for_trocr(canvas_snap):
    """
    Convert a cyan-on-black canvas to a clean dark-ink-on-white image
    suitable for TrOCR, with CLAHE contrast boost and slight stroke thickening.
    """
    # 1. Extract the green channel (cyan ink is brightest here)
    g = canvas_snap[:, :, 1].astype(np.uint8)

    # 2. CLAHE for local contrast normalisation
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g     = clahe.apply(g)

    # 3. Threshold to binary
    _, bw = cv2.threshold(g, 25, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Dilate slightly to thicken thin strokes (helps TrOCR)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw     = cv2.dilate(bw, kernel, iterations=1)

    # 5. Invert → dark ink on white background (TrOCR expects this)
    ink_on_white = cv2.bitwise_not(bw)

    # 6. Convert to BGR for letterbox
    bgr = cv2.cvtColor(ink_on_white, cv2.COLOR_GRAY2BGR)

    # 7. Tight crop around ink before letterboxing
    rows = np.any(ink_on_white < 200, axis=1)
    cols = np.any(ink_on_white < 200, axis=0)
    if not (rows.any() and cols.any()):
        return None, None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    pad = 22
    y1 = max(0, y1-pad); y2 = min(bgr.shape[0]-1, y2+pad)
    x1 = max(0, x1-pad); x2 = min(bgr.shape[1]-1, x2+pad)
    crop = bgr[y1:y2+1, x1:x2+1]
    if crop.size == 0:
        return None, None

    # 8. Letterbox to TROCR_W × TROCR_H
    final = _letterbox(crop, TROCR_W, TROCR_H)
    return final, crop   # also return crop for debug if needed

def run_trocr(canvas_snap, ai_history):
    global ai_running
    try:
        img, _ = preprocess_for_trocr(canvas_snap)
        if img is None:
            ai_history.append((time.strftime("%H:%M:%S"), "[blank]", ""))
            ai_running = False; return

        pil    = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pixels = processor(images=pil, return_tensors="pt").pixel_values.to(DEVICE)

        with torch.no_grad():
            ids = trocr.generate(
                pixels,
                max_new_tokens=32,
                num_beams=5,
                early_stopping=True,
            )
        raw = processor.batch_decode(ids, skip_special_tokens=True)[0].strip().upper()
        raw = ''.join(c for c in raw if c.isalpha() or c == ' ').strip()

        match = difflib.get_close_matches(raw, VOCAB, n=1, cutoff=0.30)
        best  = match[0] if match else (raw or "[?]")

        ai_history.append((time.strftime("%H:%M:%S"), best, raw))
        print(f"[TrOCR] raw='{raw}'  matched='{best}'")
    except Exception as e:
        ai_history.append((time.strftime("%H:%M:%S"), "ERR", str(e)))
        print(f"[TrOCR error] {e}")
    ai_running = False

# ═══════════════════════════════════════════════════════════════
# 8. HUD PANELS
# ═══════════════════════════════════════════════════════════════

# Left panel gesture guide entries  (key, line1, line2, colour)
GUIDE_STANDBY = [
    ("IDX",  "Index up  3s",  "→ enter draw mode", COL_DRAW),
    ("TWO",  "2 fingers up",  "→ erase",           COL_ERASE),
    ("FIST", "Fist  3s",      "→ AI scan",          COL_SCAN),
    ("OPEN", "Open hand  5s", "→ clear canvas",     COL_CLEAR),
]
GUIDE_DRAW = [
    ("IDX",  "Index up",      "→ drawing",          COL_DRAW),
    ("NONE", "Fingers down",  "pause (no draw)",    COL_IDLE),
    ("FIST", "Fist  3s",      "→ AI scan",           COL_SCAN),
    ("OPEN", "Open hand  5s", "→ clear canvas",      COL_CLEAR),
    ("EXIT", "Fingers down 3s","→ exit draw mode",   COL_ERASE),
]

def _card_active(gkey, raw_gest, app_state):
    if app_state == "STANDBY":
        return raw_gest == gkey
    else:  # DRAW_MODE
        if gkey == "IDX":  return raw_gest == "IDX"
        if gkey == "NONE": return raw_gest != "IDX" and raw_gest not in ("FIST","OPEN")
        if gkey == "EXIT": return raw_gest != "IDX" and raw_gest not in ("FIST","OPEN")
        return raw_gest == gkey

def draw_left_panel(frame, raw_gest, app_state,
                    draw_enter_frac, draw_exit_frac,
                    fist_frac, open_frac):
    cv2.rectangle(frame, (0, 0), (PANEL_L, WIN_H), C_PANEL, -1)
    cv2.line(frame,  (PANEL_L, 0), (PANEL_L, WIN_H), C_BORDER, 1)

    # Header
    mode_lbl = "DRAW MODE" if app_state == "DRAW_MODE" else "STANDBY"
    mode_col = COL_DRAW    if app_state == "DRAW_MODE" else CYAN_DIM
    cv2.putText(frame, mode_lbl, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, mode_col, 1, cv2.LINE_AA)
    cv2.putText(frame, "GESTURES", (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN_DIM, 1, cv2.LINE_AA)
    cv2.line(frame, (8, 44), (PANEL_L-8, 44), C_BORDER, 1)

    guide  = GUIDE_DRAW if app_state == "DRAW_MODE" else GUIDE_STANDBY
    card_h = 116
    gap    = 5

    for i, (gkey, l1, l2, col) in enumerate(guide):
        cy     = 50 + i * (card_h + gap)
        if cy + card_h > WIN_H - 4: break
        active = _card_active(gkey, raw_gest, app_state)

        fill = (16, 32, 24) if active else (9, 16, 12)
        cv2.rectangle(frame, (7, cy), (PANEL_L-7, cy+card_h), fill, -1)
        bdr = col if active else C_BORDER
        cv2.rectangle(frame, (7, cy), (PANEL_L-7, cy+card_h), bdr,
                      2 if active else 1)
        if active:
            cv2.rectangle(frame, (7, cy), (12, cy+card_h), col, -1)

        cv2.putText(frame, l1, (18, cy+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    col if active else COL_IDLE, 1, cv2.LINE_AA)
        cv2.putText(frame, l2, (18, cy+48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_BORDER, 1, cv2.LINE_AA)

        # Timer arcs
        arc_cx, arc_cy, arc_r = PANEL_L//2, cy+86, 18
        cv2.circle(frame, (arc_cx, arc_cy), arc_r, (18,36,26), 2, cv2.LINE_AA)

        frac = 0.0
        if gkey == "IDX"  and app_state == "STANDBY" and active: frac = draw_enter_frac
        if gkey == "EXIT" and app_state == "DRAW_MODE":           frac = draw_exit_frac
        if gkey == "FIST" and active:                             frac = fist_frac
        if gkey == "OPEN" and active:                             frac = open_frac

        if frac > 0:
            cv2.ellipse(frame, (arc_cx, arc_cy), (arc_r, arc_r),
                        -90, 0, int(frac*360), col, 3, cv2.LINE_AA)
            t_max = (DRAW_ENTER_T if gkey in ("IDX","EXIT") else
                     FIST_HOLD_T  if gkey == "FIST" else OPEN_HOLD_T)
            lbl = f"{frac*t_max:.1f}s"
            tw  = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0][0]
            cv2.putText(frame, lbl, (arc_cx - tw//2, arc_cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)
        elif active and gkey in ("IDX", "TWO", "NONE"):
            pr = 5 + int(3*(0.5 + 0.5*math.sin(time.time()*6)))
            cv2.circle(frame, (arc_cx, arc_cy), pr, col, -1, cv2.LINE_AA)
            cv2.putText(frame, "ACTIVE", (arc_cx-22, arc_cy+arc_r+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1, cv2.LINE_AA)

def draw_right_panel(frame, ai_history):
    rx = WIN_W - PANEL_R
    cv2.rectangle(frame, (rx, 0), (WIN_W, WIN_H), C_PANEL, -1)
    cv2.line(frame, (rx, 0), (rx, WIN_H), C_BORDER, 1)

    cv2.putText(frame, "AI LOG", (rx+14, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, CYAN_DIM, 1, cv2.LINE_AA)
    cv2.line(frame, (rx+8, 30), (WIN_W-8, 30), C_BORDER, 1)

    entry_h = 72
    visible = list(reversed(ai_history[-8:]))
    for j, (ts, best, raw) in enumerate(visible):
        ey = 36 + j*(entry_h+4)
        if ey + entry_h > WIN_H - 8: break
        newest = (j == 0)
        fill   = (13, 26, 17) if newest else (8, 14, 10)
        bdr    = COL_AI if newest else C_BORDER
        col    = COL_AI if newest else COL_IDLE
        cv2.rectangle(frame, (rx+5, ey), (WIN_W-5, ey+entry_h), fill, -1)
        cv2.rectangle(frame, (rx+5, ey), (WIN_W-5, ey+entry_h), bdr, 1)
        # Best match (large)
        cv2.putText(frame, best, (rx+12, ey+32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, col, 2, cv2.LINE_AA)
        # Raw TrOCR output (small, dimmer)
        raw_disp = f"raw: {raw[:14]}" if raw and raw != best else ""
        cv2.putText(frame, raw_disp, (rx+12, ey+52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, C_BORDER, 1, cv2.LINE_AA)
        cv2.putText(frame, ts, (rx+12, ey+66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_BORDER, 1, cv2.LINE_AA)

def draw_top_bar(frame, app_state, ai_running, frame_count):
    cv2.rectangle(frame, (CANVAS_X, 0), (WIN_W-PANEL_R, CANVAS_Y), (5,10,8), -1)
    cv2.line(frame, (CANVAS_X, CANVAS_Y), (WIN_W-PANEL_R, CANVAS_Y), C_BORDER, 1)

    cv2.putText(frame, "H O L O S C R I B E",
                (CANVAS_X+14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, CYAN, 1, cv2.LINE_AA)

    if ai_running:
        ang = int((frame_count*9)%360)
        sc  = WIN_W - PANEL_R - 68
        cv2.ellipse(frame, (sc, 21), (10,10), 0, ang, ang+270, COL_SCAN, 2, cv2.LINE_AA)
        cv2.putText(frame, "SCANNING",
                    (WIN_W-PANEL_R-158, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_SCAN, 1, cv2.LINE_AA)

    ts = time.strftime("%H:%M:%S")
    tw = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
    cv2.putText(frame, ts, (WIN_W-PANEL_R-tw-10, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_BORDER, 1, cv2.LINE_AA)

def draw_canvas_overlay(frame, raw_gest, app_state):
    # Grid
    for gx in range(CANVAS_X, CANVAS_X+CANVAS_W, 50):
        cv2.line(frame, (gx, CANVAS_Y), (gx, CANVAS_Y+CANVAS_H), C_GRID, 1)
    for gy in range(CANVAS_Y, CANVAS_Y+CANVAS_H, 50):
        cv2.line(frame, (CANVAS_X, gy), (CANVAS_X+CANVAS_W, gy), C_GRID, 1)

    # Corner brackets
    bsz = 24
    for (cx, cy), (dx, dy) in zip(
        [(CANVAS_X,CANVAS_Y),(CANVAS_X+CANVAS_W-bsz,CANVAS_Y),
         (CANVAS_X,CANVAS_Y+CANVAS_H-bsz),(CANVAS_X+CANVAS_W-bsz,CANVAS_Y+CANVAS_H-bsz)],
        [(1,1),(-1,1),(1,-1),(-1,-1)]
    ):
        cv2.line(frame, (cx,cy), (cx+dx*bsz,cy),        CYAN, 2)
        cv2.line(frame, (cx,cy), (cx,cy+dy*bsz),        CYAN, 2)

    # State badge
    if app_state == "DRAW_MODE":
        lbl = "// DRAW MODE"
        col = COL_DRAW
    else:
        lbl_map = {"IDX":"// ENTERING DRAW...","TWO":"// ERASE",
                   "FIST":"// SCANNING...","OPEN":"// CLEARING...","NONE":"// STANDBY"}
        col_map = {"IDX":COL_DRAW,"TWO":COL_ERASE,
                   "FIST":COL_SCAN,"OPEN":COL_CLEAR,"NONE":COL_IDLE}
        lbl = lbl_map.get(raw_gest, "// STANDBY")
        col = col_map.get(raw_gest, COL_IDLE)

    tw = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
    cv2.putText(frame, lbl, (CANVAS_X+CANVAS_W-tw-12, CANVAS_Y+24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

def draw_bottom_bar(frame, fist_frac, open_frac, draw_exit_frac,
                    draw_enter_frac, raw_gest, app_state):
    by = WIN_H - 22
    cv2.rectangle(frame, (CANVAS_X, by), (WIN_W-PANEL_R, WIN_H), (5,10,8), -1)
    cv2.line(frame, (CANVAS_X, by), (WIN_W-PANEL_R, by), C_BORDER, 1)

    cv2.putText(frame, "Q = quit",
                (CANVAS_X+10, WIN_H-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_BORDER, 1, cv2.LINE_AA)

    # Progress bar for active timed gesture
    frac, col, lbl = 0.0, CYAN, ""
    if app_state == "STANDBY" and raw_gest == "IDX" and draw_enter_frac > 0:
        frac, col, lbl = draw_enter_frac, COL_DRAW,  "ENTERING DRAW"
    elif app_state == "DRAW_MODE" and draw_exit_frac > 0:
        frac, col, lbl = draw_exit_frac,  COL_ERASE, "EXITING DRAW"
    elif raw_gest == "FIST" and fist_frac > 0:
        frac, col, lbl = fist_frac,       COL_SCAN,  "SCAN"
    elif raw_gest == "OPEN" and open_frac > 0:
        frac, col, lbl = open_frac,       COL_CLEAR, "CLEAR"

    if frac > 0:
        bx = CANVAS_X + 130
        bw = CANVAS_W - 260
        cv2.rectangle(frame, (bx, by+5), (bx+bw, by+13), (18,36,26), -1)
        cv2.rectangle(frame, (bx, by+5), (bx+int(bw*frac), by+13), col, -1)
        cv2.putText(frame, lbl, (CANVAS_X+14, WIN_H-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════
# 9. PARTICLE SYSTEM
# ═══════════════════════════════════════════════════════════════
particles = []

def spawn_particles(n=90):
    for _ in range(n):
        cx    = CANVAS_X + int(np.random.randint(0, CANVAS_W))
        cy    = CANVAS_Y + int(np.random.randint(0, CANVAS_H))
        angle = np.random.uniform(0, 2*math.pi)
        speed = np.random.uniform(2, 10)
        life  = np.random.uniform(0.5, 1.3)
        particles.append([cx, cy,
                          math.cos(angle)*speed,
                          math.sin(angle)*speed,
                          life, life])

def update_particles(frame, dt):
    dead = []
    for i, p in enumerate(particles):
        p[0] += p[2]; p[1] += p[3]
        p[4] -= dt * 1.5
        if p[4] <= 0: dead.append(i); continue
        a   = p[4] / p[5]
        r   = max(1, int(5*a))
        col = tuple(int(c*a) for c in COL_CLEAR)
        cv2.circle(frame, (int(p[0]), int(p[1])), r, col, -1, cv2.LINE_AA)
    for i in reversed(dead): particles.pop(i)

# ═══════════════════════════════════════════════════════════════
# 10. MAIN LOOP
# ═══════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("HoloScribe", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HoloScribe", WIN_W, WIN_H)

canvas      = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
ai_history  = []

lsx, lsy    = WIN_W//2, WIN_H//2
prev_pt     = None
frame_count = 0
last_time   = time.time()

# App state
app_state       = "STANDBY"    # "STANDBY" | "DRAW_MODE"
raw_gest        = "NONE"

# Timers
draw_enter_start = 0.0
draw_exit_start  = 0.0
fist_start       = 0.0
open_start       = 0.0

draw_enter_frac  = 0.0
draw_exit_frac   = 0.0
fist_frac        = 0.0
open_frac        = 0.0

while cap.isOpened():
    ok, frame_raw = cap.read()
    if not ok: break
    now  = time.time()
    dt   = max(0.001, now - last_time)
    last_time   = now
    frame_count += 1

    frame_raw = cv2.flip(frame_raw, 1)
    cam_h, cam_w = frame_raw.shape[:2]

    output   = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    hand_lm  = None
    pts_win  = None
    sx, sy   = lsx, lsy
    raw_gest = "NONE"

    # ── Hand tracking ────────────────────────
    rgb    = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
    result = tracker.process(rgb)

    if result.multi_hand_landmarks:
        hand_lm = result.multi_hand_landmarks[0].landmark

        rx = int(hand_lm[8].x * WIN_W)
        ry = int(hand_lm[8].y * WIN_H)
        sx = int(lsx*(1-SMOOTH_ALPHA) + rx*SMOOTH_ALPHA)
        sy = int(lsy*(1-SMOOTH_ALPHA) + ry*SMOOTH_ALPHA)
        lsx, lsy = sx, sy

        pts_win  = [(int(l.x*WIN_W), int(l.y*WIN_H)) for l in hand_lm]
        raw_gest = classify_raw(hand_lm)

    # ════════════════════════════════════════
    # STATE MACHINE
    # ════════════════════════════════════════

    # ── Draw mode ENTRY timer (STANDBY only) ─
    if app_state == "STANDBY" and raw_gest == "IDX":
        if draw_enter_start == 0.0: draw_enter_start = now
        draw_enter_frac = min(1.0, (now - draw_enter_start) / DRAW_ENTER_T)
        if draw_enter_frac >= 1.0:
            app_state        = "DRAW_MODE"
            draw_enter_start = 0.0
            draw_enter_frac  = 0.0
            prev_pt          = None
    else:
        draw_enter_start = 0.0
        draw_enter_frac  = 0.0

    # ── Draw mode EXIT timer (DRAW_MODE, index NOT up) ─
    if app_state == "DRAW_MODE" and raw_gest != "IDX":
        if draw_exit_start == 0.0: draw_exit_start = now
        draw_exit_frac = min(1.0, (now - draw_exit_start) / DRAW_EXIT_T)
        if draw_exit_frac >= 1.0:
            app_state       = "STANDBY"
            draw_exit_start = 0.0
            draw_exit_frac  = 0.0
            prev_pt         = None
    else:
        draw_exit_start = 0.0
        draw_exit_frac  = 0.0

    # ── Fist → AI scan (both states) ─────────
    if raw_gest == "FIST":
        if fist_start == 0.0: fist_start = now
        fist_frac = min(1.0, (now - fist_start) / FIST_HOLD_T)
        if fist_frac >= 1.0 and not ai_running:
            ai_running = True
            fist_start = 0.0; fist_frac = 0.0
            threading.Thread(target=run_trocr,
                             args=(canvas.copy(), ai_history),
                             daemon=True).start()
    else:
        fist_start = 0.0; fist_frac = 0.0

    # ── Open → clear (both states) ───────────
    if raw_gest == "OPEN":
        if open_start == 0.0: open_start = now
        open_frac = min(1.0, (now - open_start) / OPEN_HOLD_T)
        if open_frac >= 1.0:
            spawn_particles(90)
            canvas.fill(0)
            open_start = 0.0; open_frac = 0.0
    else:
        open_start = 0.0; open_frac = 0.0

    # ════════════════════════════════════════
    # CANVAS ACTIONS
    # ════════════════════════════════════════
    if pts_win is not None:

        # ── DRAWING (DRAW_MODE + index up) ───
        if app_state == "DRAW_MODE" and raw_gest == "IDX":
            if (CANVAS_X <= sx <= CANVAS_X+CANVAS_W and
                    CANVAS_Y <= sy <= CANVAS_Y+CANVAS_H):
                if prev_pt is not None:
                    ink_line(canvas, prev_pt, (sx, sy))
                # No dot/circle at stroke start — wait for 2nd point
                prev_pt = (sx, sy)
            else:
                prev_pt = None
        else:
            prev_pt = None

        # ── ERASE (STANDBY only, two fingers up) ─
        if app_state == "STANDBY" and raw_gest == "TWO":
            # Erase midpoint between index and middle tips
            mx = (pts_win[8][0] + pts_win[12][0]) // 2
            my = (pts_win[8][1] + pts_win[12][1]) // 2
            cv2.circle(canvas, (mx, my), ERASE_R, (0,0,0), -1)

    else:
        prev_pt = None

    # ════════════════════════════════════════
    # RENDER
    # ════════════════════════════════════════

    # 1. Ink layer
    output = cv2.add(output, canvas)

    # 2. Panels
    draw_left_panel(output, raw_gest, app_state,
                    draw_enter_frac, draw_exit_frac,
                    fist_frac, open_frac)
    draw_right_panel(output, ai_history)

    # 3. Canvas HUD
    draw_canvas_overlay(output, raw_gest, app_state)

    # 4. Bars
    draw_top_bar(output, app_state, ai_running, frame_count)
    draw_bottom_bar(output, fist_frac, open_frac, draw_exit_frac,
                    draw_enter_frac, raw_gest, app_state)

    # 5. Particles
    if particles:
        update_particles(output, dt)

    # 6. Tech hand
    if pts_win is not None:
        draw_tech_hand(output, pts_win, raw_gest, app_state,
                       max(fist_frac, open_frac, draw_enter_frac, draw_exit_frac))

    cv2.imshow("HoloScribe", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()