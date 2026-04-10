import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import random
from collections import deque

# ──────────────────────────────────────────
#  ONE-EURO FILTER
# ──────────────────────────────────────────
class OneEuroFilter:
    def __init__(self, min_cutoff=1.5, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if self.t_prev is None:
            self.t_prev, self.x_prev, self.dx_prev = t, x, 0
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        dx_hat = self._alpha(dt, self.d_cutoff) * dx + (1 - self._alpha(dt, self.d_cutoff)) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self._alpha(dt, cutoff) * x + (1 - self._alpha(dt, cutoff)) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


# ──────────────────────────────────────────
#  HAND ANALYSER — FIXED PINCH DETECTION
# ──────────────────────────────────────────
class HandAnalyzer:
    """
    Improved pinch detection with relaxed thresholds and faster tracking.
    """
    # Relaxed thresholds for easier triggering
    PINCH_CLOSE_THRESH = 0.12   
    PINCH_OPEN_THRESH  = 0.16
    DEBOUNCE_FRAMES    = 2

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,            # ← Faster tracking for higher FPS
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.fx = OneEuroFilter()
        self.fy = OneEuroFilter()
        self._pinching = False
        self._candidate = False
        self._candidate_count = 0

    @staticmethod
    def _dist2d(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _hand_scale(self, lm):
        """Wrist → middle MCP as normalisation reference."""
        return self._dist2d(lm[0], lm[9]) + 1e-6

    def process(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            self._candidate_count = 0
            self._pinching = False # Reset state when hand is lost
            return None

        lm = results.multi_hand_landmarks[0].landmark
        t  = time.perf_counter()

        # Smooth index-finger tip position
        sx = self.fx.filter(lm[8].x * w, t)
        sy = self.fy.filter(lm[8].y * h, t)

        # Normalised pinch distance
        pinch_dist = self._dist2d(lm[4], lm[8]) / self._hand_scale(lm)

        # Hysteresis state machine with debounce
        candidate = (pinch_dist < self.PINCH_CLOSE_THRESH) if not self._pinching \
                    else (pinch_dist < self.PINCH_OPEN_THRESH)

        if candidate == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = candidate
            self._candidate_count = 1

        if self._candidate_count >= self.DEBOUNCE_FRAMES:
            self._pinching = self._candidate

        return {
            "x": int(sx), "y": int(sy),
            "pinching": self._pinching,
            "pinch_dist": pinch_dist,
        }


# ──────────────────────────────────────────
#  PARTICLES
# ──────────────────────────────────────────
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size")

    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 5)
        self.x, self.y = x, y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 2
        self.life = self.max_life = random.randint(20, 45)
        self.color = color
        self.size = random.randint(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.18
        self.life -= 1

    def draw(self, surf):
        alpha = int(255 * self.life / self.max_life)
        r, g, b = self.color
        s = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (r, g, b, alpha), (self.size, self.size), self.size)
        surf.blit(s, (int(self.x) - self.size, int(self.y) - self.size))


# ──────────────────────────────────────────
#  BIRD
# ──────────────────────────────────────────
class Bird:
    W, H = 38, 30
    GRAVITY   = 0.42
    JUMP_VEL  = -8.5
    MAX_FALL  = 11
    ROTATION_SPEED = 4

    def __init__(self, win_h):
        self.x = 120
        self.y = float(win_h // 2)
        self.vel = 0.0
        self.angle = 0.0
        self.alive = True
        self.particles = []
        self._trail = deque(maxlen=10)

    @property
    def rect(self):
        return pygame.Rect(self.x - self.W // 2, int(self.y) - self.H // 2, self.W, self.H)

    def flap(self):
        self.vel = self.JUMP_VEL
        for _ in range(6):
            self.particles.append(Particle(self.x, self.y, (255, 230, 80)))

    def update(self, win_h):
        if not self.alive:
            return
        self.vel = min(self.vel + self.GRAVITY, self.MAX_FALL)
        self.y  += self.vel
        target_angle = max(-30, min(90, self.vel * 6))
        self.angle  += (target_angle - self.angle) * 0.18
        self._trail.append((self.x, int(self.y)))

        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.life > 0]

        if self.y > win_h + 40 or self.y < -40:
            self.alive = False

    def draw(self, surf):
        for i, (tx, ty) in enumerate(self._trail):
            alpha = int(120 * i / len(self._trail))
            r = max(2, 8 - i)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 255, 100, alpha), (r, r), r)
            surf.blit(s, (tx - r, ty - r))

        for p in self.particles:
            p.draw(surf)

        bsurf = pygame.Surface((self.W + 10, self.H + 10), pygame.SRCALPHA)
        cx, cy = (self.W + 10) // 2, (self.H + 10) // 2

        pygame.draw.ellipse(bsurf, (255, 210, 30), (2, 5, self.W, self.H))
        pygame.draw.ellipse(bsurf, (220, 160, 10), (2, 5, self.W, self.H), 2)
        wing_y = cy + int(self.vel * 0.5)
        pygame.draw.ellipse(bsurf, (255, 170, 0), (cx - 14, wing_y - 4, 20, 11))
        pygame.draw.circle(bsurf, (255, 255, 255), (cx + 10, cy - 3), 7)
        pygame.draw.circle(bsurf, (30, 30, 30),    (cx + 12, cy - 3), 4)
        pygame.draw.circle(bsurf, (255, 255, 255), (cx + 13, cy - 5), 2)
        pygame.draw.polygon(bsurf, (255, 120, 0), [(cx + self.W // 2 - 2, cy), (cx + self.W // 2 + 9, cy + 3), (cx + self.W // 2 - 2, cy + 7)])

        rotated = pygame.transform.rotate(bsurf, -self.angle)
        rr = rotated.get_rect(center=(int(self.x), int(self.y)))
        surf.blit(rotated, rr.topleft)


# ──────────────────────────────────────────
#  PIPE
# ──────────────────────────────────────────
class Pipe:
    WIDTH  = 68
    SPEED  = 3.8
    CAP_H  = 26

    def __init__(self, x, win_h, gap=170):
        self.x     = float(x)
        self.win_h = win_h
        self.gap   = gap
        self.top   = random.randint(80, win_h - gap - 80)
        self.passed = False
        self._col       = (34, 160, 34)
        self._col_dark  = (20, 100, 20)
        self._col_light = (60, 200, 60)

    def update(self):
        self.x -= self.SPEED

    @property
    def top_rect(self):
        return pygame.Rect(int(self.x), 0, self.WIDTH, self.top)

    @property
    def bot_rect(self):
        return pygame.Rect(int(self.x), self.top + self.gap, self.WIDTH, self.win_h)

    def draw(self, surf):
        ix = int(self.x)
        pygame.draw.rect(surf, self._col, self.top_rect)
        pygame.draw.rect(surf, self._col_dark, (ix, 0, 4, self.top))
        pygame.draw.rect(surf, self._col_light, (ix + self.WIDTH - 4, 0, 4, self.top))
        cap_y = self.top - self.CAP_H
        pygame.draw.rect(surf, self._col,      (ix - 5, cap_y, self.WIDTH + 10, self.CAP_H))
        pygame.draw.rect(surf, self._col_dark, (ix - 5, cap_y, 5, self.CAP_H))
        pygame.draw.rect(surf, self._col_light,(ix + self.WIDTH, cap_y, 5, self.CAP_H))

        bot_y = self.top + self.gap
        pygame.draw.rect(surf, self._col, self.bot_rect)
        pygame.draw.rect(surf, self._col_dark, (ix, bot_y, 4, self.win_h))
        pygame.draw.rect(surf, self._col_light, (ix + self.WIDTH - 4, bot_y, 4, self.win_h))
        pygame.draw.rect(surf, self._col,      (ix - 5, bot_y, self.WIDTH + 10, self.CAP_H))
        pygame.draw.rect(surf, self._col_dark, (ix - 5, bot_y, 5, self.CAP_H))
        pygame.draw.rect(surf, self._col_light,(ix + self.WIDTH, bot_y, 5, self.CAP_H))


# ──────────────────────────────────────────
#  BACKGROUND
# ──────────────────────────────────────────
class Background:
    def __init__(self, win_w, win_h):
        self.win_w, self.win_h = win_w, win_h
        self.clouds = [{"x": random.uniform(0, win_w), "y": random.uniform(40, win_h * 0.55), "r": random.randint(28, 65), "speed": random.uniform(0.3, 0.8)} for _ in range(10)]
        self.ground_h = 60
        self.stripe_x = 0

    def update(self):
        for c in self.clouds:
            c["x"] -= c["speed"]
            if c["x"] < -c["r"] * 2:
                c["x"] = self.win_w + c["r"]
                c["y"] = random.uniform(40, self.win_h * 0.55)
        self.stripe_x = (self.stripe_x - 4) % 40

    def draw(self, surf):
        for y in range(self.win_h - self.ground_h):
            t = y / (self.win_h - self.ground_h)
            r = int(112 + t * 60)
            g = int(197 + t * 30)
            b = int(235 - t * 30)
            pygame.draw.line(surf, (r, g, b), (0, y), (self.win_w, y))

        for c in self.clouds:
            cx, cy, cr = int(c["x"]), int(c["y"]), c["r"]
            for ox, oy, scale in [(0, 0, 1.0), (-cr * 0.6, cr * 0.3, 0.7), (cr * 0.6, cr * 0.2, 0.75), (0, cr * 0.4, 0.65)]:
                r2 = int(cr * scale)
                pygame.draw.circle(surf, (240, 248, 255), (cx + int(ox), cy + int(oy)), r2)

        gy = self.win_h - self.ground_h
        pygame.draw.rect(surf, (110, 190, 80), (0, gy, self.win_w, self.ground_h))
        pygame.draw.rect(surf, (80, 150, 50),  (0, gy, self.win_w, 8))
        for sx in range(int(self.stripe_x), self.win_w + 40, 40):
            pygame.draw.rect(surf, (90, 165, 65), (sx, gy + 14, 22, 8))


# ──────────────────────────────────────────
#  HUD
# ──────────────────────────────────────────
def draw_text_shadow(surf, text, font, color, x, y, shadow=(0, 0, 0)):
    surf.blit(font.render(text, True, shadow), (x + 2, y + 2))
    surf.blit(font.render(text, True, color),  (x, y))

def draw_pinch_indicator(surf, data, win_w, win_h):
    if data is None: return
    bar_w, bar_h = 120, 14
    bx, by = win_w - bar_w - 14, win_h - bar_h - 14
    pygame.draw.rect(surf, (0, 0, 0, 120), (bx - 2, by - 2, bar_w + 4, bar_h + 4), border_radius=6)
    ratio = min(1.0, data["pinch_dist"] / 0.15)
    fill_color = (int(ratio * 200), int((1 - ratio) * 80 + 180), 60)
    pygame.draw.rect(surf, fill_color, (bx, by, int(bar_w * (1 - ratio)), bar_h), border_radius=5)
    pygame.draw.rect(surf, (200, 200, 200), (bx, by, bar_w, bar_h), 1, border_radius=5)
    if data["pinching"]: pygame.draw.circle(surf, (255, 80, 80), (bx - 10, by + bar_h // 2), 5)


# ──────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────
def main():
    pygame.init()
    WIN_W, WIN_H = 900, 620
    GROUND_H = 60
    win   = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Flappy Hand")
    clock = pygame.time.Clock()

    font_big  = pygame.font.SysFont("Arial", 52, bold=True)
    font_med  = pygame.font.SysFont("Arial", 30, bold=True)
    font_sm   = pygame.font.SysFont("Arial", 20)

    cap      = cv2.VideoCapture(0)
    analyzer = HandAnalyzer()
    bg       = Background(WIN_W, WIN_H)

    state_menu, state_playing, state_dead = "menu", "playing", "dead"
    state = state_menu

    bird, pipes, score, best, death_particles, death_timer = Bird(WIN_H), [], 0, 0, [], 0
    prev_pinching, pinch_cooldown = False, 0
    cam_surf_w, cam_surf_h = 180, 135

    def reset_game():
        nonlocal bird, pipes, score, death_particles, death_timer
        bird  = Bird(WIN_H)
        pipes = [Pipe(WIN_W + 100, WIN_H - GROUND_H), Pipe(WIN_W + 430, WIN_H - GROUND_H)]
        score = 0; death_particles = []; death_timer = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if state == state_menu: reset_game(); state = state_playing
                    elif state == state_playing: bird.flap()
                    elif state == state_dead and death_timer > 40: reset_game(); state = state_playing
                if event.key == pygame.K_ESCAPE: running = False

        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        data  = analyzer.process(frame)

        just_pinched = False
        if data:
            if data["pinching"] and not prev_pinching and pinch_cooldown == 0:
                just_pinched, pinch_cooldown = True, 14
            prev_pinching = data["pinching"]
        if pinch_cooldown > 0: pinch_cooldown -= 1

        if state == state_menu:
            if just_pinched: reset_game(); state = state_playing
        elif state == state_playing:
            if just_pinched: bird.flap()
            bird.update(WIN_H - GROUND_H)
            for pipe in pipes: pipe.update()
            if pipes and pipes[-1].x < WIN_W - 280: pipes.append(Pipe(WIN_W + 40, WIN_H - GROUND_H, gap=max(130, 170 - score * 2)))
            pipes = [p for p in pipes if p.x > -Pipe.WIDTH - 20]
            for pipe in pipes:
                if not pipe.passed and pipe.x + Pipe.WIDTH < bird.x: pipe.passed = True; score += 1
            shrink = 5
            bird_hit = bird.rect.inflate(-shrink * 2, -shrink * 2)
            hit = any(bird_hit.colliderect(p.top_rect) or bird_hit.colliderect(p.bot_rect) for p in pipes)
            if not bird.alive: hit = True
            if hit:
                best, state, death_timer = max(best, score), state_dead, 0
                for _ in range(30): death_particles.append(Particle(bird.x, bird.y, random.choice([(255,210,30),(255,80,30),(255,255,120)])))
        elif state == state_dead:
            death_timer += 1
            for p in death_particles: p.update()
            death_particles = [p for p in death_particles if p.life > 0]
            if just_pinched and death_timer > 40: reset_game(); state = state_playing

        bg.update(); bg.draw(win)
        for pipe in pipes: pipe.draw(win)
        bird.draw(win)
        for p in death_particles: p.draw(win)
        draw_text_shadow(win, str(score), font_big, (255, 255, 255), WIN_W // 2 - 20, 12)
        draw_pinch_indicator(win, data, WIN_W, WIN_H)

        if ret:
            small = cv2.resize(frame, (cam_surf_w, cam_surf_h))
            cam_surf = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            win.blit(cam_surf, (WIN_W - cam_surf_w - 10, 10))
            pygame.draw.rect(win, (200, 200, 200), (WIN_W - cam_surf_w - 10, 10, cam_surf_w, cam_surf_h), 2)
            if data:
                fx = int(data["x"] * cam_surf_w / frame.shape[1]) + (WIN_W - cam_surf_w - 10)
                fy = int(data["y"] * cam_surf_h / frame.shape[0]) + 10
                pygame.draw.circle(win, (255, 60, 60) if data["pinching"] else (60, 255, 100), (fx, fy), 5)

        if state == state_menu:
            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA); overlay.fill((0, 0, 0, 110)); win.blit(overlay, (0, 0))
            draw_text_shadow(win, "FLAPPY HAND", font_big, (255, 230, 0), WIN_W//2 - 160, WIN_H//2 - 90)
            draw_text_shadow(win, "Pinch fingers to flap", font_med, (255,255,255), WIN_W//2 - 155, WIN_H//2 - 10)
            draw_text_shadow(win, "Pinch to start", font_sm, (200,255,200), WIN_W//2 - 80, WIN_H//2 + 40)
        elif state == state_dead and death_timer > 20:
            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA); overlay.fill((0, 0, 0, 130)); win.blit(overlay, (0, 0))
            draw_text_shadow(win, "GAME OVER", font_big, (255, 80, 60), WIN_W//2 - 140, WIN_H//2 - 90)
            draw_text_shadow(win, f"Score: {score}", font_med, (255,255,255), WIN_W//2 - 65, WIN_H//2 - 10)
            if death_timer > 40: draw_text_shadow(win, "Pinch to retry", font_sm, (180,255,180), WIN_W//2 - 80, WIN_H//2 + 80)

        pygame.display.flip(); clock.tick(60)
        cv2.imshow("Hand Tracking", cv2.resize(frame, (320, 240)))
        if cv2.waitKey(1) & 0xFF == ord("q"): running = False

    cap.release(); cv2.destroyAllWindows(); pygame.quit()

if __name__ == "__main__": main()