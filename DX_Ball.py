import pygame
import random
import math
import cv2
import mediapipe as mp
import numpy as np
import sys

# ---------- CONFIG ----------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

PADDLE_WIDTH = 120
PADDLE_HEIGHT = 16
PADDLE_Y_OFFSET = 40

BALL_RADIUS = 8
BALL_SPEED = 5

BRICK_ROWS = 6
BRICK_COLS = 10
BRICK_GAP = 6
BRICK_TOP = 80
BRICK_HEIGHT = 22
# ----------------------------

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("DX-Ball - Paddle Control Modes")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 22)

# ---------- GAME OBJECTS ----------
class Paddle:
    def __init__(self):
        self.x = (SCREEN_WIDTH - PADDLE_WIDTH) / 2
        self.y = SCREEN_HEIGHT - PADDLE_Y_OFFSET
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), int(self.width), int(self.height))

    def draw(self, surf):
        pygame.draw.rect(surf, (200, 200, 255), self.rect(), border_radius=6)


class Ball:
    def __init__(self, paddle):
        self.x = paddle.x + paddle.width / 2
        self.y = paddle.y - BALL_RADIUS - 2
        self.vx = 0
        self.vy = -BALL_SPEED
        self.radius = BALL_RADIUS
        self.stuck = True

    def rect(self):
        return pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            self.radius * 2,
            self.radius * 2,
        )

    def draw(self, surf):
        pygame.draw.circle(surf, (255, 220, 80), (int(self.x), int(self.y)), self.radius)


class Brick:
    def __init__(self, rect, color, hits=1):
        self.rect = rect
        self.color = color
        self.hits = hits

    def draw(self, surf):
        pygame.draw.rect(surf, self.color, self.rect, border_radius=4)


# ---------- HELPER FUNCTIONS ----------
def make_bricks():
    bricks = []
    total_gap = (BRICK_COLS + 1) * BRICK_GAP
    brick_width = (SCREEN_WIDTH - total_gap) / BRICK_COLS
    colors = [
        (255, 120, 120),
        (255, 180, 100),
        (255, 220, 120),
        (180, 240, 160),
        (160, 200, 255),
        (200, 160, 255),
    ]
    for row in range(BRICK_ROWS):
        for col in range(BRICK_COLS):
            x = BRICK_GAP + col * (brick_width + BRICK_GAP)
            y = BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
            rect = pygame.Rect(int(x), int(y), int(brick_width), BRICK_HEIGHT)
            bricks.append(Brick(rect, colors[row % len(colors)], hits=1))
    return bricks


def serve_ball(ball, paddle):
    if ball.stuck:
        ball.stuck = False
        ball.vx = random.choice([-BALL_SPEED, BALL_SPEED])
        ball.vy = -BALL_SPEED


def clamp(v, a, b):
    return max(a, min(b, v))


def reflect_wall(ball):
    if ball.x - ball.radius <= 0:
        ball.x = ball.radius
        ball.vx = -ball.vx
    if ball.x + ball.radius >= SCREEN_WIDTH:
        ball.x = SCREEN_WIDTH - ball.radius
        ball.vx = -ball.vx
    if ball.y - ball.radius <= 0:
        ball.y = ball.radius
        ball.vy = -ball.vy


def paddle_ball_collision(ball, paddle):
    if ball.rect().colliderect(paddle.rect()):
        ball.y = paddle.y - ball.radius
        rel = (ball.x - (paddle.x + paddle.width / 2)) / (paddle.width / 2)
        angle = rel * (math.pi / 3)
        speed = math.hypot(ball.vx, ball.vy)
        ball.vx = speed * math.sin(angle)
        ball.vy = -abs(speed * math.cos(angle))
        return True
    return False


def handle_brick_collisions(ball, bricks):
    hit = None
    for brick in bricks:
        if ball.rect().colliderect(brick.rect):
            hit = brick
            break
    if hit:
        ball.vy = -ball.vy
        bricks.remove(hit)
        return True
    return False


def draw_hud(score, lives):
    score_surf = font.render(f"Score: {score}", True, (255, 255, 255))
    lives_surf = font.render(f"Lives: {lives}", True, (255, 255, 255))
    screen.blit(score_surf, (10, 10))
    screen.blit(lives_surf, (SCREEN_WIDTH - 120, 10))


# ---------- FADE TRANSITIONS ----------
def fade_in():
    fade_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    for alpha in range(255, -1, -10):
        fade_surface.set_alpha(alpha)
        fade_surface.fill((0, 0, 0))
        screen.blit(fade_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(20)


def fade_out():
    fade_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    for alpha in range(0, 256, 10):
        fade_surface.set_alpha(alpha)
        fade_surface.fill((0, 0, 0))
        screen.blit(fade_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(20)

# ---------- SETTINGS ----------
control_mode = 0  # 0=keyboard, 1=mouse, 2=hand
key_left = pygame.K_LEFT
key_right = pygame.K_RIGHT
key_shoot = pygame.K_SPACE
mouse_sensitivity = 1.0

def settings_menu():
    global control_mode, key_left, key_right, key_shoot, mouse_sensitivity

    fade_in()
    selected = 0
    editing_key = None

    while True:
        screen.fill((15, 25, 40))
        title = pygame.font.SysFont("arialblack", 36).render("Settings", True, (255, 255, 180))
        screen.blit(title, (SCREEN_WIDTH / 2 - title.get_width() / 2, 100))

        # Tentukan daftar menu tergantung mode kontrol
        if control_mode == 0:  # Keyboard
            options = ["Control Mode", "Customize Keys", "Back"]
        elif control_mode == 1:  # Mouse
            options = ["Control Mode", "Mouse Sensitivity", "Back"]
        else:  # Hand Detection
            options = ["Control Mode", "Back"]

        # Draw options
        for i, opt in enumerate(options):
            color = (255, 255, 255) if i == selected else (180, 180, 180)
            text = font.render(opt, True, color)
            screen.blit(text, (SCREEN_WIDTH / 2 - 200, 220 + i * 40))

        # Display current values
        control_label = ["Keyboard", "Mouse", "Hand Detection"][control_mode]
        screen.blit(font.render(f"Mode: {control_label}", True, (200, 255, 200)), (SCREEN_WIDTH / 2 + 40, 220))

        if control_mode == 0:  # Keyboard display
            screen.blit(font.render(f"Left: {pygame.key.name(key_left)}", True, (200, 255, 200)), (SCREEN_WIDTH / 2 + 40, 260))
            screen.blit(font.render(f"Right: {pygame.key.name(key_right)}", True, (200, 255, 200)), (SCREEN_WIDTH / 2 + 40, 300))
            screen.blit(font.render(f"Shoot: {pygame.key.name(key_shoot)}", True, (200, 255, 200)), (SCREEN_WIDTH / 2 + 40, 340))

        elif control_mode == 1:  # Mouse display
            screen.blit(font.render(f"Sensitivity: {mouse_sensitivity:.1f}", True, (200, 255, 200)), (SCREEN_WIDTH / 2 + 40, 260))

        # Editing hint
        if editing_key:
            info = font.render("Press any key to assign...", True, (255, 255, 100))
            screen.blit(info, (SCREEN_WIDTH / 2 - info.get_width() / 2, 500))

        pygame.display.flip()

        # --- Input handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if editing_key:
                    # Assign key
                    if editing_key == "left":
                        key_left = event.key
                    elif editing_key == "right":
                        key_right = event.key
                    elif editing_key == "shoot":
                        key_shoot = event.key
                    editing_key = None

                elif event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)

                elif event.key == pygame.K_RETURN:
                    if options[selected] == "Control Mode":
                        control_mode = (control_mode + 1) % 3
                        selected = 0  # reset cursor
                    elif options[selected] == "Customize Keys" and control_mode == 0:
                        # Masuk ke mode ubah tombol
                        fade_in()
                        custom_keys_menu()
                    elif options[selected] == "Mouse Sensitivity" and control_mode == 1:
                        mouse_sensitivity += 0.1
                        if mouse_sensitivity > 2.0:
                            mouse_sensitivity = 0.5
                    elif options[selected] == "Back":
                        fade_out()
                        return

                elif event.key == pygame.K_ESCAPE:
                    fade_out()
                    return

def custom_keys_menu():
    global key_left, key_right, key_shoot
    fade_in()
    options = ["Left", "Right", "Shoot", "Back"]
    selected = 0
    editing_key = None

    while True:
        screen.fill((20, 30, 50))
        title = pygame.font.SysFont("arialblack", 32).render("Customize Keys", True, (255, 255, 180))
        screen.blit(title, (SCREEN_WIDTH / 2 - title.get_width() / 2, 100))

        for i, opt in enumerate(options):
            color = (255, 255, 255) if i == selected else (180, 180, 180)
            text = font.render(opt, True, color)
            screen.blit(text, (SCREEN_WIDTH / 2 - 150, 220 + i * 40))

        # Show assigned keys
        screen.blit(font.render(f"{pygame.key.name(key_left)}", True, (200, 255, 200)), (SCREEN_WIDTH / 2 + 100, 220))
        screen.blit(font.render(f"{pygame.key.name(key_right)}", True, (200, 255, 200)), (SCREEN_WIDTH / 2 + 100, 260))
        screen.blit(font.render(f"{pygame.key.name(key_shoot)}", True, (200, 255, 200)), (SCREEN_WIDTH / 2 + 100, 300))

        if editing_key:
            info = font.render("Press any key to assign...", True, (255, 255, 100))
            screen.blit(info, (SCREEN_WIDTH / 2 - info.get_width() / 2, 480))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if editing_key:
                    if editing_key == "Left":
                        key_left = event.key
                    elif editing_key == "Right":
                        key_right = event.key
                    elif editing_key == "Shoot":
                        key_shoot = event.key
                    editing_key = None
                elif event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if options[selected] in ["Left", "Right", "Shoot"]:
                        editing_key = options[selected]
                    elif options[selected] == "Back":
                        fade_out()
                        return
                elif event.key == pygame.K_ESCAPE:
                    fade_out()
                    return

# ---------- MAIN MENU ----------
def main_menu():
    global control_mode
    fade_in()
    title = pygame.font.SysFont("arialblack", 42).render("DX-BALL", True, (255, 255, 100))
    options = ["Play", "Settings", "Exit"]
    selected = 0

    while True:
        screen.fill((10, 20, 40))
        screen.blit(title, (SCREEN_WIDTH / 2 - title.get_width() / 2, 120))

        for i, opt in enumerate(options):
            color = (255, 255, 255) if i == selected else (180, 180, 180)
            text = font.render(opt, True, color)
            screen.blit(text, (SCREEN_WIDTH / 2 - text.get_width() / 2, 250 + i * 50))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if selected == 0:
                        fade_out()
                        return control_mode
                    elif selected == 1:
                        settings_menu()
                    elif selected == 2:
                        pygame.quit()
                        sys.exit()
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

# ---------- GAME OVER SCREEN ----------
def game_over_screen(score):
    fade_in()
    title = pygame.font.SysFont("arialblack", 36).render("Game Over", True, (255, 80, 80))
    score_text = font.render(f"Final Score: {score}", True, (255, 255, 255))
    back_text = font.render("Press ENTER to return to Menu", True, (200, 200, 255))

    while True:
        screen.fill((10, 10, 30))
        screen.blit(title, (SCREEN_WIDTH / 2 - title.get_width() / 2, 200))
        screen.blit(score_text, (SCREEN_WIDTH / 2 - score_text.get_width() / 2, 280))
        screen.blit(back_text, (SCREEN_WIDTH / 2 - back_text.get_width() / 2, 340))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                fade_out()
                return

# ---------- VICTORY SCREEN ----------
def victory_screen(score):
    fade_in()
    title = pygame.font.SysFont("arialblack", 36).render("🎉 You Win! 🎉", True, (120, 255, 120))
    score_text = font.render(f"Final Score: {score}", True, (255, 255, 255))
    back_text = font.render("Press ENTER to return to Menu", True, (200, 200, 255))

    while True:
        screen.fill((20, 40, 20))
        # Sparkle effect background
        for _ in range(30):
            pygame.draw.circle(screen, (random.randint(100,255), random.randint(200,255), random.randint(100,255)),
                               (random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)), 2)

        screen.blit(title, (SCREEN_WIDTH / 2 - title.get_width() / 2, 200))
        screen.blit(score_text, (SCREEN_WIDTH / 2 - score_text.get_width() / 2, 280))
        screen.blit(back_text, (SCREEN_WIDTH / 2 - back_text.get_width() / 2, 340))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                fade_out()
                return

def fade_transition(screen, color=(0, 0, 0), speed=10):
    """Efek transisi fade in dan fade out."""
    fade_surface = pygame.Surface((screen.get_width(), screen.get_height()))
    fade_surface.fill(color)

    # Fade out (gelap ke terang)
    for alpha in range(0, 255, speed):
        fade_surface.set_alpha(alpha)
        screen.blit(fade_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(10)

    # Fade in (terang ke gelap)
    for alpha in range(255, 0, -speed):
        fade_surface.set_alpha(alpha)
        screen.blit(fade_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(10)

def pause_menu(screen, current_frame):
    """Tampilkan menu pause dengan efek blur lembut dan animasi fade in/out."""
    paused = True
    options = ["Resume", "Main Menu", "Exit Game"]
    selected = 0

    # --- Buat background blur ---
    scale_factor = 0.1  # makin kecil = makin blur
    small = pygame.transform.smoothscale(
        current_frame,
        (int(SCREEN_WIDTH * scale_factor), int(SCREEN_HEIGHT * scale_factor))
    )
    blurred_bg = pygame.transform.smoothscale(small, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # Overlay semi-transparan
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))

    # --- Animasi Fade-In ---
    fade_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    fade_surface.fill((0, 0, 0))
    for alpha in range(255, -1, -15):
        screen.blit(blurred_bg, (0, 0))
        screen.blit(overlay, (0, 0))
        fade_surface.set_alpha(alpha)
        screen.blit(fade_surface, (0, 0))
        pygame.display.flip()
        pygame.time.delay(10)

    # --- Loop Pause Menu ---
    while paused:
        screen.blit(blurred_bg, (0, 0))
        screen.blit(overlay, (0, 0))

        # Judul
        title_font = pygame.font.SysFont("arialblack", 60)
        title = title_font.render("PAUSED", True, (255, 255, 120))
        screen.blit(title, (SCREEN_WIDTH/2 - title.get_width()/2, 160))

        # Opsi menu
        for i, opt in enumerate(options):
            color = (255, 255, 255) if i == selected else (180, 180, 180)
            text = font.render(opt, True, color)
            screen.blit(text, (SCREEN_WIDTH/2 - text.get_width()/2, 280 + i * 50))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    # --- Animasi Fade-Out sebelum keluar pause ---
                    for alpha in range(0, 256, 15):
                        screen.blit(blurred_bg, (0, 0))
                        screen.blit(overlay, (0, 0))
                        fade_surface.set_alpha(alpha)
                        screen.blit(fade_surface, (0, 0))
                        pygame.display.flip()
                        pygame.time.delay(10)

                    if options[selected] == "Resume":
                        return "resume"
                    elif options[selected] == "Main Menu":
                        return "menu"
                    elif options[selected] == "Exit Game":
                        pygame.quit()
                        sys.exit()

                elif event.key == pygame.K_ESCAPE:
                    # Fade-Out juga jika ESC ditekan untuk resume
                    for alpha in range(0, 256, 15):
                        screen.blit(blurred_bg, (0, 0))
                        screen.blit(overlay, (0, 0))
                        fade_surface.set_alpha(alpha)
                        screen.blit(fade_surface, (0, 0))
                        pygame.display.flip()
                        pygame.time.delay(10)
                    return "resume"

# ---------- GAME LOOP ----------
def game_loop(control_mode):
    paddle = Paddle()
    ball = Ball(paddle)
    bricks = make_bricks()
    score = 0
    lives = 3
    running = True

    if control_mode == 2:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        cap = cv2.VideoCapture(0)
    else:
        hands = None
        cap = None

    fade_in()

    pygame.mouse.set_visible(control_mode != 2)

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # Ambil snapshot layar saat ini
                snapshot = screen.copy()
                choice = pause_menu(screen, snapshot)
                if choice == "menu":
                    return  # keluar dari game_loop kembali ke main_menu
                elif choice == "resume":
                    fade_in()

            # Tekan spasi untuk serve (semua mode)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                serve_ball(ball, paddle)

            # Klik kanan mouse untuk serve (hanya mode mouse)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if control_mode == 1 and event.button == 3:  # 3 = right-click
                    serve_ball(ball, paddle)

        if control_mode == 0:  # Keyboard
            keys = pygame.key.get_pressed()
            if keys[key_left]:
                paddle.x -= 7
            if keys[key_right]:
                paddle.x += 7
            if keys[key_shoot]:
                serve_ball(ball, paddle)
            paddle.x = clamp(paddle.x, 0, SCREEN_WIDTH - paddle.width)

        elif control_mode == 1:  # Mouse
            mx, _ = pygame.mouse.get_pos()
            paddle.x += (mx - (paddle.x + paddle.width / 2)) * 0.1 * mouse_sensitivity
            paddle.x = clamp(paddle.x, 0, SCREEN_WIDTH - paddle.width)

        elif control_mode == 2:  # Hand
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand = result.multi_hand_landmarks[0]
                x_norm = hand.landmark[mp_hands.HandLandmark.WRIST].x
                real_x = 1 - x_norm
                paddle.x = clamp(real_x * SCREEN_WIDTH - paddle.width / 2, 0, SCREEN_WIDTH - paddle.width)
            frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame))

        # --- Game Logic ---
        if not ball.stuck:
            ball.x += ball.vx
            ball.y += ball.vy
            reflect_wall(ball)
            paddle_ball_collision(ball, paddle)
            if handle_brick_collisions(ball, bricks):
                score += 10

            if ball.y > SCREEN_HEIGHT:
                lives -= 1
                ball = Ball(paddle)
                if lives <= 0:
                    running = False
        else:
            ball.x = paddle.x + paddle.width / 2
            ball.y = paddle.y - ball.radius - 2

        if not bricks:
            fade_out()
            victory_screen(score)
            return

        # --- Draw ---
        if control_mode == 2 and 'frame_surface' in locals():
            # Gunakan kamera sebagai background
            screen.blit(frame_surface, (0, 0))
        else:
            # Background biru gelap biasa jika tidak pakai kamera
            screen.fill((10, 20, 40))

        for brick in bricks:
            brick.draw(screen)
        paddle.draw(screen)
        ball.draw(screen)
        draw_hud(score, lives)
        pygame.display.flip()

    if cap:
        cap.release()
    if hands:
        hands.close()
    cv2.destroyAllWindows()
    fade_out()
    game_over_screen(score)


# ---------- MAIN ----------
def main():
    while True:
        mode = main_menu()
        game_loop(mode)


if __name__ == "__main__":
    main()
