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
        self.x = paddle.x + paddle.width/2
        self.y = paddle.y - BALL_RADIUS - 2
        self.vx = 0
        self.vy = -BALL_SPEED
        self.radius = BALL_RADIUS
        self.stuck = True

    def rect(self):
        return pygame.Rect(int(self.x - self.radius), int(self.y - self.radius),
                           self.radius*2, self.radius*2)

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
    colors = [(255, 120, 120), (255, 180, 100), (255, 220, 120),
              (180, 240, 160), (160, 200, 255), (200, 160, 255)]
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
        rel = (ball.x - (paddle.x + paddle.width/2)) / (paddle.width/2)
        angle = rel * (math.pi/3)
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
    score_surf = font.render(f"Score: {score}", True, (255,255,255))
    lives_surf = font.render(f"Lives: {lives}", True, (255,255,255))
    screen.blit(score_surf, (10, 10))
    screen.blit(lives_surf, (SCREEN_WIDTH - 120, 10))

# ---------- MAIN MENU ----------
def main_menu():
    title = pygame.font.SysFont("arialblack", 42).render("DX-BALL", True, (255,255,100))
    options = ["1. Keyboard Control", "2. Mouse Control", "3. Hand Detection (Webcam)"]
    selected = 0

    while True:
        screen.fill((10, 20, 40))
        screen.blit(title, (SCREEN_WIDTH/2 - title.get_width()/2, 120))

        for i, opt in enumerate(options):
            color = (255,255,255) if i == selected else (180,180,180)
            text = font.render(opt, True, color)
            screen.blit(text, (SCREEN_WIDTH/2 - text.get_width()/2, 250 + i*40))

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
                    return selected  # return mode index (0=keyboard,1=mouse,2=hand)
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

# ---------- GAME LOOP ----------
def game_loop(control_mode):
    paddle = Paddle()
    ball = Ball(paddle)
    bricks = make_bricks()
    score = 0
    lives = 3
    running = True

    # Initialize hand tracking only if using webcam mode
    if control_mode == 2:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        cap = cv2.VideoCapture(0)
    else:
        hands = None
        cap = None

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    serve_ball(ball, paddle)

        # --- Control Modes ---
        if control_mode == 0:  # Keyboard
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                paddle.x -= 7
            if keys[pygame.K_RIGHT]:
                paddle.x += 7
            paddle.x = clamp(paddle.x, 0, SCREEN_WIDTH - paddle.width)

            frame_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            frame_surface.fill((20, 20, 40))

        elif control_mode == 1:  # Mouse
            mx, _ = pygame.mouse.get_pos()
            paddle.x = clamp(mx - paddle.width/2, 0, SCREEN_WIDTH - paddle.width)
            frame_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            frame_surface.fill((25, 25, 60))

        elif control_mode == 2:  # Hand Detection
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
                paddle.x = clamp(real_x * SCREEN_WIDTH - paddle.width/2, 0, SCREEN_WIDTH - paddle.width)

            frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame))

        # --- Game Updates ---
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
            ball.x = paddle.x + paddle.width/2
            ball.y = paddle.y - ball.radius - 2

        if not bricks:
            bricks = make_bricks()
            ball = Ball(paddle)

        # --- Draw Everything ---
        screen.blit(frame_surface, (0,0))
        for brick in bricks:
            brick.draw(screen)
        paddle.draw(screen)
        ball.draw(screen)
        draw_hud(score, lives)
        pygame.display.flip()

    if cap: cap.release()
    if hands: hands.close()
    cv2.destroyAllWindows()

# ---------- MAIN ----------
def main():
    while True:
        mode = main_menu()
        game_loop(mode)

if __name__ == "__main__":
    main()
