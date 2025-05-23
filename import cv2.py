import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Helper function to calculate distance
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Get pixel coordinates from normalized landmarks
def get_position(lm, id, shape):
    h, w, _ = shape
    return int(lm[id].x * w), int(lm[id].y * h)

# Detect how many fingers are up
def fingers_up(lm):
    fingers = []

    # Thumb (special case: x-coordinates)
    if lm[4].x < lm[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers (based on y-coordinates)
    tips = [8, 12, 16, 20]
    for tip in tips:
        if lm[tip].y < lm[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

# Track click cooldown
last_click_time = 0
click_delay = 0.5  # seconds

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get cursor position (index finger tip)
        index_x, index_y = get_position(lm, 8, frame.shape)
        screen_x = np.interp(index_x, [0, w], [0, screen_width])
        screen_y = np.interp(index_y, [0, h], [0, screen_height])
        pyautogui.moveTo(screen_x, screen_y, duration=0.01)

        # Draw cursor circle on webcam feed
        cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

        # Detect pinch for click
        thumb_x, thumb_y = get_position(lm, 4, frame.shape)
        if distance([index_x, index_y], [thumb_x, thumb_y]) < 30:
            current_time = time.time()
            if current_time - last_click_time > click_delay:
                pyautogui.click()
                last_click_time = current_time
                cv2.putText(frame, "Click!", (index_x + 20, index_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Scroll gestures
        finger_count = fingers_up(lm)
        if finger_count == 2:
            pyautogui.scroll(20)
            cv2.putText(frame, "Scroll Up", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif finger_count == 3:
            pyautogui.scroll(-20)
            cv2.putText(frame, "Scroll Down", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the webcam feed
    cv2.imshow("Gesture Controlled Mouse", frame)

    # Exit on 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
