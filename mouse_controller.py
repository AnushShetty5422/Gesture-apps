import numpy as np
import pyautogui
import time
import math
import cv2

LOCK_THRESHOLD = 0.45
CLICK_THRESHOLD = 0.3
CLICK_COOLDOWN = 0.5
last_click_time = 0
prev_x, prev_y = 0, 0
smoothing_factor = 0.2


def mouse_mode(landmark_list, img):
    global last_click_time, prev_x, prev_y

    if not landmark_list or len(landmark_list) < 21:
        return img, "No Hand Detected"

    index_tip = landmark_list[8]
    thumb_tip = landmark_list[4]

    lm5 = landmark_list[5]
    lm17 = landmark_list[17]
    palm_width = math.hypot(lm5[1] - lm17[1], lm5[2] - lm17[2])
    pinch_distance = math.hypot(thumb_tip[1] - index_tip[1], thumb_tip[2] - index_tip[2])
    normalized_distance = pinch_distance / (palm_width + 1e-6)

    status_text = ""
    current_time = time.time()

    if normalized_distance >= LOCK_THRESHOLD:
        img_h, img_w, _ = img.shape
        screen_w, screen_h = pyautogui.size()
        target_x = np.interp(index_tip[1], [150, img_w - 150], [0, screen_w])
        target_y = np.interp(index_tip[2], [150, img_h - 150], [0, screen_h])
        current_x = prev_x + (target_x - prev_x) * smoothing_factor
        current_y = prev_y + (target_y - prev_y) * smoothing_factor
        pyautogui.moveTo(current_x, current_y)
        prev_x, prev_y = current_x, current_y
        status_text = "Moving"
        cv2.circle(img, (index_tip[1], index_tip[2]), 10, (255, 0, 255), cv2.FILLED)
    else:
        status_text = "Cursor Locked"
        cv2.circle(img, (index_tip[1], index_tip[2]), 10, (0, 255, 255), cv2.FILLED)
        if normalized_distance < CLICK_THRESHOLD and (current_time - last_click_time > CLICK_COOLDOWN):
            pyautogui.click()
            last_click_time = current_time
            status_text = "Clicked!"
            cv2.circle(img, (index_tip[1], index_tip[2]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (thumb_tip[1], thumb_tip[2]), 15, (0, 255, 0), cv2.FILLED)

    return img, status_text