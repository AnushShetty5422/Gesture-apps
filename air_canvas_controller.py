import cv2
import numpy as np
import math
from collections import deque

canvas = None
prev_point = None
draw_color = (0, 255, 255)
draw_thickness = 10
points_history = deque(maxlen=5)


def canvas_mode(landmark_list, img):
    global prev_point
    status_text = "Air Canvas"
    if landmark_list:
        index_tip = landmark_list[8]
        thumb_tip = landmark_list[4]

        points_history.append(index_tip[1:])
        avg_point = np.mean(points_history, axis=0).astype(int)

        pinch_distance = math.hypot(thumb_tip[1] - index_tip[1], thumb_tip[2] - index_tip[2])
        is_pinching_to_draw = pinch_distance < 50

        if is_pinching_to_draw:
            status_text = "Drawing..."
            cv2.circle(img, tuple(avg_point), 10, draw_color, cv2.FILLED)
            if prev_point is None:
                prev_point = tuple(avg_point)
            cv2.line(canvas, prev_point, tuple(avg_point), draw_color, draw_thickness)
            prev_point = tuple(avg_point)
        else:
            prev_point = None
            points_history.clear()
            status_text = "Move Brush"
    else:
        prev_point = None
        points_history.clear()

    return img, status_text


def get_canvas():
    return canvas


def initialize_canvas(shape):
    global canvas
    if canvas is None:
        canvas = np.zeros(shape, dtype=np.uint8)


def clear_canvas():
    global canvas
    if canvas is not None:
        canvas = np.zeros_like(canvas)
    return "Canvas Cleared!"