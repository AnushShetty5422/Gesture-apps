import cv2
import numpy as np
import math

CUBE_VERTICES = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                          [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
CUBE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
              (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

angle_x, angle_y = 0, 0
is_grabbing = False
last_grab_pos = None
scale = 100


def cube_mode(all_hands_landmarks, img):
    global angle_x, angle_y, is_grabbing, last_grab_pos, scale
    h, w, c = img.shape
    num_hands = len(all_hands_landmarks)

    if num_hands == 2:
        hand1_center = all_hands_landmarks[0][0]
        hand2_center = all_hands_landmarks[1][0]
        inter_hand_distance = math.hypot(hand1_center[1] - hand2_center[1], hand1_center[2] - hand2_center[2])
        scale = np.interp(inter_hand_distance, [50, 400], [50, 250])
        status_text = "Two Hands: Scale"
    elif num_hands == 1:
        landmarks = all_hands_landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_distance = math.hypot(thumb_tip[1] - index_tip[1], thumb_tip[2] - index_tip[2])
        is_pinched = pinch_distance < 40
        if is_pinched:
            status_text = "Grabbed: Rotate"
            current_grab_pos = landmarks[9]
            if not is_grabbing:
                is_grabbing = True
                last_grab_pos = current_grab_pos
            else:
                dx = current_grab_pos[1] - last_grab_pos[1]
                dy = current_grab_pos[2] - last_grab_pos[2]
                angle_y += dx * 0.01
                angle_x -= dy * 0.01
                last_grab_pos = current_grab_pos
        else:
            is_grabbing = False
            status_text = "Pinch to Grab & Rotate"
    else:
        is_grabbing = False
        status_text = "Show hand(s) to interact"

    rot_x = np.array([[1, 0, 0], [0, math.cos(angle_x), -math.sin(angle_x)], [0, math.sin(angle_x), math.cos(angle_x)]])
    rot_y = np.array([[math.cos(angle_y), 0, math.sin(angle_y)], [0, 1, 0], [-math.sin(angle_y), 0, math.cos(angle_y)]])
    rotated_vertices = np.dot(CUBE_VERTICES, rot_x)
    rotated_vertices = np.dot(rotated_vertices, rot_y)
    projected_points = []
    for vertex in rotated_vertices:
        x = int(vertex[0] * scale + w / 2)
        y = int(vertex[1] * scale + h / 2)
        projected_points.append((x, y))
    for edge in CUBE_EDGES:
        p1 = projected_points[edge[0]]
        p2 = projected_points[edge[1]]
        cv2.line(img, p1, p2, (0, 200, 255), 2)
    return img, status_text