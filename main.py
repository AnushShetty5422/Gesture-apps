import cv2
import time
import hand_tracking
import mouse_controller
import air_canvas_controller
import model_controller_3d

# --- Main Application Setup ---
print("Initializing camera...")
cap = cv2.VideoCapture(0)
# --- NEW: Check if the camera opened successfully ---
if not cap.isOpened():
    print("FATAL ERROR: Could not open webcam.")
    exit()

w_cam, h_cam = 640, 480
cap.set(3, w_cam)
cap.set(4, h_cam)
print("Camera initialized successfully.")

tracker = hand_tracking.HandTracker(detection_con=0.8, max_hands=2)

# --- UI and Mode Management ---
MODES = ["Air Mouse", "Air Canvas", "3D Controller"]
current_mode_index = 0
UI_COOLDOWN = 1.0
last_ui_interaction_time = 0

UI_BG_COLOR = (44, 44, 44)
UI_TEXT_COLOR = (255, 255, 255)
UI_ACCENT_COLOR = (0, 200, 255)
UI_BUTTON_COLOR = (255, 0, 100)

print("Starting main application loop...")
# --- Main Loop ---
while True:
    success, img = cap.read()
    # --- NEW: Check if the frame was read correctly ---
    if not success:
        print("Error: Failed to read a frame from the webcam. Exiting.")
        break

    img = cv2.flip(img, 1)

    img = tracker.find_hands(img)

    all_hands_landmarks = []
    if tracker.results.multi_hand_landmarks:
        for i in range(len(tracker.results.multi_hand_landmarks)):
            all_hands_landmarks.append(tracker.find_position(img, hand_no=i))

    hand1_landmarks = all_hands_landmarks[0] if all_hands_landmarks else None

    air_canvas_controller.initialize_canvas(img.shape)
    current_mode = MODES[current_mode_index]
    status_text = "No Hand Detected"

    if hand1_landmarks:
        # print("Hand detected, processing mode:", current_mode) # Uncomment for more verbose debugging
        is_pinching = tracker.get_normalized_pinch_distance(hand1_landmarks) < 0.3
        clicked_on_ui = False
        current_time = time.time()

        if is_pinching and (current_time - last_ui_interaction_time > UI_COOLDOWN):
            index_tip = hand1_landmarks[8]
            tab_y_start, tab_height, tab_width = 10, 50, 180
            for i, mode in enumerate(MODES):
                tab_x_start = 10 + (i * (tab_width + 10))
                if tab_x_start < index_tip[1] < tab_x_start + tab_width and tab_y_start < index_tip[
                    2] < tab_y_start + tab_height:
                    current_mode_index = i
                    last_ui_interaction_time = current_time
                    clicked_on_ui = True
                    if MODES[current_mode_index] == "Air Canvas":
                        air_canvas_controller.clear_canvas()
                    break

            clear_button_rect = (500, 400, 130, 50)
            if not clicked_on_ui and current_mode == "Air Canvas":
                x, y, w, h = clear_button_rect
                if x < index_tip[1] < x + w and y < index_tip[2] < y + h:
                    status_text = air_canvas_controller.clear_canvas()
                    last_ui_interaction_time = current_time
                    clicked_on_ui = True

        if not clicked_on_ui:
            if current_mode == "Air Mouse":
                if len(all_hands_landmarks) == 1:
                    img, status_text = mouse_controller.mouse_mode(hand1_landmarks, img)
                else:
                    status_text = "Air Mouse requires one hand"
            elif current_mode == "Air Canvas":
                if len(all_hands_landmarks) == 1:
                    img, status_text = air_canvas_controller.canvas_mode(hand1_landmarks, img)
                else:
                    status_text = "Air Canvas requires one hand"
            elif current_mode == "3D Controller":
                img, status_text = model_controller_3d.cube_mode(all_hands_landmarks, img)

    # Persistent Rendering
    if current_mode == "Air Canvas":
        canvas = air_canvas_controller.get_canvas()
        if canvas is not None:
            img = cv2.addWeighted(img, 0.7, canvas, 1, 0)
    elif current_mode == "3D Controller":
        img, status_text_render = model_controller_3d.cube_mode(all_hands_landmarks, img)
        if not hand1_landmarks: status_text = status_text_render

    # Draw UI on top
    tab_y_start, tab_height, tab_width = 10, 50, 180
    for i, mode in enumerate(MODES):
        tab_x_start = 10 + (i * (tab_width + 10))
        color = UI_ACCENT_COLOR if i == current_mode_index else UI_BG_COLOR
        cv2.rectangle(img, (tab_x_start, tab_y_start), (tab_x_start + tab_width, tab_y_start + tab_height), color,
                      cv2.FILLED)
        cv2.putText(img, mode, (tab_x_start + 15, tab_y_start + 35), cv2.FONT_HERSHEY_PLAIN, 2, UI_TEXT_COLOR, 2)

    if current_mode == "Air Canvas":
        x, y, w, h = (500, 400, 130, 50)
        cv2.rectangle(img, (x, y), (x + w, y + h), UI_BUTTON_COLOR, cv2.FILLED)
        cv2.putText(img, "Clear", (x + 20, y + 35), cv2.FONT_HERSHEY_PLAIN, 2, UI_TEXT_COLOR, 2)

    cv2.putText(img, status_text, (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, UI_ACCENT_COLOR, 3)

    cv2.imshow("Gesture Control UI", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Closing application...")
cap.release()
cv2.destroyAllWindows()