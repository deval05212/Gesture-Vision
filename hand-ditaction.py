import math
import time
import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Filter", cv2.WINDOW_NORMAL)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

filters          = ["None", "GRAY", "THERMAL", "INVERT", "SKETCH", "VINTAGE"]
current_filter   = 0
last_switch_time = 0
cooldown         = 1.0


def get_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def apply_filter(frame, name):
    """Return a fully-filtered copy of frame."""
    if name == "GRAY":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif name == "THERMAL":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    elif name == "INVERT":
        return cv2.bitwise_not(frame)

    elif name == "SKETCH":
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv     = 255 - gray
        blur    = cv2.GaussianBlur(inv, (21, 21), 0)
        invblur = 255 - blur
        sketch  = cv2.divide(gray, invblur, scale=256.0)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif name == "VINTAGE":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return np.clip(cv2.transform(frame, kernel), 0, 255).astype(np.uint8)

    return frame.copy()   # "None" -> identical copy


def apply_filter_in_box(frame, left_lm, right_lm, w, h, filter_name):
    """
    Apply `filter_name` ONLY inside the quadrilateral:
      TL = left  thumb tip  (lm 4)   L4
      TR = right thumb tip  (lm 4)   R4
      BR = right index tip  (lm 8)   R8
      BL = left  index tip  (lm 8)   L8
    Outside the box the original frame is kept unchanged.
    """
    tl = get_px(left_lm.landmark[4],  w, h)
    tr = get_px(right_lm.landmark[4], w, h)
    br = get_px(right_lm.landmark[8], w, h)
    bl = get_px(left_lm.landmark[8],  w, h)

    pts = np.array([tl, tr, br, bl], dtype=np.int32)

    # Build polygon mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Apply filter to whole frame, then composite only the box region
    filtered = apply_filter(frame, filter_name)

    mask3   = cv2.merge([mask, mask, mask])
    inside  = cv2.bitwise_and(filtered, mask3)
    outside = cv2.bitwise_and(frame, cv2.bitwise_not(mask3))
    result  = cv2.add(inside, outside)

    # Draw box border and corner labels
    cv2.polylines(result, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    corners = [(tl, "L4", (-20, -12)),
               (tr, "R4", (  6, -12)),
               (bl, "L8", (-20,  18)),
               (br, "R8", (  6,  18))]
    for pt, label, (ox, oy) in corners:
        cv2.circle(result, pt, 8, (0, 0, 255), -1)
        cv2.putText(result, label, (pt[0] + ox, pt[1] + oy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    return result


# Main loop
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_hand  = None
    right_hand = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):
            label = handedness.classification[0].label
            # Labels are mirrored because we flipped the frame
            if label == "Left":
                right_hand = hand_lm
            else:
                left_hand  = hand_lm

            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            # Pinch to switch filter
            # Pinch to switch filter
            thumb_tip = hand_lm.landmark[4]
            index_tip = hand_lm.landmark[8]

            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)

            now = time.time()
            dist_index = math.hypot(ix - tx, iy - ty)

            # RIGHT HAND -> NEXT FILTER
            if label == "Left":   # mirrored camera -> actual right hand
                if dist_index < 50 and (now - last_switch_time) > cooldown:
                    current_filter = (current_filter + 1) % len(filters)
                    last_switch_time = now

                    cv2.putText(frame, "NEXT FILTER", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

            # LEFT HAND -> PREVIOUS FILTER
            else:   # mirrored camera -> actual left hand
                if dist_index < 50 and (now - last_switch_time) > cooldown:
                    current_filter = (current_filter - 1) % len(filters)
                    last_switch_time = now

                    cv2.putText(frame, "PREV FILTER", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 165, 255), 2)

    # Apply filter ONLY inside the box when both hands detected
    selected = filters[current_filter]

    if left_hand and right_hand:
        frame = apply_filter_in_box(frame, left_hand, right_hand, w, h, selected)
        cv2.putText(frame, "BOX ACTIVE", (w - 185, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # HUD
    cv2.putText(frame, "Right Pinch=Next  Left Pinch=Prev  Q=Quit",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (150, 150, 150), 1)

    cv2.imshow("Hand Filter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()