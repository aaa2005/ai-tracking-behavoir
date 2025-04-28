import cv2
import mediapipe as mp
import numpy as np
import time


def recognize_gesture(hand_landmarks):
    fingers = []
    tips_ids = [4, 8, 12, 16, 20]
    mcp_ids = [2, 5, 9, 13, 17]

    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[mcp_ids[i]].y:
            fingers.append(1)
        else:
            fingers.append(0)

    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [1, 1, 0, 0, 1]:
        return "OK Sign"
    elif fingers == [0, 0, 1, 0, 0]:
        return "FUCK YOU"
    else:
        return "Unknown"


def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=0)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face_model.yml")

    label_map = {}
    with open("label_map.txt", "r") as f:
        for line in f:
            key, val = line.strip().split(":")
            label_map[int(key)] = val

    cap = cv2.VideoCapture(0)
    cap.set(3, 1040)
    cap.set(4, 720)

    prev_time = time.time()
    prev_pos = None
    speed = 0
    direction = "Idle"
    prev_wrist = None
    wave_detected = False

    frame_skip = 3
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_pose = pose.process(rgb)
        results_hands = hands.process(rgb)

        label = "Not Enough Data"
        color = (80, 80, 80)
        current_time = time.time()
        fps = 1 / (current_time - prev_time + 1e-6)
        prev_time = current_time

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, 1.1, 6, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = grey[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)

            label_id, confidence = recognizer.predict(face_roi)
            name = label_map.get(label_id, "Unknown")
            if confidence < 50:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
                cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark
            visible = sum(1 for l in lm if l.visibility > 0.6)

            if visible >= int(len(lm) * 0.6):
                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                def get_xy(part): return np.array([lm[part].x, lm[part].y])

                left_hip = get_xy(mp_pose.PoseLandmark.LEFT_HIP)
                right_hip = get_xy(mp_pose.PoseLandmark.RIGHT_HIP)
                left_knee = get_xy(mp_pose.PoseLandmark.LEFT_KNEE)
                right_knee = get_xy(mp_pose.PoseLandmark.RIGHT_KNEE)
                left_ankle = get_xy(mp_pose.PoseLandmark.LEFT_ANKLE)
                right_ankle = get_xy(mp_pose.PoseLandmark.RIGHT_ANKLE)

                def angle(a, b, c):
                    ba = a - b
                    bc = c - b
                    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
                    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

                angle_l = angle(left_hip, left_knee, left_ankle)
                angle_r = angle(right_hip, right_knee, right_ankle)
                avg_angle = (angle_l + angle_r) / 2

                if avg_angle < 139:
                    label = "Sitting"
                    color = (0, 240, 255)
                elif avg_angle > 140:
                    label = "Standing"
                    color = (0, 255, 0)
                else:
                    label = "Transitioning"
                    color = (255, 100, 0)

                center_hip = (left_hip + right_hip) / 2
                if prev_pos is not None:
                    delta = center_hip - prev_pos
                    distance = np.linalg.norm(delta)
                    speed = distance * fps * 100

                    dx, dy = delta
                    if abs(dx) > abs(dy):
                        direction = "Right" if dx > 0 else "Left"
                    elif abs(dy) > 0.01:
                        direction = "Forward" if dy < 0 else "Backward"
                    else:
                        direction = "Idle"
                else:
                    speed = 0
                    direction = "Idle"

                prev_pos = center_hip
            else:
                label = "Body Not Fully Visible"
                color = (0, 0, 255)
                prev_pos = None
                speed = 0
                direction = "Idle"

        # === Hand detection and gestures ===
        gesture = None
        if results_hands.multi_hand_landmarks:
            for landmarks in results_hands.multi_hand_landmarks:
                wrist = np.array([landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                                  landmarks.landmark[mp_hands.HandLandmark.WRIST].y])

                index_tip = np.array([landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                      landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
                gesture = recognize_gesture(landmarks)

                if prev_wrist is not None:
                    distance = np.linalg.norm(wrist - prev_wrist)
                    if distance > 0.05:
                        wave_detected = True
                        cv2.putText(frame, "Waving!", (240, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                    else:
                        wave_detected = False

                prev_wrist = wrist
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # cv2.putText(frame, f"Status: {label} | Speed: {speed:.2f} | Direction: {direction}", (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Gesture: {gesture}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Posture: {label}", (30, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Speed: {speed:.1f} px/s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        cv2.putText(frame, f"Direction: {direction}", (30,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (30, 150), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 2)
        prev_time = current_time
        cv2.imshow("AI Vision", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
