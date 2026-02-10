import os
import cv2
import csv
import numpy as np
import shutil
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.65
)

image_dir = ""
organized_dir = ""
output_csv = "labelsTest.csv"
side = None

def get_angle(a, b, c):
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return angle


def is_valid_point(landmarks, point_name, threshold=0.05):
    if point_name not in landmarks:
        return False
    p = landmarks[point_name]
    if p.x < threshold or p.x > 1 - threshold:
        return False
    if p.y < threshold or p.y > 1 - threshold:
        return False
    return True



def classify(landmarks):
    global side
    def p(name): return [landmarks[name].x, landmarks[name].y]

    side = None
    try:

        if is_valid_point(landmarks, "LEFT_SHOULDER") and is_valid_point(landmarks, "RIGHT_SHOULDER"):
            if landmarks["LEFT_SHOULDER"].z < landmarks["RIGHT_SHOULDER"].z:
                side = "LEFT"
            else:
                side = "RIGHT"
        elif is_valid_point(landmarks, "LEFT_SHOULDER"):
            side = "LEFT"
        elif is_valid_point(landmarks, "RIGHT_SHOULDER"):
            side = "RIGHT"
        else:
            return "undetected" 

        shoulder = f"{side}_SHOULDER"
        hip = f"{side}_HIP"
        knee = f"{side}_KNEE"
        ear = f"{side}_EAR"
        nose = "NOSE" 
        opposite_knee = "RIGHT_KNEE" if side == "LEFT" else "LEFT_KNEE"


        # 1. Postural Asymmetry 
        if (is_valid_point(landmarks, "LEFT_SHOULDER") and is_valid_point(landmarks, "RIGHT_SHOULDER")):
            # ?img frontala
            dist_x = abs(landmarks["LEFT_SHOULDER"].x - landmarks["RIGHT_SHOULDER"].x)
            z_diff = abs(landmarks["LEFT_SHOULDER"].z - landmarks["RIGHT_SHOULDER"].z)
            print(f"[ASYMMETRY] Distanta X intre umeri: {dist_x:.3f}")
            print(f"[ASYMMETRY] Distanta z intre umeri: {z_diff:.3f}")

            if dist_x > 0.07 and z_diff < 0.45:
                diff_y = abs(landmarks["LEFT_SHOULDER"].y - landmarks["RIGHT_SHOULDER"].y)
                
                print(f"[ASYMMETRY] Distanta  Y intre umeri: {diff_y:.3f}")


                if diff_y > 0.017:
                    return "postural_asymmetry"
                if is_valid_point(landmarks, knee) and is_valid_point(landmarks, opposite_knee):
                    knee_dist_y = abs(p(knee)[1] - p(opposite_knee)[1])
                    print(f"[ASYMMETRY] Diferenta Y intre genunchi: {knee_dist_y:.3f}")

                    if knee_dist_y > 0.02:
                        return "postural_asymmetry"
        # 2. Symmetrical Front                
                return "symmetrical_front"

        angle_ear_shoulder_hip = 0
        # 3. Stooped posture 
        if (is_valid_point(landmarks, ear) and is_valid_point(landmarks, shoulder) and is_valid_point(landmarks, hip)):
            angle_ear_shoulder_hip = get_angle(p(ear), p(shoulder), p(hip))
            print(f"Unghi ureche-umar-sold: {angle_ear_shoulder_hip:.1f}°")

            if angle_ear_shoulder_hip < 143:
                return "stooped_posture"

        if is_valid_point(landmarks, nose) and is_valid_point(landmarks, shoulder):

            dist_nose_shoulder = abs(p(nose)[0] - p(shoulder)[0])
            print(f"[FORWARD HEAD] Distanta X nas-umar: {dist_nose_shoulder:.3f}")

            if dist_nose_shoulder  > 0.135:
                if is_valid_point(landmarks, ear) and is_valid_point(landmarks, hip):
                    print(f"[FORWARD HEAD] Unghi ureche-umar-sold: {angle_ear_shoulder_hip:.1f}°")
                    if 175 > angle_ear_shoulder_hip >= 143:
                        return "stooped_posture"
                else:
                    return "stooped_posture"

        # 4. Neutral Posture
        failed = 0
        # ureche-umar-sold deja verificat la 2
        # cap proiectat in fata verificat la 3
       
        angle_shoulder_hip_knee = 0
        if is_valid_point(landmarks, shoulder) and is_valid_point(landmarks, hip) and is_valid_point(landmarks, knee):
            angle_shoulder_hip_knee = get_angle(p(shoulder), p(hip), p(knee))
           
            print(f"[NEUTRAL] Unghi umar-sold-genunchi: {angle_shoulder_hip_knee:.1f}°")

            if 80 > angle_shoulder_hip_knee or angle_shoulder_hip_knee > 110:
                print(f" Failed")
                failed += 1

        if failed == 0:
            return "neutral_posture"


        # leaning forward ->3
        if 81 > angle_shoulder_hip_knee > 0:
            return "stooped_posture"

        # 5. Slouched Posture
        if is_valid_point(landmarks, hip):
            return "slouched_posture"

        return "undetected"

    except:
        return "undetected"



results = []
count = 1

for filename in sorted(os.listdir(image_dir)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            continue

        ext = os.path.splitext(filename)[1] 
        new_name = f"img_{count:04d}{ext}"
        label = "undetected"

        if len(image.shape) < 3 or image.shape[2] != 3:
            continue
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            landmarks = {
                mp_pose.PoseLandmark(i).name: lm
                for i, lm in enumerate(result.pose_landmarks.landmark)
            }
            label = classify(landmarks)
               
            
             relevant_all = [
                 f"{side}_EAR",
                 f"{side}_SHOULDER",
                 f"{side}_HIP",
                 f"{side}_KNEE",
                 "NOSE"
             ]

             relevant_points = [name for name in relevant_all if is_valid_point(landmarks, name)]

             annotated_image = image.copy()

             for name in relevant_points:
                 if name in landmarks:
                     cx = int(landmarks[name].x * image.shape[1])
                     cy = int(landmarks[name].y * image.shape[0])
                     cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1) 
                     cv2.putText(annotated_image, name, (cx + 5, cy - 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  

             cv2.imshow(f"{new_name} - {label}", annotated_image)

             key = cv2.waitKey(0)
             if key == 27:  
                 cv2.destroyAllWindows()
             else:
                 continue        
        
        else:
            label = "undetected"

        results.append((new_name, label))

        label_dir = os.path.join(organized_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        dest_path = os.path.join(label_dir, new_name)
        shutil.copy(image_path, dest_path)

        count += 1

file_exists = os.path.exists(output_csv)

with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['image_name', 'label'])  
    writer.writerows(results)

print("Etichetare + organizare. CSV salvat ca:", output_csv)
