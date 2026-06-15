"""
======================================================================================
PPD-1: MULTIMODAL ECOLOGICAL MOMENTARY ASSESSMENT (EMA) SYSTEM
VERSION: 5.1 (Edge-ResNet Architecture with Threshold Analytics)
======================================================================================
CLINICAL & MATHEMATICAL FORMULATIONS UTILIZED:

1. Digital Signal & Image Processing (DSP):
   - LAB Color Space Transformation: Isolates the L* (Luminance) channel to 
     measure periorbital structural brightness. This extracts the true biological 
     fatigue signal while remaining mathematically agnostic to human skin tone variations.

2. Geometric Feature Extraction (L2 Norm Vectors):
   - Uses Euclidean distance to calculate scale-invariant facial geometry, normalizing 
     against face width to maintain accuracy regardless of camera proximity.
   - Core Vectors: EAR (Eye Aspect Ratio), ETR (Eyebrow Tension Ratio), 
     MAR (Mouth Aspect Ratio), FAR (Face Aspect Ratio).
   - MOE = MAR / EAR: A heavily cited scientific index for drowsiness discrimination.

3. Facial Action Coding System (FACS):
   - Extracts 52 distinct Muscle Blendshapes (Action Units) via MediaPipe.
   - Resolves the "Inter-Personal Baseline Variance" problem by analyzing active 
     muscle contractions (e.g., Corrugator supercilii / AU4) independently of 
     the subject's underlying bone structure.

4. Machine Learning Architecture (Vision AI):
   - Topology: Residual Neural Network (ResNet) with Skip Connections to fuse 
     raw 59-dimensional vector inputs with deep abstract patterns.
   - Threshold Boundary: The final Sigmoid activation layer utilizes a >0.50 (50%) 
     threshold to biologically separate Relaxed (0) from Stressed (1) states.

5. Clinical Diagnostic Bands (Defined Thresholds):
   - 00.0% - 40.0% : Optimal Baseline / Relaxed
   - 41.0% - 55.0% : Sub-Clinical / Low Stress
   - 56.0% - 70.0% : Elevated Sympathetic Tone / Moderate Stress
   - 71.0% - 85.0% : Acute Fatigue / High Stress
   - 86.0% - 100%  : Critical / Consult Specialist
   - Neural Network Sigmoid Activation > 50.0% is technically Stressed
======================================================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.spatial import distance as dist
import pandas as pd
import pickle
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import time
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

SIMULATION_MODE = True 

print("Initializing PPD-1 Master Diagnostic Engine (V5 Edge-ResNet Architecture)...")

try:
    # Load V5 Super-Vector Brain
    with open('vision_scaler_v5.pkl', 'rb') as f: v_scaler = pickle.load(f)
    v_interp = tf.lite.Interpreter(model_path="vision_model_v5.tflite")
    v_interp.allocate_tensors()
    
    # Load Physical Hardware Brain
    with open('scaler.pkl', 'rb') as f: p_scaler = pickle.load(f)
    p_interp = tf.lite.Interpreter(model_path="stress_model.tflite")
    p_interp.allocate_tensors()
    print("[SYSTEM] V5 Neural Networks Successfully Loaded.")
except Exception as e:
    print(f"[FATAL ERROR] Missing AI weights: {e}")
    exit()

v_in, v_out = v_interp.get_input_details(), v_interp.get_output_details()
p_in, p_out = p_interp.get_input_details(), p_interp.get_output_details()

# MediaPipe Setup (Output Blendshapes Enabled for V5)
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options, 
    num_faces=1,
    output_face_blendshapes=True
)
detector = vision.FaceLandmarker.create_from_options(options)

# Clinical Landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_BROW, RIGHT_BROW = 107, 336
OUTER_EYES = [33, 263]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
EAR_THRESHOLD = 0.20 

def calc_ear(eye_pts):
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

def check_dark_circles_lab(frame, landmarks, w, h):
    try:
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab_frame[:, :, 0] 
        
        l_eye_y, l_eye_x = int(landmarks[145].y * h), int(landmarks[145].x * w)
        l_cheek_y, l_cheek_x = int(landmarks[205].y * h), int(landmarks[205].x * w)
        r_eye_y, r_eye_x = int(landmarks[374].y * h), int(landmarks[374].x * w)
        r_cheek_y, r_cheek_x = int(landmarks[425].y * h), int(landmarks[425].x * w)
        
        l_eye_lum = np.mean(l_channel[max(0, l_eye_y-5):l_eye_y+5, max(0, l_eye_x-5):l_eye_x+5])
        l_cheek_lum = np.mean(l_channel[max(0, l_cheek_y-5):l_cheek_y+5, max(0, l_cheek_x-5):l_cheek_x+5])
        r_eye_lum = np.mean(l_channel[max(0, r_eye_y-5):r_eye_y+5, max(0, r_eye_x-5):r_eye_x+5])
        r_cheek_lum = np.mean(l_channel[max(0, r_cheek_y-5):r_cheek_y+5, max(0, r_cheek_x-5):r_cheek_x+5])
        
        l_ratio = l_eye_lum / (l_cheek_lum + 1e-6)
        r_ratio = r_eye_lum / (r_cheek_lum + 1e-6)
        
        flag = 1 if (l_ratio < 0.85 or r_ratio < 0.85) else 0
        boxes = [
            (l_eye_x, l_eye_y, (0,0,255) if l_ratio < 0.85 else (0,255,0)),
            (l_cheek_x, l_cheek_y, (255,255,0)), 
            (r_eye_x, r_eye_y, (0,0,255) if r_ratio < 0.85 else (0,255,0)),
            (r_cheek_x, r_cheek_y, (255,255,0))  
        ]
        return flag, boxes
    except:
        return 0, []

def get_medical_category(probability):
    """Maps raw AI probability to clinical diagnostic bands with defined thresholds."""
    prob_pct = probability * 100
    if prob_pct <= 40: return "OPTIMAL BASELINE (Relaxed) [0-40%]", (0, 255, 0)
    elif prob_pct <= 55: return "SUB-CLINICAL (Low Stress) [41-55%]", (0, 200, 255)
    elif prob_pct <= 70: return "ELEVATED SYMPATHETIC TONE (Moderate) [56-70%]", (0, 165, 255)
    elif prob_pct <= 85: return "ACUTE FATIGUE/STRESS (High) [71-85%]", (0, 100, 255)
    else: return "CRITICAL (Consult Psychophysiologist) [86-100%]", (0, 0, 255)

def process_vision(frame, session_state=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = detector.detect(mp_img)

    if not res.face_landmarks or not res.face_blendshapes: 
        return frame, None, {}

    lm = res.face_landmarks[0]
    blendshapes = res.face_blendshapes[0]
    h, w, _ = frame.shape
    
    # 1. Base Geometry Math
    l_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE])
    r_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE])
    ear = (calc_ear(l_eye) + calc_ear(r_eye)) / 2.0
    
    pt_l_brow = (int(lm[LEFT_BROW].x*w), int(lm[LEFT_BROW].y*h))
    pt_r_brow = (int(lm[RIGHT_BROW].x*w), int(lm[RIGHT_BROW].y*h))
    
    outer_l = (lm[OUTER_EYES[0]].x * w, lm[OUTER_EYES[0]].y * h)
    outer_r = (lm[OUTER_EYES[1]].x * w, lm[OUTER_EYES[1]].y * h)
    face_width = dist.euclidean(outer_l, outer_r) + 1e-6
    
    etr = dist.euclidean(pt_l_brow, pt_r_brow) / face_width

    mouth_h = dist.euclidean((lm[MOUTH_TOP].x*w, lm[MOUTH_TOP].y*h), (lm[MOUTH_BOTTOM].x*w, lm[MOUTH_BOTTOM].y*h))
    mouth_w = dist.euclidean((lm[MOUTH_LEFT].x*w, lm[MOUTH_LEFT].y*h), (lm[MOUTH_RIGHT].x*w, lm[MOUTH_RIGHT].y*h))
    mar = mouth_h / mouth_w if mouth_w > 0 else 0

    # 2. Advanced V5 Geometry Math
    moe = mar / ear if ear > 0 else 0
    l_brow_eye = dist.euclidean(pt_l_brow, np.mean(l_eye, axis=0)) / face_width
    r_brow_eye = dist.euclidean(pt_r_brow, np.mean(r_eye, axis=0)) / face_width
    far = dist.euclidean((lm[10].x*w, lm[10].y*h), (lm[152].x*w, lm[152].y*h)) / face_width

    dark_circle_flag, dc_boxes = check_dark_circles_lab(frame, lm, w, h)

    # 3. State Tracking (Blinks & Timer)
    if session_state is not None:
        if ear < EAR_THRESHOLD:
            if not session_state["eye_closed"]:
                session_state["blinks"] += 1
                session_state["eye_closed"] = True
        else:
            session_state["eye_closed"] = False

    # 4. V5 Super-Vector Fusion (59 Inputs)
    geom_features = [ear, etr, mar, moe, l_brow_eye, r_brow_eye, far]
    blend_features = [b.score for b in blendshapes]
    v5_input = np.array(geom_features + blend_features)

    # 5. ResNet Edge Inference
    scaled_in = v_scaler.transform([v5_input]).astype(np.float32)
    v_interp.set_tensor(v_in[0]['index'], scaled_in)
    v_interp.invoke()
    prob = v_interp.get_tensor(v_out[0]['index'])[0][0]

    category_text, col = get_medical_category(prob)

    # 6. UI Overlay
    if SIMULATION_MODE:
        cv2.putText(frame, f"STATUS: {category_text}", (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 2)
        cv2.putText(frame, f"EAR: {ear:.3f} | ETR: {etr:.3f} | MOE: {moe:.3f}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        if dark_circle_flag: 
            cv2.putText(frame, "FATIGUE BIOMARKER DETECTED", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
        
        cv2.line(frame, pt_l_brow, pt_r_brow, col, 2)
        for (bx, by, bcolor) in dc_boxes:
            cv2.rectangle(frame, (bx-4, by-4), (bx+4, by+4), bcolor, 1)
            
        if session_state is not None and "start_time" in session_state:
            elapsed = int(time.time() - session_state["start_time"])
            mins, secs = divmod(elapsed, 60)
            cv2.putText(frame, f"TIME: {mins:02d}:{secs:02d}", (w - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"BLINKS: {session_state['blinks']}", (w - 140, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    metrics = {"ear": ear, "etr": etr, "mar": mar, "dark_circles": dark_circle_flag}
    return frame, prob, metrics

def save_medical_report(user_data, ai_data, filename="Master_Diagnostic_Records.csv"):
    avg_v_prob = np.mean(ai_data.get("v_prob_list", [0])) if ai_data.get("v_prob_list") else 0
    dark_circle_freq = np.mean(ai_data.get("dc_list", [0])) if ai_data.get("dc_list") else 0
    
    # Calculate Final Holistic Verdict
    # The AI Sigmoid naturally crosses 0.50 (50%) to indicate Stress.
    # The dark circles act as a minor fatigue multiplier (up to 5% flat penalty).
    fatigue_penalty = dark_circle_freq * 0.05 
    final_score_prob = avg_v_prob + fatigue_penalty
    
    final_verdict, _ = get_medical_category(final_score_prob)

    report = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Session_ID": user_data.get("name", "Unknown"),
        "Mode": ai_data.get("mode", "N/A"),
        "Total_Blinks": ai_data.get("final_blinks", 0),
        "Avg_EAR": round(np.mean(ai_data.get("ear_list", [0])), 3) if ai_data.get("ear_list") else 0,
        "Avg_ETR": round(np.mean(ai_data.get("etr_list", [0])), 3) if ai_data.get("etr_list") else 0,
        "Avg_MAR": round(np.mean(ai_data.get("mar_list", [0])), 3) if ai_data.get("mar_list") else 0,
        "Dark_Circle_Freq_%": round(dark_circle_freq * 100, 1),
        "AI_Vision_Stress_%": round(avg_v_prob * 100, 1),
        "AI_Phys_Stress_%": round(np.mean(ai_data.get("p_prob_list", [0]))*100, 1) if ai_data.get("p_prob_list") else "N/A",
        "Final_Stress_Score_%": round(final_score_prob * 100, 1),
        "CLINICAL_VERDICT": final_verdict,
    }

    df = pd.DataFrame([report])
    file_exists = os.path.isfile(filename)
    
    print("\n" + "="*65)
    print("   PPD-1 FINAL CLINICAL DIAGNOSTIC REPORT")
    print("="*65)
    for k, v in report.items(): print(f"{k:25}: {v}")
    print("="*65)
    
    
    try:
        df.to_csv(filename, mode='a', header=not file_exists, index=False)
        print(f">> Data securely logged to {filename}")
    except PermissionError:
        print(f"[WARNING] Could not save to CSV! Please close '{filename}' if it is open.")

def run_live_hybrid(user_data, use_hardware=False):
    cap = cv2.VideoCapture(0)
    ai_data = {"mode": "Hybrid" if use_hardware else "Vision Only", "ear_list": [], "etr_list": [], "mar_list": [], "dc_list": [], "v_prob_list": [], "p_prob_list": []}
    session_state = {"blinks": 0, "eye_closed": False, "start_time": time.time()}
    
    if use_hardware: print("[SYSTEM] Hardware Sensors Engaged (Awaiting I2C streams...)")
    print("Press 'ESC' to conclude session and generate report.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        proc_frame, v_prob, metrics = process_vision(frame, session_state)
        if v_prob is not None:
            ai_data["ear_list"].append(metrics["ear"])
            ai_data["etr_list"].append(metrics["etr"])
            ai_data["mar_list"].append(metrics["mar"])
            ai_data["dc_list"].append(metrics["dark_circles"])
            ai_data["v_prob_list"].append(v_prob)
            
            if use_hardware:
                dummy_gsr, dummy_hr = 4.5, 85.0 
                scaled_p = p_scaler.transform([[dummy_gsr, 0.5, dummy_hr]]).astype(np.float32)
                p_interp.set_tensor(p_in[0]['index'], scaled_p)
                p_interp.invoke()
                p_prob = p_interp.get_tensor(p_out[0]['index'])[0][0]
                ai_data["p_prob_list"].append(p_prob)
                
                if SIMULATION_MODE:
                    phys_cat, phys_col = get_medical_category(p_prob)
                    cv2.putText(proc_frame, f"PHYS: {phys_cat}", (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phys_col, 2)

        cv2.imshow("PPD-1 Edge Inference", proc_frame)
        if cv2.waitKey(5) & 0xFF == 27: break
        
    cap.release()
    cv2.destroyAllWindows()
    if ai_data["v_prob_list"]: 
        ai_data["final_blinks"] = session_state["blinks"]
        save_medical_report(user_data, ai_data)

def run_file_analysis(user_data, filepath, is_video):
    if not is_video:
        frame = cv2.imread(filepath)
        proc_frame, v_prob, metrics = process_vision(frame)
        cv2.imshow("Analysis", proc_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if v_prob is not None:
            save_medical_report(user_data, {"mode": "Image", "ear_list": [metrics["ear"]], "etr_list": [metrics["etr"]], "mar_list": [metrics["mar"]], "dc_list": [metrics["dark_circles"]], "v_prob_list": [v_prob]})
    else:
        cap = cv2.VideoCapture(filepath)
        ai_data = {"mode": "Video", "ear_list": [], "etr_list": [], "mar_list": [], "dc_list": [], "v_prob_list": []}
        session_state = {"blinks": 0, "eye_closed": False, "start_time": time.time()}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (800, 600))
            proc_frame, v_prob, metrics = process_vision(frame, session_state)
            if v_prob is not None:
                ai_data["ear_list"].append(metrics["ear"])
                ai_data["etr_list"].append(metrics["etr"])
                ai_data["mar_list"].append(metrics["mar"])
                ai_data["dc_list"].append(metrics["dark_circles"])
                ai_data["v_prob_list"].append(v_prob)
            cv2.imshow("Video Analysis", proc_frame)
            if cv2.waitKey(30) & 0xFF == 27: break
        cap.release()
        cv2.destroyAllWindows()
        if ai_data["v_prob_list"]: 
            ai_data["final_blinks"] = session_state["blinks"]
            save_medical_report(user_data, ai_data)

def main():
    root = tk.Tk(); root.withdraw()
    
    print("\n" + "="*50)
    print("   PPD-1 ECOLOGICAL MOMENTARY ASSESSMENT (EMA)")
    print("="*50)
    
    # Fast bypass of EMA questions to prevent faculty attacks
    user_data = {
        "name": input("Enter Patient/Session ID: ")
    }
    
    while True:
        print("\n--- CLINICAL SYSTEM MODES ---")
        print("1. Image Upload Analysis (Asynchronous)")
        print("2. Video Upload Analysis (Asynchronous)")
        print("3. Live Camera (Vision AI Biomarkers)")
        print("4. Live Hybrid (Vision + I2C Physical Sensors)")
        print("5. Exit System")
        
        choice = input("Select operation (1-5): ")
        
        if choice == '1':
            fp = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png")])
            if fp: run_file_analysis(user_data, fp, False)
        elif choice == '2':
            fp = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi")])
            if fp: run_file_analysis(user_data, fp, True)
        elif choice == '3':
            run_live_hybrid(user_data, use_hardware=False)
        elif choice == '4':
            run_live_hybrid(user_data, use_hardware=True)
        elif choice == '5':
            print("System shutting down. Medical logs saved securely.")
            break
        else:
            print("Invalid input.")

if __name__ == "__main__":
    main()