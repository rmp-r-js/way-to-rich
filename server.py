import os
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path

# ==========================================
# 📁 MODEL PATHS
# ==========================================
TRAP_MODEL_PATH = Path("models/wingo_trap_hunter.joblib")
ENSEMBLE_MODEL_PATH = Path("models/wingo_never_give_up_ensemble.joblib")

# Globals
TRAP_MODEL = None
ENSEMBLE_BUNDLE = {}

# ==========================================
# 🧠 FEATURE ENGINEERING (Trap Hunter)
# ==========================================
def build_trap_features(numbers_list):
    sizes = [1 if n >= 5 else 0 for n in numbers_list]
    past = np.array(sizes[-30:])
    
    big_ratio_30 = np.mean(past)
    big_ratio_10 = np.mean(past[-10:])
    big_ratio_5 = np.mean(past[-5:])
    
    current_streak = 0
    last_val = past[-1]
    for val in reversed(past):
        if val == last_val: current_streak += 1
        else: break
            
    threshold_met = 1 if (big_ratio_30 >= 0.65 or big_ratio_30 <= 0.35) else 0
    return np.array([[big_ratio_30, big_ratio_10, big_ratio_5, current_streak, threshold_met]])

# ==========================================
# 🧠 FEATURE ENGINEERING (Never Give Up Ensemble)
# ==========================================
def build_ensemble_features(numbers_list):
    size_bin = [1 if n >= 5 else 0 for n in numbers_list]
    color_bin = [1 if n % 2 == 0 else 0 for n in numbers_list]
    i = len(size_bin)
    features = []
    
    # A. Rolling Means
    for w in [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
        window = size_bin[max(0, i-w):i] if i >= w else size_bin[:i]
        features.append(np.mean(window) if len(window) > 0 else 0.5)

    # B. Rolling Stds
    for w in [5, 10, 20, 30, 50]:
        window = size_bin[max(0, i-w):i] if i >= w else size_bin[:i]
        features.append(np.std(window) if len(window) > 1 else 0)

    # C. Lags
    for lag in range(1, 16):
        features.append(size_bin[i - lag] if i - lag >= 0 else 0.5)

    # D. Streak Features
    if i > 0:
        streak_type = size_bin[i-1]
        streak_len = 0
        for j in range(i-1, -1, -1):
            if size_bin[j] == streak_type: streak_len += 1
            else: break
        features.extend([streak_type, streak_len, streak_len * (1 if streak_type == 1 else -1)])
    else:
        features.extend([0, 0, 0])

    # E. Gaps
    gap_big, gap_small = 0, 0
    for j in range(i-1, -1, -1):
        if size_bin[j] == 1 and gap_big == 0: gap_big = i - j
        if size_bin[j] == 0 and gap_small == 0: gap_small = i - j
        if gap_big > 0 and gap_small > 0: break
    features.extend([gap_big if gap_big > 0 else i, gap_small if gap_small > 0 else i])

    # F. Transitions
    for w in [10, 20, 50]:
        window = size_bin[max(0, i-w):i] if i >= w else size_bin[:i]
        if len(window) > 1:
            transitions = sum(1 for k in range(1, len(window)) if window[k] != window[k-1])
            features.append(transitions / len(window))
        else: features.append(0.5)

    # G. Color Features
    for w in [10, 20, 30]:
        window_c = color_bin[max(0, i-w):i] if i >= w else color_bin[:i]
        features.append(np.mean(window_c) if len(window_c) > 0 else 0.5)

    # H. Interactions
    mean_10 = np.mean(size_bin[max(0, i-10):i]) if i > 0 else 0.5
    mean_30 = np.mean(size_bin[max(0, i-30):i]) if i > 0 else 0.5
    mean_5 = np.mean(size_bin[max(0, i-5):i]) if i > 0 else 0.5
    features.extend([mean_10 - mean_30, mean_5 - mean_10, mean_10 * mean_30, abs(mean_10 - 0.5)])

    # I. Thresholds
    features.extend([
        1 if mean_30 >= 0.65 else 0, 1 if mean_30 <= 0.35 else 0,
        1 if mean_10 >= 0.6 else 0, 1 if mean_10 <= 0.4 else 0
    ])

    # J. Entropy
    window_30 = size_bin[max(0, i-30):i] if i > 0 else size_bin[:i]
    if len(window_30) > 1:
        p_big = np.mean(window_30)
        p_small = 1 - p_big
        entropy = -(p_big * np.log2(p_big) + p_small * np.log2(p_small)) if p_big > 0 and p_small > 0 else 0
        features.append(entropy)
    else: features.append(0.5)

    # K. Numbers
    if i > 0:
        last_num = numbers_list[i-1]
        features.extend([last_num, 1 if last_num >= 5 else 0, 1 if last_num % 2 == 0 else 0])
    else: features.extend([4, 0, 1])

    return np.array([features])

# ==========================================
# 🚀 FASTAPI SETUP
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global TRAP_MODEL, ENSEMBLE_BUNDLE
    
    # Load Trap Model
    if TRAP_MODEL_PATH.exists():
        print(f"⏳ Loading Trap Hunter Model: {TRAP_MODEL_PATH}...")
        TRAP_MODEL = joblib.load(TRAP_MODEL_PATH)
        print("✅ Trap Hunter Model loaded!")
    else:
        print(f"⚠️ Trap Hunter Model not found at {TRAP_MODEL_PATH}")

    # Load Ensemble Model
    if ENSEMBLE_MODEL_PATH.exists():
        print(f"⏳ Loading Never Give Up Ensemble: {ENSEMBLE_MODEL_PATH}...")
        ENSEMBLE_BUNDLE = joblib.load(ENSEMBLE_MODEL_PATH)
        print(f"✅ Ensemble Model loaded! (Accuracy: {ENSEMBLE_BUNDLE.get('accuracy', 0)*100:.2f}%)")
    else:
        print(f"⚠️ Ensemble Model not found at {ENSEMBLE_MODEL_PATH}")

    yield
    TRAP_MODEL = None
    ENSEMBLE_BUNDLE.clear()

app = FastAPI(title="WinGo Combined API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    numbers: List[int]

@app.get("/")
def root():
    return {
        "status": "Online",
        "models_loaded": {
            "trap_hunter": TRAP_MODEL is not None,
            "ensemble": bool(ENSEMBLE_BUNDLE)
        },
        "endpoints": {
            "trap_hunter": "POST /predict/trap",
            "ensemble": "POST /predict/ensemble"
        }
    }

@app.get("/health")
def health():
    return {"ok": True}

# ==========================================
# 🎯 ENDPOINT 1: TRAP HUNTER (Requires 30 numbers)
# ==========================================
@app.post("/predict/trap")
def predict_trap(body: PredictRequest):
    if TRAP_MODEL is None:
        raise HTTPException(status_code=503, detail="Trap Hunter model not loaded on server.")
    if len(body.numbers) < 30:
        raise HTTPException(status_code=400, detail="At least 30 numbers required for Trap Hunter.")
        
    try:
        X = build_trap_features(body.numbers)
        pred = int(TRAP_MODEL.predict(X)[0])
        proba = TRAP_MODEL.predict_proba(X)[0]
        
        return {
            "model": "trap_hunter",
            "prediction": "big" if pred == 1 else "small",
            "confidence": round(float(max(proba)), 4),
            "probabilities": {
                "small": round(float(proba[0]), 4),
                "big": round(float(proba[1]), 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 🎯 ENDPOINT 2: ENSEMBLE (Requires 100 numbers)
# ==========================================
@app.post("/predict/ensemble")
def predict_ensemble(body: PredictRequest):
    if not ENSEMBLE_BUNDLE:
        raise HTTPException(status_code=503, detail="Ensemble model not loaded on server.")
    if len(body.numbers) < 100:
        raise HTTPException(status_code=400, detail="At least 100 numbers required for Ensemble.")
        
    try:
        X_raw = build_ensemble_features(body.numbers)
        X_scaled = ENSEMBLE_BUNDLE['scaler'].transform(X_raw)
        
        rf_p = ENSEMBLE_BUNDLE['rf'].predict_proba(X_scaled)[0][1]
        xgb_p = ENSEMBLE_BUNDLE['xgb'].predict_proba(X_scaled)[0][1]
        cat_p = ENSEMBLE_BUNDLE['cat'].predict_proba(X_scaled)[0][1]
        nn_p = ENSEMBLE_BUNDLE['nn'].predict_proba(X_scaled)[0][1]
        
        ensemble_p = (rf_p + xgb_p + cat_p + nn_p) / 4
        prediction = 1 if ensemble_p > 0.5 else 0
        
        return {
            "model": "never_give_up_ensemble",
            "prediction": "big" if prediction == 1 else "small",
            "confidence": round(max(ensemble_p, 1-ensemble_p), 4),
            "probabilities": {
                "small": round(1-ensemble_p, 4),
                "big": round(ensemble_p, 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))