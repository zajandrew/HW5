import xgboost as xgb
import numpy as np
import ctypes

# 1. Apply your fix
try:
    ctypes.WinDLL("nvrtc-builtins64_129.dll")
    print("DLL Loaded.")
except:
    print("DLL Load Failed (ignoring if testing).")

# 2. Create dummy data
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)
dtrain = xgb.DMatrix(X, label=y)

# 3. Try to train on GPU
params = {
    'tree_method': 'hist',
    'device': 'cuda',  # This forces GPU. If it fails, it will crash here.
    'max_bin': 256
}

try:
    print("Attempting XGBoost training on GPU...")
    model = xgb.train(params, dtrain, num_boost_round=10)
    print("\nSUCCESS: XGBoost successfully trained on your RTX A4000!")
except xgb.core.XGBoostError as e:
    print("\nFAILURE: XGBoost could not access the GPU.")
    print(f"Error details: {e}")
