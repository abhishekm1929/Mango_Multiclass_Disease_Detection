import os
import glob
import requests
from inference import engine

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "Mango S data")
classes = ['Healthy', 'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Powdery Mildew', 'Sooty Mould']

print(f"{'Class':<18} | {'Image File':<20} | {'Local Result':<26} | {'Live Result':<26} | Match?")
print("-" * 105)

for c in classes:
    imgs = glob.glob(os.path.join(DATA_DIR, c, "*.*"))
    if not imgs:
        continue
    img_path = imgs[0]
    img_name = os.path.basename(img_path)
    
    with open(img_path, "rb") as f:
        file_bytes = f.read()

    # Local inference
    local_res = engine.predict(file_bytes)
    local_str = f"{local_res['disease']} ({local_res['confidence']}%)"

    # Live inference (with retry for network stability)
    live_res = {}
    for attempt in range(2):
        files = {'file': (img_name, file_bytes, 'image/jpeg')}
        try:
            r = requests.post('https://mango-multiclass-disease-detection.onrender.com/predict', files=files, timeout=45)
            if r.status_code == 200:
                live_res = r.json()
                live_str = f"{live_res['disease']} ({live_res['confidence']}%)"
                break
            else:
                live_str = f"HTTP {r.status_code}"
        except Exception as e:
            live_str = f"Error: {e}"

    exp = 'Sooty Mold' if c == 'Sooty Mould' else c
    match = (local_res['disease'] == exp) and (live_res.get('disease') == exp)
    print(f"{c:<18} | {img_name:<20} | {local_str:<26} | {live_str:<26} | {'YES' if match else 'NO'}")
