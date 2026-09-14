import os
import sys
import glob

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath("backend"))
from inference import engine, DISEASE_CLASSES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "Mango S data")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join("backend", "data", "Mango S data")
data_dir = DATA_DIR
classes = [
    "Healthy", "Anthracnose", "Bacterial Canker", "Powdery Mildew",
    "Sooty Mould", "Die Back", "Gall Midge", "Cutting Weevil"
]

results = {}
for cls in classes:
    folder = os.path.join(data_dir, cls)
    imgs = glob.glob(os.path.join(folder, "*.*"))[:25]
    correct = 0
    total = len(imgs)
    preds = []
    multi_count = 0
    for img_path in imgs:
        with open(img_path, "rb") as f:
            res = engine.predict(f.read())
            pred_disease = res["disease"]
            is_healthy = res.get("is_healthy", False)
            is_multi = res.get("is_multiple_diseases", False)
            if is_multi:
                multi_count += 1
            preds.append(pred_disease)
            expected_name = "Sooty Mold" if cls == "Sooty Mould" else cls
            has_expected = (pred_disease == expected_name) or any(d["name"] == expected_name for d in res.get("predicted_diseases", []))
            if (has_expected and not is_healthy and cls != "Healthy") or (cls == "Healthy" and is_healthy):
                correct += 1
    acc = (correct / max(1, total)) * 100
    results[cls] = {
        "acc": acc,
        "correct": correct,
        "total": total,
        "multi_count": multi_count,
        "sample_preds": preds[:5]
    }

print("=" * 70)
print("End-to-End Inference Pipeline Accuracy on Real Images:")
print(f"{'Class Name':<18} | {'Accuracy':<10} | {'Multi Freq':<12} | Sample Predictions")
print("-" * 70)
for cls, r in results.items():
    print(f"{cls:<18} | {r['acc']:>6.1f}%    | {r['multi_count']:>2d}/{r['total']} multi  | {r['sample_preds']}")
print("=" * 70)
