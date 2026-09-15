import os
import glob
import time
import requests
import json

classes = ['Healthy', 'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Powdery Mildew', 'Sooty Mould']
data_dir = r'backend/data/Mango S data'

print('--- Testing Live Render Backend API (https://mango-multiclass-disease-detection.onrender.com/predict) ---')
total = 0
correct = 0
results = {}

for cls in classes:
    pattern = os.path.join(data_dir, f'{cls}*', '*.jpg')
    imgs = glob.glob(pattern)
    if not imgs:
        pattern = os.path.join(data_dir, f'{cls}*', '*.png')
        imgs = glob.glob(pattern)
    if not imgs:
        print(f'No images found for {cls}')
        continue
    
    # Test 2 representative images per class
    cls_correct = 0
    cls_total = 0
    for img_path in imgs[:2]:
        time.sleep(1.2)
        try:
            with open(img_path, 'rb') as f:
                r = requests.post(
                    'https://mango-multiclass-disease-detection.onrender.com/predict',
                    files={'file': ('image.jpg', f, 'image/jpeg')},
                    timeout=30
                )
        except Exception as e:
            print(f"  [ERROR] {cls} -> Request exception: {e}")
            continue

        total += 1
        cls_total += 1
        if r.status_code == 200:
            res = r.json()
            pred_disease = res.get('disease', '')
            is_healthy = res.get('is_healthy', False)
            is_multi = res.get('is_multiple_diseases', False)
            conf = res.get('confidence', 0.0)
            boxes = len(res.get('detections', []))
            
            if cls == 'Healthy':
                passed = is_healthy and ('healthy' in pred_disease.lower()) and not is_multi
            elif cls == 'Sooty Mould':
                passed = ('sooty' in pred_disease.lower()) and not is_multi
            else:
                passed = (cls.lower() in pred_disease.lower()) and not is_multi
            
            if passed:
                correct += 1
                cls_correct += 1
                print(f"  [PASS] {cls} -> {pred_disease} | Conf: {conf:.1f}% | Multi: {is_multi} | Boxes: {boxes}")
            else:
                print(f"  [FAIL] {cls} -> {pred_disease} | Conf: {conf:.1f}% | Multi: {is_multi} | Boxes: {boxes}")
        else:
            print(f"  [ERROR] {cls} -> HTTP {r.status_code}: {r.text[:100]}")
    results[cls] = f"{cls_correct}/{cls_total}"

print(f"\nLive Render Backend Summary:")
for k, v in results.items():
    print(f"  {k}: {v}")
print(f"Total Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
