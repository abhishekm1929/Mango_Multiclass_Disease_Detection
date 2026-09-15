import os
import io
import glob
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from inference import engine, DISEASE_CLASSES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "Mango S data")

def get_sample_image(class_folder, index=0):
    """Retrieve sample image for a given class from the dataset."""
    folder_path = os.path.join(DATA_DIR, class_folder)
    images = glob.glob(os.path.join(folder_path, "*.*"))
    if not images:
        raise FileNotFoundError(f"No sample images found in {folder_path}")
    img_idx = index % len(images)
    with open(images[img_idx], "rb") as f:
        return f.read()

def create_sunlit_healthy_specimen():
    """Create a sunlit/high-brightness variant of a real healthy leaf."""
    folder_path = os.path.join(DATA_DIR, "Healthy")
    images = glob.glob(os.path.join(folder_path, "*.*"))
    img = Image.open(images[0]).convert("RGB")
    # Increase brightness & contrast to simulate harsh outdoor sunlight
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.25)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.15)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def create_leaf_on_printed_paper_background():
    """
    Simulates a mango leaf placed on a notebook/printed page with text.
    The system MUST isolate the leaf, reject the text/paper background, and classify the leaf.
    """
    anthracnose_imgs = glob.glob(os.path.join(DATA_DIR, "Anthracnose", "*.*"))
    leaf_img = Image.open(anthracnose_imgs[0]).convert("RGB").resize((220, 300))
    
    # Create white/off-white notebook paper canvas with text lines
    paper = Image.new("RGB", (500, 420), color=(248, 248, 245))
    draw = ImageDraw.Draw(paper)
    # Draw ruled lines and simulated notes
    for y in range(40, 400, 30):
        draw.line([(20, y), (480, y)], fill=(210, 220, 235), width=1)
        draw.text((30, y - 15), "Agronomic Mango Leaf Pathological Sampling Notes - Section 4.2", fill=(45, 50, 60))

    # Paste leaf in the center
    paper.paste(leaf_img, (140, 60))
    buf = io.BytesIO()
    paper.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def create_pure_paper_non_leaf_image():
    """
    Simulates a non-leaf background image (e.g. document/paper with text, no mango leaf).
    The system MUST reject this image and report 'No Mango Leaf Detected'.
    """
    paper = Image.new("RGB", (400, 300), color=(245, 245, 240))
    draw = ImageDraw.Draw(paper)
    for y in range(30, 280, 25):
        draw.line([(15, y), (385, y)], fill=(200, 210, 225), width=1)
        draw.text((25, y - 12), "General agricultural research notebook notes page 14", fill=(30, 35, 45))
    buf = io.BytesIO()
    paper.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def create_multi_disease_composite():
    """
    Synthesize a realistic multi-disease leaf specimen by compositing a real
    Anthracnose diseased patch onto the left side and a real Powdery Mildew
    diseased patch onto the right side of a mango leaf canvas.
    """
    anthracnose_imgs = glob.glob(os.path.join(DATA_DIR, "Anthracnose", "*.*"))
    powdery_imgs = glob.glob(os.path.join(DATA_DIR, "Powdery Mildew", "*.*"))
    healthy_imgs = glob.glob(os.path.join(DATA_DIR, "Healthy", "*.*"))

    canvas = Image.open(healthy_imgs[0]).convert("RGB").resize((600, 450))
    anth_patch = Image.open(anthracnose_imgs[0]).convert("RGB").resize((180, 180))
    pwd_patch = Image.open(powdery_imgs[0]).convert("RGB").resize((180, 180))

    # Paste patches onto canvas in distinct spatial quadrants
    canvas.paste(anth_patch, (40, 120))
    canvas.paste(pwd_patch, (360, 120))

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def calculate_iou(box1, box2):
    """Calculates Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / max(1, union_area)


def run_diagnostic_tests():
    engine.reload_weights()
    print("=" * 75)
    print("MANGO LEAF HIGH-PRECISION INFERENCE & LOCALIZATION DIAGNOSTIC SUITE")
    print(f"Model Engine: {engine.model_version}")
    print(f"Dataset Path: {DATA_DIR}")
    print("=" * 75)

    test_cases = [
        ("Healthy Specimen (Sample 1)", lambda: get_sample_image("Healthy", 0), "Healthy", True, True),
        ("Healthy Specimen (Sample 2)", lambda: get_sample_image("Healthy", 15), "Healthy", True, True),
        ("Harsh Sunlit Healthy Leaf (Glare Invariance)", create_sunlit_healthy_specimen, "Healthy", True, True),
        ("Anthracnose Pathological Specimen", lambda: get_sample_image("Anthracnose", 0), "Anthracnose", False, True),
        ("Bacterial Canker Pathological Specimen", lambda: get_sample_image("Bacterial Canker", 0), "Bacterial Canker", False, True),
        ("Cutting Weevil Specimen", lambda: get_sample_image("Cutting Weevil", 0), "Cutting Weevil", False, True),
        ("Die Back Specimen", lambda: get_sample_image("Die Back", 0), "Die Back", False, True),
        ("Gall Midge Specimen", lambda: get_sample_image("Gall Midge", 0), "Gall Midge", False, True),
        ("Powdery Mildew Specimen", lambda: get_sample_image("Powdery Mildew", 0), "Powdery Mildew", False, True),
        ("Sooty Mold Specimen", lambda: get_sample_image("Sooty Mould", 0), "Sooty Mold", False, True),
        ("Anthracnose Leaf on Book/Printed Paper Background", create_leaf_on_printed_paper_background, "Anthracnose", False, True),
        ("Composite Multi-Disease Leaf (Anthracnose + Powdery Mildew)", create_multi_disease_composite, "multi_disease", False, True),
        ("Pure Paper Non-Leaf Image (Negative Background Rejection)", create_pure_paper_non_leaf_image, "No Mango Leaf Detected", False, False),
    ]

    for name, img_fn, expected_disease, expected_healthy, expect_leaf in test_cases:
        print(f"\n[TEST CASE] {name}")
        img_bytes = img_fn()
        result = engine.predict(img_bytes)

        print(f"  * Primary Disease Detected: {result['disease']} ({result['confidence']}%)")
        print(f"  * Status: {result['status']}")
        print(f"  * Risk Level: {result['risk']}")
        print(f"  * Is Leaf Detected: {result.get('is_leaf_detected', False)}")
        print(f"  * Is Multiple Diseases: {result['is_multiple_diseases']}")
        
        det_list = [f"{d['name']} ({d.get('cnn_confidence', d['confidence'])}%)" for d in result['predicted_diseases']]
        print(f"  * Detected Disease List ({len(result['predicted_diseases'])}): {det_list}")
        print(f"  * Localized Bounding Box Count: {len(result['detections'])}")
        for idx, det in enumerate(result['detections']):
            print(f"     -> Box {idx+1}: [{det['disease']}] Conf={det.get('confidence', det.get('cnn_confidence'))}% @ bbox={det['bbox']}")

        # Non-overlap check: Verify no two bounding boxes have excessive IoU overlap
        dets = result["detections"]
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                iou = calculate_iou(dets[i]["bbox"], dets[j]["bbox"])
                assert iou < 0.45, f"Boxes {i+1} and {j+1} overlap excessively (IoU: {iou:.2f}) in {name}"

        # Assertions
        if not expect_leaf:
            assert result.get("is_leaf_detected") is False or result["disease"] == "No Mango Leaf Detected", f"{name} should reject non-leaf image"
            assert len(result["detections"]) == 0, f"{name} should produce 0 bounding boxes"
        elif expected_healthy:
            assert result["is_healthy"] is True, f"{name} should be identified as healthy"
            assert result["disease"] == "Healthy", f"{name} should have primary disease 'Healthy'"
            assert result["is_multiple_diseases"] is False, f"{name} healthy leaf must set is_multiple_diseases=False"
            assert len(result["detections"]) == 0, f"{name} should have 0 disease detections, got {len(result['detections'])}"
        elif expected_disease == "multi_disease":
            assert len(result["detections"]) >= 2, "Multi-disease leaf must localize at least 2 distinct regions"
            assert result["is_multiple_diseases"] is True, "Multi-disease specimen must set is_multiple_diseases=True"
            assert len(result["predicted_diseases"]) >= 2, "Multi-disease specimen must report at least 2 distinct diseases"
        else:
            assert result["is_healthy"] is False, f"{expected_disease} should not be marked healthy"
            assert result["is_multiple_diseases"] is False, f"Single-disease specimen '{name}' must have is_multiple_diseases=False"
            assert len(result["predicted_diseases"]) == 1, f"Single-disease specimen '{name}' must report only 1 disease, got: {[d['name'] for d in result['predicted_diseases']]}"
            assert result["disease"].lower() == expected_disease.lower() or any(d["name"].lower() == expected_disease.lower() for d in result["predicted_diseases"]), f"Expected {expected_disease}, got {result['disease']}"

    print("\n" + "=" * 75)
    print("ALL 13 HIGH-PRECISION REAL-IMAGE & BACKGROUND-INVARIANT TESTS PASSED!")
    print("=" * 75)

if __name__ == "__main__":
    run_diagnostic_tests()
