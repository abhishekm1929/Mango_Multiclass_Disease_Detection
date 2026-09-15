import io
import time
import os
import numpy as np
from PIL import Image, ImageOps

try:
    from detector import YOLOv8Detector
    from classifier import (
        EfficientNetMangoClassifier,
        DISEASE_CLASSES,
        CLASS_MAP,
        CLASS_NAMES,
        NAME_TO_META,
        INDEX_TO_CLASS,
        ID_TO_INDEX
    )
except ImportError:
    from backend.detector import YOLOv8Detector
    from backend.classifier import (
        EfficientNetMangoClassifier,
        DISEASE_CLASSES,
        CLASS_MAP,
        CLASS_NAMES,
        NAME_TO_META,
        INDEX_TO_CLASS,
        ID_TO_INDEX
    )


class MangoLeafInferenceEngine:
    """
    Two-Stage Deep Learning Inference Pipeline for Mango Leaf Disease Diagnosis:
    1. EfficientNet-B0 Whole-Leaf Neural Classification & Multi-Class Activation Mapping (CAM).
    2. High-Precision Spatial Lesion Localization (CAM + Morphological Contours + Strict NMS).
    3. Independent Crop Patch Verification via CNN Forward Passes.
    4. Multi-Disease vs Single-Disease vs Healthy Diagnostic Resolution.
    """
    def __init__(self, models_dir=None, cnn_weights_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if models_dir is None:
            models_dir = os.path.join(base_dir, "models")
        if cnn_weights_path is None:
            cnn_weights_path = os.path.join(base_dir, "models", "mango_cnn_efficientnet.pth")

        self.models_dir = models_dir
        self.cnn_weights_path = cnn_weights_path

        # Stage 1 & 3: CNN Classifier & Saliency Engine
        self.classifier = EfficientNetMangoClassifier(weights_path=self.cnn_weights_path)

        # Stage 2: Spatial Object & Lesion Detector
        self.detector = YOLOv8Detector(models_dir=self.models_dir)

        self.model_version = f"{self.detector.model_version} + {self.classifier.model_name}"

    def reload_weights(self):
        """Reloads trained model weights if updated."""
        self.classifier = EfficientNetMangoClassifier(weights_path=self.cnn_weights_path)
        self.detector = YOLOv8Detector(models_dir=self.models_dir)
        self.model_version = f"{self.detector.model_version} + {self.classifier.model_name}"

    def preprocess_image(self, file_bytes, max_dim=800):
        """Decode, auto-rotate EXIF, and downscale to efficient dimension for fast CPU inference & low RAM."""
        try:
            raw_img = Image.open(io.BytesIO(file_bytes))
            pil_img = ImageOps.exif_transpose(raw_img).convert("RGB")
            w, h = pil_img.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                new_w = max(16, int(w * scale))
                new_h = max(16, int(h * scale))
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            np_rgb = np.array(pil_img)
            return np_rgb, pil_img.size
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image format: {str(e)}")

    def predict(self, file_bytes):
        """
        Executes end-to-end multi-region inference:
          Image -> Whole-Leaf CNN & CAM -> Saliency Lesion Detection -> Crop Verification -> Resolution
        """
        start_time = time.time()
        np_rgb, (w_orig, h_orig) = self.preprocess_image(file_bytes)

        # ---------------------------------------------------------------------
        # STAGE 1: Global Whole-Leaf Neural Classification & CAM Saliency
        # ---------------------------------------------------------------------
        whole_leaf_cnn = self.classifier.classify_crop(np_rgb)
        leaf_disease = whole_leaf_cnn["disease"]
        leaf_id = whole_leaf_cnn["disease_id"]
        leaf_conf = whole_leaf_cnn["cnn_confidence"]

        # Generate Class Activation Map (CAM) heatmaps for top classes
        cam_heatmaps = []
        top1_idx = ID_TO_INDEX.get(leaf_id, 0)
        cam1, _ = self.classifier.generate_saliency_cam(np_rgb, target_class_idx=top1_idx)
        if cam1 is not None:
            cam_heatmaps.append(cam1)

        # Check for top secondary candidate classes in whole-leaf distribution
        sorted_dist = sorted(
            [(c_name, c_prob) for c_name, c_prob in whole_leaf_cnn["distribution"].items() if c_name != "Healthy" and c_name != leaf_disease],
            key=lambda x: x[1],
            reverse=True
        )
        for top_name, top_prob in sorted_dist[:2]:
            if top_prob >= 5.0:
                top_id = NAME_TO_META.get(top_name, {}).get("id")
                top_idx = ID_TO_INDEX.get(top_id, 0)
                cam_sec, _ = self.classifier.generate_saliency_cam(np_rgb, target_class_idx=top_idx)
                if cam_sec is not None:
                    cam_heatmaps.append(cam_sec)

        # ---------------------------------------------------------------------
        # STAGE 2: Spatial Lesion Localization (CAM + Morphological Contours)
        # ---------------------------------------------------------------------
        raw_detections = []
        if leaf_id != "healthy" or leaf_conf < 80.0:
            raw_detections = self.detector.detect_regions(
                np_rgb,
                cam_heatmaps=cam_heatmaps,
                target_disease=leaf_id
            )

        # ---------------------------------------------------------------------
        # STAGE 3: Crop EACH Detected Bounding Box & Classify Independently
        # ---------------------------------------------------------------------
        enriched_detections = []
        for idx, det in enumerate(raw_detections):
            x1, y1, x2, y2 = det["bbox"]
            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < 16 or box_h < 16:
                continue

            pad = 6
            cx1 = max(0, int(x1) - pad)
            cy1 = max(0, int(y1) - pad)
            cx2 = min(w_orig, int(x2) + pad)
            cy2 = min(h_orig, int(y2) + pad)

            crop_patch = np_rgb[cy1:cy2, cx1:cx2]
            if crop_patch.shape[0] < 12 or crop_patch.shape[1] < 12:
                continue

            cnn_result = self.classifier.classify_crop(crop_patch)
            crop_disease = cnn_result["disease"]
            crop_id = cnn_result["disease_id"]
            crop_conf = cnn_result["cnn_confidence"]

            if crop_id == "healthy" or crop_conf < 50.0:
                continue

            disease_meta = CLASS_MAP.get(crop_id, cnn_result)

            enriched_detections.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "relative_bbox": det.get("relative_bbox", [
                    round(x1 / w_orig, 4),
                    round(y1 / h_orig, 4),
                    round(x2 / w_orig, 4),
                    round(y2 / h_orig, 4)
                ]),
                "area": int(det.get("area", box_w * box_h)),
                "disease": crop_disease,
                "disease_id": crop_id,
                "scientific_name": disease_meta.get("scientific_name", ""),
                "category": disease_meta.get("category", ""),
                "risk": disease_meta.get("risk", "Moderate"),
                "confidence": crop_conf,
                "cnn_confidence": crop_conf,
                "distribution": cnn_result.get("distribution", {})
            })

        # ---------------------------------------------------------------------
        # STAGE 4: Diagnostic Resolution (Healthy vs Single-Disease vs Multi-Disease)
        # ---------------------------------------------------------------------
        # Case 1: Healthy Specimen (Whole leaf is healthy and no strong high-confidence lesion was confirmed)
        if leaf_id == "healthy" and not [d for d in enriched_detections if d["cnn_confidence"] >= 80.0]:
            is_healthy = True
            is_multiple = False
            primary_disease = "Healthy"
            primary_id = "healthy"
            primary_conf = leaf_conf
            status = "Healthy Specimen"
            risk = "None"
            summary = "The leaf exhibits healthy, uniform pigmentation and laminar structure with no active pathological lesions detected."
            predicted_diseases = [{
                "name": "Healthy",
                "disease": "Healthy",
                "confidence": primary_conf,
                "disease_id": "healthy",
                "cnn_confidence": primary_conf,
                "risk": "None",
                "count": 0
            }]
            final_detections = []

        # Case 2: Disease Present
        else:
            is_healthy = False
            confirmed_diseases = {}

            # 1. Register whole-leaf predicted pathology as primary confirmed disease
            if leaf_id != "healthy":
                leaf_meta = CLASS_MAP.get(leaf_id, {})
                confirmed_diseases[leaf_id] = {
                    "name": leaf_disease,
                    "disease": leaf_disease,
                    "disease_id": leaf_id,
                    "confidence": round(float(leaf_conf), 1),
                    "cnn_confidence": round(float(leaf_conf), 1),
                    "risk": leaf_meta.get("risk", "Moderate"),
                    "scientific_name": leaf_meta.get("scientific_name", ""),
                    "category": leaf_meta.get("category", ""),
                    "count": 0,
                    "boxes": []
                }

            # 2. Register localized boxes matching primary or genuine secondary diseases
            valid_detections = []
            for d in enriched_detections:
                d_id = d["disease_id"]
                d_name = d["disease"]
                d_conf = d["confidence"]

                # Case A: Box matches the primary whole-leaf disease
                if d_id == leaf_id:
                    valid_detections.append(d)
                    if d_id in confirmed_diseases:
                        confirmed_diseases[d_id]["count"] += 1
                        confirmed_diseases[d_id]["boxes"].append(d["bbox"])
                        if d_conf > confirmed_diseases[d_id]["confidence"]:
                            confirmed_diseases[d_id]["confidence"] = d_conf
                            confirmed_diseases[d_id]["cnn_confidence"] = d_conf

                # Case B: Box proposes a SECONDARY disease (requires strong evidence >= 82% AND non-trivial global activation)
                else:
                    global_secondary_prob = float(whole_leaf_cnn["distribution"].get(d_name, 0.0))
                    # Secondary disease confirmed ONLY if high local confidence and non-trivial support
                    if d_conf >= 82.0 and (global_secondary_prob >= 3.0 or d_conf >= 90.0):
                        valid_detections.append(d)
                        if d_id not in confirmed_diseases:
                            confirmed_diseases[d_id] = {
                                "name": d_name,
                                "disease": d_name,
                                "disease_id": d_id,
                                "confidence": d_conf,
                                "cnn_confidence": d_conf,
                                "risk": d["risk"],
                                "scientific_name": d.get("scientific_name", ""),
                                "category": d.get("category", ""),
                                "count": 1,
                                "boxes": [d["bbox"]]
                            }
                        else:
                            confirmed_diseases[d_id]["count"] += 1
                            confirmed_diseases[d_id]["boxes"].append(d["bbox"])
                            if d_conf > confirmed_diseases[d_id]["confidence"]:
                                confirmed_diseases[d_id]["confidence"] = d_conf
                                confirmed_diseases[d_id]["cnn_confidence"] = d_conf
                    elif d_conf >= 60.0 and leaf_id != "healthy":
                        # If the crop had lower confidence on secondary, re-assign box to the confirmed primary pathology
                        d_remapped = dict(d)
                        d_remapped["disease"] = leaf_disease
                        d_remapped["disease_id"] = leaf_id
                        leaf_meta = CLASS_MAP.get(leaf_id, {})
                        d_remapped["scientific_name"] = leaf_meta.get("scientific_name", "")
                        d_remapped["category"] = leaf_meta.get("category", "")
                        d_remapped["risk"] = leaf_meta.get("risk", "Moderate")
                        valid_detections.append(d_remapped)
                        if leaf_id in confirmed_diseases:
                            confirmed_diseases[leaf_id]["count"] += 1
                            confirmed_diseases[leaf_id]["boxes"].append(d_remapped["bbox"])

            final_detections = valid_detections

            # If no local box was found but whole-leaf is diseased, create canonical leaf focus box
            if not final_detections and confirmed_diseases:
                top_k = list(confirmed_diseases.keys())[0]
                top_d = confirmed_diseases[top_k]
                final_detections.append({
                    "x1": int(w_orig * 0.12),
                    "y1": int(h_orig * 0.12),
                    "x2": int(w_orig * 0.88),
                    "y2": int(h_orig * 0.88),
                    "bbox": [int(w_orig * 0.12), int(h_orig * 0.12), int(w_orig * 0.88), int(h_orig * 0.88)],
                    "relative_bbox": [0.12, 0.12, 0.88, 0.88],
                    "area": int(w_orig * h_orig * 0.58),
                    "disease": top_d["name"],
                    "disease_id": top_d["disease_id"],
                    "scientific_name": top_d.get("scientific_name", ""),
                    "category": top_d.get("category", ""),
                    "risk": top_d.get("risk", "Moderate"),
                    "confidence": top_d["confidence"],
                    "cnn_confidence": top_d["confidence"]
                })
                top_d["count"] = 1

            predicted_diseases = sorted(confirmed_diseases.values(), key=lambda x: x["confidence"], reverse=True)
            is_multiple = len(predicted_diseases) > 1

            if is_multiple:
                primary_disease = "Multiple Diseases Detected"
                primary_id = "multiple-diseases"
                primary_conf = round(float(np.mean([d["confidence"] for d in predicted_diseases])), 1)
                status = "Multiple Diseases Detected"
                dis_summary_list = ", ".join([f"{d['name']} ({d['confidence']}%)" for d in predicted_diseases])
                summary = f"Multiple distinct foliar pathologies detected across the leaf blade: {dis_summary_list}. Each affected region was independently localized and verified."
                risk = "High" if any(d.get("risk") == "High" for d in predicted_diseases) else "Moderate"
            else:
                top_d = predicted_diseases[0] if predicted_diseases else {
                    "name": leaf_disease, "disease_id": leaf_id, "confidence": leaf_conf
                }
                primary_disease = top_d["name"]
                primary_id = top_d["disease_id"]
                primary_conf = top_d["confidence"]
                meta = CLASS_MAP.get(primary_id, {})
                status = meta.get("status", "Disease Detected")
                risk = meta.get("risk", "Moderate")
                summary = f"Localized foliar pathology identified {len(final_detections)} lesion region(s) and diagnosed {primary_disease} with {primary_conf}% confidence."

        # Calibrate all_predictions distribution consistently from model
        all_predictions = sorted(
            [{"name": name, "confidence": round(float(conf), 2), "isTop": (name == primary_disease or any(d["name"] == name for d in predicted_diseases))} 
             for name, conf in whole_leaf_cnn["distribution"].items()],
            key=lambda x: x["confidence"],
            reverse=True
        )

        exec_time = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "disease": primary_disease,
            "disease_id": primary_id,
            "confidence": primary_conf,
            "status": status,
            "risk": risk,
            "is_healthy": is_healthy,
            "is_multiple_diseases": is_multiple,
            "predicted_diseases": predicted_diseases,
            "all_predictions": all_predictions,
            "predictions": all_predictions,
            "detections": final_detections,
            "summary": summary,
            "execution_time_ms": exec_time,
            "executionTimeMs": exec_time,
            "model_version": self.model_version,
            "modelVersion": self.model_version
        }


engine = MangoLeafInferenceEngine()
