import os
import glob
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

try:
    from leaf_segmenter import segmenter
except ImportError:
    from backend.leaf_segmenter import segmenter


class YOLOv8Detector:
    """
    High-Precision Mango Leaf Lesion Localization Engine.
    Combines botanical leaf foreground segmentation, deep Class Activation Mapping (CAM),
    morphological pathology segmentation across distinct disease phenotypes,
    and strict Non-Maximum Suppression (NMS) to precisely locate diseased areas.
    """
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.models_dir = os.path.join(base_dir, "models")
        else:
            self.models_dir = models_dir

        self.yolo_model = None
        self.model_version = "YOLOv8-Localization-Engine"
        self._load_yolo_model()

    def _load_yolo_model(self):
        """Discovers custom trained YOLO weights if present."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            return

        pt_files = glob.glob(os.path.join(self.models_dir, "*mango*.pt"))
        if pt_files and HAS_ULTRALYTICS:
            try:
                self.yolo_model = YOLO(pt_files[0])
                self.model_version = f"YOLO-{os.path.basename(pt_files[0])}"
                print(f"[YOLO Detector] Initialized detection engine with: {pt_files[0]}")
            except Exception as e:
                print(f"[YOLO Detector] Error initializing YOLO model: {e}")
        else:
            self.yolo_model = None
            self.model_version = "CAM-Morphology-Localization"

    def detect_regions(self, np_rgb, cam_heatmaps=None, target_disease=None, leaf_mask=None):
        """
        Detects candidate pathological lesion regions strictly within the leaf lamina.
        Returns a clean list of non-overlapping dicts:
          - 'bbox': [x1, y1, x2, y2]
          - 'relative_bbox': [rx1, ry1, rx2, ry2]
          - 'confidence': float (percentage)
          - 'area': int
        """
        h_img, w_img = np_rgb.shape[:2]

        # 0. If leaf mask not provided, segment leaf foreground
        if leaf_mask is None:
            seg_res = segmenter.segment(np_rgb)
            leaf_mask = seg_res["leaf_mask"]
            if not seg_res["is_leaf_present"]:
                # Pure background / non-leaf image -> return 0 detections
                return []

        raw_candidates = []

        # 1. CAM Saliency Contours
        if cam_heatmaps is not None and HAS_CV2:
            if isinstance(cam_heatmaps, list):
                for cam in cam_heatmaps:
                    if cam is not None:
                        raw_candidates.extend(self._extract_cam_regions(cam, w_img, h_img, leaf_mask))
            elif isinstance(cam_heatmaps, np.ndarray):
                raw_candidates.extend(self._extract_cam_regions(cam_heatmaps, w_img, h_img, leaf_mask))

        # 2. Individual Botanical Pathology Morphological Lesion Masks
        morph_dets = self._detect_pathology_morphology(np_rgb, w_img, h_img, target_disease, leaf_mask)
        raw_candidates.extend(morph_dets)

        # 3. Optional YOLO proposals
        if self.yolo_model is not None:
            raw_candidates.extend(self._detect_with_yolo(np_rgb, w_img, h_img))

        if not raw_candidates:
            return []

        # 4. Filter all boxes to lie strictly on the leaf blade (reject paper, table, ground)
        leaf_filtered = segmenter.filter_boxes_to_leaf(raw_candidates, leaf_mask, min_overlap=0.60)

        # 5. Apply strict Non-Maximum Suppression (IoU 0.35)
        clean_detections = self.apply_nms(leaf_filtered, iou_threshold=0.35)

        return clean_detections[:10]

    def _extract_cam_regions(self, cam_heatmap, w_img, h_img, leaf_mask=None):
        """Extracts bounding boxes from neural class activation heatmaps."""
        detections = []
        try:
            cam_resized = cv2.resize(cam_heatmap, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            
            # Mask CAM with leaf mask so activations on background are zeroed out
            if leaf_mask is not None:
                cam_resized[leaf_mask == 0] = 0.0

            thresh_val = max(0.35, float(np.percentile(cam_resized[cam_resized > 0] if np.any(cam_resized > 0) else cam_resized, 70)))
            binary_mask = (cam_resized >= thresh_val).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_area = w_img * h_img
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < (total_area * 0.008) or area > (total_area * 0.65):
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 16 or h < 16:
                    continue

                pad_x = int(w * 0.08)
                pad_y = int(h * 0.08)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w_img, x + w + pad_x)
                y2 = min(h_img, y + h + pad_y)

                mean_act = float(np.mean(cam_resized[y1:y2, x1:x2])) * 100.0

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "relative_bbox": [
                        round(x1 / w_img, 4),
                        round(y1 / h_img, 4),
                        round(x2 / w_img, 4),
                        round(y2 / h_img, 4)
                    ],
                    "confidence": round(min(98.0, max(75.0, mean_act)), 1),
                    "area": int((x2 - x1) * (y2 - y1)),
                    "source": "cam"
                })
        except Exception:
            pass

        return detections

    def _detect_pathology_morphology(self, np_rgb, w_img, h_img, target_disease=None, leaf_mask=None):
        """
        Adaptive multi-phenotype lesion contour extraction strictly within leaf lamina.
        Extracts necrotic spots, canker halos, powdery mycelium, sooty molds, gall pustules,
        cutting damage, and margin die-back desiccation.
        """
        if not HAS_CV2:
            return []

        total_pixels = h_img * w_img
        r = np_rgb[:, :, 0].astype(np.float32)
        g = np_rgb[:, :, 1].astype(np.float32)
        b = np_rgb[:, :, 2].astype(np.float32)
        brightness = 0.299 * r + 0.587 * g + 0.114 * b

        img_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        s_channel = img_hsv[:, :, 1].astype(np.float32)
        v_channel = img_hsv[:, :, 2].astype(np.float32)

        # Restrict strictly to inside the segmented leaf
        if leaf_mask is not None:
            is_leaf_blade = (leaf_mask > 0)
        else:
            is_leaf_blade = (v_channel < 235) & (s_channel > 15)

        # 1. Anthracnose (Dark necrotic spots & shot holes)
        dark_necrotic = (
            (((brightness < 80) & (brightness > 8)) |
             ((r > b + 10) & (r > g - 12) & (brightness < 170) & (s_channel > 20))) &
            is_leaf_blade
        )

        # 2. Bacterial Canker (Angular water-soaked lesions + yellow chlorotic halos)
        chlorotic_canker = (
            ((r > 120) & (g > 95) & (b < 115) & (r > b + 15) & (brightness > 40) & (brightness < 210)) &
            is_leaf_blade
        )

        # 3. Powdery Mildew (Whitish mycelial fungal coating on leaf surface)
        powdery = (
            (brightness > 160) & (brightness <= 240) &
            (np.abs(r - g) < 22) & (np.abs(g - b) < 22) &
            (s_channel < 48) & (s_channel > 10) &
            is_leaf_blade
        )

        # 4. Sooty Mold (Superficial dense black coating)
        sooty = (
            (brightness < 48) & (brightness > 5) &
            (s_channel < 65) & (r < 60) & (g < 60) & (b < 60) &
            is_leaf_blade
        )

        # 5. Gall Midge (Elevated pustules/galls, blister bumps)
        gall_midge = (
            ((r > 130) & (r > b + 25) & (g > 90) & (g < 165) & (brightness > 65) & (brightness < 175)) &
            is_leaf_blade
        )

        # 6. Die Back (Brown necrosis / margin desiccation)
        die_back = (
            ((r > b + 14) & (r > g - 6) & (brightness > 35) & (brightness < 195) & (s_channel > 22)) &
            is_leaf_blade
        )

        phenotype_masks = [
            ("necrotic", dark_necrotic),
            ("canker", chlorotic_canker),
            ("powdery", powdery),
            ("sooty", sooty),
            ("galls", gall_midge),
            ("dieback", die_back)
        ]

        min_area = max(80, int(total_pixels * 0.0008))
        max_area = int(total_pixels * 0.60)
        detections = []

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for name, p_mask in phenotype_masks:
            mask_u8 = (p_mask.astype(np.uint8) * 255)
            if np.count_nonzero(mask_u8) < min_area:
                continue

            cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w < 16 or h < 16 or (w >= w_img * 0.90 and h >= h_img * 0.90):
                        continue

                    pad_x = max(6, int(w * 0.10))
                    pad_y = max(6, int(h * 0.10))
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w_img, x + w + pad_x)
                    y2 = min(h_img, y + h + pad_y)

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "relative_bbox": [
                            round(x1 / w_img, 4),
                            round(y1 / h_img, 4),
                            round(x2 / w_img, 4),
                            round(y2 / h_img, 4)
                        ],
                        "confidence": 85.0,
                        "area": int((x2 - x1) * (y2 - y1)),
                        "source": f"morphology_{name}"
                    })

        return detections

    def _detect_with_yolo(self, np_rgb, w_img, h_img):
        """Runs fine-tuned YOLO bounding box inference."""
        detections = []
        if self.yolo_model is None:
            return detections

        try:
            results = self.yolo_model.predict(np_rgb, conf=0.25, verbose=False)
            for res in results:
                boxes = res.boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy()) * 100.0

                        x1 = max(0, min(w_img - 1, int(xyxy[0])))
                        y1 = max(0, min(h_img - 1, int(xyxy[1])))
                        x2 = max(x1 + 1, min(w_img, int(xyxy[2])))
                        y2 = max(y1 + 1, min(h_img, int(xyxy[3])))
                        area = (x2 - x1) * (y2 - y1)

                        if area >= (w_img * h_img * 0.65):
                            continue

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "relative_bbox": [
                                round(x1 / w_img, 4),
                                round(y1 / h_img, 4),
                                round(x2 / w_img, 4),
                                round(y2 / h_img, 4)
                            ],
                            "confidence": round(conf, 1),
                            "area": int(area),
                            "source": "yolo"
                        })
        except Exception:
            pass

        return detections

    def apply_nms(self, detections, iou_threshold=0.35):
        """Non-Maximum Suppression (NMS) merging overlapping candidate boxes."""
        if not detections:
            return []

        dets = sorted(detections, key=lambda d: d.get("confidence", 80.0), reverse=True)
        keep = []

        while dets:
            current = dets.pop(0)
            keep.append(current)

            x1_a, y1_a, x2_a, y2_a = current["bbox"]
            area_a = max(1, (x2_a - x1_a) * (y2_a - y1_a))

            remaining = []
            for other in dets:
                x1_b, y1_b, x2_b, y2_b = other["bbox"]
                area_b = max(1, (x2_b - x1_b) * (y2_b - y1_b))

                inter_x1 = max(x1_a, x1_b)
                inter_y1 = max(y1_a, y1_b)
                inter_x2 = min(x2_a, x2_b)
                inter_y2 = min(y2_a, y2_b)

                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                union = area_a + area_b - inter_area
                iou = inter_area / max(1, union)

                # If overlap exceeds threshold, suppress
                if iou < iou_threshold:
                    remaining.append(other)
            dets = remaining

        return keep
