import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class MangoLeafSegmenter:
    """
    Botanical Mango Leaf Foreground Segmentation & Background Elimination Engine.
    
    Isolates the mango leaf lamina from distracting backgrounds such as:
      - Paper (white sheets, ruled notebooks, printed text, books)
      - Ground (soil, mud, gravel, dry foliage)
      - Tables / desks (wood grain, laminate, tiles)
      - Hands, shadows, and outdoor glare
    
    Generates a binary leaf mask and ensures that subsequent lesion detection
    and neural classification are strictly focused on the leaf tissue.
    """
    def __init__(self, min_leaf_coverage_pct=3.0):
        self.min_leaf_coverage_pct = min_leaf_coverage_pct

    def segment(self, np_rgb):
        """
        Segments the mango leaf from the background.
        
        Args:
            np_rgb: numpy array of shape (H, W, 3) in RGB format.
            
        Returns:
            dict containing:
              - 'leaf_mask': binary uint8 mask (H, W) where 255 = leaf, 0 = background
              - 'masked_rgb': np_rgb with background replaced by neutral dark background
              - 'is_leaf_present': bool indicating if a valid leaf is detected
              - 'leaf_coverage_pct': float percentage of image area covered by the leaf
              - 'leaf_bbox': [x1, y1, x2, y2] bounding box encompassing the leaf
              - 'leaf_contour': largest contour representing the leaf blade
        """
        if not HAS_CV2 or np_rgb is None:
            h, w = (np_rgb.shape[:2] if np_rgb is not None else (224, 224))
            ones_mask = np.ones((h, w), dtype=np.uint8) * 255
            return {
                "leaf_mask": ones_mask,
                "masked_rgb": np_rgb if np_rgb is not None else np.zeros((h, w, 3), dtype=np.uint8),
                "is_leaf_present": True,
                "leaf_coverage_pct": 100.0,
                "leaf_bbox": [0, 0, w, h],
                "leaf_contour": None
            }

        h, w = np_rgb.shape[:2]
        total_pixels = h * w

        img_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

        r = np_rgb[:, :, 0].astype(np.float32)
        g = np_rgb[:, :, 1].astype(np.float32)
        b = np_rgb[:, :, 2].astype(np.float32)

        # 1. Color and Vegetation Chrominance Signals
        # Excess Green Index: 2G - R - B
        exg = 2.0 * g - r - b
        exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, exg_thresh = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Lab a* channel: green is negative (low in 0-255 scaling, typically < 126)
        a_chan = img_lab[:, :, 1]
        _, a_thresh = cv2.threshold(a_chan, 126, 255, cv2.THRESH_BINARY_INV)

        # HSV Hue and Saturation: Green-yellow vegetation range (Hue 18-95, Sat > 22)
        h_chan = img_hsv[:, :, 0]
        s_chan = img_hsv[:, :, 1]
        v_chan = img_hsv[:, :, 2]
        hsv_veg = (h_chan >= 18) & (h_chan <= 95) & (s_chan >= 22) & (v_chan >= 20)

        # Diseased brown/necrotic tissue inside leaf: lower green but within foliage structure
        necrotic_tissue = (r > 35) & (g > 25) & (b < 140) & (s_chan > 25) & (v_chan < 225)

        # 2. Background Rejection Masks
        # A) White paper / bright page / background glare (High brightness, Low saturation)
        is_paper_or_glare = (v_chan > 200) & (s_chan < 45)

        # B) Printed text / dark lines on paper (very dark text against high background)
        is_printed_text = (v_chan < 60) & (s_chan < 40)

        # C) Gray floor / tile grout / concrete (neutral low saturation)
        is_neutral_background = (s_chan < 18) & (np.abs(r - g) < 14) & (np.abs(g - b) < 14)

        # 3. Combine Foreground Candidates
        candidate_fg = (
            (exg_thresh > 0) |
            (a_thresh > 0) |
            hsv_veg |
            necrotic_tissue
        ) & ~is_paper_or_glare & ~is_printed_text & ~is_neutral_background

        fg_mask = candidate_fg.astype(np.uint8) * 255

        # 4. Morphological Refinement
        # Fill holes in the leaf blade and connect separate lesion patches
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_refined = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
        mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, kernel_open)

        # 5. Extract Largest Connected Components (The Mango Leaf)
        contours, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        leaf_mask = np.zeros((h, w), dtype=np.uint8)
        is_leaf_present = False
        leaf_coverage_pct = 0.0
        leaf_bbox = [0, 0, w, h]
        primary_contour = None

        if contours:
            # Sort contours by area descending
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            primary_contour = sorted_contours[0]
            max_area = cv2.contourArea(primary_contour)
            leaf_coverage_pct = (max_area / float(total_pixels)) * 100.0

            if leaf_coverage_pct >= self.min_leaf_coverage_pct:
                is_leaf_present = True
                
                # Draw the main leaf contour and any closely associated leaf lobes/fragments
                for cnt in sorted_contours:
                    if cv2.contourArea(cnt) >= (max_area * 0.10):
                        cv2.drawContours(leaf_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                
                # Fill any internal holes enclosed by the leaf contour
                leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
                
                # Compute bounding box of the leaf
                x, y, bw, bh = cv2.boundingRect(primary_contour)
                pad_x = int(bw * 0.05)
                pad_y = int(bh * 0.05)
                leaf_bbox = [
                    max(0, x - pad_x),
                    max(0, y - pad_y),
                    min(w, x + bw + pad_x),
                    min(h, y + bh + pad_y)
                ]
            else:
                # Leaf too small or pure background image
                is_leaf_present = False
        
        # If no valid leaf is detected, create fallback mask
        if not is_leaf_present:
            leaf_mask = np.zeros((h, w), dtype=np.uint8)
            masked_rgb = np_rgb.copy()
        else:
            # Apply leaf mask to RGB image (neutral background: 25, 30, 25)
            bg_color = np.array([25, 30, 25], dtype=np.uint8)
            masked_rgb = np_rgb.copy()
            masked_rgb[leaf_mask == 0] = bg_color

        return {
            "leaf_mask": leaf_mask,
            "masked_rgb": masked_rgb,
            "is_leaf_present": is_leaf_present,
            "leaf_coverage_pct": round(float(leaf_coverage_pct), 2),
            "leaf_bbox": leaf_bbox,
            "leaf_contour": primary_contour
        }

    def filter_boxes_to_leaf(self, detections, leaf_mask, min_overlap=0.55):
        """
        Filters candidate bounding boxes to only keep those located on the leaf lamina.
        Discards boxes located on background paper, ground, wood, or margins.
        """
        if leaf_mask is None or (leaf_mask > 0).sum() == 0:
            return detections

        valid_detections = []
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            box_area = max(1, (x2 - x1) * (y2 - y1))
            
            # Extract mask region for this box
            sub_mask = leaf_mask[y1:y2, x1:x2]
            if sub_mask.size == 0:
                continue
            leaf_pixels_in_box = (sub_mask > 0).sum()
            overlap_ratio = float(leaf_pixels_in_box) / float(box_area)

            if overlap_ratio >= min_overlap:
                det_copy = det.copy()
                det_copy["leaf_overlap"] = round(overlap_ratio, 3)
                valid_detections.append(det_copy)

        return valid_detections


# Singleton instance
segmenter = MangoLeafSegmenter()
