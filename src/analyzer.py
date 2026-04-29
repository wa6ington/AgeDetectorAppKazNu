import cv2
import numpy as np
import os
import sys

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        # Use project-root-relative path instead of current working directory.
        # This makes model loading stable when app is launched from another folder.
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class FaceAnalyzer:
    """
    Main engine for face detection and age-range prediction.
    Optimized for accuracy and performance.
    """
    
    def __init__(self):
        # Configuration
        self.MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
        self.AGE_LIST = ['0-2 лет', '4-6 лет', '8-12 лет', '15-20 лет', '25-32 года', '38-43 года', '48-53 года', '60-100 лет']
        self.AGE_LIST_ASCII = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
        self.AGE_MIDPOINTS = [1, 5, 10, 17.5, 28.5, 40.5, 50.5, 80]
        self.AGE_BOUNDS = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]
        # Mild prior: reduce overprediction of adult classes on young faces.
        self.AGE_PRIOR = np.array([1.20, 1.18, 1.15, 1.08, 0.95, 0.80, 0.65, 0.45], dtype=np.float64)
        # Thresholds
        self.FACE_CONFIDENCE_THRESHOLD = 0.90
        self.MIN_FACE_SIZE = 70
        self.MAX_INPUT_SIDE = 960
        self.MAX_FACES = 1
        
        # Model paths
        self.face_proto = get_resource_path("assets/opencv_face_detector.pbtxt")
        self.face_model = get_resource_path("assets/opencv_face_detector_uint8.pb")
        self.age_proto = get_resource_path("assets/age_deploy.prototxt")
        self.age_model = get_resource_path("assets/age_net.caffemodel")
        
        # Load networks
        self.face_net = cv2.dnn.readNet(self.face_model, self.face_proto)
        self.age_net = cv2.dnn.readNet(self.age_model, self.age_proto)

    def _normalize_probs(self, probs):
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        probs = np.clip(probs, 1e-9, None)
        probs = probs / probs.sum()
        return probs

    def _prediction_confidence(self, probs):
        probs = self._normalize_probs(probs)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max(max_entropy, 1e-9)
        return max(1.0 - normalized_entropy, 0.0)

    def _estimate_precise_age(self, probs):
        probs = self._normalize_probs(probs)

        # Smooth neighboring bins to reduce unstable jumps between adjacent classes.
        smoothed = probs.copy()
        for i in range(len(probs)):
            left = probs[i - 1] if i > 0 else probs[i]
            right = probs[i + 1] if i < len(probs) - 1 else probs[i]
            smoothed[i] = 0.2 * left + 0.6 * probs[i] + 0.2 * right
        smoothed = self._normalize_probs(smoothed)

        weighted_age = float(np.dot(smoothed, np.array(self.AGE_MIDPOINTS, dtype=np.float64)))
        variance = float(np.dot(smoothed, (np.array(self.AGE_MIDPOINTS) - weighted_age) ** 2))
        age_std = variance ** 0.5

        top_idx = int(np.argmax(smoothed))
        bin_min, bin_max = self.AGE_BOUNDS[top_idx]

        # Keep estimate close to dominant age region while allowing soft transitions.
        clamp_margin = 4.0
        precise_age = float(np.clip(weighted_age, bin_min - clamp_margin, bin_max + clamp_margin))
        return int(round(precise_age)), float(age_std)

    def _calibrate_age_probs(self, probs):
        """Apply light calibration without forcing a fixed dominant age bucket."""
        probs = self._normalize_probs(probs)
        calibrated = self._normalize_probs(probs * self.AGE_PRIOR)
        return calibrated

    def _select_age_bucket(self, probs):
        """
        Pick stable age bucket while avoiding class collapse:
        if top-2 classes are very close, use weighted midpoint and map back to nearest bucket.
        """
        probs = self._normalize_probs(probs)
        weighted_age = float(np.dot(probs, np.array(self.AGE_MIDPOINTS, dtype=np.float64)))

        # Child/teen safeguard: if younger bins hold meaningful mass, avoid hard jump to 25-32.
        child_teen_mass = float(np.sum(probs[:4]))
        if child_teen_mass >= 0.45:
            return int(np.argmax(probs[:4]))

        top_two = np.argsort(probs)[-2:]
        top = int(top_two[-1])
        second = int(top_two[-2])
        if abs(float(probs[top]) - float(probs[second])) < 0.06:
            nearest = int(np.argmin(np.abs(np.array(self.AGE_MIDPOINTS, dtype=np.float64) - weighted_age)))
            return nearest
        # If mean age is still young, do not allow forced jump into older classes.
        if weighted_age < 22.0 and top >= 4:
            return int(np.argmax(probs[:5]))
        return top

    def _iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter_area / float(area_a + area_b - inter_area)

    def _collect_face_candidates(self, detections, w, h):
        boxes = []
        scores = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.FACE_CONFIDENCE_THRESHOLD:
                continue
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w - 1, int(detections[0, 0, i, 5] * w))
            y2 = min(h - 1, int(detections[0, 0, i, 6] * h))
            fw, fh = x2 - x1, y2 - y1
            if fw < self.MIN_FACE_SIZE or fh < self.MIN_FACE_SIZE:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # NMS expects x,y,w,h
            scores.append(confidence)

        if not boxes:
            return []

        keep = cv2.dnn.NMSBoxes(boxes, scores, self.FACE_CONFIDENCE_THRESHOLD, 0.30)
        if keep is None or len(keep) == 0:
            return []

        filtered = []
        for idx in np.array(keep).reshape(-1):
            x, y, bw, bh = boxes[idx]
            filtered.append((x, y, x + bw, y + bh, scores[idx]))
        filtered.sort(key=lambda item: item[4], reverse=True)
        return self._suppress_nested_faces(filtered)

    def _suppress_nested_faces(self, boxes_with_scores):
        kept = []
        for current in boxes_with_scores:
            x1, y1, x2, y2, score = current
            area = max(1, (x2 - x1) * (y2 - y1))
            is_nested = False
            for other in kept:
                ox1, oy1, ox2, oy2, oscore = other
                oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
                inter_x1 = max(x1, ox1)
                inter_y1 = max(y1, oy1)
                inter_x2 = min(x2, ox2)
                inter_y2 = min(y2, oy2)
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                containment = inter_area / float(area)
                smaller_ratio = area / float(oarea)
                # Drop mostly-contained smaller boxes (typical duplicate inside same face).
                if containment > 0.85 and smaller_ratio < 0.70 and score <= oscore:
                    is_nested = True
                    break
            if not is_nested:
                kept.append(current)
        return self._suppress_center_inside_duplicates(kept)

    def _suppress_center_inside_duplicates(self, boxes_with_scores):
        if len(boxes_with_scores) <= 1:
            return boxes_with_scores

        kept = []
        for i, current in enumerate(boxes_with_scores):
            x1, y1, x2, y2, score = current
            area = max(1, (x2 - x1) * (y2 - y1))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            duplicate = False

            for j, other in enumerate(boxes_with_scores):
                if i == j:
                    continue
                ox1, oy1, ox2, oy2, oscore = other
                oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
                # If current box center lies in a bigger box and they overlap, treat as duplicate.
                center_inside = (ox1 <= cx <= ox2) and (oy1 <= cy <= oy2)
                area_ratio = area / float(oarea)
                overlap = self._iou((x1, y1, x2, y2), (ox1, oy1, ox2, oy2))
                if center_inside and area_ratio < 0.78 and overlap > 0.10 and score <= oscore + 0.03:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(current)

        return kept

    def preprocess_image(self, frame):
        """ Improve image quality and normalize contrast/exposure """
        # Auto-gamma correction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mid = np.median(gray)
        gamma = 1.0
        if mid < 50: gamma = 1.5
        elif mid > 200: gamma = 0.7
        
        if gamma != 1.0:
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            frame = cv2.LUT(frame, table)

        # Convert to LAB to equalize lightness
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def get_multi_crops(self, face_crop):
        """ Generate multi-crops for more stable age inference """
        try:
            # Gentle contrast normalization (aggressive sharpening inflates age estimates).
            lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(6, 6))
            cl = clahe.apply(l)
            face_crop = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
            
            base = cv2.resize(face_crop, (256, 256))
            crops = []
            positions = [(14, 14), (0, 0), (29, 0), (0, 29), (29, 29)]
            for x, y in positions:
                c = base[y:y+227, x:x+227]
                crops.append(c)
                crops.append(cv2.flip(c, 1))
            return crops
        except:
            return []

    def detect_and_analyze(self, frame):
        """
        Detects faces and predicts age range using multi-scale padding ensemble.
        """
        # Resize very large images to keep inference responsive on weaker CPUs.
        h0, w0 = frame.shape[:2]
        max_side = max(h0, w0)
        if max_side > self.MAX_INPUT_SIDE:
            scale = self.MAX_INPUT_SIDE / float(max_side)
            frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

        frame = self.preprocess_image(frame)
        h, w = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        results = []
        annotated = frame.copy()
        
        face_candidates = self._collect_face_candidates(detections, w, h)
        for x1, y1, x2, y2, _ in face_candidates[: self.MAX_FACES]:
            fw, fh = x2 - x1, y2 - y1
            
            # Multi-scale Padding Ensemble (Inference at 3 different crops)
            # This combats the model's sensitivity to face scale/framing.
            all_age_preds = []
            all_age_weights = []
            
            # Two scales improve robustness for child/teen faces.
            for pad_ratio in [0.03, 0.10]:
                padding = int(max(fw, fh) * pad_ratio)
                face_crop = frame[
                    max(0, y1 - padding): min(y2 + padding, h - 1),
                    max(0, x1 - padding): min(x2 + padding, w - 1)
                ]
                
                if face_crop.size == 0: continue
                
                crops = self.get_multi_crops(face_crop)
                if not crops: continue
                
                blob_batch = cv2.dnn.blobFromImages(crops, 1.0, (227, 227), self.MODEL_MEAN, swapRB=False)
                
                self.age_net.setInput(blob_batch)
                age_pred = self.age_net.forward().mean(axis=0)
                all_age_preds.append(age_pred)
                all_age_weights.append(self._prediction_confidence(age_pred) + 1e-3)
            
            if not all_age_preds: continue
            
            # Confidence-weighted aggregation for more stable predictions.
            final_age_preds = np.average(np.array(all_age_preds), axis=0, weights=np.array(all_age_weights))

            probs = self._calibrate_age_probs(final_age_preds)
            age_idx = self._select_age_bucket(probs)
            age_category = self.AGE_LIST[age_idx]
            age_ascii = self.AGE_LIST_ASCII[age_idx]

            # Store result
            results.append({
                'box': (x1, y1, x2, y2),
                'age_category': age_category,
                'age_ascii': age_ascii
            })
            
            # Annotation - only age range
            color = (124, 58, 237) # Purple accent
            # ASCII-only label for reliable OpenCV rendering on all systems.
            label = f"Age {age_ascii}"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        
        return annotated, results
