import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.Feature_matching import (
    create_sift_detector,
    create_flann_matcher,
    extract_sift_features,
    match_descriptors_flann,
    compute_similarity_score,
    visualise_matches,
    visualise_keypoints,
)


class SIFTFLANNClassifier:
   
    def __init__(self,
                 ratio_threshold: float = 0.75,
                 min_match_count: int = 10,
                 inlier_threshold: float = 0.25,
                 sift_params: dict | None = None,
                 flann_params: dict | None = None):
       
        self.ratio_threshold = ratio_threshold
        self.min_match_count = min_match_count
        self.inlier_threshold = inlier_threshold

        # Default SIFT params for more distinctive features
        default_sift_params = {
            'contrast_threshold': 0.06,  # Higher = fewer but more distinctive
            'edge_threshold': 8,         # Lower = fewer edge-like keypoints
            'n_features': 0,             # Keep all features
            'n_octave_layers': 3,
            'sigma': 1.6
        }
        
        # Merge user params with defaults
        if sift_params:
            default_sift_params.update(sift_params)
        
        # Build SIFT detector and FLANN matcher once (reused across all calls)
        self._sift = create_sift_detector(**default_sift_params)
        self._flann = create_flann_matcher(**(flann_params or {}))

        # Internal reference database: list of dicts, one per fitted image
        self._database: list[dict] = []

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        
        if image is None:
            return None
        
        # If image has 3 channels (RGB), convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # If image has 1 channel (grayscale), return as is
        return image
    
    def fit(self, image: np.ndarray, label: str) -> "SIFTFLANNClassifier":
       
        if image is None:
            raise ValueError(f"fit(): image for label '{label}' is None.")

        # Convert to grayscale if needed (SIFT requires single channel)
        gray = self._to_grayscale(image)

        # Extract keypoints and 128-D descriptors for this reference image
        keypoints, descriptors = extract_sift_features(gray, self._sift)

        if descriptors is None or len(descriptors) == 0:
            raise ValueError(
                f"fit(): SIFT found no keypoints in the reference image for '{label}'. "
                "Check that the image is not blank or heavily blurred."
            )

        self._database.append({
            "label": label,
            "image": gray,          # store grayscale version
            "keypoints": keypoints,
            "descriptors": descriptors,
        })

        return self

    def fit_with_rotations(self, image: np.ndarray, label: str, 
                          angles=[0, 90, 180, 270]) -> "SIFTFLANNClassifier":
       
        # Convert to grayscale
        gray = self._to_grayscale(image)
        
        # Add original orientation
        self.fit(gray, label)
        
        # Add rotated versions
        for angle in angles:
            if angle == 0:
                continue
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h))
            self.fit(rotated, label)
        
        return self

    def predict(self, query_image: np.ndarray) -> tuple[str, float, dict]:
       
        if not self._database:
            raise RuntimeError(
                "predict(): no reference images have been fitted yet. "
                "Call fit() with at least one reference image first."
            )

        if query_image is None:
            raise ValueError("predict(): query_image is None.")

        # Convert query to grayscale (SIFT requires single channel)
        query_gray = self._to_grayscale(query_image)

        # Extract query features
        kp_q, ds_q = extract_sift_features(query_gray, self._sift)

        if ds_q is None or len(ds_q) == 0:
            # No keypoints found in the query
            return "unknown", 0.0, {"error": "No SIFT keypoints detected in query image."}

        best_label = "unknown"
        best_score = 0.0
        best_detail = {}

        # Compare query against every reference in the database
        for entry in self._database:
            # Match query descriptors against this reference's descriptors
            good_matches, _ = match_descriptors_flann(
                ds_q,
                entry["descriptors"],
                self._flann,
                self.ratio_threshold,
            )

            # Score the match using RANSAC homography verification
            result = compute_similarity_score(
                good_matches,
                kp_q,
                entry["keypoints"],
                self.min_match_count,
            )

            # Track the reference with the highest inlier score
            if result["inlier_ratio"] > best_score:
                best_score = result["inlier_ratio"]
                best_label = entry["label"]
                best_detail = result
                
                # Attach artifacts needed for external visualization calls
                best_detail["label"] = entry["label"]
                best_detail["keypoints_query"] = kp_q
                best_detail["descriptors_query"] = ds_q
                best_detail["keypoints_reference"] = entry["keypoints"]
                best_detail["reference_image"] = entry["image"]
                best_detail["good_matches"] = good_matches

        # Apply confidence gate (reject if below threshold)
        if best_score < self.inlier_threshold:
            best_label = "unknown"

        return best_label, best_score, best_detail

    def predict_all(self, query_image: np.ndarray) -> list[dict]:
        
        if not self._database:
            raise RuntimeError("predict_all(): database is empty. Call fit() first.")

        # Convert query to grayscale
        query_gray = self._to_grayscale(query_image)
        
        kp_q, ds_q = extract_sift_features(query_gray, self._sift)

        if ds_q is None:
            return [{
                "label": ref["label"],
                "inlier_ratio": 0.0,
                "num_good_matches": 0,
                "match_ratio": 0.0,
                "is_match": False
            } for ref in self._database]

        all_results = []
        for entry in self._database:
            good_matches, _ = match_descriptors_flann(
                ds_q, entry["descriptors"], self._flann, self.ratio_threshold
            )
            result = compute_similarity_score(
                good_matches, kp_q, entry["keypoints"], self.min_match_count
            )
            all_results.append({
                "label": entry["label"],
                "num_good_matches": result["num_good_matches"],
                "match_ratio": result["match_ratio"],
                "inlier_ratio": result["inlier_ratio"],
                "is_match": result["is_match"],
            })

        # Sort so the best candidate appears first
        all_results.sort(key=lambda x: x["inlier_ratio"], reverse=True)
        return all_results

    def visualise_prediction(self, query_image: np.ndarray,
                            output_path: str | None = None) -> np.ndarray:
        
        # Convert query to grayscale for prediction
        query_gray = self._to_grayscale(query_image)
        
        label, score, detail = self.predict(query_gray)

        if "keypoints_query" not in detail:
            raise RuntimeError(
                f"visualise_prediction(): predict() returned no "
                f"match artifacts (label='{label}', score={score:.3f})."
            )

        vis = visualise_matches(
            query_gray,
            detail["keypoints_query"],
            detail["reference_image"],
            detail["keypoints_reference"],
            detail["good_matches"],
            inlier_mask=detail.get("inlier_mask"),
            output_path=output_path,
        )
        return vis

    def database_summary(self) -> None:
        """Print a summary of all fitted reference images."""
        if not self._database:
            print("Database is empty – call fit() to add reference images.")
            return
        
        print(f"{'#':<4} {'Label':<25} {'Keypoints':>10} {'Descriptors':>12}")
        print("-" * 55)
        for i, entry in enumerate(self._database):
            n_kp = len(entry["keypoints"])
            n_des = entry["descriptors"].shape[0]
            print(f"{i:<4} {entry['label']:<25} {n_kp:>10} {n_des:>12}")

    def clear_database(self) -> None:
        """Clear all reference images from the database."""
        self._database = []
        print("Database cleared.")

    def __len__(self) -> int:
        """Return the number of reference images in the database."""
        return len(self._database)

    def __repr__(self) -> str:
        return (
            f"SIFTFLANNClassifier("
            f"references={len(self._database)}, "
            f"ratio_threshold={self.ratio_threshold}, "
            f"inlier_threshold={self.inlier_threshold})"
        )


if __name__ == "__main__":
    # Example usage
    print("SIFT-FLANN Classifier Test")
    print("=" * 50)
    
    # Create classifier
    classifier = SIFTFLANNClassifier(
        ratio_threshold=0.75,
        min_match_count=15,
        inlier_threshold=0.35
    )
    
    print(f"Classifier initialized: {classifier}")
    print(f"Database size: {len(classifier)}")
    
    # Test would require actual images
    print("\nTo use this classifier:")
    print("1. Load reference images: classifier.fit(image, label)")
    print("2. Predict: label, confidence, details = classifier.predict(query_image)")
    print("3. Visualize: classifier.visualise_prediction(query_image)")
