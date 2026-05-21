
import cv2
import numpy as np

from Feature_matching import (
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
                 min_match_count: int   = 10,
                 inlier_threshold: float = 0.25,
                 sift_params:  dict | None = None,
                 flann_params: dict | None = None):

        self.ratio_threshold  = ratio_threshold
        self.min_match_count  = min_match_count
        self.inlier_threshold = inlier_threshold

        # Build the shared SIFT detector and FLANN matcher once so they
        # are reused across all fit() and predict() calls – avoids
        # redundant index construction on every comparison.
        self._sift  = create_sift_detector(**(sift_params  or {}))
        self._flann = create_flann_matcher(**(flann_params or {}))

        # Internal reference database: list of dicts, one per fitted image.
        # Each entry holds the label, raw image, keypoints, and descriptors.
        self._database: list[dict] = []


    # FIT
    def fit(self, image: np.ndarray, label: str) -> "SIFTFLANNClassifier":
        """
        Add a reference image to the classifier's database.
        Called once per reference image
        image : np.ndarray-Preprocessed grayscale reference image.
        label : str
        """
        if image is None:
           raise ValueError(f"fit(): image for label '{label}' is None.")

        # Extract keypoints and 128-D descriptors for this reference image.
        # These are stored once so predict() never re-extracts reference features.
        keypoints, descriptors = extract_sift_features(image, self._sift)

        if descriptors is None or len(descriptors) == 0:
            raise ValueError(
                f"fit(): SIFT found no keypoints in the reference image for '{label}'. "
                "Check that the image is not blank or heavily blurred."
            )

        self._database.append({
            "label":       label,
            "image":       image,        # kept for visualisation
            "keypoints":   keypoints,
            "descriptors": descriptors,
        })

        return self  # allow chaining

    # PREDICT (single best match)
    def predict(self, query_image: np.ndarray) -> tuple[str, float, dict]:
        """
        Classify a query banknote image against all fitted references.
        query_image : np.ndarray-Preprocessed grayscale query image
        """
        if not self._database:
            raise RuntimeError("predict(): no reference images have been fitted yet. "
                               "Call fit() with at least one reference image first.")

        if query_image is None:
            raise ValueError("predict(): query_image is None.")

        # Extract query features once
        kp_q, ds_q = extract_sift_features(query_image, self._sift)

        if ds_q is None or len(ds_q) == 0:
            # No keypoints found in the query
            return "unknown", 0.0, {"error": "No SIFT keypoints detected in query image."}

        best_label  = "unknown"
        best_score  = 0.0
        best_detail = {}

        # Compare the query against every reference entry in the database.
        for entry in self._database:
            # Match query descriptors against this reference's descriptors.
            good_matches, _ = match_descriptors_flann(
                ds_q,
                entry["descriptors"],
                self._flann,
                self.ratio_threshold,
            )

            # Score the match using RANSAC homography verification.
            result = compute_similarity_score(
                good_matches,
                kp_q,
                entry["keypoints"],
                self.min_match_count,
            )

            # Track the reference with the highest  inlier score.
            if result["inlier_ratio"] > best_score:
                best_score  = result["inlier_ratio"]
                best_label  = entry["label"]
                best_detail = result
                # Attach artefacts needed for external visualisation calls.
                best_detail["label"]              = entry["label"]
                best_detail["keypoints_query"]    = kp_q
                best_detail["descriptors_query"]  = ds_q
                best_detail["keypoints_reference"]= entry["keypoints"]
                best_detail["reference_image"]    = entry["image"]
                best_detail["good_matches"]       = good_matches

        # Apply confidence gate
        if best_score < self.inlier_threshold:
            best_label = "unknown"

        return best_label, best_score, best_detail

    # PREDICT_ALL (scores against every reference)
    def predict_all(self, query_image: np.ndarray) -> list[dict]:
        """
        Return similarity scores against every reference in the database.
        query_image : np.ndarray – Preprocessed grayscale query image.
        """
        if not self._database:
            raise RuntimeError("predict_all(): database is empty. Call fit() first.")

        kp_q, ds_q = extract_sift_features(query_image, self._sift)

        if ds_q is None:
            return [{"label": ref["label"], "inlier_ratio": 0.0,
                     "num_good_matches": 0, "match_ratio": 0.0,
                     "is_match": False} for ref in self._database]

        all_results = []
        for entry in self._database:
            good_matches, _ = match_descriptors_flann(
                ds_q, entry["descriptors"], self._flann, self.ratio_threshold
            )
            result = compute_similarity_score(
                good_matches, kp_q, entry["keypoints"], self.min_match_count
            )
            all_results.append({
                "label":            entry["label"],
                "num_good_matches": result["num_good_matches"],
                "match_ratio":      result["match_ratio"],
                "inlier_ratio":     result["inlier_ratio"],
                "is_match":         result["is_match"],
            })

        # Sort so the best candidate appears first
        all_results.sort(key=lambda x: x["inlier_ratio"], reverse=True)
        return all_results


    # VISUALISE (convenience wrapper for the winning match)
    def visualise_prediction(self, query_image: np.ndarray,
                              output_path: str | None = None) -> np.ndarray:
        """
        Runs predict() and immediately draw matches for the best reference.
        query_image : np.ndarray – Preprocessed grayscale query image.
        output_path : str -If given, saves the image to disk.
        """
        label, score, detail = self.predict(query_image)

        if "keypoints_query" not in detail:
            raise RuntimeError(f"visualise_prediction(): predict() returned no "
                               f"match artefacts (label='{label}', score={score:.3f}).")

        vis = visualise_matches(
            query_image,              detail["keypoints_query"],
            detail["reference_image"],detail["keypoints_reference"],
            detail["good_matches"],
            inlier_mask=detail.get("inlier_mask"),
            output_path=output_path,
        )
        return vis

    # HELPERS
    def database_summary(self) -> None:
        #Print a summary of all fitted reference images.
        if not self._database:
            print("Database is empty – call fit() to add reference images.")
            return
        print(f"{'#':<4} {'Label':<25} {'Keypoints':>10} {'Descriptors':>12}")
        print("-" * 55)
        for i, entry in enumerate(self._database):
            n_kp  = len(entry["keypoints"])
            n_des = entry["descriptors"].shape[0]
            print(f"{i:<4} {entry['label']:<25} {n_kp:>10} {n_des:>12}")

    def __len__(self) -> int:
        #Return the number of reference images in the database.
        return len(self._database)

    def __repr__(self) -> str:
        return (f"SIFTFLANNClassifier("
                f"references={len(self._database)}, "
                f"ratio_threshold={self.ratio_threshold}, "
                f"inlier_threshold={self.inlier_threshold})")



