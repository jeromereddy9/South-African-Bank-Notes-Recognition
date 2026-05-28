import cv2
import numpy as np

#SIFT Feature Extraction

def create_sift_detector(n_features=0, n_octave_layers=3,
                         contrast_threshold=0.04, edge_threshold=10, sigma=1.6):

    sift = cv2.SIFT_create(
        nfeatures=n_features,
        nOctaveLayers=n_octave_layers,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )
    return sift


def extract_sift_features(image, sift_detector):

    if image is None:
        raise ValueError("extract_sift_features: received a None image.")
    
    # detectAndCompute does both detection and description in one pass
    keypoints, descriptors = sift_detector.detectAndCompute(image, mask=None)
    return keypoints, descriptors


def visualise_keypoints(image, keypoints, output_path=None):

    vis_image = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image


#FLANN Matching 

def create_flann_matcher(algorithm="kdtree", trees=5, checks=50):

    if algorithm == "kdtree":
        # FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=1, trees=trees)
    else:
        raise ValueError(f"Unknown FLANN algorithm '{algorithm}'. Use kdtree")
    
    # Controls how many tree nodes FLANN visits during matching
    search_params = dict(checks=checks)
    
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return matcher


def match_descriptors_flann(descriptors_query, descriptors_reference,
                            flann_matcher, ratio_threshold=0.75):

    if descriptors_query is None or descriptors_reference is None:
        raise ValueError("match_descriptors_flann: one or both descriptor arrays are None.")
    
    # knnMatch returns 2 nearest neighbours for each query descriptor
    # k=2 is required for Lowe's ratio test
    all_matches = flann_matcher.knnMatch(
        descriptors_query.astype(np.float32),
        descriptors_reference.astype(np.float32),
        k=2
    )
    
    # Lowe's ratio test: keep match only if it's distinctly closer than the 2nd best
    good_matches = []
    for pair in all_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches, all_matches


def compute_similarity_score(good_matches, keypoints_query, keypoints_reference,
                             min_match_count=10):

    n_query = len(keypoints_query)
    n_good = len(good_matches)
    
    # Basic ratio: what percentage of query keypoints were matched?
    match_ratio = n_good / max(n_query, 1)
    
    result = {
        "num_good_matches": n_good,
        "match_ratio": match_ratio,
        "homography": None,
        "inlier_mask": None,
        "inlier_ratio": 0.0,
        "is_match": False,
    }
    
    if n_good < min_match_count:
        return result  # Not enough matches for reliable homography
    
    # Extract coordinates of matched keypoints
    pts_query = np.float32(
        [keypoints_query[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    
    pts_ref = np.float32(
        [keypoints_reference[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    
    # RANSAC finds the best homography while rejecting outliers
    homography, inlier_mask = cv2.findHomography(
        pts_query, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0  # Max reprojection error in pixels
    )
    
    if homography is not None and inlier_mask is not None:
        n_inliers = int(inlier_mask.sum())
        inlier_ratio = n_inliers / max(n_good, 1)
        
        result.update({
            "homography": homography,
            "inlier_mask": inlier_mask,
            "inlier_ratio": inlier_ratio,
            "is_match": inlier_ratio > 0.3,  # Confident match if ≥30% inliers
        })
    
    return result


def visualise_matches(image_query, keypoints_query,
                      image_reference, keypoints_reference,
                      good_matches, inlier_mask=None,
                      max_draw=50, output_path=None):

    draw_params = dict(
        matchColor=(0, 255, 0),       # Green for matches
        singlePointColor=(255, 0, 0), # Blue for unmatched keypoints
        flags=cv2.DrawMatchesFlags_DEFAULT
    )
    
    # Limit drawn matches to avoid clutter
    matches_to_draw = good_matches[:max_draw]
    
    if inlier_mask is not None:
        # Color inliers green, outliers red
        inlier_flags = inlier_mask[:max_draw].ravel().tolist()
        draw_params["matchesMask"] = inlier_flags
        draw_params["matchColor"] = (0, 255, 0)  # Inliers → green
    
    vis_image = cv2.drawMatches(
        image_query, keypoints_query,
        image_reference, keypoints_reference,
        matches_to_draw, None,
        **draw_params
    )
    
    if output_path:
        cv2.imwrite(output_path, vis_image)
    
    return vis_image


#Comparison Method

def compare_banknotes(image_query, image_reference,
                      sift_params=None, flann_params=None,
                      ratio_threshold=0.75, min_match_count=10):

    sift_params = sift_params or {}
    flann_params = flann_params or {}
    
    # Detect and describe keypoints
    sift = create_sift_detector(**sift_params)
    kp_q, ds_q = extract_sift_features(image_query, sift)
    kp_r, ds_r = extract_sift_features(image_reference, sift)
    
    # Match descriptors with FLANN
    flann = create_flann_matcher(**flann_params)
    good_matches, _ = match_descriptors_flann(ds_q, ds_r, flann, ratio_threshold)
    
    # Score the match using homography
    result = compute_similarity_score(good_matches, kp_q, kp_r, min_match_count)
    
    # Attach artifacts for visualization
    result.update({
        "keypoints_query": kp_q,
        "keypoints_reference": kp_r,
        "descriptors_query": ds_q,
        "descriptors_reference": ds_r,
        "good_matches": good_matches,
    })
    
    return result
