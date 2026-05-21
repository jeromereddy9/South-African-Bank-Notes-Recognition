
import cv2
import numpy as np

# SIFT FEATURE EXTRACTION
def create_sift_detector(n_features=0, n_octave_layers=3,
                         contrast_threshold=0.04, edge_threshold=10, sigma=1.6):
    """
    Creates and returns configured SIFT detector.
    n_features: int– Max keypoints to retain.
    n_octave_layers: int– Gaussian pyramid levels per octave.
    contrast_threshold : float – discard keypoints below this contrast.
    edge_threshold : int– Suppresses edge-like keypoints (e.g. note borders).
    sigma: float – Gaussian blur applied to the first octave.
    """
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        nOctaveLayers=n_octave_layers,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )
    return sift


def extract_sift_features(image, sift_detector):

    #Detect keypoints and compute 128-dimensional SIFT descriptors.
    """
    image: np.ndarray – Grayscale image (uint8, single channel)
    sift_detector: cv2.SIFT   – Detector created by create_sift_detector().                                   None if no keypoints were found.
    """
    if image is None:
        raise ValueError("extract_sift_features: received a None image.")

    # detects and computes keypoints as well as computes descriptors in one pass
    keypoints, descriptors = sift_detector.detectAndCompute(image, mask=None)

    return keypoints, descriptors


def visualise_keypoints(image, keypoints, output_path=None):
    """
    detect SIFT keypoints on the image for visual inspection.
    image       : np.ndarray – Original grayscale image.
    keypoints   : list– Keypoints returned by extract_sift_features().
    """
    vis_image = cv2.drawKeypoints(
        image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    if output_path:
        cv2.imwrite(output_path, vis_image)

    return vis_image



# FLANN MATCHING
def create_flann_matcher(algorithm="kdtree", trees=5, checks=50):
    """
    Creates a FLANN-based descriptor matcher.
    FLANN uses approximate nearest-neighbour search
    algorithm : str – 'kdtree'.
    trees     : int – Number of parallel kd-trees
    checks    : int – Leaf nodes visited per query during search.
    """

    if algorithm == "kdtree":
        # FLANN_INDEX_KDTREE = 1
        index_params  = dict(algorithm=1, trees=trees)
    else:
        raise ValueError(f"Unknown FLANN algorithm '{algorithm}'. Use kdtree")

    # controls how many tree nodes FLANN visits during matching.
    search_params = dict(checks=checks)

    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return matcher


def match_descriptors_flann(descriptors_query, descriptors_reference,
                             flann_matcher, ratio_threshold=0.75):
    """
    Match SIFT descriptors between a query image and a reference image using
    FLANN + Lowe's ratio test.
    descriptors_query: np.ndarray (N×128) – Query note descriptors.
    descriptors_reference: np.ndarray (M×128) – Reference note descriptors.
    flann_matcher : cv2.FlannBasedMatcher – Created by create_flann_matcher().
    ratio_threshold: float – Lowe's ratio (0–1).
    """
    if descriptors_query is None or descriptors_reference is None:
        raise ValueError("match_descriptors_flann: one or both descriptor arrays are None.")

    # knnMatch returns the 2 nearest neighbours for each query descriptor.
    # We need k=2 specifically for Lowe's ratio test.
    all_matches = flann_matcher.knnMatch(
        descriptors_query.astype(np.float32),
        descriptors_reference.astype(np.float32),
        k=2
    )

    # Lowe's ratio test keeps match m only if it is distinctly closer than its nearest competitor n.
    good_matches = []
    for pair in all_matches:
        # Guard against cases where FLANN returns fewer than 2 neighbours
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches, all_matches


def compute_similarity_score(good_matches, keypoints_query, keypoints_reference,
                              min_match_count=10):
    """
    Converts good matches into a similarity score and attempt homography estimation.
    returns two scores  match_ratio and inlier_ratio.
    good_matches: list – Filtered matches from match_descriptors_flann().
    keypoints_query: list – Query image keypoints.
    keypoints_reference: list – Reference image keypoints.
    min_match_count: int  – Minimum good matches required to attempt homography.

    returns
    dict with keys:
        'num_good_matches'  : int
        'match_ratio'       : float  (good_matches / query_keypoints)
        'homography'        : np.ndarray
        'inlier_mask'       : np.ndarray(boolean mask over good_matches)
        'inlier_ratio'      : float
        'is_match'          : bool (True if homography was found)
    """
    n_query = len(keypoints_query)
    n_good  = len(good_matches)

    # Basic ratio: how many of the query's keypoints were matched?
    match_ratio = n_good / max(n_query, 1)

    result = {
        "num_good_matches": n_good,
        "match_ratio":      match_ratio,
        "homography":       None,
        "inlier_mask":      None,
        "inlier_ratio":     0.0,
        "is_match":         False,
    }

    if n_good < min_match_count:
        # Not enough evidence for a reliable match
        return result

    # Extract the 2-D coordinates of each matched keypoint pair
    pts_query = np.float32(
        [keypoints_query[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    pts_ref = np.float32(
        [keypoints_reference[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # RANSAC-based homography: finds the best perspective transform that maps query keypoints to reference keypoints while rejecting outliers.
    homography, inlier_mask = cv2.findHomography(
        pts_query, pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0   # Max reprojection error in pixels
    )

    if homography is not None and inlier_mask is not None:
        n_inliers    = int(inlier_mask.sum())
        inlier_ratio = n_inliers / max(n_good, 1)

        result.update({
            "homography":   homography,
            "inlier_mask":  inlier_mask,
            "inlier_ratio": inlier_ratio,
            "is_match":     inlier_ratio > 0.3,  # ≥30 % inliers = confident match
        })

    return result


def visualise_matches(image_query, keypoints_query,
                      image_reference, keypoints_reference,
                      good_matches, inlier_mask=None,
                      max_draw=50, output_path=None):
    """
    Draws matching keypoints for qualitative evaluation.
    image_query: np.ndarray – Query grayscale image.
    keypoints_query : list– Query keypoints.
    image_reference : np.ndarray – Reference grayscale image.
    keypoints_reference : list– Reference keypoints.
    good_matches: list– Matches from match_descriptors_flann().
    inlier_mask: np.ndarray -Boolean mask from compute_similarity_score().
    max_draw: int– Cap on how many matches to render to avoid clutter.
    output_path: str
    """
    draw_params = dict(
        matchColor      = (0, 255, 0), # Green for inlier shows all matches
        singlePointColor= (255, 0, 0), # Blue for unmatched keypoints
        flags           = cv2.DrawMatchesFlags_DEFAULT
    )

    #limit drawn matches to avoid an unreadable tangle of lines
    matches_to_draw = good_matches[:max_draw]

    if inlier_mask is not None:
        #colour inliers green, outliers red for easy visual distinction
        inlier_flags = inlier_mask[:max_draw].ravel().tolist()
        draw_params["matchesMask"] = inlier_flags
        draw_params["matchColor"]  = (0, 255, 0)    # inliers  → green

    vis_image = cv2.drawMatches(
        image_query,     keypoints_query,
        image_reference, keypoints_reference,
        matches_to_draw, None,
        **draw_params
    )

    if output_path:
        cv2.imwrite(output_path, vis_image)

    return vis_image


# Comparison
def compare_banknotes(image_query, image_reference,
                      sift_params=None, flann_params=None,
                      ratio_threshold=0.75, min_match_count=10):
    """
    SIFT + FLANN comparison between two preprocessed banknote images.
    image_query: np.ndarray – Preprocessed grayscale query note.
    image_reference: np.ndarray – Preprocessed grayscale reference note.
    sift_params: dict
    flann_params: dict
    ratio_threshold : float – Lowe's ratio for match filtering
    min_match_count : int– Minimum good matches for homography .

    returns:
    result : dict – All scores
                    compute_similarity_score() plus:
                    'keypoints_query': list
                    'keypoints_reference': list
                    'descriptors_query': np.ndarray
                    'descriptors_reference': np.ndarray
                    'good_matches': list
    """
    sift_params  = sift_params  or {}
    flann_params = flann_params or {}

    # Detect and describe keypoints with SIFT ---
    sift= create_sift_detector(**sift_params)
    kp_q, ds_q = extract_sift_features(image_query,sift)
    kp_r, ds_r = extract_sift_features(image_reference,sift)

    #Match descriptors with FLANN ---
    flann= create_flann_matcher(**flann_params)
    good_matches, _ = match_descriptors_flann(ds_q, ds_r, flann, ratio_threshold)

    # Score the match using homography and inlier ratio
    result = compute_similarity_score(good_matches, kp_q, kp_r, min_match_count)

    # raw artefacts to visualise or log it
    result.update({
        "keypoints_query":      kp_q,
        "keypoints_reference":  kp_r,
        "descriptors_query":    ds_q,
        "descriptors_reference":ds_r,
        "good_matches":         good_matches,
    })

    return result
