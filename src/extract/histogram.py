"""
HSV Color Histogram Feature Extractor
- Baseline method cho Track A (Image → Image)
- Chuyển ảnh sang HSV rồi tính histogram
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def extract_histogram(image_path: str, bins: tuple = (8, 8, 8)) -> np.ndarray:
    """
    Extract HSV color histogram từ 1 ảnh
    
    Args:
        image_path: đường dẫn đến ảnh
        bins: số bins cho mỗi channel (H, S, V)
    
    Returns:
        histogram vector (normalized)
    """
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # Chuyển BGR -> HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Tính histogram
    hist = cv2.calcHist(
        [hsv], 
        [0, 1, 2],  # channels H, S, V
        None, 
        bins, 
        [0, 180, 0, 256, 0, 256]  # ranges
    )
    
    # Flatten và normalize
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-7)  # L1 normalize
    
    return hist.astype(np.float32)


def extract_all_histograms(image_dir: str, output_path: str, bins: tuple = (8, 8, 8)):
    """
    Extract histogram cho tất cả ảnh trong thư mục
    
    Args:
        image_dir: thư mục chứa ảnh
        output_path: đường dẫn save file .npy
        bins: số bins cho histogram
    """
    # Lấy danh sách ảnh
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images")
    
    # Extract features
    features = []
    valid_paths = []
    
    for path in tqdm(image_paths, desc="Extracting histograms"):
        try:
            feat = extract_histogram(path, bins)
            features.append(feat)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # Stack và save
    features = np.stack(features)
    print(f"Features shape: {features.shape}")
    
    # Save features
    np.save(output_path, features)
    print(f"✓ Saved features to: {output_path}")
    
    # Save image paths
    paths_file = output_path.replace('.npy', '_paths.txt')
    with open(paths_file, 'w') as f:
        for p in valid_paths:
            f.write(p + '\n')
    print(f"✓ Saved paths to: {paths_file}")
    
    return features, valid_paths


def compute_similarity(query_hist: np.ndarray, database_hists: np.ndarray, method: str = "cosine") -> np.ndarray:
    """
    Tính similarity giữa query và database
    
    Args:
        query_hist: histogram của query image
        database_hists: histograms của database (N x D)
        method: "cosine", "chi_square", "correlation", "intersection"
    
    Returns:
        similarity scores (N,)
    """
    if method == "cosine":
        # Cosine similarity
        query_norm = query_hist / (np.linalg.norm(query_hist) + 1e-7)
        db_norm = database_hists / (np.linalg.norm(database_hists, axis=1, keepdims=True) + 1e-7)
        scores = np.dot(db_norm, query_norm)
        
    elif method == "chi_square":
        # Chi-square distance (convert to similarity)
        eps = 1e-7
        diff = query_hist - database_hists
        sum_val = query_hist + database_hists + eps
        distances = np.sum((diff ** 2) / sum_val, axis=1)
        scores = 1 / (1 + distances)  # convert distance to similarity
        
    elif method == "correlation":
        # Correlation
        query_centered = query_hist - query_hist.mean()
        db_centered = database_hists - database_hists.mean(axis=1, keepdims=True)
        
        numerator = np.dot(db_centered, query_centered)
        denominator = np.sqrt(np.sum(db_centered**2, axis=1)) * np.sqrt(np.sum(query_centered**2))
        scores = numerator / (denominator + 1e-7)
        
    elif method == "intersection":
        # Histogram intersection
        scores = np.minimum(query_hist, database_hists).sum(axis=1)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return scores


# ====================
# MAIN
# ====================
if __name__ == "__main__":
    # Tạo thư mục
    os.makedirs(FEATURES_TRACK_A, exist_ok=True)
    
    # Extract features cho DeepFashion
    output_path = os.path.join(FEATURES_TRACK_A, "histogram.npy")
    
    print("=" * 50)
    print("Extracting HSV Histograms for Track A")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists(DEEPFASHION_IMAGES):
        print(f"⚠ Dataset not found at: {DEEPFASHION_IMAGES}")
        print("Please download DeepFashion dataset first!")
    else:
        features, paths = extract_all_histograms(
            image_dir=DEEPFASHION_IMAGES,
            output_path=output_path,
            bins=(8, 8, 8)  # 8*8*8 = 512 dimensions
        )
