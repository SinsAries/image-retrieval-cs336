"""
CLIP Feature Extractor
- Main method cho cả Track A và Track B
- Hỗ trợ cả image và text embedding
"""
import os
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *


class CLIPExtractor:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP feature extractor
        
        Args:
            model_name: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"
            device: "cuda" hoặc "cpu"
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        print(f"✓ Loaded CLIP {model_name}")
        
    def extract_image(self, image_path: str) -> np.ndarray:
        """
        Extract CLIP features từ 1 ảnh
        
        Args:
            image_path: đường dẫn đến ảnh
        
        Returns:
            feature vector (512,) for ViT-B/32
        """
        # Load và preprocess ảnh
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.encode_image(img_tensor)
        
        # Chuyển về numpy và normalize
        features = features.squeeze().cpu().numpy()
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features.astype(np.float32)
    
    def extract_text(self, text: str) -> np.ndarray:
        """
        Extract CLIP features từ text
        
        Args:
            text: câu text query
        
        Returns:
            feature vector (512,) for ViT-B/32
        """
        # Tokenize text
        text_token = clip.tokenize([text], truncate=True).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.encode_text(text_token)
        
        # Chuyển về numpy và normalize
        features = features.squeeze().cpu().numpy()
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features.astype(np.float32)
    
    def extract_images_batch(self, image_paths: list, batch_size: int = 32) -> np.ndarray:
        """
        Extract CLIP features cho nhiều ảnh (batch processing)
        
        Args:
            image_paths: list đường dẫn ảnh
            batch_size: số ảnh mỗi batch
        
        Returns:
            features array (N x 512)
        """
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    img_tensor = self.preprocess(img)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    # Tạo zero tensor nếu lỗi
                    batch_tensors.append(torch.zeros(3, 224, 224))
            
            # Stack thành batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model.encode_image(batch)
            
            features = features.cpu().numpy()
            all_features.append(features)
        
        # Concatenate tất cả
        all_features = np.vstack(all_features)
        
        # L2 normalize
        norms = np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-7
        all_features = all_features / norms
        
        return all_features.astype(np.float32)
    
    def extract_texts_batch(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Extract CLIP features cho nhiều texts (batch processing)
        
        Args:
            texts: list các câu text
            batch_size: số texts mỗi batch
        
        Returns:
            features array (N x 512)
        """
        all_features = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            text_tokens = clip.tokenize(batch_texts, truncate=True).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model.encode_text(text_tokens)
            
            features = features.cpu().numpy()
            all_features.append(features)
        
        # Concatenate tất cả
        all_features = np.vstack(all_features)
        
        # L2 normalize
        norms = np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-7
        all_features = all_features / norms
        
        return all_features.astype(np.float32)


def extract_all_clip_images(image_dir: str, output_path: str, batch_size: int = 32):
    """
    Extract CLIP features cho tất cả ảnh trong thư mục
    
    Args:
        image_dir: thư mục chứa ảnh
        output_path: đường dẫn save file .npy
        batch_size: số ảnh mỗi batch
    """
    # Lấy danh sách ảnh
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []
    
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images")
    
    # Initialize extractor
    extractor = CLIPExtractor(model_name=CLIP_MODEL)
    
    # Extract features
    features = extractor.extract_images_batch(image_paths, batch_size=batch_size)
    print(f"Features shape: {features.shape}")
    
    # Save features
    np.save(output_path, features)
    print(f"✓ Saved features to: {output_path}")
    
    # Save image paths
    paths_file = output_path.replace('.npy', '_paths.txt')
    with open(paths_file, 'w') as f:
        for p in image_paths:
            f.write(p + '\n')
    print(f"✓ Saved paths to: {paths_file}")
    
    return features, image_paths


# ====================
# MAIN
# ====================
if __name__ == "__main__":
    # Tạo thư mục
    os.makedirs(FEATURES_TRACK_A, exist_ok=True)
    os.makedirs(FEATURES_TRACK_B, exist_ok=True)
    
    print("=" * 50)
    print("Extracting CLIP Features")
    print("=" * 50)
    
    # Track A - DeepFashion
    if os.path.exists(DEEPFASHION_IMAGES):
        print("\n[Track A] DeepFashion")
        output_path = os.path.join(FEATURES_TRACK_A, "clip.npy")
        extract_all_clip_images(
            image_dir=DEEPFASHION_IMAGES,
            output_path=output_path,
            batch_size=BATCH_SIZE
        )
    else:
        print(f"⚠ DeepFashion not found at: {DEEPFASHION_IMAGES}")
    
    # Track B - Flickr30k
    if os.path.exists(FLICKR_IMAGES):
        print("\n[Track B] Flickr30k")
        output_path = os.path.join(FEATURES_TRACK_B, "clip.npy")
        extract_all_clip_images(
            image_dir=FLICKR_IMAGES,
            output_path=output_path,
            batch_size=BATCH_SIZE
        )
    else:
        print(f"⚠ Flickr30k not found at: {FLICKR_IMAGES}")
