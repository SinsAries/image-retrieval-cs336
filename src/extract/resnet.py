"""
ResNet50 Feature Extractor
- Baseline 2 cho Track A (Image → Image)
- Dùng pre-trained ResNet50, lấy features từ layer cuối
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *


class ResNetExtractor:
    def __init__(self, model_name: str = "resnet50", device: str = None):
        """
        Initialize ResNet feature extractor
        
        Args:
            model_name: "resnet50", "resnet101", "resnet152"
            device: "cuda" hoặc "cpu"
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load pre-trained model
        if model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif model_name == "resnet101":
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        elif model_name == "resnet152":
            self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Bỏ layer classification cuối, giữ features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform cho input
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"✓ Loaded {model_name}")
    
    def extract_single(self, image_path: str) -> np.ndarray:
        """
        Extract features từ 1 ảnh
        
        Args:
            image_path: đường dẫn đến ảnh
        
        Returns:
            feature vector (2048,) for ResNet50
        """
        # Load và transform ảnh
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Flatten và chuyển về numpy
        features = features.squeeze().cpu().numpy()
        
        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features.astype(np.float32)
    
    def extract_batch(self, image_paths: list, batch_size: int = 32) -> np.ndarray:
        """
        Extract features cho nhiều ảnh (batch processing)
        
        Args:
            image_paths: list đường dẫn ảnh
            batch_size: số ảnh mỗi batch
        
        Returns:
            features array (N x 2048)
        """
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting ResNet features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    # Tạo zero tensor nếu lỗi
                    batch_tensors.append(torch.zeros(3, 224, 224))
            
            # Stack thành batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(batch)
            
            features = features.squeeze(-1).squeeze(-1).cpu().numpy()
            all_features.append(features)
        
        # Concatenate tất cả
        all_features = np.vstack(all_features)
        
        # L2 normalize
        norms = np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-7
        all_features = all_features / norms
        
        return all_features.astype(np.float32)


def extract_all_resnet(image_dir: str, output_path: str, batch_size: int = 32):
    """
    Extract ResNet features cho tất cả ảnh trong thư mục
    
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
    extractor = ResNetExtractor(model_name="resnet50")
    
    # Extract features
    features = extractor.extract_batch(image_paths, batch_size=batch_size)
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
    
    # Extract features cho DeepFashion
    output_path = os.path.join(FEATURES_TRACK_A, "resnet.npy")
    
    print("=" * 50)
    print("Extracting ResNet50 Features for Track A")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists(DEEPFASHION_IMAGES):
        print(f"⚠ Dataset not found at: {DEEPFASHION_IMAGES}")
        print("Please download DeepFashion dataset first!")
    else:
        features, paths = extract_all_resnet(
            image_dir=DEEPFASHION_IMAGES,
            output_path=output_path,
            batch_size=BATCH_SIZE
        )
