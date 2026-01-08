"""
Unified Search Module
- Kết hợp tất cả methods và indexes
- Cung cấp API đơn giản cho UI
"""
import os
import numpy as np
from typing import List, Tuple, Optional
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *
from src.index.faiss_index import FAISSIndex
from src.extract.histogram import extract_histogram, compute_similarity
from src.extract.resnet import ResNetExtractor
from src.extract.clip_feat import CLIPExtractor


class ImageSearchEngine:
    """
    Unified search engine cho Track A (Image → Image)
    """
    def __init__(self):
        self.methods = {}
        self.features = {}
        self.image_paths = []
        self.indexes = {}
        
        # Lazy loading extractors
        self._resnet_extractor = None
        self._clip_extractor = None
    
    @property
    def resnet_extractor(self):
        if self._resnet_extractor is None:
            self._resnet_extractor = ResNetExtractor()
        return self._resnet_extractor
    
    @property
    def clip_extractor(self):
        if self._clip_extractor is None:
            self._clip_extractor = CLIPExtractor()
        return self._clip_extractor
    
    def load_features(self, method: str, features_path: str, paths_file: str):
        """
        Load pre-extracted features
        
        Args:
            method: "histogram", "resnet", "clip"
            features_path: đường dẫn file .npy
            paths_file: đường dẫn file chứa image paths
        """
        print(f"Loading {method} features...")
        self.features[method] = np.load(features_path).astype(np.float32)
        
        with open(paths_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        print(f"✓ Loaded {len(self.features[method])} features for {method}")
    
    def load_index(self, method: str, index_path: str):
        """
        Load FAISS index
        
        Args:
            method: "histogram", "resnet", "clip"
            index_path: đường dẫn file index
        """
        print(f"Loading {method} index...")
        dimension = self.features[method].shape[1]
        index = FAISSIndex(dimension=dimension, index_type="Flat")
        index.load(index_path)
        self.indexes[method] = index
        print(f"✓ Loaded index for {method}")
    
    def build_index(self, method: str, index_type: str = "Flat"):
        """
        Build index từ features đã load
        
        Args:
            method: "histogram", "resnet", "clip"
            index_type: "Flat", "IVF", "HNSW"
        """
        if method not in self.features:
            raise ValueError(f"Features for {method} not loaded")
        
        features = self.features[method]
        features = np.ascontiguousarray(features)
        
        index = FAISSIndex(dimension=features.shape[1], index_type=index_type)
        index.add(features)
        self.indexes[method] = index
        
        print(f"✓ Built {index_type} index for {method}")
    
    def search(self, query_image_path: str, method: str = "clip", 
               top_k: int = 20, use_faiss: bool = True) -> Tuple[List[str], List[float], float]:
        """
        Search similar images
        
        Args:
            query_image_path: đường dẫn ảnh query
            method: "histogram", "resnet", "clip"
            top_k: số kết quả trả về
            use_faiss: dùng FAISS index hay brute-force
        
        Returns:
            result_paths: list đường dẫn ảnh kết quả
            scores: similarity scores
            latency: thời gian search (ms)
        """
        start = time.time()
        
        # Extract query features
        if method == "histogram":
            query_feat = extract_histogram(query_image_path)
        elif method == "resnet":
            query_feat = self.resnet_extractor.extract_single(query_image_path)
        elif method == "clip":
            query_feat = self.clip_extractor.extract_image(query_image_path)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Search
        if use_faiss and method in self.indexes:
            scores, indices, _ = self.indexes[method].search(query_feat, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Brute-force cosine similarity
            db_features = self.features[method]
            scores = np.dot(db_features, query_feat)
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        
        latency = (time.time() - start) * 1000  # ms
        
        # Get paths
        result_paths = [self.image_paths[i] for i in indices]
        
        return result_paths, scores.tolist(), latency


class TextImageSearchEngine:
    """
    Search engine cho Track B (Text → Image) với CLIP
    """
    def __init__(self):
        self._clip_extractor = None
        self.image_features = None
        self.image_paths = []
        self.index = None
    
    @property
    def clip_extractor(self):
        if self._clip_extractor is None:
            self._clip_extractor = CLIPExtractor()
        return self._clip_extractor
    
    def load_features(self, features_path: str, paths_file: str):
        """Load CLIP image features"""
        print("Loading CLIP image features...")
        self.image_features = np.load(features_path).astype(np.float32)
        
        with open(paths_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        print(f"✓ Loaded {len(self.image_features)} features")
    
    def build_index(self, index_type: str = "Flat"):
        """Build FAISS index"""
        features = np.ascontiguousarray(self.image_features)
        self.index = FAISSIndex(dimension=features.shape[1], index_type=index_type)
        self.index.add(features)
        print(f"✓ Built {index_type} index")
    
    def search_by_text(self, query_text: str, top_k: int = 20) -> Tuple[List[str], List[float], float]:
        """
        Search images by text query
        
        Args:
            query_text: text query (e.g., "a red dress")
            top_k: số kết quả
        
        Returns:
            result_paths, scores, latency_ms
        """
        start = time.time()
        
        # Extract text features
        text_feat = self.clip_extractor.extract_text(query_text)
        
        # Search
        if self.index:
            scores, indices, _ = self.index.search(text_feat, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Brute-force
            scores = np.dot(self.image_features, text_feat)
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        
        latency = (time.time() - start) * 1000
        
        result_paths = [self.image_paths[i] for i in indices]
        
        return result_paths, scores.tolist(), latency
    
    def search_by_image(self, query_image_path: str, top_k: int = 20) -> Tuple[List[str], List[float], float]:
        """
        Search images by image query (using CLIP)
        
        Args:
            query_image_path: đường dẫn ảnh query
            top_k: số kết quả
        
        Returns:
            result_paths, scores, latency_ms
        """
        start = time.time()
        
        # Extract image features
        img_feat = self.clip_extractor.extract_image(query_image_path)
        
        # Search
        if self.index:
            scores, indices, _ = self.index.search(img_feat, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            scores = np.dot(self.image_features, img_feat)
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        
        latency = (time.time() - start) * 1000
        
        result_paths = [self.image_paths[i] for i in indices]
        
        return result_paths, scores.tolist(), latency


# ====================
# DEMO
# ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Search Engine Demo")
    print("=" * 50)
    
    # Demo Track A
    clip_features = os.path.join(FEATURES_TRACK_A, "clip.npy")
    clip_paths = os.path.join(FEATURES_TRACK_A, "clip_paths.txt")
    
    if os.path.exists(clip_features):
        print("\n[Track A] Image Search Demo")
        engine = ImageSearchEngine()
        engine.load_features("clip", clip_features, clip_paths)
        engine.build_index("clip", index_type="Flat")
        
        # Test search
        test_image = engine.image_paths[0]
        results, scores, latency = engine.search(test_image, method="clip", top_k=5)
        
        print(f"\nQuery: {test_image}")
        print(f"Latency: {latency:.2f} ms")
        print("Results:")
        for path, score in zip(results, scores):
            print(f"  {score:.4f} - {os.path.basename(path)}")
    else:
        print("⚠ Features not found. Run extract scripts first!")
