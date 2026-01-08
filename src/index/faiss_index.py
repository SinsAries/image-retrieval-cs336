"""
FAISS Index Builder
- Tạo và quản lý FAISS indexes cho search
- Hỗ trợ nhiều loại index: Flat, IVF, HNSW
"""
import os
import numpy as np
import faiss
import time
from typing import Tuple, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *


class FAISSIndex:
    def __init__(self, dimension: int, index_type: str = "Flat", nlist: int = 100):
        """
        Initialize FAISS index
        
        Args:
            dimension: chiều của feature vector (512 cho CLIP, 2048 cho ResNet)
            index_type: "Flat", "IVF", "HNSW", "PQ"
            nlist: số clusters cho IVF
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.index = None
        self.is_trained = False
        
        self._create_index()
    
    def _create_index(self):
        """Tạo FAISS index theo loại được chọn"""
        
        if self.index_type == "Flat":
            # Brute-force search (chính xác nhất, chậm nhất)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine với normalized vectors)
            self.is_trained = True
            
        elif self.index_type == "IVF":
            # Inverted File Index (nhanh hơn, approximate)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
            self.is_trained = False
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World (rất nhanh, approximate)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            self.is_trained = True
            
        elif self.index_type == "PQ":
            # Product Quantization (tiết kiệm memory)
            self.index = faiss.IndexPQ(self.dimension, 16, 8, faiss.METRIC_INNER_PRODUCT)
            self.is_trained = False
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        print(f"✓ Created {self.index_type} index (dim={self.dimension})")
    
    def train(self, features: np.ndarray):
        """
        Train index (cần thiết cho IVF, PQ)
        
        Args:
            features: training features (N x D)
        """
        if self.is_trained:
            print("Index doesn't need training")
            return
        
        print(f"Training index with {len(features)} vectors...")
        start = time.time()
        
        self.index.train(features)
        self.is_trained = True
        
        print(f"✓ Training done in {time.time() - start:.2f}s")
    
    def add(self, features: np.ndarray):
        """
        Thêm features vào index
        
        Args:
            features: feature vectors (N x D)
        """
        if not self.is_trained:
            print("Training index first...")
            self.train(features)
        
        print(f"Adding {len(features)} vectors to index...")
        start = time.time()
        
        self.index.add(features)
        
        print(f"✓ Added in {time.time() - start:.2f}s")
        print(f"Total vectors in index: {self.index.ntotal}")
    
    def search(self, query: np.ndarray, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search top-K similar vectors
        
        Args:
            query: query vector (D,) hoặc (N x D)
            top_k: số kết quả trả về
        
        Returns:
            scores: similarity scores (N x K)
            indices: indices của kết quả (N x K)
        """
        # Đảm bảo query là 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Set nprobe cho IVF
        if self.index_type == "IVF":
            self.index.nprobe = FAISS_NPROBE
        
        # Search
        start = time.time()
        scores, indices = self.index.search(query, top_k)
        latency = (time.time() - start) * 1000  # ms
        
        return scores, indices, latency
    
    def save(self, path: str):
        """Save index to file"""
        faiss.write_index(self.index, path)
        print(f"✓ Saved index to: {path}")
    
    def load(self, path: str):
        """Load index from file"""
        self.index = faiss.read_index(path)
        self.is_trained = True
        print(f"✓ Loaded index from: {path}")
        print(f"Total vectors: {self.index.ntotal}")


def build_index(features_path: str, index_path: str, index_type: str = "Flat"):
    """
    Build FAISS index từ file features
    
    Args:
        features_path: đường dẫn file .npy chứa features
        index_path: đường dẫn save index
        index_type: loại index
    """
    # Load features
    print(f"Loading features from: {features_path}")
    features = np.load(features_path)
    print(f"Features shape: {features.shape}")
    
    # Đảm bảo features là float32 và contiguous
    features = np.ascontiguousarray(features.astype(np.float32))
    
    # Create index
    dimension = features.shape[1]
    index = FAISSIndex(dimension=dimension, index_type=index_type)
    
    # Add features
    index.add(features)
    
    # Save index
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    index.save(index_path)
    
    return index


def benchmark_indexes(features_path: str, num_queries: int = 100, top_k: int = 20):
    """
    Benchmark các loại index khác nhau
    
    Args:
        features_path: đường dẫn file features
        num_queries: số queries để test
        top_k: số kết quả trả về
    """
    # Load features
    features = np.load(features_path).astype(np.float32)
    features = np.ascontiguousarray(features)
    
    # Random queries
    query_indices = np.random.choice(len(features), num_queries, replace=False)
    queries = features[query_indices]
    
    print("\n" + "=" * 60)
    print("FAISS Index Benchmark")
    print("=" * 60)
    print(f"Database size: {len(features)}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Num queries: {num_queries}")
    print(f"Top-K: {top_k}")
    print("=" * 60)
    
    results = []
    index_types = ["Flat", "IVF", "HNSW"]
    
    for idx_type in index_types:
        print(f"\n--- {idx_type} Index ---")
        
        # Create index
        index = FAISSIndex(dimension=features.shape[1], index_type=idx_type)
        index.add(features)
        
        # Benchmark
        latencies = []
        for q in queries:
            _, _, latency = index.search(q, top_k)
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Avg latency: {avg_latency:.2f} ms")
        print(f"P99 latency: {p99_latency:.2f} ms")
        
        results.append({
            "type": idx_type,
            "avg_latency_ms": avg_latency,
            "p99_latency_ms": p99_latency
        })
    
    return results


# ====================
# MAIN
# ====================
if __name__ == "__main__":
    os.makedirs(INDEXES_DIR, exist_ok=True)
    
    print("=" * 50)
    print("Building FAISS Indexes")
    print("=" * 50)
    
    # Build indexes cho Track A
    track_a_features = [
        ("histogram", os.path.join(FEATURES_TRACK_A, "histogram.npy")),
        ("resnet", os.path.join(FEATURES_TRACK_A, "resnet.npy")),
        ("clip", os.path.join(FEATURES_TRACK_A, "clip.npy")),
    ]
    
    for name, feat_path in track_a_features:
        if os.path.exists(feat_path):
            print(f"\n[Track A] Building index for {name}")
            index_path = os.path.join(INDEXES_DIR, f"track_a_{name}_flat.index")
            build_index(feat_path, index_path, index_type="Flat")
        else:
            print(f"⚠ Features not found: {feat_path}")
    
    # Build indexes cho Track B
    track_b_clip = os.path.join(FEATURES_TRACK_B, "clip.npy")
    if os.path.exists(track_b_clip):
        print(f"\n[Track B] Building index for CLIP")
        index_path = os.path.join(INDEXES_DIR, "track_b_clip_flat.index")
        build_index(track_b_clip, index_path, index_type="Flat")
