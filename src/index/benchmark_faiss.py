"""
FAISS Index Benchmark - Flat vs IVF vs HNSW
"""
import os
import sys
import numpy as np
import time
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

def load_features(method):
    feat_path = os.path.join(FEATURES_TRACK_A, f"{method}.npy")
    features = np.load(feat_path).astype(np.float32)
    return np.ascontiguousarray(features)

def benchmark_index(name, index, queries, top_k=20):
    """Benchmark search latency"""
    latencies = []
    results = []
    
    for query in queries:
        query = query.reshape(1, -1)
        start = time.time()
        scores, indices = index.search(query, top_k)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        results.append(indices[0])
    
    return {
        'name': name,
        'avg_ms': np.mean(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p99_ms': np.percentile(latencies, 99),
        'results': results
    }

def compute_recall(baseline_results, test_results, k=20):
    recalls = []
    for base, test in zip(baseline_results, test_results):
        base_set = set(base[:k])
        test_set = set(test[:k])
        recall = len(base_set & test_set) / k
        recalls.append(recall)
    return np.mean(recalls)

def main():
    print("="*70)
    print("FAISS INDEX BENCHMARK - CLIP Features")
    print("="*70)
    
    # Load CLIP features
    print("\nLoading CLIP features...")
    features = load_features('clip')
    print(f"Shape: {features.shape}")
    
    # Normalize for cosine similarity (use inner product)
    faiss.normalize_L2(features)
    
    # Sample queries
    num_queries = 500
    np.random.seed(42)
    query_indices = np.random.choice(len(features), num_queries, replace=False)
    queries = features[query_indices].copy()
    
    dim = features.shape[1]
    
    # Build indexes
    print("\nBuilding indexes...")
    indexes = {}
    
    # 1. Flat (brute-force) - baseline
    print("  [1/5] Flat (brute-force)...")
    start = time.time()
    index_flat = faiss.IndexFlatIP(dim)
    index_flat.add(features)
    print(f"        Built in {time.time()-start:.2f}s")
    indexes['Flat'] = index_flat
    
    # 2. IVF-50
    print("  [2/5] IVF (nlist=50)...")
    start = time.time()
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf50 = faiss.IndexIVFFlat(quantizer, dim, 50, faiss.METRIC_INNER_PRODUCT)
    index_ivf50.train(features)
    index_ivf50.add(features)
    index_ivf50.nprobe = 10
    print(f"        Built in {time.time()-start:.2f}s")
    indexes['IVF-50'] = index_ivf50
    
    # 3. IVF-100
    print("  [3/5] IVF (nlist=100)...")
    start = time.time()
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf100 = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)
    index_ivf100.train(features)
    index_ivf100.add(features)
    index_ivf100.nprobe = 10
    print(f"        Built in {time.time()-start:.2f}s")
    indexes['IVF-100'] = index_ivf100
    
    # 4. IVF-200
    print("  [4/5] IVF (nlist=200)...")
    start = time.time()
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf200 = faiss.IndexIVFFlat(quantizer, dim, 200, faiss.METRIC_INNER_PRODUCT)
    index_ivf200.train(features)
    index_ivf200.add(features)
    index_ivf200.nprobe = 10
    print(f"        Built in {time.time()-start:.2f}s")
    indexes['IVF-200'] = index_ivf200
    
    # 5. HNSW
    print("  [5/5] HNSW (M=32)...")
    start = time.time()
    index_hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.add(features)
    print(f"        Built in {time.time()-start:.2f}s")
    indexes['HNSW-32'] = index_hnsw
    
    # Benchmark
    print(f"\nBenchmarking with {num_queries} queries...")
    results = {}
    
    for name, index in indexes.items():
        print(f"  Testing {name}...")
        results[name] = benchmark_index(name, index, queries, top_k=20)
    
    # Compute recall vs Flat
    baseline_results = results['Flat']['results']
    
    # Print results
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Index':<15} | {'Avg (ms)':>10} | {'P50 (ms)':>10} | {'P99 (ms)':>10} | {'Recall@20':>10}")
    print("-"*70)
    
    for name in ['Flat', 'IVF-50', 'IVF-100', 'IVF-200', 'HNSW-32']:
        r = results[name]
        recall = compute_recall(baseline_results, r['results']) if name != 'Flat' else 1.0
        print(f"{name:<15} | {r['avg_ms']:>10.3f} | {r['p50_ms']:>10.3f} | {r['p99_ms']:>10.3f} | {recall:>10.4f}")
    
    print("="*70)
    
    # Speedup analysis
    flat_latency = results['Flat']['avg_ms']
    print("\nSPEEDUP vs Flat:")
    for name in ['IVF-50', 'IVF-100', 'IVF-200', 'HNSW-32']:
        r = results[name]
        speedup = flat_latency / r['avg_ms']
        recall = compute_recall(baseline_results, r['results'])
        print(f"  {name}: {speedup:.2f}x faster, {recall*100:.1f}% recall")
    
    # Save
    import json
    save_data = {name: {k: v for k, v in r.items() if k != 'results'} for name, r in results.items()}
    with open('benchmark_faiss_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print("\nSaved to: benchmark_faiss_results.json")

if __name__ == "__main__":
    main()
