"""
Final Evaluation - With Chi-square for Histogram + P@20/R@20
"""
import os
import sys
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from src.evaluate.metrics import precision_at_k, recall_at_k, mean_average_precision

def load_eval_partition(partition_file):
    queries = []
    gallery = []
    
    with open(partition_file, 'r') as f:
        lines = f.readlines()[2:]
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            img_name = parts[0]
            item_id = parts[1]
            status = parts[2]
            img_path = os.path.join(DEEPFASHION_DIR, img_name)
            
            if status == 'query':
                queries.append((img_path, item_id))
            elif status == 'gallery':
                gallery.append((img_path, item_id))
    
    return queries, gallery

def load_features_with_filter(method, valid_paths):
    feat_path = os.path.join(FEATURES_TRACK_A, f"{method}.npy")
    paths_file = os.path.join(FEATURES_TRACK_A, f"{method}_paths.txt")
    
    all_features = np.load(feat_path)
    with open(paths_file, 'r') as f:
        all_paths = [line.strip() for line in f]
    
    valid_set = set(valid_paths)
    indices = [i for i, p in enumerate(all_paths) if p in valid_set]
    
    filtered_features = all_features[indices]
    filtered_paths = [all_paths[i] for i in indices]
    
    print(f"    Filtered: {len(filtered_paths)}/{len(valid_paths)} paths matched")
    
    return filtered_features, filtered_paths, all_features, all_paths

def cosine_search(query_feat, gallery_feats, top_k=20):
    """Cosine similarity search (for ResNet/CLIP)"""
    query_norm = query_feat / (np.linalg.norm(query_feat) + 1e-7)
    gallery_norms = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-7)
    scores = np.dot(gallery_norms, query_norm)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices

def chi_square_search(query_feat, gallery_feats, top_k=20):
    """Chi-square distance search (for Histogram)"""
    eps = 1e-10
    # Chi-square distance: sum((q-g)^2 / (q+g))
    distances = 0.5 * np.sum(
        ((gallery_feats - query_feat) ** 2) / (gallery_feats + query_feat + eps),
        axis=1
    )
    top_indices = np.argsort(distances)[:top_k]  # smaller distance = better
    return top_indices

def histogram_intersection_search(query_feat, gallery_feats, top_k=20):
    """Histogram Intersection search (for Histogram)"""
    # Intersection: sum(min(q, g))
    scores = np.sum(np.minimum(gallery_feats, query_feat), axis=1)
    top_indices = np.argsort(scores)[::-1][:top_k]  # larger = better
    return top_indices

def evaluate_method(method, queries, ground_truth, gallery_feats, gallery_paths, all_feats, path_to_idx, top_k=20, num_queries=None, similarity='cosine'):
    valid_queries = [(q, item_id) for q, item_id in queries if q in path_to_idx and len(ground_truth.get(q, [])) > 0]
    
    if num_queries and len(valid_queries) > num_queries:
        sampled_queries = random.sample(valid_queries, num_queries)
    else:
        sampled_queries = valid_queries
    
    all_retrieved = []
    all_relevant = []
    
    for query_path, item_id in tqdm(sampled_queries, desc=f"{method} ({similarity})", leave=False):
        relevant = ground_truth.get(query_path, [])
        
        query_idx = path_to_idx[query_path]
        query_feat = all_feats[query_idx]
        
        # Choose search method
        if similarity == 'chi_square':
            top_indices = chi_square_search(query_feat, gallery_feats, top_k)
        elif similarity == 'intersection':
            top_indices = histogram_intersection_search(query_feat, gallery_feats, top_k)
        else:
            top_indices = cosine_search(query_feat, gallery_feats, top_k)
        
        results = [gallery_paths[i] for i in top_indices]
        
        all_retrieved.append(results)
        all_relevant.append(relevant)
    
    k_values = [1, 5, 10, 20]
    metrics = {'method': method, 'similarity': similarity, 'num_queries': len(sampled_queries)}
    
    for k in k_values:
        p_at_k = np.mean([precision_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)])
        r_at_k = np.mean([recall_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)])
        metrics[f'P@{k}'] = p_at_k
        metrics[f'R@{k}'] = r_at_k
    
    metrics['mAP'] = mean_average_precision(all_retrieved, all_relevant)
    
    return metrics

def print_results(all_metrics, num_queries):
    print(f"\n{'='*100}")
    print(f"RESULTS WITH {num_queries} QUERIES")
    print(f"{'='*100}")
    
    header = f"{'Method':<20} | {'P@1':>6} | {'P@5':>6} | {'P@10':>6} | {'P@20':>6} | {'R@1':>6} | {'R@5':>6} | {'R@10':>6} | {'R@20':>6} | {'mAP':>6}"
    print(header)
    print("-"*100)
    
    for m in all_metrics:
        name = f"{m['method']} ({m['similarity'][:3]})"
        row = f"{name:<20} | "
        row += f"{m.get('P@1', 0):>6.4f} | "
        row += f"{m.get('P@5', 0):>6.4f} | "
        row += f"{m.get('P@10', 0):>6.4f} | "
        row += f"{m.get('P@20', 0):>6.4f} | "
        row += f"{m.get('R@1', 0):>6.4f} | "
        row += f"{m.get('R@5', 0):>6.4f} | "
        row += f"{m.get('R@10', 0):>6.4f} | "
        row += f"{m.get('R@20', 0):>6.4f} | "
        row += f"{m.get('mAP', 0):>6.4f}"
        print(row)

def main():
    partition_file = os.path.join(DEEPFASHION_DIR, "list_eval_partition.txt")
    print(f"Loading partition...")
    
    queries, gallery = load_eval_partition(partition_file)
    print(f"Queries: {len(queries)}, Gallery: {len(gallery)}")
    
    # Build ground-truth
    gallery_by_item = defaultdict(list)
    for img_path, item_id in gallery:
        gallery_by_item[item_id].append(img_path)
    
    ground_truth = {}
    for img_path, item_id in queries:
        ground_truth[img_path] = gallery_by_item.get(item_id, [])
    
    # Count avg relevant per query
    avg_rel = np.mean([len(v) for v in ground_truth.values() if len(v) > 0])
    print(f"Avg relevant per query: {avg_rel:.2f}")
    
    # Pre-load all features
    print("\nLoading features...")
    gallery_paths_list = [g[0] for g in gallery]
    
    features_cache = {}
    for method in ['histogram', 'resnet', 'clip']:
        print(f"  Loading {method}...")
        gallery_feats, gallery_paths, all_feats, all_paths = load_features_with_filter(
            method, gallery_paths_list
        )
        path_to_idx = {p: i for i, p in enumerate(all_paths)}
        features_cache[method] = (gallery_feats, gallery_paths, all_feats, path_to_idx)
    
    # Evaluate with ALL queries
    print(f"\n{'='*50}")
    print(f"Evaluating with ALL queries...")
    print(f"{'='*50}")
    
    random.seed(42)
    
    metrics_list = []
    
    # Histogram with different similarities
    gallery_feats, gallery_paths, all_feats, path_to_idx = features_cache['histogram']
    for sim in ['cosine', 'chi_square', 'intersection']:
        metrics = evaluate_method(
            'histogram', queries, ground_truth,
            gallery_feats, gallery_paths, all_feats, path_to_idx,
            top_k=20, num_queries=None, similarity=sim
        )
        metrics_list.append(metrics)
    
    # ResNet with cosine
    gallery_feats, gallery_paths, all_feats, path_to_idx = features_cache['resnet']
    metrics = evaluate_method(
        'resnet', queries, ground_truth,
        gallery_feats, gallery_paths, all_feats, path_to_idx,
        top_k=20, num_queries=None, similarity='cosine'
    )
    metrics_list.append(metrics)
    
    # CLIP with cosine
    gallery_feats, gallery_paths, all_feats, path_to_idx = features_cache['clip']
    metrics = evaluate_method(
        'clip', queries, ground_truth,
        gallery_feats, gallery_paths, all_feats, path_to_idx,
        top_k=20, num_queries=None, similarity='cosine'
    )
    metrics_list.append(metrics)
    
    # Print results
    print_results(metrics_list, "ALL")
    
    # Summary table for report
    print("\n" + "="*80)
    print("SUMMARY TABLE FOR REPORT")
    print("="*80)
    print(f"{'Method':<25} | {'P@1':>8} | {'R@20':>8} | {'mAP':>8}")
    print("-"*60)
    
    for m in metrics_list:
        name = f"{m['method']} ({m['similarity']})"
        print(f"{name:<25} | {m['P@1']:>8.4f} | {m['R@20']:>8.4f} | {m['mAP']:>8.4f}")
    
    print("="*80)
    
    # Save
    import json
    with open('evaluation_results_final.json', 'w') as f:
        json.dump(metrics_list, f, indent=2)
    print("\nSaved to: evaluation_results_final.json")

if __name__ == "__main__":
    main()
