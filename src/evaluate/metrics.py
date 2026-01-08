"""
Evaluation Metrics
- Precision@K, Recall@K, mAP
- Cho cả Track A và Track B
"""
import os
import numpy as np
from typing import List, Dict, Tuple
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Tính Precision@K
    
    Args:
        retrieved: list các items retrieved (theo thứ tự)
        relevant: list các items relevant (ground-truth)
        k: số items xét
    
    Returns:
        precision@k score
    """
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    
    num_relevant = sum(1 for item in retrieved_k if item in relevant_set)
    return num_relevant / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Tính Recall@K
    
    Args:
        retrieved: list các items retrieved (theo thứ tự)
        relevant: list các items relevant (ground-truth)
        k: số items xét
    
    Returns:
        recall@k score
    """
    if len(relevant) == 0:
        return 0.0
    
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    
    num_relevant = sum(1 for item in retrieved_k if item in relevant_set)
    return num_relevant / len(relevant)


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """
    Tính Average Precision cho 1 query
    
    Args:
        retrieved: list các items retrieved (theo thứ tự)
        relevant: list các items relevant (ground-truth)
    
    Returns:
        AP score
    """
    if len(relevant) == 0:
        return 0.0
    
    relevant_set = set(relevant)
    precisions = []
    num_relevant = 0
    
    for i, item in enumerate(retrieved):
        if item in relevant_set:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / len(relevant)


def mean_average_precision(all_retrieved: List[List[str]], 
                           all_relevant: List[List[str]]) -> float:
    """
    Tính Mean Average Precision
    
    Args:
        all_retrieved: list of retrieved lists cho mỗi query
        all_relevant: list of relevant lists cho mỗi query
    
    Returns:
        mAP score
    """
    aps = [average_precision(ret, rel) for ret, rel in zip(all_retrieved, all_relevant)]
    return np.mean(aps)


def evaluate_retrieval(search_engine, queries: List[Dict], 
                       method: str, top_k: int = 20) -> Dict:
    """
    Evaluate retrieval system
    
    Args:
        search_engine: ImageSearchEngine instance
        queries: list of query dicts với 'query_path' và 'relevant_paths'
        method: method name
        top_k: số kết quả trả về
    
    Returns:
        dict chứa các metrics
    """
    all_retrieved = []
    all_relevant = []
    latencies = []
    
    for query in queries:
        query_path = query['query_path']
        relevant_paths = query['relevant_paths']
        
        # Search
        results, scores, latency = search_engine.search(
            query_path, method=method, top_k=top_k
        )
        
        all_retrieved.append(results)
        all_relevant.append(relevant_paths)
        latencies.append(latency)
    
    # Calculate metrics
    k_values = K_VALUES
    metrics = {
        'method': method,
        'num_queries': len(queries),
        'top_k': top_k,
        'avg_latency_ms': np.mean(latencies),
        'p99_latency_ms': np.percentile(latencies, 99),
    }
    
    # Precision@K và Recall@K
    for k in k_values:
        if k <= top_k:
            precisions = [precision_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)]
            recalls = [recall_at_k(ret, rel, k) for ret, rel in zip(all_retrieved, all_relevant)]
            
            metrics[f'precision@{k}'] = np.mean(precisions)
            metrics[f'recall@{k}'] = np.mean(recalls)
    
    # mAP
    metrics['mAP'] = mean_average_precision(all_retrieved, all_relevant)
    
    return metrics


def print_metrics(metrics: Dict):
    """Pretty print metrics"""
    print("\n" + "=" * 50)
    print(f"Evaluation Results - {metrics['method']}")
    print("=" * 50)
    print(f"Queries: {metrics['num_queries']}")
    print(f"Top-K: {metrics['top_k']}")
    print(f"Avg Latency: {metrics['avg_latency_ms']:.2f} ms")
    print(f"P99 Latency: {metrics['p99_latency_ms']:.2f} ms")
    print("-" * 50)
    
    for k in K_VALUES:
        if f'precision@{k}' in metrics:
            print(f"Precision@{k}: {metrics[f'precision@{k}']:.4f}")
            print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")
    
    print("-" * 50)
    print(f"mAP: {metrics['mAP']:.4f}")
    print("=" * 50)


def compare_methods(all_metrics: List[Dict]):
    """
    So sánh nhiều methods
    
    Args:
        all_metrics: list of metrics dicts
    """
    print("\n" + "=" * 80)
    print("METHOD COMPARISON")
    print("=" * 80)
    
    # Header
    header = f"{'Method':<15} | {'mAP':>8} | {'R@1':>8} | {'R@5':>8} | {'R@10':>8} | {'R@20':>8} | {'Latency':>10}"
    print(header)
    print("-" * 80)
    
    # Rows
    for m in all_metrics:
        row = f"{m['method']:<15} | {m['mAP']:>8.4f} | "
        row += f"{m.get('recall@1', 0):>8.4f} | "
        row += f"{m.get('recall@5', 0):>8.4f} | "
        row += f"{m.get('recall@10', 0):>8.4f} | "
        row += f"{m.get('recall@20', 0):>8.4f} | "
        row += f"{m['avg_latency_ms']:>8.2f}ms"
        print(row)
    
    print("=" * 80)


def save_results(all_metrics: List[Dict], output_path: str):
    """Save evaluation results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Saved results to: {output_path}")


# ====================
# DEMO với mock data
# ====================
if __name__ == "__main__":
    print("Evaluation Metrics Demo")
    
    # Mock data
    retrieved = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg']
    relevant = ['img1.jpg', 'img3.jpg', 'img6.jpg']
    
    print(f"Retrieved: {retrieved}")
    print(f"Relevant: {relevant}")
    print()
    
    for k in [1, 3, 5]:
        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        print(f"P@{k}: {p:.4f}, R@{k}: {r:.4f}")
    
    ap = average_precision(retrieved, relevant)
    print(f"\nAP: {ap:.4f}")
