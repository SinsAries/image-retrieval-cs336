# 🖼️ Image Retrieval System

**CS336: Multimedia Information Retrieval 2025**

## 📌 Overview

Hệ thống tìm kiếm ảnh tương tự (Content-Based Image Retrieval - CBIR) trên dataset DeepFashion.

- **Input:** Ảnh query (upload từ user)
- **Output:** Top-K ảnh tương tự nhất trong database

## 🏗️ Architecture
```
Query Image
       │
       ▼
┌─────────────────┐
│  Feature        │
│  Extraction     │
│  (Histogram/    │
│   ResNet/CLIP)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Index    │
│  (Flat/IVF/HNSW)│
└────────┬────────┘
         │
         ▼
   Top-K Results
```

## 🛠️ Methods

| Method | Description | Dimension | Similarity |
|--------|-------------|-----------|------------|
| **HSV Histogram** | Color-based | 256 | Chi-square ⭐ |
| HSV Histogram | Color-based | 256 | Intersection |
| HSV Histogram | Color-based | 256 | Cosine |
| ResNet50 | Deep features | 2048 | Cosine |
| CLIP ViT-B/32 | Multi-modal | 512 | Cosine |

## 📊 Evaluation Results

### Image → Image Retrieval (DeepFashion, 14,218 queries)

| Method | P@1 | P@5 | P@10 | R@20 | mAP |
|--------|-----|-----|------|------|-----|
| **Histogram + Chi-square** | **0.6042** | **0.2474** | **0.1470** | **0.4807** | **0.3549** |
| Histogram + Intersection | 0.4678 | 0.1950 | 0.1203 | 0.3953 | 0.2648 |
| CLIP + Cosine | 0.4553 | 0.2194 | 0.1415 | 0.4310 | 0.2667 |
| Histogram + Cosine | 0.2987 | 0.1206 | 0.0753 | 0.2534 | 0.1526 |
| ResNet + Cosine | 0.2773 | 0.1277 | 0.0836 | 0.2454 | 0.1364 |

### 💡 Key Insight

**Histogram + Chi-square đạt kết quả tốt nhất** (mAP = 0.3549) vì:
- DeepFashion có ảnh studio với nền trắng → màu sắc là đặc trưng phân biệt mạnh
- Chi-square là metric chuẩn cho so sánh histogram (đúng lý thuyết CBIR)
- CLIP/ResNet tối ưu cho semantic similarity, không phải instance retrieval

### FAISS Index Benchmark (52,712 vectors, 500 queries)

| Index | Avg Latency | Speedup | Recall@20 |
|-------|-------------|---------|-----------|
| Flat (brute-force) | 5.39ms | 1x | 100% |
| IVF-50 | 1.18ms | 4.6x | 98.9% |
| IVF-100 | 0.60ms | 9x | 96.8% |
| IVF-200 | 0.30ms | 18x | 94.0% |
| **HNSW-32** | **0.10ms** | **54.6x** | **98.1%** |

**Kết luận:** HNSW-32 là lựa chọn tối ưu cho production (54.6x speedup, 98.1% recall).

## 📦 Installation
```bash
# Clone repo
git clone https://github.com/SinsAries/image-retrieval-cs336.git
cd image-retrieval-cs336

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Create directories
python3 config.py
```

## 📂 Dataset: DeepFashion In-shop

| Thông số | Giá trị |
|----------|---------|
| Tổng số ảnh | 52,712 |
| Số ảnh query | 14,218 |
| Số ảnh gallery | 12,612 |
| Số danh mục | 7,982 |

### Download

1. Download từ [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html):
   - `img.zip` (792 MB)
   - `list_eval_partition.txt`

2. Giải nén vào `data/deepfashion/`:
```
data/deepfashion/
├── img/
│   ├── MEN/
│   └── WOMEN/
└── list_eval_partition.txt
```

## 🚀 Usage

### Step 1: Extract Features
```bash
# Histogram HSV (256-dim)
PYTHONPATH=. python3 src/extract/histogram.py

# ResNet50 (2048-dim)
PYTHONPATH=. python3 src/extract/resnet.py

# CLIP ViT-B/32 (512-dim)
PYTHONPATH=. python3 src/extract/clip_feat.py
```

### Step 2: Run UI
```bash
PYTHONPATH=. streamlit run app.py
```

Mở browser: http://localhost:8501

### Step 3: Evaluation
```bash
# Full evaluation (14,218 queries)
PYTHONPATH=. python3 src/evaluate/evaluate_track_a.py

# FAISS benchmark
PYTHONPATH=. python3 src/index/benchmark_faiss.py
```

## 📁 Project Structure
```
image-retrieval-cs336/
├── data/
│   └── deepfashion/         # Dataset (52,712 images)
├── features/
│   └── track_a/             # Extracted features (.npy)
│       ├── histogram.npy    # 52712 x 256
│       ├── resnet.npy       # 52712 x 2048
│       └── clip.npy         # 52712 x 512
├── indexes/                 # FAISS indexes (.faiss)
├── src/
│   ├── extract/             # Feature extraction
│   │   ├── histogram.py     # HSV color histogram
│   │   ├── resnet.py        # ResNet50 features
│   │   └── clip_feat.py     # CLIP features
│   ├── index/
│   │   ├── faiss_index.py   # FAISS index builder
│   │   └── benchmark_faiss.py
│   ├── search/
│   │   └── search.py        # Search engine
│   └── evaluate/
│       └── evaluate_track_a.py
├── app.py                   # Streamlit UI
├── config.py                # Configuration
├── requirements.txt
└── README.md
```

## 🔬 Technical Details

### Similarity Metrics

| Feature Type | Metric | Formula | Best for |
|--------------|--------|---------|----------|
| Histogram | Chi-square | $\frac{1}{2}\sum\frac{(a_i-b_i)^2}{a_i+b_i}$ | Histogram comparison ⭐ |
| Histogram | Intersection | $\sum\min(a_i, b_i)$ | Histogram comparison |
| Deep features | Cosine | $\frac{a \cdot b}{\|a\|\|b\|}$ | Normalized embeddings |

### Why Histogram + Chi-square beats Deep Learning?

1. **Dataset characteristics:** DeepFashion có ảnh studio, nền trắng đồng nhất → color là tín hiệu mạnh
2. **Metric matching:** Chi-square được thiết kế cho histogram, trong khi Cosine không phù hợp
3. **Task mismatch:** CLIP/ResNet tối ưu cho semantic similarity, không phải instance-level retrieval

### FAISS Index Types

| Index | Use case | Trade-off |
|-------|----------|-----------|
| Flat | Small dataset, exact search | Slow but accurate |
| IVF | Medium dataset | Tunable speed/accuracy |
| HNSW | Production, real-time | Fast with high recall |

## ✅ Project Checklist

### Yêu cầu cơ bản

| Yêu cầu | Status |
|---------|--------|
| Giao diện người dùng | ✅ Streamlit UI |
| Module nhập query + hiển thị kết quả | ✅ Upload → Grid |
| Dataset ≥ 5K ảnh, 50 queries | ✅ 52,712 ảnh, 14,218 queries |
| Đánh giá kết quả | ✅ P@K, R@K, mAP |
| So sánh với các phương pháp khác | ✅ 5 methods |
| Phân tích ưu/nhược điểm | ✅ Trong báo cáo |

### Điểm cộng

| Yêu cầu | Status |
|---------|--------|
| Kĩ thuật tìm kiếm CSDL lớn | ✅ FAISS (Flat/IVF/HNSW) |
| Dataset > 20K ảnh | ✅ 52,712 ảnh |

## 👥 Team

| Member | MSSV | Role |
|--------|------|------|
| Nguyễn Trọng Tất Thành | 23521455 | Feature extraction, FAISS, Evaluation, UI |
| Trần Vạn Tấn | XXXXXXXX | Documentation, Report |

## 📄 License

MIT License

---

**CS336: Multimedia Information Retrieval 2025 - UIT**
EOF
