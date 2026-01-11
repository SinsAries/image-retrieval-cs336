# 🖼️ Image Retrieval System

**CS336: Multimedia Information Retrieval 2025**

## 📌 Overview

Hệ thống tìm kiếm ảnh hỗ trợ 2 tracks:
- **Track A:** Image → Image (tìm ảnh tương tự từ ảnh query)
- **Track B:** Text → Image (tìm ảnh từ text query)

## 🏗️ Architecture
```
Query (Image/Text)
       │
       ▼
┌─────────────────┐
│  Feature        │
│  Extraction     │
│  (CLIP/ResNet)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Index    │
│  (Flat/IVF)     │
└────────┬────────┘
         │
         ▼
   Top-K Results
```

## 🛠️ Methods

### Track A (Image → Image)
| Method | Description | Dimension | Similarity |
|--------|-------------|-----------|------------|
| HSV Histogram | Color-based baseline | 512 | Chi-square |
| ResNet50 | Deep learning baseline | 2048 | Cosine |
| **CLIP** | Multi-modal (main) | 512 | Cosine |

### Track B (Text → Image)
| Method | Description | Similarity |
|--------|-------------|------------|
| BM25 | TF-IDF on captions | BM25 Score |
| SBERT | Sentence embeddings | Cosine |
| **CLIP** | Multi-modal (main) | Cosine |

## 📊 Evaluation Results

### Track A: Image → Image (DeepFashion, 14,218 queries)

| Method | P@1 | P@5 | P@10 | R@20 | mAP |
|--------|-----|-----|------|------|-----|
| **Histogram (chi-square)** | **0.6042** | **0.2474** | **0.1470** | **0.4807** | **0.3549** |
| Histogram (intersection) | 0.4678 | 0.1950 | 0.1203 | 0.3953 | 0.2648 |
| CLIP (cosine) | 0.4553 | 0.2194 | 0.1415 | 0.4310 | 0.2667 |
| Histogram (cosine) | 0.2987 | 0.1206 | 0.0753 | 0.2534 | 0.1526 |
| ResNet (cosine) | 0.2773 | 0.1277 | 0.0836 | 0.2454 | 0.1364 |

**Key Insight:** Histogram + Chi-square đạt kết quả tốt nhất trên DeepFashion vì:
- Fashion phụ thuộc nhiều vào màu sắc và texture
- Chi-square là similarity metric chuẩn cho histogram (đúng theo lý thuyết CBIR)

### FAISS Index Benchmark (52,712 vectors, 500 queries)

| Index | Avg Latency | Speedup | Recall@20 |
|-------|-------------|---------|-----------|
| Flat (brute-force) | 5.39ms | 1x | 100% |
| IVF-50 | 1.18ms | 4.6x | 98.9% |
| IVF-100 | 0.60ms | 9x | 96.8% |
| IVF-200 | 0.30ms | 18x | 94.0% |
| **HNSW-32** | **0.10ms** | **54.6x** | **98.1%** |

**Key Insight:** HNSW-32 là lựa chọn tối ưu với speedup 54.6x và vẫn giữ 98.1% recall.

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

## 📂 Dataset

### Track A: DeepFashion In-shop
1. Download từ [Google Drive](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html):
   - `img.zip` (792 MB)
   - `list_eval_partition.txt`
   - `list_item_inshop.txt`
2. Giải nén vào `data/deepfashion/`
```
data/deepfashion/
├── img/
│   ├── MEN/
│   └── WOMEN/
├── list_eval_partition.txt
└── list_item_inshop.txt
```

### Track B: Flickr30k
1. Download từ [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
2. Giải nén vào `data/flickr30k/`
```
data/flickr30k/
├── images/
└── captions.txt
```

## 🚀 Usage

### Step 1: Extract Features
```bash
# Track A - Histogram
PYTHONPATH=. python3 src/extract/histogram.py

# Track A - ResNet50
PYTHONPATH=. python3 src/extract/resnet.py

# Track A & B - CLIP
PYTHONPATH=. python3 src/extract/clip_feat.py
```

### Step 2: Build Index
```bash
PYTHONPATH=. python3 src/index/faiss_index.py
```

### Step 3: Run UI
```bash
PYTHONPATH=. streamlit run app.py
```

Mở browser: http://localhost:8501

### Step 4: Evaluation
```bash
# Track A evaluation
PYTHONPATH=. python3 src/evaluate/evaluate_track_a.py

# FAISS benchmark
PYTHONPATH=. python3 src/index/benchmark_faiss.py
```

## 📁 Project Structure
```
image-retrieval-cs336/
├── data/
│   ├── deepfashion/         # Track A dataset (52,712 images)
│   └── flickr30k/           # Track B dataset (31K images)
├── features/
│   ├── track_a/             # Extracted features (.npy)
│   └── track_b/
├── indexes/                 # FAISS indexes
├── src/
│   ├── extract/             # Feature extraction
│   │   ├── histogram.py     # HSV color histogram
│   │   ├── resnet.py        # ResNet50 features
│   │   └── clip_feat.py     # CLIP image/text features
│   ├── index/
│   │   ├── faiss_index.py   # FAISS index builder
│   │   └── benchmark_faiss.py
│   ├── search/
│   │   └── search.py        # Search engine
│   └── evaluate/
│       ├── metrics.py       # P@K, R@K, mAP
│       └── evaluate_track_a.py
├── app.py                   # Streamlit UI
├── config.py                # Configuration
├── requirements.txt
└── README.md
```

## 🔬 Technical Details

### Similarity Metrics
| Feature Type | Recommended Metric | Reason |
|--------------|-------------------|--------|
| Histogram | Chi-square / Intersection | So sánh phân bố, đúng lý thuyết CBIR |
| Deep Embeddings | Cosine | Phù hợp với normalized vectors |
| BM25 | BM25 Score | Ranking truyền thống cho text |

### Why Histogram + Chi-square > CLIP on DeepFashion?
1. **Dataset bias**: DeepFashion có ảnh studio, nền trắng → màu sắc là tín hiệu mạnh
2. **Metric matching**: Chi-square được thiết kế để so sánh histogram
3. **CLIP semantic**: CLIP tối ưu cho semantic similarity, không phải instance retrieval

## 👥 Team

| Member | MSSV | Role |
|--------|------|------|
| Nguyễn Trọng Tất Thành | 23521455 | Track A + UI + FAISS |
| Trần Vạn Tấn | - | Track B + Evaluation + Report |

## 📄 License

MIT License

---

**CS336: Multimedia Information Retrieval 2025 - UIT**
