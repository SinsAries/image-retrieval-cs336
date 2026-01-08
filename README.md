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
| Method | Description | Dimension |
|--------|-------------|-----------|
| HSV Histogram | Color-based baseline | 512 |
| ResNet50 | Deep learning baseline | 2048 |
| **CLIP** | Multi-modal (main) | 512 |

### Track B (Text → Image)
| Method | Description |
|--------|-------------|
| BM25 | TF-IDF on captions |
| SBERT | Sentence embeddings |
| **CLIP** | Multi-modal (main) |

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/image-retrieval-cs336.git
cd image-retrieval-cs336

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create directories
python config.py
```

## 📂 Dataset

### Track A: DeepFashion In-shop
1. Download từ [official link](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)
2. Giải nén vào `data/deepfashion/`

### Track B: Flickr30k
1. Download từ [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
2. Giải nén vào `data/flickr30k/`

## 🚀 Usage

### Step 1: Extract Features

```bash
# Track A - Histogram
python src/extract/histogram.py

# Track A - ResNet50
python src/extract/resnet.py

# Track A & B - CLIP
python src/extract/clip_feat.py
```

### Step 2: Build Index

```bash
python src/index/faiss_index.py
```

### Step 3: Run UI

```bash
streamlit run app.py
```

Mở browser: http://localhost:8501

## 📊 Evaluation

```bash
python src/evaluate/metrics.py
```

Metrics:
- Precision@K (K=1,5,10,20)
- Recall@K
- mAP (Mean Average Precision)

## 📁 Project Structure

```
image-retrieval-cs336/
├── data/
│   ├── deepfashion/         # Track A dataset
│   └── flickr30k/           # Track B dataset
├── features/
│   ├── track_a/             # Extracted features
│   └── track_b/
├── indexes/                 # FAISS indexes
├── src/
│   ├── extract/             # Feature extraction
│   │   ├── histogram.py
│   │   ├── resnet.py
│   │   └── clip_feat.py
│   ├── index/
│   │   └── faiss_index.py
│   ├── search/
│   │   └── search.py
│   └── evaluate/
│       └── metrics.py
├── app.py                   # Streamlit UI
├── config.py               # Configuration
├── requirements.txt
└── README.md
```

## 👥 Team

| Member | Role |
|--------|------|
| Người A | Track A + UI |
| Người B | Track B + Report |

## 📄 License

MIT License

---

**CS336: Multimedia Information Retrieval 2025**
