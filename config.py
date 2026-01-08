"""
Config file - Cấu hình cho toàn bộ project
"""
import os

# ====================
# PATHS
# ====================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
FEATURES_DIR = os.path.join(ROOT_DIR, "features")
INDEXES_DIR = os.path.join(ROOT_DIR, "indexes")
EVALUATION_DIR = os.path.join(ROOT_DIR, "evaluation")

# Track A - DeepFashion
DEEPFASHION_DIR = os.path.join(DATA_DIR, "deepfashion")
DEEPFASHION_IMAGES = os.path.join(DEEPFASHION_DIR, "img")
FEATURES_TRACK_A = os.path.join(FEATURES_DIR, "track_a")

# Track B - Flickr30k
FLICKR_DIR = os.path.join(DATA_DIR, "flickr30k")
FLICKR_IMAGES = os.path.join(FLICKR_DIR, "images")
FLICKR_CAPTIONS = os.path.join(FLICKR_DIR, "captions.txt")
FEATURES_TRACK_B = os.path.join(FEATURES_DIR, "track_b")

# ====================
# MODEL SETTINGS
# ====================
DEVICE = "cuda"  # hoặc "cpu" nếu không có GPU
BATCH_SIZE = 32
IMAGE_SIZE = 224

# CLIP
CLIP_MODEL = "ViT-B/32"  # hoặc "ViT-L/14" cho accuracy cao hơn

# ResNet
RESNET_MODEL = "resnet50"

# ====================
# FAISS SETTINGS
# ====================
FAISS_INDEX_TYPE = "Flat"  # "Flat", "IVF", "HNSW"
FAISS_NLIST = 100  # số clusters cho IVF
FAISS_NPROBE = 10  # số clusters search cho IVF

# ====================
# SEARCH SETTINGS
# ====================
TOP_K = 20  # số kết quả trả về

# ====================
# EVALUATION
# ====================
NUM_QUERIES = 50  # số queries để evaluate
K_VALUES = [1, 5, 10, 20]  # Recall@K

# ====================
# CREATE DIRECTORIES
# ====================
def create_dirs():
    """Tạo các thư mục cần thiết"""
    dirs = [
        DATA_DIR,
        DEEPFASHION_DIR,
        FLICKR_DIR,
        FEATURES_DIR,
        FEATURES_TRACK_A,
        FEATURES_TRACK_B,
        INDEXES_DIR,
        EVALUATION_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Created: {d}")

if __name__ == "__main__":
    create_dirs()
