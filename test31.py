import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import os
from PIL import Image

# ================== 配置部分 ==================
# 模型路径
MODEL_PATH = r"D:\face2face\inswapper_128.onnx"
# 图片路径
SOURCE_IMG_PATH = r"D:\face2face\inputs\source.jpg"  # 源图，提供要换上去的脸
TARGET_IMG_PATH = r"D:\face2face\inputs\3.png"  # 目标图，要被替换的脸
# 输出路径
OUTPUT_IMG_PATH = r"D:\face2face\outputs\swapped.jpg"
# 人脸检测器名称
FACE_ANALYSIS_NAME = 'buffalo_l'
    
# ================== 模型加载 ==================
print("[1/4] 正在初始化人脸检测器和换脸模型...")
# 初始化人脸检测器 (RetinaFace)
face_app = FaceAnalysis(name=FACE_ANALYSIS_NAME, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# 准备检测器，设置检测尺寸
face_app.prepare(ctx_id=0, det_size=(640, 640))

# 加载换脸模型
swapper = get_model(MODEL_PATH)

# ================== 辅助函数 ==================
def sort_faces_by_x(face_info):
    """
    Sort facial detection results according to the X coordinate of the bounding box
    """
    return face_info.bbox[0]

def load_and_detect(img_path):
    """
    Load an image and perform face detection, returning a sorted list of faces
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Read image (BGR format)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read the image: {img_path}")
    
    # Face detection
    faces = face_app.get(img)
    if len(faces) == 0:
        raise ValueError(f"No faces were detected in the image: {img_path}")
    
    # Sort according to X coordinate (from left to right)
    faces_sorted = sorted(faces, key=sort_faces_by_x)
    return img, faces_sorted

# ================== 核心处理 ==================
print("[2/4] Load and sort the faces of the source and target images...")
# 1. Load and sort images
source_img, source_faces = load_and_detect(SOURCE_IMG_PATH)
target_img, target_faces = load_and_detect(TARGET_IMG_PATH)

# Check if the number of faces matches the 11:11 requirement
if len(source_faces) != 11:
    print(f"Warning: The source image detected {len(source_faces)} faces, expected 11")
if len(target_faces) != 11:
    print(f"Warning: The target image detected {len(target_faces)} faces, expected 11")

# 2. Perform sequential swapping
print("[3/4] Performing batch face swapping...")
result_img = target_img.copy()
num_swaps = min(len(source_faces), len(target_faces))
for i in range(num_swaps):
    print(f"  - Face {i+1}/{num_swaps} swap...")
    source_face = source_faces[i]
    target_face = target_faces[i]
    # The swapper.get method will return a new image after swapping
    result_img = swapper.get(result_img, target_face, source_face, paste_back=True)

# 3. Save the result
print("[4/4] Saving the result...")
os.makedirs(os.path.dirname(OUTPUT_IMG_PATH), exist_ok=True)
cv2.imwrite(OUTPUT_IMG_PATH, result_img)
print(f"Batch face swap completed! The result has been saved to: {OUTPUT_IMG_PATH}")