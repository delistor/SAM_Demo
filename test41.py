import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# ================== 配置部分（请根据实际路径修改） ==================
# 换脸模型路径（inswapper_128.onnx 的绝对路径）
MODEL_PATH = r"D:\face2face\inswapper_128.onnx"

# 源图（提供要换上去的脸）
SOURCE_IMG_PATH = r"D:\face2face\inputs\source.jpg"

# 目标图（要被替换的脸，本例中需要包含11张脸）
TARGET_IMG_PATH = r"D:\face2face\inputs\3.png"

# 输出结果路径
OUTPUT_IMG_PATH = r"D:\face2face\outputs\swapped.jpg"

# 人脸检测模型名称（buffalo_l）
FACE_ANALYSIS_NAME = 'buffalo_l'

# 本地存放 buffalo_l 模型的根目录（注意：是父目录，里面应有 buffalo_l 子文件夹）
LOCAL_MODEL_ROOT = r"D:\face2face"

# 是否使用 GPU（若没有 CUDA 或想强制 CPU，设为 False）
USE_GPU = True

# ================== 模型加载（完全离线） ==================
print("[1/4] 正在从本地加载人脸检测器和换脸模型...")

# 根据是否使用 GPU 选择推理后端
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if USE_GPU else ['CPUExecutionProvider']
# ctx_id: 0 表示第一个 GPU，-1 表示 CPU（当 providers 只有 CPU 时建议设为 -1）
ctx_id = 0 if USE_GPU else -1

# 初始化 FaceAnalysis，指定本地模型根目录
face_app = FaceAnalysis(
    name=FACE_ANALYSIS_NAME,
    root=LOCAL_MODEL_ROOT,
    providers=providers
)
face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

# 加载换脸模型
swapper = get_model(MODEL_PATH)

# ================== 辅助函数 ==================
def sort_faces_by_x(face_info):
    """
    根据人脸边界框的 X 坐标排序（从左到右）
    """
    return face_info.bbox[0]

def load_and_detect(img_path):
    """
    加载图像并进行人脸检测，返回图像和按 X 坐标排序的人脸列表
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图像不存在：{img_path}")

    # 读取图像（BGR 格式）
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像：{img_path}")

    # 人脸检测
    faces = face_app.get(img)
    if len(faces) == 0:
        raise ValueError(f"未检测到任何人脸：{img_path}")

    # 按 X 坐标升序排序（从左到右）
    faces_sorted = sorted(faces, key=sort_faces_by_x)
    return img, faces_sorted

# ================== 核心处理 ==================
print("[2/4] 正在加载源图和目标图，并检测人脸...")

# 1. 加载并检测源图、目标图
source_img, source_faces = load_and_detect(SOURCE_IMG_PATH)
target_img, target_faces = load_and_detect(TARGET_IMG_PATH)

# 检查人脸数量是否满足要求（本例期望 11 张，但只做警告，不中断）
if len(source_faces) != 11:
    print(f"警告：源图检测到 {len(source_faces)} 张人脸，期望 11 张")
if len(target_faces) != 11:
    print(f"警告：目标图检测到 {len(target_faces)} 张人脸，期望 11 张")

# 2. 按位置一一换脸
print("[3/4] 正在进行批量换脸...")
result_img = target_img.copy()
num_swaps = min(len(source_faces), len(target_faces))

for i in range(num_swaps):
    print(f"  - 正在交换第 {i+1}/{num_swaps} 张人脸...")
    source_face = source_faces[i]
    target_face = target_faces[i]
    # swapper.get 返回换脸后的新图像
    result_img = swapper.get(result_img, target_face, source_face, paste_back=True)

# 3. 保存结果
print("[4/4] 正在保存结果...")
os.makedirs(os.path.dirname(OUTPUT_IMG_PATH), exist_ok=True)
cv2.imwrite(OUTPUT_IMG_PATH, result_img)
print(f"批量换脸完成！结果已保存至：{OUTPUT_IMG_PATH}")


# pip show insightface
# Name: insightface
# Version: 0.7.3
# D:\face2face\models\buffalo_l