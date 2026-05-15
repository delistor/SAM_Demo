import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay

# =========================
# MediaPipe 初始化
# =========================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# =========================
# 人脸轮廓
# =========================

FACE_OUTLINE = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

# 使用更多点增强稳定性
USED_POINTS = list(range(468))

FEATHER_AMOUNT = 31

# =========================
# 获取 landmarks
# =========================

def get_landmarks(image):

    h, w = image.shape[:2]

    results = face_mesh.process(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    if not results.multi_face_landmarks:
        raise ValueError("未检测到人脸")

    face = results.multi_face_landmarks[0]

    points = []

    for lm in face.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    return np.array(points, dtype=np.int32)

# =========================
# Face mask
# =========================

def get_face_mask(image, landmarks):

    mask = np.zeros(image.shape[:2], dtype=np.float32)

    outline = landmarks[FACE_OUTLINE]

    cv2.fillConvexPoly(mask, outline, 1.0)

    mask = cv2.GaussianBlur(
        mask,
        (FEATHER_AMOUNT, FEATHER_AMOUNT),
        0
    )

    return mask

# =========================
# Delaunay Triangles
# =========================

def get_triangles(points):

    tri = Delaunay(points)

    return tri.simplices

# =========================
# Warp Triangle
# =========================

def warp_triangle(img1, img2, t1, t2):

    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    t1_rect = []
    t2_rect = []

    for i in range(3):
        t1_rect.append((
            t1[i][0] - x1,
            t1[i][1] - y1
        ))

        t2_rect.append((
            t2[i][0] - x2,
            t2[i][1] - y2
        ))

    mask = np.zeros((h2, w2, 3), dtype=np.float32)

    cv2.fillConvexPoly(
        mask,
        np.int32(t2_rect),
        (1.0, 1.0, 1.0),
        16,
        0
    )

    img1_rect = img1[y1:y1+h1, x1:x1+w1]

    size = (w2, h2)

    warp_mat = cv2.getAffineTransform(
        np.float32(t1_rect),
        np.float32(t2_rect)
    )

    warped = cv2.warpAffine(
        img1_rect,
        warp_mat,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    warped = warped * mask

    img2_section = img2[y2:y2+h2, x2:x2+w2]

    img2_section *= (1.0 - mask)

    img2_section += warped

    img2[y2:y2+h2, x2:x2+w2] = img2_section

# =========================
# 颜色校正
# =========================

def correct_colors(dst_img, warped_img):

    blur = 61

    if blur % 2 == 0:
        blur += 1

    dst_blur = cv2.GaussianBlur(dst_img, (blur, blur), 0)
    warped_blur = cv2.GaussianBlur(warped_img, (blur, blur), 0)

    warped_blur = np.where(warped_blur <= 1, 1, warped_blur)

    corrected = (
        warped_img.astype(np.float32)
        * dst_blur.astype(np.float32)
        / warped_blur.astype(np.float32)
    )

    corrected = np.clip(corrected, 0, 255)

    return corrected.astype(np.uint8)

# =========================
# Face Swap
# =========================

def face_swap(src_img, dst_img):

    src_points = get_landmarks(src_img)
    dst_points = get_landmarks(dst_img)

    src_used = src_points[USED_POINTS]
    dst_used = dst_points[USED_POINTS]

    triangles = get_triangles(dst_used)

    warped_face = np.zeros_like(dst_img, dtype=np.float32)

    # =========================
    # Piecewise affine warp
    # =========================

    for tri in triangles:

        t1 = np.float32([
            src_used[tri[0]],
            src_used[tri[1]],
            src_used[tri[2]]
        ])

        t2 = np.float32([
            dst_used[tri[0]],
            dst_used[tri[1]],
            dst_used[tri[2]]
        ])

        warp_triangle(
            src_img,
            warped_face,
            t1,
            t2
        )

    warped_face = np.clip(warped_face, 0, 255).astype(np.uint8)

    # =========================
    # 颜色校正
    # =========================

    warped_face = correct_colors(dst_img, warped_face)

    # =========================
    # Mask
    # =========================

    mask = get_face_mask(dst_img, dst_points)

    mask_uint8 = (mask * 255).astype(np.uint8)

    # =========================
    # Seamless Clone
    # =========================

    r = cv2.boundingRect(mask_uint8)

    center = (
        r[0] + r[2] // 2,
        r[1] + r[3] // 2
    )

    output = cv2.seamlessClone(
        warped_face,
        dst_img,
        mask_uint8,
        center,
        cv2.NORMAL_CLONE
    )

    return output

# =========================
# Main
# =========================

if __name__ == "__main__":

    src = cv2.imread(r"D:\face2face\5.png")
    dst = cv2.imread(r"D:\face2face\2.png")

    if src is None or dst is None:
        print("图片读取失败")
        exit()

    try:

        result = face_swap(src, dst)

        cv2.imwrite(
            r"D:\face2face\swapped_result.jpg",
            result
        )

        print("换脸完成")

        cv2.imshow("Result", result)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    finally:
        face_mesh.close()