import dlib
import numpy as np
import cv2
import os
from imutils import face_utils
from tqdm import tqdm

def get_face_mask_dlib_batch(
    input_dir,
    output_dir,
    predictor_path='/path/to/your/shape_predictor_68_face_landmarks.dat'
):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    os.makedirs(output_dir, exist_ok=True)

    img_list = os.listdir(input_dir)
    for img_name in tqdm(img_list, desc="Processing images"):
        if not (img_name.endswith('.jpg') or img_name.endswith('.png')):
            continue

        img_path = os.path.join(input_dir, img_name)
        save_path = os.path.join(output_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        mask = np.zeros_like(gray)

        for rect in faces:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            hull = cv2.convexHull(shape)
            cv2.fillConvexPoly(mask, hull, 255)

        cv2.imwrite(save_path, mask)

# 调用示例
get_face_mask_dlib_batch(
    input_dir='/path/to/your/input/images',  # 输入图片文件夹
    output_dir='/path/to/your/output/mask'  # 输出mask文件夹
)

