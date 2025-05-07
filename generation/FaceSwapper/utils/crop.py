import os
import cv2
import dlib
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import pickle
from imutils import face_utils

def get_detect(image, iter):
    detector = dlib.get_frontal_face_detector()
    for i in range(iter + 1):
        faces = detector(image, i)
        if len(faces) >= 1:
            break
    return faces

def crop_ffhq(input_path, output_path, landmark_path=None, output_size=256, transform_size=1024, enable_padding=False, rotate_level=True):
    """
    简化版的crop_ffhq函数，用于人脸对齐和裁剪
    
    Args:
        input_path (str): 输入图片目录
        output_path (str): 输出图片目录
        landmark_path (str, optional): 人脸关键点文件路径
        output_size (int, optional): 输出图片大小，默认256
        transform_size (int, optional): 变换大小，默认1024
        enable_padding (bool, optional): 是否启用填充，默认False
        rotate_level (bool, optional): 是否启用旋转，默认True
    """
    print('Processing images...')
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 加载人脸关键点检测器
    landmark_predictor = dlib.shape_predictor('/path/to/your/shape_predictor_68_face_landmarks.dat')
    
    # 处理每张图片
    img_list = os.listdir(input_path)
    for img in tqdm(img_list, desc='Processing images'):
        try:
            # 读取图片
            image = cv2.imread(os.path.join(input_path, img))
            if image is None:
                print(f"Error reading image: {img}")
                continue
                
            # 转换为灰度图进行人脸检测
            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = get_detect(imgray, 2)
            print(f"faces: {len(faces)}")
            if len(faces) == 0:
                print(f"No face detected in: {img}")
                continue
                
            # 选择最大的人脸
            if len(faces) > 1:
                face = dlib.rectangle(faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom())
                for i in range(1, len(faces)):
                    if abs(faces[i].right() - faces[i].left()) * abs(faces[i].top() - faces[i].bottom()) > \
                       abs(face.right() - face.left()) * abs(face.top() - face.bottom()):
                        face = dlib.rectangle(faces[i].left(), faces[i].top(), faces[i].right(), faces[i].bottom())
            else:
                face = faces[0]
                
            # 获取人脸关键点
            landmark = landmark_predictor(image, face)
            landmark = face_utils.shape_to_np(landmark)
            
            # 提取关键点
            lm_chin = landmark[0:17]  # 下巴
            lm_eyebrow_left = landmark[17:22]  # 左眉毛
            lm_eyebrow_right = landmark[22:27]  # 右眉毛
            lm_nose = landmark[27:31]  # 鼻子
            lm_nostrils = landmark[31:36]  # 鼻孔
            lm_eye_left = landmark[36:42]  # 左眼
            lm_eye_right = landmark[42:48]  # 右眼
            lm_mouth_outer = landmark[48:60]  # 外嘴唇
            lm_mouth_inner = landmark[60:68]  # 内嘴唇
            
            # 计算辅助向量
            eye_left = np.mean(lm_eye_left, axis=0)
            eye_right = np.mean(lm_eye_right, axis=0)
            eye_avg = (eye_left + eye_right) * 0.5
            eye_to_eye = eye_right - eye_left
            mouth_left = lm_mouth_outer[0]
            mouth_right = lm_mouth_outer[6]
            mouth_avg = (mouth_left + mouth_right) * 0.5
            eye_to_mouth = mouth_avg - eye_avg
            
            # 选择裁剪矩形
            if rotate_level:
                x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
                x /= np.hypot(*x)
                x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
                y = np.flipud(x) * [-1, 1]
                c0 = eye_avg + eye_to_mouth * 0.1
            else:
                x = np.array([1, 0], dtype=np.float64)
                x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
                y = np.flipud(x) * [-1, 1]
                c0 = eye_avg + eye_to_mouth * 0.1
                
            # 加载原始图片
            src_file = os.path.join(input_path, img)
            image = Image.open(src_file)
            
            # 计算裁剪区域
            quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
            qsize = np.hypot(*x) * 2
            
            # 缩小
            shrink = int(np.floor(qsize / output_size * 0.5))
            if shrink > 1:
                rsize = (int(np.rint(float(image.size[0]) / shrink)), int(np.rint(float(image.size[1]) / shrink)))
                image = image.resize(rsize, Image.BICUBIC)
                quad /= shrink
                qsize /= shrink
                
            # 裁剪
            border = max(int(np.rint(qsize * 0.1)), 3)
            crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), 
                   int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), 
                   min(crop[2] + border, image.size[0]), min(crop[3] + border, image.size[1]))
            
            if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
                crop = tuple(map(round, crop))
                image = image.crop(crop)
                quad -= crop[0:2]
                
            # 变换
            quad = (quad + 0.5).flatten()
            affine = (-(quad[0] - quad[6])/transform_size, -(quad[0] - quad[2])/transform_size, quad[0],
                     -(quad[1] - quad[7])/transform_size, -(quad[1] - quad[3])/transform_size, quad[1])
            image = image.transform((transform_size, transform_size), Image.AFFINE, affine, Image.BICUBIC)
            
            if output_size < transform_size:
                image = image.resize((output_size, output_size), Image.BICUBIC)
                
            # 保存对齐后的图片
            image.save(os.path.join(output_path, img))
            
        except Exception as e:
            print(f"Error processing image {img}: {str(e)}")
            continue
            
    print('Processing completed!')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Input image directory',default='/path/to/your/input/images')
    parser.add_argument('--output_path', type=str, help='Output image directory',default='../data/CelebDF/CelebDF')
    parser.add_argument('--output_size', type=int, default=256, help='Output image size')
    parser.add_argument('--transform_size', type=int, default=1024, help='Transform size')
    parser.add_argument('--enable_padding', action='store_true', help='Enable padding',default=False)
    parser.add_argument('--rotate_level', action='store_true', help='Enable rotation',default=True)
    args = parser.parse_args()
    args.output_size=256
    args.transform_size=1024
    args.enable_padding=False
    args.rotate_level=True
    crop_ffhq(args.input_path, args.output_path, 
              output_size=args.output_size,
              transform_size=args.transform_size,
              enable_padding=args.enable_padding,
              rotate_level=args.rotate_level) 