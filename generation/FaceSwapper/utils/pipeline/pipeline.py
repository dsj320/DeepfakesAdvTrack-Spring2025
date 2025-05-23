import dlib
import matplotlib.pyplot as plt
import matplotlib
import cv2
import pickle
import os
import json
import torch
import re
import copy
import time
import numpy as np
import scipy
import random
from omegaconf import OmegaConf
from PIL import Image, ImageChops
from tqdm import tqdm
from utils.portrait import Portrait
from einops import rearrange, repeat
from utils.blending.blending_mask import gaussian_pyramid, laplacian_pyramid, laplacian_pyr_join, laplacian_collapse
from skimage import io
from imutils import face_utils
from torchvision import transforms
from einops import rearrange


detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./checkpoints/shape_predictor_68_face_landmarks.dat')

def get_detect(image, iter):
    for i in range(iter + 1): # the bigger, the slower
        faces = detector(image,i)
        if len(faces) >= 1:
            break
    return faces
    
def get_lmk_ori(data_path = 'data/portrait', save_path = 'data/portrait/landmark'):
    all_lmk = {}
    for type in ['source', 'target']:
        all_lmk[type] = {}
        img_count = 0
        img_list = os.listdir(os.path.join(data_path, type))
        for img in tqdm(img_list, desc='image'):
            resize_flag = False
            image = cv2.imread(os.path.join(data_path, type, img))
            while image.shape[0] > 2000 or image.shape[1] > 2000:
                resize_flag = True
                image = cv2.resize(image, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            while image.shape[0] < 400 or image.shape[1] < 400:
                resize_flag = True
                image = cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = get_detect(imgray, 2)
            if len(faces) == 0:
                print('error', type, img)
                os.remove(os.path.join(data_path, type, img))
                continue
            if len(faces) > 1:
                print('> 1', type, img)
                face = dlib.rectangle(faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom())
                for i in range(1, len(faces)):
                    if abs(faces[i].right() - faces[i].left()) * abs(faces[i].top() - faces[i].bottom()) > abs(face.right() - face.left()) * abs(face.top() - face.bottom()):
                        face = dlib.rectangle(faces[i].left(), faces[i].top(), faces[i].right(), faces[i].bottom())
            else:
                face = faces[0]
            landmark = landmark_predictor(image, face)
            landmark = face_utils.shape_to_np(landmark)
            # new_name = f'{img_count:04d}.png'
            # os.rename(os.path.join(data_path, type, img), os.path.join(data_path, type, new_name))
            # if resize_flag:
            cv2.imwrite(os.path.join(data_path, type, img), image)
            all_lmk[type][img] = landmark
            img_count += 1
        print(all_lmk)
        print('type', type, 'img_count', img_count)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pickle.dump(all_lmk, open(os.path.join(save_path, 'landmark_ori.pkl'), 'wb'))

def get_lmk_256(data_path = 'data/portrait/align', save_path = 'data/portrait/landmark', error_path = 'data/portrait/error_img.json'):
    all_lmk = {}
    if os.path.exists(error_path):
        error_img = json.load(open(error_path, 'r'))
    else:
        error_img = {'source': [], 'target': []}
    for type in ['source', 'target']:
        all_lmk[type] = {}
        img_count = 0
        
        img_list = os.listdir(os.path.join(data_path, type))
        for img in tqdm(img_list, desc='image'):
            if os.path.exists(error_path):
                if img in error_img[type]:
                    continue
            image = cv2.imread(os.path.join(data_path, type, img))

            imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = get_detect(imgray, 2)
            if len(faces) == 0:
                print('error', type, img)
                error_img[type].append(img)
                continue
            if len(faces) > 1:
                print('> 1', type, img)
                face = dlib.rectangle(faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom())
                for i in range(1, len(faces)):
                    if abs(faces[i].right() - faces[i].left()) * abs(faces[i].top() - faces[i].bottom()) > abs(face.right() - face.left()) * abs(face.top() - face.bottom()):
                        face = dlib.rectangle(faces[i].left(), faces[i].top(), faces[i].right(), faces[i].bottom())
            else:
                face = faces[0]
            landmark = landmark_predictor(image, face)
            landmark = face_utils.shape_to_np(landmark)
            
            all_lmk[type][img] = landmark
            
            img_count += 1
        print('type', type, 'img_count', img_count)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pickle.dump(all_lmk, open(os.path.join(save_path, 'landmark_256.pkl'), 'wb'))
    json.dump(error_img, open(error_path, 'w'), indent=4)
    

    

def crop_ffhq(data_path = 'data/portrait', save_path = 'data/portrait/align', affine_path = 'data/portrait/affines.json', landmark_path = 'data/portrait/landmark/landmark_ori.pkl',\
            output_size=256, transform_size=1024, enable_padding=False, rotate_level=True, random_shift=0, retry_crops=False):
    print('Recreating aligned images...')
    # Fix random seed for reproducibility
    np.random.seed(12345)
    landmarks = pickle.load(open(landmark_path, 'rb'))
    affine_all = {}
    for type in ['source', 'target']:
        img_count = 0
        affine_all[type] = {}
        
        img_list = os.listdir(os.path.join(data_path, type))
        for img in tqdm(img_list, desc='image'): 
            lm = landmarks[type][img]
            lm_chin          = lm[0  : 17]  # left-right
            lm_eyebrow_left  = lm[17 : 22]  # left-right
            lm_eyebrow_right = lm[22 : 27]  # left-right
            lm_nose          = lm[27 : 31]  # top-down
            lm_nostrils      = lm[31 : 36]  # top-down
            lm_eye_left      = lm[36 : 42]  # left-clockwise
            lm_eye_right     = lm[42 : 48]  # left-clockwise
            lm_mouth_outer   = lm[48 : 60]  # left-clockwise
            lm_mouth_inner   = lm[60 : 68]  # left-clockwise

            # Calculate auxiliary vectors.
            eye_left     = np.mean(lm_eye_left, axis=0)
            eye_right    = np.mean(lm_eye_right, axis=0)
            eye_avg      = (eye_left + eye_right) * 0.5
            eye_to_eye   = eye_right - eye_left
            mouth_left   = lm_mouth_outer[0]
            mouth_right  = lm_mouth_outer[6]
            mouth_avg    = (mouth_left + mouth_right) * 0.5
            eye_to_mouth = mouth_avg - eye_avg

            # Choose oriented crop rectangle.
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

            # Load in ori image.
            src_file = os.path.join(data_path, type, img)

            image = Image.open(src_file)
            quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
            qsize = np.hypot(*x) * 2

            # Keep drawing new random crop offsets until we find one that is contained in the image
            # and does not require padding
            if random_shift != 0:
                for _ in range(1000):
                    # Offset the crop rectange center by a random shift proportional to image dimension
                    # and the requested standard deviation
                    c = (c0 + np.hypot(*x)*2 * random_shift * np.random.normal(0, 1, c0.shape))
                    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
                    if not retry_crops or not (crop[0] < 0 or crop[1] < 0 or crop[2] >= image.width or crop[3] >= image.height):
                        # We're happy with this crop (either it fits within the image, or retries are disabled)
                        break
                else:
                    # rejected N times, give up and move to next image
                    # (does not happen in practice with the FFHQ data)
                    print('rejected image')
                    return
            # Shrink.
            shrink = int(np.floor(qsize / output_size * 0.5))
            if shrink > 1:
                rsize = (int(np.rint(float(image.size[0]) / shrink)), int(np.rint(float(image.size[1]) / shrink)))
                # print(f'first opretion: resize, from {image.size} to {rsize}')
                image = image.resize(rsize, Image.BICUBIC)
                quad /= shrink
                qsize /= shrink
            # Crop.
            border = max(int(np.rint(qsize * 0.1)), 3)
            crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, image.size[0]), min(crop[3] + border, image.size[1]))
            IsCrop = False
            if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
                IsCrop = True
                crop = tuple(map(round, crop))
                # print(f'second operation: crop, {crop}')
                image = image.crop(crop) # (left, upper, right, lower)
                # location = [crop[0], crop[1], crop[2], crop[3]]
                quad -= crop[0:2]
            # Pad.
            pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - image.size[0] + border, 0), max(pad[3] - image.size[1] + border, 0))
            if enable_padding and max(pad) > border - 4:
                pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
                image = np.pad(np.float32(image), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                h, w, _ = image.shape
                y, x, _ = np.ogrid[:h, :w, :1]
                mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
                blur = qsize * 0.02
                image += (scipy.ndimage.gaussian_filter(image, [blur, blur, 0]) - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                image += (np.median(image, axis=(0,1)) - image) * np.clip(mask, 0.0, 1.0)
                image = Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255)), 'RGB')
                quad += pad[:2]

            # Transform(with rotation)
            quad = (quad + 0.5).flatten()
            assert(abs((quad[2] - quad[0]) - (quad[4] - quad[6])) < 1e-6 and abs((quad[3] - quad[1]) - (quad[5] - quad[7])) < 1e-6)
            
            if IsCrop:
                quad_new = [quad[0] + crop[0], quad[1] + crop[1], quad[2] + crop[0], quad[3] + crop[1], quad[4] + crop[0], quad[5] + crop[1], quad[6] + crop[0], quad[7] + crop[1]]
            else:
                quad_new = quad
            if shrink > 1:
                quad_new *= shrink
            # print(f'quad_new: {quad_new}', 'type', type, 'img', img)
            affine_rev = ((256*(quad_new[1] - quad_new[3]))/(quad_new[0]*quad_new[3] - quad_new[1]*quad_new[2] - quad_new[0]*quad_new[5] + quad_new[1]*quad_new[4] + quad_new[2]*quad_new[5] - quad_new[3]*quad_new[4]),
                        -(256*(quad_new[0] - quad_new[2]))/(quad_new[0]*quad_new[3] - quad_new[1]*quad_new[2] - quad_new[0]*quad_new[5] + quad_new[1]*quad_new[4] + quad_new[2]*quad_new[5] - quad_new[3]*quad_new[4]),
                        (256*(quad_new[0]*quad_new[3] - quad_new[1]*quad_new[2]))/(quad_new[0]*quad_new[3] - quad_new[1]*quad_new[2] - quad_new[0]*quad_new[5] + quad_new[1]*quad_new[4] + quad_new[2]*quad_new[5] - quad_new[3]*quad_new[4]),
                        -(256*(quad_new[3] - quad_new[5]))/(quad_new[0]*quad_new[3] - quad_new[1]*quad_new[2] - quad_new[0]*quad_new[5] + quad_new[1]*quad_new[4] + quad_new[2]*quad_new[5] - quad_new[3]*quad_new[4]),
                        (256*(quad_new[2] - quad_new[4]))/(quad_new[0]*quad_new[3] - quad_new[1]*quad_new[2] - quad_new[0]*quad_new[5] + quad_new[1]*quad_new[4] + quad_new[2]*quad_new[5] - quad_new[3]*quad_new[4]),
                        (256*(quad_new[0]*quad_new[3] - quad_new[1]*quad_new[2] - quad_new[0]*quad_new[5] + quad_new[1]*quad_new[4]))/(quad_new[0]*quad_new[3] - quad_new[1]*quad_new[2] - quad_new[0]*quad_new[5] + quad_new[1]*quad_new[4] + quad_new[2]*quad_new[5] - quad_new[3]*quad_new[4]))
            affine_all[type][img] = affine_rev
            # use affine to transform image
            affine = (-(quad[0] - quad[6])/transform_size, -(quad[0] - quad[2])/transform_size, quad[0],
                    -(quad[1] - quad[7])/transform_size, -(quad[1] - quad[3])/transform_size, quad[1])
            image = image.transform((transform_size, transform_size), Image.AFFINE, affine, Image.BICUBIC) # a, b, c, d, e, f
            
            if output_size < transform_size:
                image = image.resize((output_size, output_size), Image.BICUBIC)
            
            # Save aligned image.
            dst_subdir = os.path.join(save_path, type)
            os.makedirs(dst_subdir, exist_ok=True)
            image.save(os.path.join(dst_subdir, img))
            
            img_count += 1
        print('type {} finished, processed {} images'.format(type, img_count))
    # All done.
    json.dump(affine_all, open(affine_path, 'w'), indent=4)
    
# get affine_theta by 'data_preprocessing/': bash data_preprocessing/detection/run_detect_faces_portrait.sh && python data_preprocessing/detection/marge_mtcnn_portrait.py && python data_preprocessing/align/face_align_portrait.py

# swap faces by 'tests/faceswap_portrait.py': bash tests/face_swap.sh

# before paste, we need to solve the problem that the mask area of the generated face is not the same as the mask area of the original face


# repair image by mask and multiband blending. ref: https://aitechtogether.com/article/36091.html

#mask是target的mask路径
def repair_by_mask(tgt_path = 'data/portrait/align/target', swap_path = 'data/portrait/swap_res', save_path = 'data/portrait/swap_res_repair', mask_path = 'data/portrait/mask'):
    gen_type_list = os.listdir(swap_path)
    for type in tqdm(gen_type_list):
        src_list = os.listdir(os.path.join(swap_path, type))
        mask_type = 'mask'
        print('type: {}, mask_type: {}'.format(type, mask_type))
        for src in tqdm(src_list, desc = type, leave = False):
            img_list = os.listdir(os.path.join(swap_path, type, src))
            for img in tqdm(img_list, desc = src, leave = False):
                # if ((img == '0013.png' and src == '0006') is False) and ((img == '1086.png' and src == '0005') is False) and \
                #     ((img == '0208.png' and src == '0006') is False) and ((img == '0092.png' and src == '0005') is False) and \
                #         ((img == '0021.png' and src == '0002') is False) and ((img == '0021.png' and src == '0006') is False):
                #     continue
                swap_img = cv2.imread(os.path.join(swap_path, type, src, img))
                im1 = cv2.cvtColor(swap_img, cv2.COLOR_BGR2RGB)
                tgt_img = cv2.imread(os.path.join(tgt_path, img))
                im2 = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
                mask = matplotlib.image.imread(os.path.join(mask_path, mask_type, img))
                im1, im2 = np.int32(im1), np.int32(im2)
                mask = np.uint8(mask)
                
                gp_1, gp_2 = [gaussian_pyramid(im) for im in [im1, im2]]
                mask_gp  = [cv2.resize(mask, (gp.shape[1], gp.shape[0])) for gp in gp_1]
                lp_1, lp_2 = [laplacian_pyramid(gp) for gp in [gp_1, gp_2]]
                lp_join = laplacian_pyr_join(lp_1, lp_2, mask_gp)
                im_join = laplacian_collapse(lp_join)
                np.clip(im_join, 0, 255, out=im_join)
                im_join = np.uint8(im_join)
                
                os.makedirs(os.path.join(save_path, type, src), exist_ok=True)
                plt.imsave(os.path.join(save_path, type, src, img), im_join)

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



# paste images by affine matrix
def paste(data_root = 'data/portrait/swap_res_repair', dst_dir = 'data/portrait/swap_res_ori', ori_tgt_path = 'data/portrait/target'):
    gen_type_list = os.listdir(data_root)
    affine_all = json.load(open('data/portrait/affines.json', 'r'))
    for type in tqdm(gen_type_list):
        src_list = os.listdir(os.path.join(data_root, type))
        for src in tqdm(src_list, desc = type, leave = False):
            img_list = os.listdir(os.path.join(data_root, type, src))
            for img in tqdm(img_list, desc = src, leave = False):
                tgt_img = Image.open(os.path.join(ori_tgt_path, img)).convert('RGB')
                gen_img = tgt_img.copy()            
                gen_img256 = Image.open(os.path.join(data_root, type, src, img)).convert('RGB') # 256x256
                mask = Image.new('RGBA', (256, 256), (255, 255, 255))
                mask = mask.transform(tgt_img.size, Image.AFFINE, affine_all['target'][img], Image.BICUBIC)
                affine_img = gen_img256.transform(tgt_img.size, Image.AFFINE, affine_all['target'][img], Image.BICUBIC)
                gen_img.paste(affine_img, (0, 0), mask = mask)
                os.makedirs(os.path.join(dst_dir, type, src), exist_ok=True)
                gen_img.save(os.path.join(dst_dir, type, src, img))

def paste_v2(data_root='data/portrait/swap_res', 
             dst_dir='data/portrait/swap_res_ori', 
             ori_tgt_path='data/portrait/target', 
             affine_path='data/portrait/affines.json'):
    os.makedirs(dst_dir, exist_ok=True)
    affine_all = json.load(open(affine_path, 'r'))
    img_list = os.listdir(data_root)

    for img_name in tqdm(img_list, desc="Pasting back"):
        # 解析 target 名字
        if "_to_" not in img_name:
            print(f"Warning: unexpected file name format: {img_name}")
            continue
        tgt_name = img_name.split("_to_")[1]  # 得到 id4_0000_00000.png
        #这里我源图和目标图前面分别加了source_和target_，所以这里需要去掉
        tgt_name = tgt_name.replace('target_', '')
        tgt_name = tgt_name.replace('.png', '0.png')

        # 加载图像
        tgt_img = Image.open(os.path.join(ori_tgt_path, tgt_name)).convert('RGB')
        gen_img = tgt_img.copy()
        gen_img256 = Image.open(os.path.join(data_root, img_name)).convert('RGB')  # 256x256换脸修复结果

        # 生成mask
        mask = Image.new('RGBA', (256, 256), (255, 255, 255))
        mask = mask.transform(tgt_img.size, Image.AFFINE, affine_all['target'][tgt_name], Image.BICUBIC)

        # 生成affine warp后的小图
        affine_img = gen_img256.transform(tgt_img.size, Image.AFFINE, affine_all['target'][tgt_name], Image.BICUBIC)

        # 粘回去
        gen_img.paste(affine_img, (0, 0), mask=mask)

        # 保存
        out_path = os.path.join(dst_dir, img_name)
        gen_img.save(out_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            
if __name__ == '__main__':
    # get_lmk_ori()
    # crop_ffhq()
    # get_lmk_256()
    #paste_v2()