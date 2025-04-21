#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从image_list.txt读取每一行，解析格式并打印源图像和目标图像路径
文件命名格式：id目标_id源_视频编号_帧编号.png
例如：id0_id1_0000_00060.png 表示source为ID1，target为ID0，视频0的第60帧
然后执行diffswap，将结果保存到result目录下，命名为id目标_源id_视频编号_帧编号.png
"""

import os
import re
import shutil
import cv2
from tqdm import tqdm
import numpy as np
# 配置路径
IMAGE_LIST_PATH = 'process_data/image_list.txt'
BASE_DIR = 'process_data/extracted_frames'


def parse_image_name(filename):
    """解析图像文件名，提取目标ID和源ID"""
    pattern = r'id(\d+)_id(\d+)_(\d+)_(\d+)\.png'
    match = re.match(pattern, filename)
    if match:
        target_id = match.group(1)
        source_id = match.group(2)
        video_id = match.group(3)
        frame_id = match.group(4)
        return {
            'target_id': target_id,
            'source_id': source_id,
            'video_id': video_id,
            'frame_id': frame_id,
            'filename': filename
        }
    return None



#添加进度

def process_image(source_id,target_id,video_id,frame_id,source_path,target_path):
    # """处理图像"""
    print(source_path,target_path)
    # 清理/设置workspace
    if os.path.exists('data/portrait_jpg'):
        os.system('rm -rf data/portrait_jpg')
    if os.path.exists('data/portrait'):
        os.system('rm -rf data/portrait')
    os.makedirs('data/portrait_jpg/source')
    os.makedirs('data/portrait_jpg/target')

    #import pdb;pdb.set_trace()

    # 复制图像
    shutil.copy(source_path, 'data/portrait_jpg/source')
    shutil.copy(target_path, 'data/portrait_jpg/target')

    # 运行diffswap
    os.system('python3 pipeline.py')

    # 确保结果目录存在
    os.makedirs('result', exist_ok=True)
    os.makedirs('result_compare', exist_ok=True)

    # 保存结果，其中结果在data/portrait/swap_res_ori/diffswap_0.01/id源_0000_00000/下面，图片为id目标_视频编号_帧编号.png
    source_result_path = os.path.join('data/portrait/swap_res_ori', f"diffswap_0.01", f"id{source_id}_0000_00000")
    source_image_path = os.path.join(source_result_path, f"id{target_id}_{video_id}_{frame_id}.png")

    # 保存为id目标_源id_视频编号_帧编号.png
    dest_image_path = os.path.join('result', f"id{target_id}_id{source_id}_{video_id}_{frame_id}.png")
    
    # 复制源图像到目标位置
    shutil.copy(source_image_path, dest_image_path)
    print(f"结果已保存到: {dest_image_path}")

    # 将源图，目标图，结果图拼接成一张图，1x3的图，放到result_compare目录下,命名为id目标_源id_视频编号_帧编号.png
    #为了方便拼接，采用对齐后的，在路径data/portrait/align/下面

    # source_path = os.path.join('data/portrait/align/source', f"id{source_id}_0000_00000.png")
    # target_path = os.path.join('data/portrait/align/target', f"id{target_id}_{video_id}_{frame_id}.png")
    # #参考这个
    # source_result_path = os.path.join('data/portrait/swap_res_ori', f"diffswap_0.01", f"id{source_id}_0000_00000")
    # source_image_path = os.path.join(source_result_path, f"id{target_id}_{video_id}_{frame_id}.png")
    # dest_image_path = os.path.join('data/portrait/swap_res', f"diffswap_0.01",f"id{source_id}_0000_00000")
    # img_source = cv2.imread(source_path)
    # img_target = cv2.imread(target_path)
    # img_result = cv2.imread(dest_image_path)
    # # 拼接成一张图
    # compare_image_path = os.path.join('result_compare', f"id{target_id}_id{source_id}_{video_id}_{frame_id}.png")
    # img = np.concatenate((img_source, img_target, img_result), axis=1)
    # cv2.imwrite(compare_image_path, img)

    # print(f"处理完成: {source_path} -> {target_path} -> {compare_image_path}")
    
    

def process_image_list():
    """处理图像列表文件"""
    if not os.path.exists(IMAGE_LIST_PATH):
        print(f"错误：文件 {IMAGE_LIST_PATH} 不存在")
        return
    with open(IMAGE_LIST_PATH, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    for id,line in tqdm(enumerate(lines),total=len(lines)):
        info = parse_image_name(line)
        #import pdb;pdb.set_trace()
        print(f"line: {id}")
        if info:
            # 源图像路径 (格式为 id源_0000_00000.png)
            source_path = os.path.join(BASE_DIR, 'source', f"id{info['source_id']}_0000_00000.png")
            # 目标图像路径 (格式为 id目标_视频编号_帧编号.png)
            target_path = os.path.join(BASE_DIR, 'target', f"id{info['target_id']}_{info['video_id']}_{info['frame_id']}.png")
            # 执行diffswap
            process_image(info['source_id'],info['target_id'],info['video_id'],info['frame_id'],source_path,target_path)
    return ans
if __name__ == '__main__':
    process_image_list()
    


    
        