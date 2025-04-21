#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import re
import argparse
from tqdm import tqdm
import numpy as np

def create_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_frame(video_path, frame_num):
    """Extract a specific frame from a video."""
    if not os.path.exists(video_path):
        print(f"Warning: Video not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_num >= total_frames:
        print(f"Warning: Frame {frame_num} exceeds total frames {total_frames} in {video_path}")
        cap.release()
        return None
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Warning: Could not read frame {frame_num} from {video_path}")
        return None
    
    return frame

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos based on image_list.txt')
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to the CelebDF-v2 dataset directory containing the videos')
    parser.add_argument('--output_dir', type=str, default='extracted_frames',
                        help='Directory to save extracted frames')
    parser.add_argument('--image_list', type=str, default='image_list.txt',
                        help='Path to the image_list.txt file')
    
    args = parser.parse_args()
    
    # Create output directories
    target_dir = os.path.join(args.output_dir, 'target')
    source_dir = os.path.join(args.output_dir, 'source')
    create_dir(target_dir)
    create_dir(source_dir)
    
    # Set of source identities we've already processed
    processed_sources = set()
    
    # Read image list
    with open(args.image_list, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]
    
    print(f"Processing {len(image_names)} images...")
    
    # Process each image in the list
    for img_name in tqdm(image_names):
        # Parse the image name, e.g., id0_id1_0000_00120.png
        parts = img_name.split('_')
        if len(parts) < 4 or not img_name.endswith('.png'):
            print(f"Warning: Invalid image name format: {img_name}")
            continue
        
        target_id = parts[0]  # id0 (目标人脸ID)
        source_id = parts[1]  # id1 (源身份ID)
        video_num = parts[2]  # 0000 (视频编号)
        frame_num = int(parts[3].split('.')[0])  # 00120 (帧号)
        
        # 提取目标帧 (从id0_0000.mp4中提取特定帧)
        target_video_path = os.path.join(args.dataset_path, f"{target_id}_{video_num}.mp4")
        target_frame = extract_frame(target_video_path, frame_num)
        
        if target_frame is not None:
            # 保存目标帧为id0_0000_00120.png格式
            target_frame_name = f"{target_id}_{video_num}_{parts[3].split('.')[0]:05d}.png"
            target_frame_path = os.path.join(target_dir, target_frame_name)
            cv2.imwrite(target_frame_path, target_frame)
        
        # 提取源身份帧 (只需处理每个源身份一次)
        if source_id not in processed_sources:
            # 从id1_0000.mp4中提取第0帧作为源身份
            source_video_path = os.path.join(args.dataset_path, f"{source_id}_0000.mp4")
            source_frame = extract_frame(source_video_path, 0)
            
            if source_frame is not None:
                # 保存源身份帧为id1_0000_00000.png格式
                source_frame_path = os.path.join(source_dir, f"{source_id}_0000_00000.png")
                cv2.imwrite(source_frame_path, source_frame)
                processed_sources.add(source_id)
    
    print(f"提取完成!")
    print(f"目标帧保存在: {target_dir}")
    print(f"源身份帧保存在: {source_dir}")

if __name__ == "__main__":
    main() 