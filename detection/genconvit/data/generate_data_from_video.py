import os
import cv2
import argparse
from tqdm import tqdm
from facenet_pytorch import MTCNN
import multiprocessing

def extract_frames_from_video(args):
    video_path, label, img_dir, fps_extract = args
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video: {video_path}")
        return [], []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(int(video_fps / fps_extract), 1)
    frame_idx = 0
    saved_idx = 0

    img_list = []
    img_labels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            img_name = f"{video_name}_frame{saved_idx:05d}.jpg"
            save_path = os.path.join(img_dir, img_name)
            cv2.imwrite(save_path, frame)
            img_list.append(img_name)
            img_labels.append(label)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return img_list, img_labels

def extract_frames(video_dir, video_list_path, label, img_dir, fps_extract=1, num_workers=8):
    os.makedirs(img_dir, exist_ok=True)

    with open(video_list_path, 'r') as f:
        video_list = [line.strip() for line in f if line.strip()]

    video_paths = [os.path.join(video_dir, fn) for fn in video_list]
    args_list = [(vp, label, img_dir, fps_extract) for vp in video_paths]

    img_list_all = []
    img_labels_all = []

    import multiprocessing
    with multiprocessing.Pool(num_workers) as pool:
        for img_list, img_labels in tqdm(pool.imap_unordered(extract_frames_from_video, args_list), total=len(args_list), desc="Extracting frames"):
            img_list_all.extend(img_list)
            img_labels_all.extend(img_labels)

    return img_list_all, img_labels_all

def face_rec(img_dir, img_list, device='cpu'):
    """
    Detect faces in images using MTCNN and return a list of bounding boxes.
    """
    mtcnn = MTCNN(keep_all=False, device=device)
    face_infos = []

    for img_name in tqdm(img_list, desc="Detecting faces in images"):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            face_infos.append([0, 0, 0, 0])
        else:
            x1, y1, x2, y2 = boxes[0]
            face_infos.append([x1, y1, x2, y2])

    return face_infos



def save_data(img_list, img_labels, face_infos, 
              img_list_path, img_label_path, face_info_path):
    """
    Append image filenames, labels, and face bounding boxes to respective files.
    """
    with open(img_list_path, 'a') as f:
        for name in img_list:
            f.write(name + '\n')
    print(f"Appended image list to {img_list_path}.")

    with open(img_label_path, 'a') as f:
        for label in img_labels:
            f.write(label + '\n')
    print(f"Appended image labels to {img_label_path}.")

    with open(face_info_path, 'a') as f:
        for box in face_infos:
            f.write(" ".join([f"{coord:.2f}" for coord in box]) + '\n')
    print(f"Appended face infos to {face_info_path}.")

def main():
    parser = argparse.ArgumentParser(description='Prepare deepfake dataset: extract frames and detect faces.')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing input videos')
    parser.add_argument('--video_list_path', type=str, required=True, help='File containing video names')
    parser.add_argument('--fake', type=str, required=True, choices=['true', 'false'], help='Indicates if videos are fake')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for imgs/, img_list.txt, img_label.txt, face_info.txt')
    parser.add_argument('--fps', type=float, default=1.0, help='Number of frames to extract per second (default: 1)')
    parser.add_argument('--device', type=str, default='cpu', help='Device for face detection ("cpu" or "cuda")')
    args = parser.parse_args()

    # Determine label based on 'fake' argument
    label = '0' if args.fake.lower() == 'true' else '1'

    # Define output paths
    img_dir = os.path.join(args.output_dir, 'imgs')
    img_list_path = os.path.join(args.output_dir, 'img_list.txt')
    img_label_path = os.path.join(args.output_dir, 'labels.txt')
    face_info_path = os.path.join(args.output_dir, 'face_info.txt')

    # Extract frames and detect faces
    img_list, img_labels = extract_frames(args.video_dir, args.video_list_path, label, img_dir, fps_extract=args.fps)
    face_infos = face_rec(img_dir, img_list, device=args.device)
    save_data(img_list, img_labels, face_infos, img_list_path, img_label_path, face_info_path)

if __name__ == '__main__':
    main()
