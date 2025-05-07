import os

def generate_training_list(data_root, subdir, test_list_file):
    # Read test videos list
    test_list_path = os.path.join(data_root, test_list_file)
    test_videos = set()
    with open(test_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                _, video_path = parts
                test_videos.add(video_path)

    subdir_path = os.path.join(data_root, subdir)
    if not os.path.isdir(subdir_path):
        print(f"Warning: Directory {subdir_path} does not exist.")
        return

    # Get all video names
    all_videos = [f for f in os.listdir(subdir_path) if f.endswith('.mp4')]
    relative_video_paths = [os.path.join(subdir, v) for v in all_videos]

    # Filter out test videos
    train_videos = [v for v in relative_video_paths if v not in test_videos]
    train_videos = [os.path.basename(v) for v in train_videos]

    # Save training videos list
    output_file = os.path.join(data_root, f"{subdir}_training_video_list.txt")
    with open(output_file, 'w') as f:
        for video in sorted(train_videos):
            f.write(f"{video}\n")

    print(f"Saved {len(train_videos)} training videos from {len(all_videos)} videos to {output_file}")

if __name__ == "__main__":
    data_root = "data/CelebDF-v2"
    subdirs = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]
    test_list_file = "List_of_testing_videos.txt"
    for subdir in subdirs:
        generate_training_list(data_root, subdir, test_list_file)
