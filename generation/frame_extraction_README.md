# 帧提取工具 (Frame Extraction Tool)

这个脚本用于从CelebDF-v2数据集中提取图像，根据DeepfakesAdvTrack-Spring2025的要求。

## 功能

1. 从`image_list.txt`中读取所有图片名称（格式如`id0_id1_0000_00120.png`）
2. 从目标视频中提取特定帧（从`id0_0000.mp4`中提取第120帧）并保存到`target`文件夹
3. 从源身份视频中提取第0帧（从`id1_0000.mp4`中提取第0帧）并保存到`source`文件夹

## 示例说明

对于图片名称 `id0_id1_0000_00120.png`：
- `id0` 是目标人脸ID
- `id1` 是源身份ID
- `0000` 是视频编号
- `00120` 是帧号

脚本会：
1. 提取 `id0_0000.mp4` 中的第120帧，保存为 `target/id0_0000_00120.png`
2. 提取 `id1_0000.mp4` 中的第0帧，保存为 `source/id1_0000_00000.png`

## 依赖项

- Python 3.7+
- OpenCV (`opencv-python`)
- tqdm
- numpy
- argparse

安装依赖项：

```bash
pip install opencv-python tqdm numpy
```

## 使用方法

1. 确保已下载CelebDF-v2数据集并放置在可访问的位置
2. 运行以下命令提取帧：

```bash
python extract_frames.py --dataset_path /path/to/celebdf_v2_videos --output_dir extracted_frames
```

### 参数说明

- `--dataset_path` (必需): CelebDF-v2数据集中包含视频的目录路径
- `--output_dir` (可选): 保存提取帧的目录，默认为 'extracted_frames'
- `--image_list` (可选): 图像列表文件的路径，默认为 'image_list.txt'

## 输出结构

```
extracted_frames/
├── target/
│   ├── id0_0000_00000.png
│   ├── id0_0000_00060.png
│   └── ...
└── source/
    ├── id1_0000_00000.png
    ├── id16_0000_00000.png
    └── ...
```

## 注意事项

1. 脚本会自动跳过无法处理的文件名或找不到的视频
2. 对于每个源身份（如id1），只会提取一次，以避免重复处理
3. 如果指定的帧号超出视频总帧数，将输出警告信息 