## 📦 预训练权重下载

请从 HuggingFace 下载 GenConViT 所需的预训练模型权重文件：

🔗 [https://huggingface.co/Deressa/GenConViT/tree/main](https://huggingface.co/Deressa/GenConViT/tree/main)

下载完成后，请将下列两个 `.pth` 文件放入项目根目录下的 `./genconvit_weights/` 文件夹中。

最终的目录结构应如下所示：
```
genconvit_weights/
├── genconvit_ed_inference.pth
├── genconvit_vae_inference.pth
```
## Data Preparation

### Training Data


请将 CelebDF-v2 数据集整理为以下目录结构：

```
data/
└── CelebDF-v2/
    ├── Celeb-real/
    ├── Celeb-synthesis/
    ├── YouTube-real/
    └── List_of_testing_videos.txt
```

首次使用时，请依次运行以下命令生成训练图像及标签数据（**仅需运行一次**）：

```
python generate_celeb_v2_training_list.py
# 从 Celeb-real 提取人脸图像帧
python generate_data_from_video.py --video_dir=CelebDF-v2/Celeb-real --video_list_path=CelebDF-v2/Celeb-real_training_video_list.txt --fake=false --output_dir=train_data --fps=1 --device=cuda

# 从 Celeb-synthesis 提取伪造人脸图像帧
python generate_data_from_video.py --video_dir=CelebDF-v2/Celeb-synthesis --video_list_path=CelebDF-v2/Celeb-synthesis_training_video_list.txt --fake=true --output_dir=train_data --fps=1 --device=cuda

# 从 YouTube-real 提取额外真实图像帧
python generate_data_from_video.py --video_dir=CelebDF-v2/YouTube-real --video_list_path=CelebDF-v2/YouTube-real_training_video_list.txt --fake=false --output_dir=train_data --fps=1 --device=cuda
```


如果你希望在 CelebDF-v2 的基础上**增加其他自定义数据**用于训练，可以按照以下步骤操作：

1. **参考上述脚本结构**从自定义视频中提取图像帧（或直接使用已有图像）；
2. 确保生成的数据目录下包含以下文件：
```
train_data/
├── imgs/ # 存放所有图像文件
├── img_list.txt # 每行是一个图像文件名（如 xxx.png）
├── face_info.txt # 每行是对应图像的人脸框，格式为：x1 y1 x2 y2
└── labels.txt # 每行是图像对应的标签，0 表示真实，1 表示伪造
```
3. 所有文件应按行一一对应，例如第 `n` 行的 `img_list.txt`、`face_info.txt` 和 `labels.txt` 都对应同一张图像；其中face_info.txt中的人脸信息我们使用`MTCNN`中的人脸识别模型自动检测
4. 将上述内容直接整合到 `train_data/` 目录下，训练脚本将自动加载。

在我们的实验中，除了使用 CelebDF-v2 数据集作为主要训练来源，我们还引入了通过多种换脸方法（如FaceSwapper、E4S、DiffSwap 等）生成的伪造数据，进一步丰富伪造样本类型。



### Validation Data

Place the validation data as follows:

```
data/
└── val_data/
    ├── imgs/
    ├── img_list.txt
    ├── face_info.txt
    └── gts.xlsx (rename val_gts.xlsx to gts.xlsx)
```

## 🏋️‍♀️ 模型训练（Train）

使用以下命令开始训练 GenConViT 模型：

```
python train_genconvit.py --train_dir=data/train_data --val_dir=data/val_data --save_dir=checkpoints/genconvit --batch_size=128 --lr=1e-4 --epochs=20 --device=cuda:0
```


训练完成后，在./checkpoints/genconvit下面会保存有best_model.pth,后续用于推理



### test 1上的推理

将test1的数据放到data中，结构如下：

```
data/
└── test1_data/
    ├── imgs/
    ├── img_list.txt
    └── face_info.txt
```

在genconvit/下运行以下命令

# 请根据实际路径修改 --model-weights 的权重文件位置
```
python inference.py \
  --your-team-name=MoyuSquad \
  --data-folder=./data/test1_data \
  --model-weights=./checkpoints/genconvit/best_model.pth \
  --result-path=./results/test1_result
```

查看`test1_results/MoyuSquad.xlsx`（即为test1的预测结果）


