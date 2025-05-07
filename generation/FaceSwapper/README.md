## 配置环境
```bash
conda env create -f env.yml
conda activate face
```




## 数据集与预训练模型

原始 FaceSwapper 所使用的预训练模型下载链接。

### 预训练模型

以下是各模块对应的预训练模型下载链接：

- **人脸识别模型（face recognition model）**  
  - [百度网盘下载（提取码：2g78）](https://pan.baidu.com/s/11qcEiBjAsQPXwIqKjOE-rQ?pwd=2g78)  
  - [Google Drive 下载](https://drive.google.com/file/d/1-lxc-jZGIFNdwFUXQ9tDS9OSuhadj6AC/view?usp=sharing)

- **人脸对齐模型（face alignment model）**  
  - [百度网盘下载（提取码：ejmj）](https://pan.baidu.com/s/1htwmXDi2Gev8l09oJpr_Mg?pwd=ejmj)  
  - [Google Drive 下载](https://drive.google.com/file/d/1lBt4x4P5qaClB2ZN_POBV-ue41hdlaoJ/view?usp=sharing)

- **换脸模型（face swapping model）**  
  - [百度网盘下载（提取码：bkru）](https://pan.baidu.com/s/1aIRX0twylUJ42z4sYhUaVA?pwd=bkru)  
  - [Google Drive 下载](https://drive.google.com/file/d/1Tb3V09wbaGe6SaiN3BZkOcCy7VJ0KYC8/view?usp=sharing)

下载完成后，请将预训练模型放入 `checkpoints/`，结构如下
```
./checkpoints
├── model_ir_se50.pth
├── wing.ckpt
└── faceswapper.ckpt
```



### 数据集准备

由于直接使用预训练模型效果有限，我们在 Celeb-DF-V2 数据集上进行了微调。本实验仅使用其中的**真实人脸图像**进行训练，总共约包含 1 万张图像。

目录结构如下：

```
data/
└── CelebDF/
    ├── CelebDF/                # 裁剪后的人脸图像（源自 Celeb-DF-V2 真实视频）
    ├── CelebDF_lm_images/      # 人脸关键点图像（供 FaceSwapper 使用）
    └── CelebDF_mask_images/    # 人脸区域遮罩图（供伪造区域提取）
```

其中：

- `CelebDF/` 下的图像需先通过 `utils/crop.py` 进行人脸裁剪，裁剪模型可从以下链接下载：

  > [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)

  请将模型文件放置到 `utils/` 脚本所需的默认路径下。

- `CelebDF_lm_images/` 为 FaceSwapper 所需的关键点图像，可使用脚本 `./core/wing.py` 生成。

- `CelebDF_mask_images/` 为实验所需的人脸遮罩图，可通过 `utils/get_mask.py` 生成，对应图像将保存在该目录中。

确保以上文件结构与路径设置一致，即可用于后续训练与测试。

## 训练模型


在开始训练前，请确保已正确设置 param.yaml 中的训练集数据路径，并将参数项 mode 修改为 'train'。完成配置后，执行以下命令启动训练：

```bash
python main.py
```


## 使用训练好的模型进行推理


训练完成后，可以使用以下命令生成换脸图像。生成的图像将保存在 `expr/results` 目录下。

请先在 `param.yaml` 中设置测试参数，**特别是将 `mode` 设置为 `'test'`**，并确认 `# directory for testing` 部分的路径已正确填写。


推理阶段，数据除了mask和landmark,还需要人脸解析图（我也不懂为啥，好像可以直接用训练过程中的验证代码，但我懒得改就直接用作者的了），具体来说，数据格式如下


```
test/
├── test/
├── test_lm_images/
├── test_mask_images/
├── test_parsing_images/
└── list.txt
```
其中，`test/`、`test_lm_images/`、`test_mask_images/` 与之前的数据结构保持一致。  
`test_parsing_images/` 可通过脚本 `utils/face_parsing/test.py` 生成。

然后运行以下命令开始推理：

```bash
python main.py
```

推理结果将保存在 `expr/results/CelebDF/` 目录下，包含以下三个子文件夹：

- `swapped_result_single`：生成的换脸图像；
- `swapped_result_afterps`：后处理后的换脸图像；
- `swapped_result_all`：拼接图像，包括源图（source）、目标图（target）、换脸图和后处理图。

图像命名格式为：

```
source_to_target.png
```

其中，source 图像提供身份信息，target 图像提供属性信息。
list.txt中表示需要交换的人脸，每一行为source_image.png  target_image.png,


## 完整推理流程

本代码在裁剪后的人脸图像上进行换脸操作，但最终我们希望将换脸结果贴回到完整原图中。该过程通过关键点的仿射变换实现，整体推理流程如下：

1. 将待换脸的图像分别放入 `utils/pipeline/data/portrait/source/` 和 `target/` 文件夹中。

2. 依次运行 `pipeline.py` 中的以下三个函数完成对齐与裁剪：

   - `get_lmk_ori()`
   - `crop_ffhq()`
   - `get_lmk_256()`

3. 生成的对齐图像将保存在 `utils/pipeline/data/portrait/align/` 目录中，可作为前述推理部分中 `test/test/` 的输入图像。

4. 推理完成后，将换脸生成的图像放入 `utils/pipeline/data/portrait/swap_res/` 文件夹。文件命名格式应为：

```
source_to_target.png
```

5. 最后，运行 `pipeline.py` 中的 `paste_v2()` 函数，该函数会将换脸结果通过仿射变换粘贴回原图，对应的完整输出图像将保存在：

```
utils/pipeline/data/portrait/swap_res_ori/
```

至此，即可获得贴回源图的最终换脸图像结果。














