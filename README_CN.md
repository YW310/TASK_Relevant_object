# SAM 3.1 ModelScope 本地模型测试包

本测试包不下载模型。它直接读取已经下载到本地的 ModelScope snapshot 目录。

## 目录结构

```text
sam31_modelscope_demo/
├── test_sam31_modelscope_local.py
├── run_quick_test.sh
├── run_sam31_examples.sh
├── requirements.txt
└── inputs/
    ├── coffee_scene.png
    ├── coffee_mask_prompt.png
    ├── shapes_scene.png
    ├── shapes_green_rectangle_mask.png
    └── prompts.json
```

## 前提

1. 已安装 SAM 3/SAM 3.1 源码环境，Python 中可以导入：

```python
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
```

2. 已经通过 ModelScope 下载模型，并知道 snapshot 的本地目录。

3. 推荐 CUDA 推理。CPU 推理资源消耗很大。

## 快速测试

```bash
unzip sam31_modelscope_demo.zip
cd sam31_modelscope_demo

MODEL_DIR=/path/to/modelscope/snapshot \
bash run_quick_test.sh
```

输出目录：

```text
outputs/quick_test/
```

## 批量测试所有输入方式

```bash
MODEL_DIR=/path/to/modelscope/snapshot \
bash run_sam31_examples.sh
```

可选环境变量：

```bash
DEVICE=cuda
PYTHON=python
OUT_ROOT=/path/to/output
```

CPU 测试：

```bash
MODEL_DIR=/path/to/modelscope/snapshot \
DEVICE=cpu \
bash run_quick_test.sh
```

## 内置测试输入

### coffee_scene.png

真实图像测试，目标是杯子：

- 文本：`cup`
- 前景点：`292 206 1`
- 背景点：`392 286 0`
- 框：`164 16 430 322`
- 外部 mask：`coffee_mask_prompt.png`

### shapes_scene.png

程序生成的几何图像，目标是中央绿色矩形：

- 文本：`green rectangle`
- 前景点：`372 245 1`
- 背景点：`160 255 0`、`655 260 0`
- 框：`244 114 500 378`
- 外部 mask：`shapes_green_rectangle_mask.png`

全部坐标保存在 `inputs/prompts.json`。

## 单项命令

### 文本提示

```bash
python test_sam31_modelscope_local.py \
  --model-dir /path/to/model \
  --image inputs/coffee_scene.png \
  --mode text \
  --text "cup" \
  --threshold 0.30 \
  --output-dir outputs/text
```

### 点提示

```bash
python test_sam31_modelscope_local.py \
  --model-dir /path/to/model \
  --image inputs/coffee_scene.png \
  --mode point \
  --point 292 206 1 \
  --point 392 286 0 \
  --output-dir outputs/point
```

### 框提示

```bash
python test_sam31_modelscope_local.py \
  --model-dir /path/to/model \
  --image inputs/coffee_scene.png \
  --mode box \
  --box 164 16 430 322 \
  --output-dir outputs/box
```

### 外部 mask 提示

```bash
python test_sam31_modelscope_local.py \
  --model-dir /path/to/model \
  --image inputs/coffee_scene.png \
  --mode mask \
  --mask-input inputs/coffee_mask_prompt.png \
  --point 292 206 1 \
  --point 392 286 0 \
  --output-dir outputs/mask
```

## 输出

每个模式会生成：

```text
overlay.png
mask_00.png
mask_01.png
result.json
```

测试根目录会生成 `summary.json`。模型推理结果不会预先包含在压缩包中。
