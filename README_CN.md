# SAM 3 ModelScope 本地单图测试包（修正版）

本包直接读取本地 `sam3.pt`，不下载模型。

## 修正内容

SAM 3 的单图 point/box/mask 正确调用路径是：

```python
state = processor.set_image(image)
masks, scores, logits = model.predict_inst(
    state,
    point_coords=points,
    point_labels=labels,
    box=box,
)
```

不要直接调用：

```python
model.inst_interactive_predictor.set_image(image)
```

该 tracker 默认没有独立 backbone，直接调用会出现：

```text
AttributeError: 'NoneType' object has no attribute 'forward_image'
```

## 快速测试

```bash
cd sam3_modelscope_demo_fixed

MODEL_DIR=/common-data-32t/.cache/facebook/sam3 \
bash run_quick_test.sh
```

首次测试建议使用 FP32：

```bash
python test_sam3_modelscope_local.py \
  --model-dir /common-data-32t/.cache/facebook/sam3 \
  --checkpoint /common-data-32t/.cache/facebook/sam3/sam3.pt \
  --image inputs/coffee_scene.png \
  --mode point_box \
  --point 292 206 1 \
  --point 392 286 0 \
  --box 164 16 430 322 \
  --device cuda \
  --no-bf16 \
  --output-dir outputs/quick_test
```

## 支持输入

- text
- exemplar_box
- text_box
- point
- box
- point_box
- mask
- mask_refine


## BF16

Shell 脚本默认加入 `--no-bf16`。确认 FP32 正常后，可启用 BF16：

```bash
MODEL_DIR=/common-data-32t/.cache/facebook/sam3 \
USE_BF16=1 \
bash run_quick_test.sh
```

## 导出给 Qwen3-VL 的实例候选

新增 `qwen_candidates` 模式。它不会让 SAM 决定 target/reference，而是分别用
短概念生成候选实例，并输出稳定的候选 ID、mask、crop、候选拼图和 Qwen prompt。

### 直接指定 target/reference 概念

```bash
python test_sam3_modelscope_local.py \
  --model-dir /common-data-32t/.cache/facebook/sam3 \
  --checkpoint /common-data-32t/.cache/facebook/sam3/sam3.pt \
  --image /path/to/front_rgb/0.png \
  --mode qwen_candidates \
  --instruction "pick up the light bulb on the black socket" \
  --target-text "light bulb" \
  --reference-text "socket" \
  --threshold 0.25 \
  --candidate-pool-size 20 \
  --candidate-top-k 6 \
  --device cuda \
  --no-bf16 \
  --output-dir outputs/light_bulb_front
```

### 读取前一步 Qwen role_spec.json

如果已有 target/reference 语义解析结果：

```bash
python test_sam3_modelscope_local.py \
  --model-dir /common-data-32t/.cache/facebook/sam3 \
  --checkpoint /common-data-32t/.cache/facebook/sam3/sam3.pt \
  --image /path/to/front_rgb/0.png \
  --mode qwen_candidates \
  --role-spec-json /path/to/role_spec.json \
  --device cuda \
  --no-bf16 \
  --output-dir outputs/light_bulb_front
```

脚本会优先使用命令行参数；未显式提供时，从以下字段读取：

```json
{
  "instruction": "pick up the light bulb on the black socket",
  "role_spec": {
    "relation": "mounted on",
    "target": {"name": "light bulb"},
    "reference": {"name": "socket"}
  }
}
```

### 输出目录

```text
outputs/light_bulb_front/qwen_candidates/
├── original.png
├── numbered_candidates.png
├── candidate_grid.png
├── candidates.json
├── qwen_prompt.txt
├── masks/
│   ├── T0.png
│   ├── T1.png
│   └── R0.png
├── crops/
└── masked_crops/
```

候选 ID 约定：

- `T0, T1, ...`：target 类别候选；
- `R0, R1, ...`：reference 类别候选。

`candidates.json` 包含每个实例的 SAM score、像素 bbox、归一化 bbox、中心、面积、
mask 路径和 crop 路径。Qwen 只需要选择 ID，不再回归 bbox。

### 直接调用 Qwen3-VL 选择候选

```bash
python select_qwen3vl_candidate.py \
  --model-path /new-common-data/new-common-data/huggingface/Qwen3-VL-8B-Instruct \
  --candidate-dir outputs/light_bulb_front/qwen_candidates
```

输出：

```text
outputs/light_bulb_front/qwen_candidates/qwen_selection.json
```

典型结果：

```json
{
  "target_id": "T1",
  "reference_id": "R0",
  "relation": "mounted on",
  "target_evidence": "T1 is directly attached to R0",
  "reference_evidence": "R0 is the black socket",
  "uncertain": false,
  "uncertain_reason": null
}
```

### 候选过滤参数

- `--candidate-pool-size`：SAM 原始候选池大小；
- `--candidate-top-k`：每种角色最终保留数量；
- `--min-mask-area`：去除过小碎片；
- `--max-mask-area-ratio`：去除大背景 mask；
- `--mask-iou-threshold`：去除高度重复 mask；
- `--crop-padding-ratio`：候选 crop 的扩边比例。

对于 RLBench 小物体，建议先使用：

```text
--threshold 0.20~0.30
--candidate-pool-size 20
--candidate-top-k 6
--min-mask-area 20~40
```
