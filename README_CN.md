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

## Stage 4：对象级角色决策

`qwen3vl_object_role_decision.py` 默认使用 `DECISION_SCOPE=all`，为 episode 中的
**每一帧**输出一条 decision。每个当前帧 `t` 分别触发一次模型调用，并使用
`DECISION_WINDOW_FRAMES=3` 的滑动窗口 `[t-2, t-1, t]`（episode 开头按实际帧数
截断）。窗口内每一帧都会提供完整候选、空间关系和带 fused object ID 的 contact
sheet；输出 JSON 的 `frame_decisions` 保存全部逐帧结果，顶层 `decision` 保留最后一帧
结果用于兼容后续可视化。

每个候选对象最多提供两个不同相机视角，contact sheet 会显示 target/reference
语义先验。候选截断也优先保留具有角色语义证据的对象。默认
`USE_DECISION_HISTORY=0`，不会把上一帧的模型答案反馈给下一帧，以免首帧误判在整段
episode 中自我强化；确实需要该连续性先验时才设置为 `1`。

调试单帧时设置 `DECISION_SCOPE=single`，再用 `--decision-frame-id` 或
`--decision-frame first|last` 选择帧。对于没有独立参考物的单对象任务，
`reference_object_id=null` 是正常且可以高置信度的判断；仅用于识别 target 的颜色、
底座或局部结构不应被强制解释成 reference。

Target 选择采用明确的两阶段流程：Qwen 先根据 instruction 和视觉身份线索输出
`instruction_compatible_object_ids`；代码只在该集合内优先选择当前帧夹爪距离最小的
对象，距离相同时再选择在 `[t-2, t-1, t]` 中持续接近、接近幅度更大的对象。输出会
保留 `model_target_object_id` 和 `target_selection`，便于检查是否发生距离重排。

Stage 2 默认设置 `MAX_CENTROID_TO_CLOUD_DISTANCE_M=0.02`。如果融合中心到点云
最近点的距离超过 2 cm，说明中心落在较大的空隙中，该 candidate 会在分配 object ID
前被删除。设置为 `0` 可关闭此过滤；具体删除原因写入每帧的
`diagnostics.filtered_clusters`。

此外，Stage 2 会以 `COMPONENT_VOXEL_SIZE_M=0.008` 对点云做 3D voxel 连通区域
分析。如果最大区域少于 75%、第二大区域超过 20%，并且两个区域中心相距至少 2 cm，
就以 `multiple_large_3d_components` 原因删除 candidate。少于
`MIN_COMPONENT_POINTS=20` 的小区域按噪声忽略。相关阈值都可以通过 pipeline 环境
变量调整。

Stage 5 可视化会在原始 RGB 上重新绘制一次所有对象。若 `O2` 被判为 target，原标签
直接替换为 `T2`，不会同时保留 `O2` 或再绘制带透明底色的标签方块。bbox 默认使用
1 像素线宽，bbox、文字、中心点和点云高亮均采用透明叠加；可通过
`DECISION_VIZ_BOX_WIDTH` 和 `DECISION_VIZ_ANNOTATION_ALPHA` 调整。
