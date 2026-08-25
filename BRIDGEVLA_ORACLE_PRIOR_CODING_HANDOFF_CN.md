# BridgeVLA Oracle Prior Coding 交接文档（精简版）

> 更新时间：2026-08-06  
> 目标分支：BridgeVLA 官方仓库 `bridgevla` 分支  
> 目标：用 RLBench 真值 entity/site prior 测量 BridgeVLA 的性能上限

## 1. 实验目标与首版范围

回答一个问题：如果 BridgeVLA 在当前时刻知道正确的任务相关实体或 interaction site，它的 translation prediction 与 closed-loop success 最多能改善多少？

首版只做：

1. 从 RLBench segmentation mask 获取 simulator handle truth；
2. 将 handle 映射为 `target/reference/site`；
3. 为点云附加 role，并投影到 BridgeVLA 相同的三个正交视图；
4. 冻结 BridgeVLA，只给 translation heatmap 加 soft prior；
5. 先跑离线 keyframe replay，再跑 paired online evaluation。

首版不运行 Qwen/SAM，不改 rotation、gripper、collision heads，不实现完整 world repair。

## 2. Oracle 等级与泄漏边界

| ID | 信息 | 用途 |
| --- | --- | --- |
| B0 | 无 prior | 原始 BridgeVLA baseline |
| O1 | 所有当前合法 target/reference 实体 | 主实验，最干净 |
| O2 | 当前应操作的精确实例 | 单列，包含实例选择 oracle |
| O3-V | 当前 RGB-D 中可见的真实 interaction site | 主实验 |
| O3-F | simulator 完整 site，包括遮挡部分 | 理论上界，单列 |
| O4 | 专家下一 keyframe 位置高斯图 | 仅诊断，不得作为 relevant-object 主结果 |

主实验允许读取当前及历史 observation、instruction、task/variation、当前 gripper 和任务谓词。禁止读取未来帧、专家下一动作、下一 waypoint、episode 最终结果。

例如 `stack_blocks`：O1 应标出所有尚未放置且合法的方块；若根据专家轨迹只标下一步抓取的 `block2`，应归为 O2/O4，不能写成 O1。

## 3. 数据流与代码归属

```text
RLBench RGB / point cloud / handle mask
                 ↓
OracleRoleProvider：handle → target/reference/site
                 ↓
LabeledPointCloud：xyz + rgb + role bits
                 ↓
BridgeVLA 原有 preprocessing
                 ↓
同一 renderer、同一三视图像素坐标
        ┌────────┴────────┐
        ↓                 ↓
BridgeVLA RGB views   Oracle soft prior
        └────────┬────────┘
                 ↓
translation logits + prior bias
                 ↓
BridgeVLA 原 action decoder
```

Oracle 首版应直接实现于 BridgeVLA 仓库，因为它需要 PyRep scene handles、BridgeVLA preprocessing 和 renderer。本仓库只提供设计参考：[`task_schema.py`](task_schema.py)、[`dynamic_role_reasoning.py`](dynamic_role_reasoning.py) 和 [`CONSERVE3D_IDEA_OVERVIEW_CN.md`](CONSERVE3D_IDEA_OVERVIEW_CN.md)。

## 4. 开工前定位 BridgeVLA 插入点

```bash
git checkout bridgevla
git rev-parse HEAD
rg -n heatmap|trans.*logit|translation finetune/RLBench
rg -n render|orthographic|point.*cloud finetune/RLBench
rg -n decode|act\(|agent|eval.*episode finetune/RLBench
rg -n replay|dataset|augmentation|transform finetune/RLBench
```

必须找到点云 preprocessing、三视图 renderer、decode 前 translation logits、offline/online observation 入口。prior 必须在 translation decode 前融合，不要在 3D action 生成后做最近点吸附。

## 5. 最小模块与接口

```text
oracle_prior/
  role_registry.py       # task/variation → handle selector
  role_provider.py       # 当前状态 → active handles
  labeled_points.py      # mask + point cloud → point roles
  prior_renderer.py      # labeled points → three-view prior
  fusion.py              # translation logit fusion
  leakage_guard.py       # 禁止未来信息
  metrics.py             # paired metrics
tools/
  eval_oracle_prior_offline.py
  eval_oracle_prior_online.py
```

role 使用 bit flags，因为 site 同时属于 entity：

```python
ROLE_TARGET    = 1 << 0
ROLE_REFERENCE = 1 << 1
ROLE_SITE      = 1 << 2
ROLE_ATTACHED  = 1 << 3
```

核心契约：

```python
@dataclass
class LabeledPointCloud:
    xyz: Tensor          # [N, 3]
    rgb: Tensor          # [N, 3]
    handle_id: Tensor    # [N]
    role_bits: Tensor    # [N]
    site_score: Tensor   # [N], [0, 1]

@dataclass
class OraclePriorViews:
    target: Tensor       # [B, 3, H, W]
    reference: Tensor    # [B, 3, H, W]
    site: Tensor         # [B, 3, H, W]
    active: Tensor       # [B, 3, H, W]
```

所有 crop、filter、sampling 和 augmentation 必须对 `xyz/rgb/handle/role` 使用相同索引和几何变换。

## 6. RLBench truth 与 role registry

```python
CameraConfig(
    rgb=True,
    depth=True,
    point_cloud=True,
    mask=True,
    masks_as_one_channel=True,
)
```

RGB 编码 mask 使用 `handle = R + G*256 + B*256*256`；若已是单通道 handle mask，不要再次解码。官方参考：

- <https://github.com/stepjam/RLBench/blob/master/rlbench/observation_config.py>
- <https://github.com/stepjam/RLBench/blob/master/rlbench/backend/observation.py>

registry 示例：

```yaml
tasks:
  stack_blocks:
    task_family: pick_place_repeated
    target_selectors: [{name_regex: block.*}]
    reference_selectors: [{name_regex: stack.*|base.*}]
    site_strategy: {type: top_surface, surface_band_m: 0.015}
```

reset 后解析 name→handle；0 匹配或意外多匹配必须报错；variation override 显式配置；registry hash 写入结果。

## 7. Site 与三视图 prior

site 生成优先级：独立 shape handle → entity 几何子区域 → 完整 simulator mesh（O3-F）→ 专家下一 action point（O4）。

top surface 示例：

```python
z_threshold = torch.quantile(entity_xyz[:, 2], 0.95) - surface_band_m
is_site = is_entity & (xyz[:, 2] >= z_threshold)
```

prior renderer 必须使用 BridgeVLA 相同的 workspace bounds、normalization、augmentation、view matrices、分辨率与像素约定。禁止直接 resize RLBench 相机 mask。

hard mask `M` 转 soft prior：

\[
P_v(x)=\exp\left(-\frac{d(x,M_v)^2}{2\sigma^2}\right)
\]

要求 `P∈[0,1]`、mask 内最大值为 1、空 prior 有 fallback；测试 `sigma_px ∈ {4,8,16}`。

阶段选择：未持有物体时使用 target/site；持有后使用 reference/site；press/open/turn 优先 site。分别评测 inferred phase 与 oracle phase。

## 8. Translation heatmap 融合

设原 logits 为 `L`，active prior 为 `P`：`Q = floor + (1-floor)P`，然后在 decode 前计算 `L' = L + alpha * log(Q + eps)`。

核心要求：`alpha=0` 原样返回 logits；prior 与 logits shape 必须一致；prior 转到相同 device/dtype 并截断到 `[0,1]`；`floor>0` 防止 mask 外概率被清零。

validation sweep：`alpha={0,0.5,1,2,4}`、`floor={0.05,0.10,0.20}`、`sigma_px={4,8,16}`。只能在 validation seeds 上选参数，其他 action heads 保持不变。

## 9. 评测 protocol

### 离线 replay

同一 keyframe 分别解码原 logits 和 fused logits，与专家下一 translation 比较。报告：

- mean/median 3D error；
- PCK@2cm、PCK@5cm；
- correct-entity/site；
- rescue：baseline 错、oracle 对；
- harm：baseline 对、oracle 错；
- 按 task、phase、oracle level 分组结果。

### 在线 paired evaluation

同一个 task、variation、seed 运行：

```text
B0   baseline
O1   entity-set + inferred phase
O2   exact instance + inferred phase
O3V  visible site + inferred phase
O3P  visible site + oracle phase
O3F  full-state site + oracle phase
O4   action-point，仅诊断
```

pilot 每任务 25 个 paired episodes；正式建议至少 `25 episodes × 3 seeds`。保持 checkpoint、instruction、最大步数、renderer、decoder 和其他 heads 一致，报告 success、steps、wrong-entity/site、rescue/harm、failure taxonomy 和 paired bootstrap 95% CI。

## 10. 测试与验收

必须测试：

- registry 0/多匹配报错，O1 不访问专家下一实例；
- point attributes 经过 crop/sample 后仍对齐；
- 合成 cube/plane 三视图投影正确；
- prior 范围正确，空 prior 无 NaN；
- `alpha=0` 返回原 logits，uniform prior 不改变 softmax；
- `evidence_time <= action_time`，O4 必须显式授权；
- 一个 replay batch 可生成 RGB/prior views；
- 三视图 overlay 对齐；
- `alpha=0` action 差异 `<=1e-6`；
- online 两 episode smoke test 可完成；
- prior 关闭后 baseline 独立运行。

Go/No-Go 建议：hard-task 离线 error 相对下降 ≥10% 或 PCK@5cm 提升 ≥5pp；在线 hard-task 提升 ≥10pp 或总体 ≥5pp；简单任务 harm ≤2pp。

## 11. 实施顺序与后续

1. 固定 BridgeVLA commit、checkpoint、RLBench 环境和 baseline seeds；
2. 实现 handle mask、registry 和 O1；
3. 实现 point roles 与三视图 renderer，人工验证对齐；
4. 插入 frozen logit fusion，验证 `alpha=0` parity；
5. 跑 offline paired evaluation 并选参数；
6. 跑 online B0/O1/O2；
7. 扩展 O3-V/O3-F；
8. 最后加入隔离的 O4。

Oracle 达到 Go 标准后，保持相同 prior 接口，依次替换 provider：

```text
SimulatorOracleProvider
  → CurrentProjectProvider（SAM/Qwen/3D fusion）
  → LearnedStudentProvider
  → Persistent World/Repair Provider
```

结果解释：O1 提升说明相关实体选择有价值；只有 O3 提升说明应重点做 interaction-site；O3-F 显著高于 O3-V 说明需要持久世界记忆；translation 改善但 success 不变，则瓶颈已转移到 rotation/gripper/collision。
