# RLBench 任务相关物体识别项目交接文档

## 1. 交接目标

本文档用于帮助接手人完成以下事项：

1. 理解项目的六个处理阶段及其输入输出关系。
2. 准备 Python、模型和 RLBench 数据环境。
3. 完成一次小样本端到端运行。
4. 能够跳过高成本阶段，单独调试融合、决策或可视化。
5. 根据输出和日志定位常见问题。

项目阶段总结见 [`PROJECT_REPORT_CN.md`](PROJECT_REPORT_CN.md)，完整参数说明以 [`README.md`](README.md) 和 [`run_full_pipeline.sh`](run_full_pipeline.sh) 为准。

## 2. 当前状态

| 项目 | 状态 |
| --- | --- |
| 主分支 | `main` |
| 主入口 | `run_full_pipeline.sh` |
| Python 要求 | Python 3.10 或更高版本 |
| 推荐运行环境 | Linux 或 Git Bash + CUDA GPU |
| 默认运行阶段 | 阶段 1、2、3 |
| 可选阶段 | 阶段 4、5、6 |
| 自动化测试 | 7 个 `pytest` 测试模块 |
| 模型权重 | 不在仓库内，不自动下载 |
| 样例输出 | 当前仓库未提交 `outputs/` |

本次交接检查中，`run_full_pipeline.sh` 已通过 `bash -n` 语法检查。本机基础 Conda 环境为 Python 3.10.12，但未安装 `pytest`，因此需要在正式模型环境中重新执行测试集。

研究主线已收敛为 **Retrospective World Topology Repair**：当前 `frame_fused_candidates.json`、逐帧 O-ID 和 `object_predictions.json` 仍是现有 pipeline 的兼容输出，不应被解释为最终物理身份。高层入口是 [`CONSERVE3D_THESIS_CARD_CN.md`](CONSERVE3D_THESIS_CARD_CN.md)，完整方案是 [`RESEARCH_PROPOSAL_CONSERVE3D_CN.md`](RESEARCH_PROPOSAL_CONSERVE3D_CN.md)，整体叙事是 [`CONSERVE3D_IDEA_OVERVIEW_CN.md`](CONSERVE3D_IDEA_OVERVIEW_CN.md)。

## 3. 系统架构与数据流

| 阶段 | 入口文件 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| 1. 角色解析与候选生成 | `qwen_role_sam3_candidate_episode.py` | 指令、RGB、Qwen3-VL、SAM3 | `role_spec.json`、`episode_candidates.json`、掩码与裁剪图 |
| 2. 多视角融合与跟踪 | `multiview_candidate_fusion.py` | 候选、深度、相机参数 | 当前 observation-level `frame_fused_candidates.json`、逐帧物体 JSON/NPZ、可选 `object_summary.json`；后续作为 world belief 输入 |
| 3. 融合可视化 | `visualize_fused_candidates.py` | 融合结果、RGB、相机参数 | `viz/*_montage.png`、`sanity_report.json` |
| 4. 目标/参考物体决策 | `qwen3vl_object_role_decision.py` | 当前 `object_summary.json`、Qwen3-VL | `object_predictions.json`、`decision_inputs/`；目标架构改为读取 typed world queries |
| 5. 决策可视化 | `stage4_visualize_decision.py` | 决策结果、融合结果 | `viz_decision/`、`decision_visualization.json` |
| 6. 阶段对比 | `stage6_visualize_stage_montage.py` | 阶段 1、3、5 结果 | `viz_compare/` |

默认执行阶段 1、2、3。设置 `SKIP_DECISION=0` 后执行阶段 4 和 5；再设置 `SKIP_STAGE_COMPARE=0` 执行阶段 6。

### 3.1 接手时必须区分的两层

- **当前工程层：** SAM3 候选、融合 O-ID、时域传播和 Qwen 角色输出，用于运行、可视化和回归；
- **目标研究层：** immutable evidence ledger、versioned entity–site–event belief、`Conserve–Test–Repair` 以及 typed world-query addresses。

不要通过延长 TTL、增加 appearance feature 或强制单一 ID 来掩盖不可辨识性。优先检查候选 evidence、world hypothesis、topology edit provenance 和 role query 是否一致。

## 4. 仓库目录与模块职责

### 主流程入口

| 文件 | 职责 |
| --- | --- |
| `run_full_pipeline.sh` | 统一编排所有阶段，集中管理环境变量和输出路径 |
| `qwen_role_sam3_candidate_episode.py` | episode 级语义角色解析和 SAM3 候选生成 |
| `multiview_candidate_fusion.py` | 深度反投影、多视角融合、过滤与跨帧跟踪 |
| `qwen3vl_object_role_decision.py` | 逐帧目标/参考物体判断和动态角色推理 |
| `visualize_fused_candidates.py` | 融合结果回投影、三维点云和诊断报告 |
| `stage4_visualize_decision.py` | 最终决策叠加可视化 |
| `stage6_visualize_stage_montage.py` | 候选、融合和决策结果的对比图 |

### 公共模块

| 文件 | 职责 |
| --- | --- |
| `common_io.py` | 原子 JSON 输出、CSV 解析、自然排序 |
| `sam3_runtime.py` | SAM3 checkpoint 搜索、autocast 和张量标准化 |
| `mask_geometry.py` | 掩码、连通分量和二维包围盒计算 |
| `camera_geometry.py` | RLBench 深度解码、相机元数据、反投影和重投影 |
| `fusion_types.py` | 融合数据结构和角色常量 |
| `fusion_matching.py` | 同相机 NMS、跨相机兼容性和 Hungarian 分配 |
| `fused_candidate_io.py` | 版本化融合结果读取和点云延迟加载 |
| `dynamic_role_reasoning.py` | 任务谓词、夹爪事件、物体状态和关系推理 |
| `task_schema.py` | 任务动作族和目标谓词结构 |
| `visualization_utils.py` | 多阶段共享的颜色和标注工具 |

### 其他入口

- `run_quick_test.sh`：本地 SAM3 单图 smoke test。
- `run_qwen_s1.sh`：批量运行较早的 Qwen episode grounding 入口。
- `demo_sam3.py`：SAM3 直接调用示例。
- `qwen3_bbox_guided_sam3_demo.py`：Qwen 框提示结合 SAM3 的示例。
- `schemas/`：阶段 2 三层 JSON 输出 Schema。
- `tests/`：不加载完整模型的单元测试和回归测试。

## 5. 环境准备

### 5.1 基础要求

- Python 3.10 或更高版本。
- 推荐使用 CUDA GPU；CPU 仅适合部分检查或小规模调试。
- 本地 Qwen3-VL checkpoint。
- 本地 SAM3 代码、配置和 checkpoint，默认从 `<SAM_MODEL_DIR>/sam3.pt` 等位置搜索。
- 包含 RGB、深度和相机元数据的 RLBench episode。
- Bash 环境：Linux shell、WSL 或 Git for Windows 自带的 Git Bash。

### 5.2 Python 依赖

先安装仓库列出的基础依赖：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 当前只包含基础计算、绘图和测试依赖。运行模型还需要目标环境能够导入：

- SAM3 实现及其依赖；
- 支持 Qwen3-VL 的 Transformers 及相关依赖；
- 与服务器 CUDA 驱动兼容的 PyTorch；
- 可选 FlashAttention 2；未安装时阶段 4 可退回 SDPA 或 eager。

仓库没有锁定这些模型依赖的准确版本。交接时应优先导出已验证服务器环境，而不是在生产环境中临时选择最新版。

建议在已验证环境执行并保存以下信息：

```bash
python --version
python -m pip freeze > environment-freeze.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

如果 SAM3 不是通过 Pip 安装，还应记录其源码目录、commit 和加入 `PYTHONPATH` 的方式。

### 5.3 模型与数据权限

在开始运行前确认：

- 接手人能够读取 Qwen3-VL 模型目录。
- 接手人能够读取 SAM3 模型目录和 checkpoint。
- 接手人能够读取 RLBench episode 数据。
- 输出目录所在磁盘具有足够空间；逐帧掩码、裁剪图和点云可能占用较多空间。

模型路径和数据路径不要提交到 Git；通过环境变量传入。

## 6. 输入数据规范

一个 episode 至少应包含所选相机的 RGB、深度和相机元数据。例如：

```text
episode0/
├── variation_descriptions.pkl
├── front_rgb/
│   ├── 0.png
│   └── 1.png
├── front_depth/
├── left_shoulder_rgb/
├── left_shoulder_depth/
├── right_shoulder_rgb/
├── right_shoulder_depth/
└── low_dim_obs.pkl
```

注意事项：

- 所选相机的 RGB 帧文件名需要有可匹配的 frame stem。
- 融合阶段需要相应的深度帧和相机参数。
- 指令通常从 RLBench 描述文件读取，也可以设置 `INSTRUCTION` 显式覆盖。
- 如需自定义相机参数，可使用 `CAMERA_PARAMS_JSON`。
- 相机外参方向不一致时，可尝试 `INVERT_RLBENCH_EXTRINSICS=1`，但必须同时保持融合和可视化配置一致。

## 7. 第一次运行建议

### 7.1 仅检查数据发现

该命令不加载 Qwen3-VL 或 SAM3，用于确认 episode、帧和相机目录能够被识别：

```bash
python qwen_role_sam3_candidate_episode.py \
  --episode-dir /path/to/episode0 \
  --sam-model-dir /path/to/sam3 \
  --dry-run
```

### 7.2 小样本端到端运行

首次不要直接运行完整 episode。建议只处理 3 个采样帧，并开启完整决策和阶段对比：

```bash
EPISODE_DIR=/path/to/episode0 \
SAM_MODEL_DIR=/path/to/sam3 \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
OUTPUT_DIR=outputs/handover_smoke \
FRAME_INTERVAL=10 \
MAX_FRAMES=3 \
SKIP_DECISION=0 \
SKIP_STAGE_COMPARE=0 \
./run_full_pipeline.sh
```

成功后重点检查：

1. `episode_candidates.json` 中是否存在所选帧和相机候选。
2. `frame_fused_candidates.json` 中是否存在角色中立的 `O*` 物体。
3. `frames/*/fused_geometry.npz` 是否成功生成。
4. `viz/` 中不同相机的同一物体颜色和 ID 是否一致。
5. `object_predictions.json` 是否包含完整的 `frame_decisions`。
6. `viz_decision/` 中目标与参考物体标签是否合理。
7. `viz_compare/` 是否能够清楚显示问题来自候选、融合还是决策阶段。

### 7.3 默认主流程

仅执行候选、融合和融合可视化：

```bash
EPISODE_DIR=/path/to/episode0 \
SAM_MODEL_DIR=/path/to/sam3 \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
./run_full_pipeline.sh
```

包含目标/参考物体决策：

```bash
EPISODE_DIR=/path/to/episode0 \
SAM_MODEL_DIR=/path/to/sam3 \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
SKIP_DECISION=0 \
./run_full_pipeline.sh
```

## 8. Windows 运行说明

主入口是 Bash 脚本，不能直接由纯 CMD 解释。Windows 下推荐使用 Git Bash：

```cmd
"C:\Program Files\Git\bin\bash.exe" -lc "cd /c/Users/yiwei.chen/TASK_Relevant_object-main && EPISODE_DIR='/path/to/episode0' SAM_MODEL_DIR='/path/to/sam3' MODEL_PATH='/path/to/qwen' MAX_FRAMES=3 SKIP_DECISION=0 ./run_full_pipeline.sh"
```

路径转换规则示例：

- Windows：`C:\data\episode0`
- Git Bash：`/c/data/episode0`

当前机器可通过以下方式激活基础 Conda 环境：

```cmd
call C:\ProgramData\miniforge3\condabin\conda.bat activate base
```

基础环境不等于模型运行环境。完整模型回归仍建议在配置好 CUDA、SAM3 和 Qwen3-VL 的 Linux 服务器上执行。

如果 VS Code 反复恢复旧 PowerShell 终端，应先执行 `Terminal: Kill All Terminals`，再新建 Command Prompt；必要时关闭 persistent terminal session。

## 9. 关键环境变量

### 9.1 必需路径

| 变量 | 说明 |
| --- | --- |
| `EPISODE_DIR` | RLBench episode 目录，必需 |
| `SAM_MODEL_DIR` | SAM3 模型目录；跳过阶段 1 时可不提供 |
| `MODEL_PATH` | Qwen3-VL 模型目录；建议显式设置 |
| `OUTPUT_DIR` | 显式输出目录；未设置时使用 `outputs/<episode-name>` |
| `CAMERAS` | 相机列表，例如 `front,left_shoulder,right_shoulder` |

### 9.2 采样与候选

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `FRAME_INTERVAL` | `1` | 每 N 个源帧处理一帧 |
| `MAX_FRAMES` | 未设置 | 限制处理帧数量 |
| `THRESHOLD` | `0.25` | SAM3 候选置信度阈值 |
| `MIN_MASK_AREA` | `4` | 最小掩码像素面积 |
| `TOP_K_PER_ROLE` | `8` | 每个语义角色保留的候选上限 |
| `MASK_NMS_IOU` | `0.80` | 阶段 1 掩码 NMS 阈值 |
| `ROLE_SPEC_JSON` | 未设置 | 复用已有角色解析结果 |

### 9.3 融合与跟踪

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `CLUSTER_DISTANCE_M` | `0.03` | 多视角融合中心点距离阈值 |
| `MIN_FUSED_CAMERA_COUNT` | `2` | 在其他相机可见时要求的最小相机支持数 |
| `TRACK_DISTANCE_M` | `0.15` | 跨帧 ID 匹配距离 |
| `TRACK_MAX_MISSED_FRAMES` | `2` | 允许短时丢失的采样帧数 |
| `MAX_CENTROID_TO_CLOUD_DISTANCE_M` | `0.02` | 中心点到点云的最大允许空隙 |
| `COMPONENT_VOXEL_SIZE_M` | `0.008` | 点云连通分量体素尺寸 |

### 9.4 决策与可视化

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `SKIP_DECISION` | `1` | 设为 `0` 开启阶段 4 |
| `DECISION_POLICY` | `adaptive` | 自适应关键帧或逐帧模型调用 |
| `DECISION_WINDOW_FRAMES` | `3` | 当前帧及历史帧窗口大小 |
| `DECISION_VISUAL_MODE` | `scene` | 使用全图场景；`patches` 使用候选裁剪图 |
| `DECISION_REFRESH_INTERVAL` | `5` | 自适应策略最大刷新间隔 |
| `SKIP_DECISION_VIZ` | `0` | 开启决策后是否跳过阶段 5 |
| `SKIP_STAGE_COMPARE` | `1` | 设为 `0` 开启阶段 6 |

完整参数及默认值见 `run_full_pipeline.sh` 文件头部。修改默认参数前，应先在固定小样本上保留对比结果。

## 10. 输出目录说明

典型完整输出如下：

```text
outputs/<episode-name>/
├── role_spec.json
├── episode_candidates.json
├── frames/
│   └── <frame_key>/
│       ├── fused_objects.json
│       └── fused_geometry.npz
├── frame_fused_candidates.json
├── object_summary.json
├── object_predictions.json
├── decision_inputs/
├── viz/
│   ├── <frame_id>_montage.png
│   └── sanity_report.json
├── viz_decision/
│   └── decision_visualization.json
└── viz_compare/
```

重要约定：

- `frame_fused_candidates.json` 是轻量 episode 索引。
- 每帧的对象元数据保存在 `frames/<frame_key>/fused_objects.json`。
- 大体积点云保存在 `fused_geometry.npz`，通过 `geometry_path` 和 `points_key` 延迟加载。
- 阶段 4 的顶层 `decision` 表示最后一帧结果，完整 episode 结果在 `frame_decisions`。
- 不要只根据最终 JSON 判断质量；应同时检查 `viz/` 和 `viz_decision/`。

## 11. 复用中间结果

调试下游阶段时，不要重复执行高成本模型阶段。

### 复用候选和融合结果

```bash
EPISODE_DIR=/path/to/episode0 \
SAM_MODEL_DIR=/path/to/sam3 \
OUTPUT_DIR=outputs/episode0 \
SKIP_CANDIDATES=1 \
SKIP_FUSION=1 \
SKIP_DECISION=0 \
./run_full_pipeline.sh
```

### 只复用角色描述

设置：

```bash
ROLE_SPEC_JSON=outputs/episode0/role_spec.json
```

这会跳过 Qwen 的阶段 1 角色解析，但仍重新生成 SAM3 候选。

### 复用时的注意事项

- `OUTPUT_DIR`、`CANDIDATES_JSON` 和 `FUSED_JSON` 必须指向同一组实验产物。
- 修改相机列表、帧采样、深度模式或外参方向后，不应直接复用旧融合结果。
- 修改候选阈值后，需要重跑阶段 1 和后续阶段。
- 仅修改阶段 4 提示、过滤器或决策参数时，可复用阶段 1 和 2。
- 仅修改可视化样式时，可跳过候选和融合。

## 12. 测试与验收

### 12.1 静态与单元测试

```bash
bash -n ./run_full_pipeline.sh
python -m pytest -q
```

如出现 `No module named pytest`：

```bash
python -m pip install pytest
```

应在正式模型环境中执行测试，避免“测试使用的 Python”和“运行模型的 Python”不是同一个解释器。

### 12.2 交接验收清单

接手人完成以下项目后，可认为基础交接完成：

- [ ] 能解释六个阶段及主要输入输出。
- [ ] 能访问 RLBench 数据、SAM3 和 Qwen3-VL 模型。
- [ ] `bash -n ./run_full_pipeline.sh` 通过。
- [ ] `python -m pytest -q` 通过。
- [ ] 数据发现 `--dry-run` 通过。
- [ ] 3 帧小样本完整流程成功。
- [ ] 能从 `viz_compare/` 判断候选、融合或决策错误。
- [ ] 能通过 `SKIP_CANDIDATES`、`SKIP_FUSION` 复用中间结果。
- [ ] 已保存验证环境的 Python、CUDA、PyTorch、Transformers 和 SAM3 版本。

## 13. 常见问题排查

### 13.1 Python 或 pytest 不可用

现象：`python` 命令不存在，或提示 `No module named pytest`。

处理：

1. 确认已激活正确 Conda 环境。
2. 使用 `python -c "import sys; print(sys.executable)"` 确认解释器。
3. 在同一环境安装 `requirements.txt`。
4. Windows 当前基础解释器路径为 `C:\ProgramData\miniforge3\python.exe`，但该基础环境不是完整模型环境。

### 13.2 模型无法加载

检查：

- `MODEL_PATH` 和 `SAM_MODEL_DIR` 是否为当前机器可访问的绝对路径。
- SAM3 checkpoint 是否存在。
- Transformers 是否支持当前 Qwen3-VL 类。
- PyTorch、CUDA 和显卡驱动是否匹配。
- 是否错误地使用了本地 Windows 基础环境运行服务器模型配置。

### 13.3 候选漏检

按以下顺序检查：

1. 查看阶段 1 的候选网格和掩码，而不是直接调整融合参数。
2. 适当降低 `THRESHOLD`，或使用 `CAMERA_THRESHOLD_OVERRIDES` 单独调整困难相机。
3. 增加 `TOP_K_PER_ROLE` 或候选池大小。
4. 检查 `MIN_MASK_AREA`、多实例掩码抑制和 NMS 是否过强。
5. 确认角色描述是否正确；必要时复查 `role_spec.json`。

### 13.4 同一物体被拆成多个 ID

检查：

- 深度解码和相机外参是否正确。
- `viz/` 中不同相机投影是否对齐。
- `CLUSTER_DISTANCE_M` 是否过小。
- 同相机是否存在重复候选未被 NMS 去除。
- 尺寸比例、点云直径或最近点限制是否过严。

不要只提高 `CLUSTER_DISTANCE_M`；距离过大会增加不同物体误合并风险。

### 13.5 不同物体被错误合并

检查：

- `CLUSTER_DISTANCE_M` 是否过大。
- `MAX_HYPOTHESIS_DIAMETER_M` 和尺寸比例限制是否过松。
- `sanity_report.json` 中中心点到点云距离是否异常。
- 主连通分量和次级连通分量比例是否提示污染点云。
- 候选掩码本身是否覆盖了多个实例。

### 13.6 物体跨帧 ID 跳变

检查：

- `TRACK_DISTANCE_M` 是否适合当前采样间隔。
- `FRAME_INTERVAL` 增大后，相邻采样帧的物体位移是否超过阈值。
- `TRACK_MAX_MISSED_FRAMES` 是否足以覆盖短时间遮挡。
- 三维包围盒尺寸是否因深度噪声发生异常变化。

### 13.7 抓取后物体检测丢失

典型现象：夹爪闭合并拿起物体后，阶段 1 不再输出该物体，或掩码包含部分夹爪；阶段 2 随后丢失原有 `O*` ID。物体颜色、纹理或亮度与夹爪相近时更容易出现该问题。

可能原因：

- 夹爪遮挡了物体的大部分可见区域。
- SAM3 难以区分外观相近的夹爪和物体边界。
- 粘连后的掩码使点云尺寸、中心点或连通分量发生突变，被融合质量过滤器拒绝。
- 当前跟踪主要依赖重新检测后的空间匹配；`TRACK_MAX_MISSED_FRAMES` 只能容忍短时漏检，不能表示“物体已刚性附着于夹爪”。

排查顺序：

1. 查看阶段 1 抓取前后的原始掩码，区分“完全漏检”和“掩码与夹爪粘连”。
2. 对比多个相机；确认是否仅某一视角受遮挡。
3. 检查阶段 2 诊断字段，确认候选是未生成，还是因尺寸、中心点或连通分量异常被过滤。
4. 临时调试时可适当增加 `TRACK_MAX_MISSED_FRAMES`，但必须检查是否引入错误 ID 关联。
5. 不要仅通过降低全局 `THRESHOLD` 解决；这通常会同时增加夹爪和背景噪声候选。

推荐的正式改进方向：在检测到“夹爪闭合、目标靠近末端执行器、随后与夹爪共同运动”的可靠抓取事件后，将物体标记为 `attached-to-gripper`。处于该状态时，即使视觉候选短暂消失，也根据夹爪位姿传播物体身份；物体重新出现或释放后，再通过几何和多视角证据完成重关联。机器人本体/夹爪掩码和外观重识别可作为后续增强，但不应替代附着状态建模。

### 13.8 目标或参考物体判断错误

按顺序区分问题来源：

1. 阶段 1 是否生成了正确实例。
2. 阶段 2 是否保持了正确的 `O*` 身份。
3. 阶段 4 的 `instruction_compatible_object_ids` 是否包含正确对象。
4. `target_selection` 是否因末端执行器距离或动态状态调整了模型结果。
5. `decision_source` 是模型调用还是传播结果。
6. `reference_object_id=null` 是否符合非关系型指令，而不是被误判为缺失。

必要时将 `DECISION_POLICY=every-frame` 用于对比，但该设置会显著增加 Qwen 调用次数。

### 13.9 VS Code 异常退出或卡顿

当前 Windows 机器曾出现系统虚拟内存不足。建议：

- 不在 VS Code 中同时打开多个大规模图片输出目录。
- 在工作区排除 `inputs/`、输出目录、缓存和 `__pycache__/` 的文件监听与搜索。
- 关闭不需要的终端和扩展。
- 大规模模型运行优先放在服务器，通过日志和输出文件检查结果。

## 14. 已知限制与待办事项

### P0

- 在正式 GPU 模型环境执行全部测试。
- 固化完整依赖版本，包括 CUDA、PyTorch、Transformers 和 SAM3 commit。
- 建立固定 RLBench 评测集和量化指标。

### P1

- 对小物体、遮挡、相似实例和单相机可见场景进行专项评测。
- 增加抓取后检测丢失专项测试，并实现 `attached-to-gripper` 身份传播与释放后的重关联。
- 记录默认参数在代表性任务上的候选召回、误融合、ID 跳变和最终决策准确率。
- 增加从输入检查到小样本运行的一键 smoke test。

### P2

- 将大量环境变量整理为版本化配置文件。
- 补充可分享的脱敏样例输入与期望输出。
- 清理历史入口和重复说明，明确推荐入口与兼容入口的生命周期。

## 15. 交接时必须传递的外部信息

以下内容不在 Git 仓库中，必须由原负责人单独交接：

1. 正式运行服务器及登录方式。
2. 已验证 Conda 环境名称或容器镜像。
3. Qwen3-VL 模型实际路径和访问权限。
4. SAM3 源码、配置、checkpoint 路径和 commit。
5. RLBench 数据路径、任务范围及数据权限。
6. 已验证成功的完整运行命令。
7. 一组成功样例和一组典型失败样例的输出目录。
8. 当前用于判断结果好坏的人工标准或标注数据。

缺少上述信息时，接手人只能验证代码结构，无法完成等价的模型结果复现。

## 16. 推荐接手顺序

1. 阅读本文件和 `PROJECT_REPORT_CN.md`，理解范围与限制。
2. 在已验证服务器环境执行测试集。
3. 对已有成功输出执行阶段 3、5、6，熟悉结果结构。
4. 使用固定 3 帧样本完成一次全流程。
5. 复用候选和融合结果，单独修改一个决策参数并比较输出。
6. 最后再运行完整 episode 或批量任务。

该顺序能够把环境问题、数据问题和算法问题分开，降低首次接手时的排查成本。
