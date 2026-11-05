# Slot-guided SAM3 离线快速开始 / Offline quickstart

这条独立路径运行当前帧的 `Coarse Slot → 原始 RGB Prompt → SAM3 → RGB-D 对象几何`。
先用下面的 Mock 示例检查数据和软件接口，再准备真实模型实验。示例无需 SAM3 权重、CUDA 或机器人环境；它不执行 BridgeVLA checkpoint 或真实 SAM3。

This separate offline path provides a reproducible current-frame software check.
The demo supplies mock masks and mock coarse points; its metrics describe that
fixture. Real model accuracy, action error and RLBench success remain unmeasured.

接口细节见 [Guidance 与坐标审计](slot_guidance_interface_audit.md)、
[单帧 Prompt/SAM3](slot_guided_offline.md)、[3D 融合与 A/B/C 实验](slot_geometry_experiments.md)和[本轮审查记录](slot_guided_review.md)。
BridgeVLA 的可选动作条件仍是[默认关闭的 PR4 草稿](https://github.com/YW310/BridgeVLA/blob/codex/slot-guided-sam3-action-draft/docs/slot_guided_sam3_action_draft.md)，需要独立的真实实验验收。

## 1. 本机 Windows 环境 / Prepared Windows environment

以下命令在 PowerShell 中执行，不需要激活脚本。脚本文件命令均从本仓库根目录运行，不能直接把新脚本当作已安装的 `relevant-object` 子命令。

```powershell
$TaskRepo = 'C:\Users\yiwei.chen\TASK_Relevant_object-main'
$BridgeRepo = 'C:\Users\yiwei.chen\BridgeVLA'
$Python = Join-Path $BridgeRepo '.venv-oht\Scripts\python.exe'
$env:PYTHONPATH = Join-Path $BridgeRepo 'tmp\slot-guidance-test-deps'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:BRIDGEVLA_REPO = $BridgeRepo
if (Test-Path -LiteralPath (Join-Path $BridgeRepo 'tmp\sam3-api-source')) {
    $env:SAM3_AUDIT_ROOT = Join-Path $BridgeRepo 'tmp\sam3-api-source'
}
Set-Location -LiteralPath $TaskRepo
$RunDir = Join-Path $BridgeRepo ('tmp\slot-guided-quickstart-' + [guid]::NewGuid().ToString('N'))
```

该 Python 与 `PYTHONPATH` 指向本机已准备的 CPU 环境和隔离测试依赖。可选 `SAM3_AUDIT_ROOT` 指向已审计的本地 SAM3 源码，用于源码契约测试；它不是 SAM3 安装或模型权重，没有该 checkout 时对应测试跳过。输出写入 BridgeVLA 已忽略的 `tmp`，每次生成新目录；Guidance 写入器会拒绝隐式覆盖已有输入。

其他机器可以在已安装 Python 3.10+ 的环境中使用下面的替代准备命令，然后继续第 2 步。先进入仓库根目录；这会安装项目声明的依赖和测试 extra，不会下载模型权重。

```powershell
$TaskRepo = (Get-Location).Path
python -m venv .venv
$Python = Join-Path $TaskRepo '.venv\Scripts\python.exe'
& $Python -m pip install -e '.[test]'
$env:PYTHONPATH = (Join-Path $TaskRepo 'src') + [System.IO.Path]::PathSeparator + $TaskRepo
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$RunDir = Join-Path ([System.IO.Path]::GetTempPath()) ('slot-guided-quickstart-' + [guid]::NewGuid().ToString('N'))
```

`BRIDGEVLA_REPO` 仅用于可选的跨仓库坐标实现对照；另一台机器可将其设置为真实 BridgeVLA checkout，没有该变量时相关测试跳过。

## 2. 运行完整合成示例 / Run the synthetic demo

```powershell
& $Python slot_guided_synthetic_demo.py --config configs/slot_guided_offline.json --output-dir $RunDir
```

示例创建同类别的两个实例、front 与动态 wrist 相机、原始 RGB-D、Oracle Guidance、Mock Coarse Guidance 和多个 Mock mask 候选。它运行 A/B 各自的 box-only 与 point+box，保存对象几何及评测结果。

| 输出 / Output | 路径，相对于 `$RunDir` / Relative path |
| --- | --- |
| 当前帧、Guidance、Mock mask 和实验清单 | `inputs/` |
| A/B 四种配置、明确缺失的 C/动作/rollout 指标 | `results/experiment_report.json` |
| 可导入表格的逐项指标 | `results/metrics.csv` |
| 分相机候选 mask、Prompt、分数、诊断图 | `results/synthetic_two_instances/7/<configuration>/<cache_key>/` |
| 对象点云 JSON + 摘要关联的 NPZ | `results/synthetic_two_instances/7/<configuration>/object_geometry.json` |

打印结果中的 `evidence_kind` 应为 `synthetic_mock`，`real_offline_acceptance_passed` 应为 `false`。C 表示另外执行的原 OCRR Slot/动作路径；没有该 baseline 运行记录时显示 `not_executed`。提供 C 时必须使用 `ocrr_baseline_evaluation` 约定，并匹配当前帧、时间、标定和来源，完整列出九项指标；缺失测量需要显式空值及原因。Mock 的耗时也不是 SAM3 推理延迟。

示例元数据的 `source_git_sha` 记录生成仓库的 Git HEAD。存在未提交修改时，HEAD 不能单独代表工作区代码；复现实验还需保留所用配置、实际源码状态、输入摘要和报告。不要把审查过程中的 dirty 工作区产物作为已提交版本的真实模型证据。

## 3. 单帧推理与缓存 / One frame and cache replay

复用第 2 步生成的当前帧输入，独立运行 Prompt/SAM3 路径：

```powershell
$FrameJson = Join-Path $RunDir 'inputs\frame.json'
$GuidanceJson = Join-Path $RunDir 'inputs\predicted_mock.json'
$MockMasksJson = Join-Path $RunDir 'inputs\mock_masks.json'
$SingleOutput = Join-Path $RunDir 'single-frame'
& $Python run_slot_guided_sam3.py --frame $FrameJson --guidance $GuidanceJson --backend mock --mock-candidates $MockMasksJson --mode point+box --config configs/slot_guided_offline.json --output-dir $SingleOutput
& $Python run_slot_guided_sam3.py --frame $FrameJson --guidance $GuidanceJson --backend mock --mock-candidates $MockMasksJson --mode point+box --config configs/slot_guided_offline.json --output-dir $SingleOutput
```

输入和配置未改变时，第二次打印 `cache_hit: true`，不会再次执行 image encoder。RGB、深度、相机参数、Guidance、后端候选/权重、提示模式和相关推理设置都参与缓存身份。检查 `inference.json` 中每个角色的 `failure_reason` 与候选诊断，不能仅凭最高 SAM 分数确定实例。

重新执行独立 A/B 实验清单，可用另一个输出目录：

```powershell
$ExperimentJson = Join-Path $RunDir 'inputs\experiment.json'
$RerunOutput = Join-Path $RunDir 'rerun'
& $Python run_slot_geometry_experiments.py --manifest $ExperimentJson --backend mock --config configs/slot_guided_offline.json --output-dir $RerunOutput
```

## 4. 配置 / Configuration

[configs/slot_guided_offline.json](../configs/slot_guided_offline.json) 使用严格 schema-1 JSON：`schema_version`、`adapter`、`selection`、`fusion`。三个设置区分别对应 `AdapterConfig`、`SelectionConfig`、`FusionConfig`；未知字段、错误类型和非法阈值会报错。省略 `--config` 时保留原来的 dataclass 默认值。该文件不修改已有 Qwen/SAM3 pipeline 配置。

| 设置区 / Section | 常用参数 / Parameter | 默认值 / Default | 含义 / Meaning |
| --- | --- | --- | --- |
| `adapter` | `min_confidence` | `0.25` | 角色定位最低置信度 |
| `adapter` | `depth_tolerance_m` | `0.03` | 3D 提示点与当前深度的表面残差 |
| `adapter` | `padding_px` / `padding_fraction` | `4.0` / `0.1` | 原始 RGB 粗框扩边 |
| `selection` | `min_mask_pixels` | `4` | 最小候选 mask 面积 |
| `selection` | `max_center_error_m` | `0.15` | 候选中心与 Guidance 中心误差上限 |
| `fusion` | `min_support_cameras` | `1` | 最少支持视角；单视角不能证明跨视角一致性 |
| `fusion` | `voxel_size_m` | `0.005` | 对象点云体素降采样间距 |
| `fusion` | `cross_view_distance_m` | `0.04` | 跨视角表面距离门限 |

打印实际生效的完整设置和配置 SHA256：

```powershell
& $Python slot_guided_config.py --config configs/slot_guided_offline.json
```
修改配置后，重新运行同一命令并比较诊断及带标签结果。阈值调整本身不能证明机器人成功率提高。

## 5. 软件验证 / Software checks

下面列出当前核心测试及配置校验的完整文件集合，实际数量以本次 pytest 输出为准。运行前沿用第 1 步环境设置；本轮结果见[审查记录](slot_guided_review.md)。

```powershell
& $Python -m pytest tests/test_slot_guidance.py tests/test_slot_guidance_geometry.py tests/test_slot_prompt_adapter.py tests/test_slot_guided_sam3.py tests/test_slot_guided_io.py tests/test_slot_guidance_producers.py tests/test_slot_geometry_fusion.py tests/test_slot_geometry_experiments.py tests/test_slot_guided_config.py tests/test_camera_geometry.py tests/compatibility/test_pipeline_baseline.py -q
& $Python -m pytest tests/test_multiview_candidate_fusion.py -q
```

原 Qwen 候选测试需要单独进程，避免其全局 torch stub 影响其他测试。此前审查发现一个在未修改 baseline 上同样存在的 wheel-prompt 断言失败；BridgeVLA reduced-mode 另有两个已复现的 baseline 失败。它们不应被描述成全仓测试通过；具体名称与已执行范围分别见[PR3 验证记录](slot_geometry_experiments.md#validation-and-remaining-limits)和 [BridgeVLA 验证记录](https://github.com/YW310/BridgeVLA/blob/codex/slot-guided-sam3-action-draft/docs/slot_guided_sam3_action_draft.md#freeze-coordinate-and-compatibility-behavior)。

## 6. 准备真实实验 / Prepare real experiments

真实运行需要经过 API 审计的 SAM3 安装与本地权重、真实 Coarse checkpoint 导出的 detached Slot 点、实际完整反归一化变换、当前帧原始 RGB-D 与相机标定，以及角色/实例标签。采样序号与 `source_frame_index` 不同；wrist 相机必须使用该源帧自己的外参。文档中的示例 K/T 只属于合成数据，不能复制为真实相机参数。

准备好实际数据后，独立单帧命令使用：

```powershell
& $Python run_slot_guided_sam3.py --frame snapshots/current.json --guidance guidance/current.json --backend sam3 --checkpoint 'C:\models\sam3.pt' --device cuda --mode point+box --config configs/slot_guided_offline.json --output-dir artifacts/slot_sam3
```

这里的输入与权重路径需要替换为实际文件。完整真实 A/B/C 评测按[实验清单约定](slot_geometry_experiments.md#abc-and-ablation-harness)提供标签和独立 baseline 记录。缺失的动作误差、配对任务成功率或 GPU 测量继续保留为空并记录原因；合成示例不会解锁 BridgeVLA 动作接入。
