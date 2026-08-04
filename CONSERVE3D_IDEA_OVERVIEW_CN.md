# Conserve3D：整体研究 Idea

> 文档定位：面向讨论、组会和论文 framing 的高层版本。技术定义与实验细节分别见 [`RESEARCH_PROPOSAL_CONSERVE3D_CN.md`](RESEARCH_PROPOSAL_CONSERVE3D_CN.md) 和 [`TCIO_METHOD_SPEC_CN.md`](TCIO_METHOD_SPEC_CN.md)。

## 一句话

**机器人不应把每帧分割当成 object truth，而应把物体维护为能够解释并预测交互结果的持久物理假设；后续证据可以回溯修订 world interpretation，而不改写原始 observations。**

## 0. 当前收敛版本

这一阶段先固定以下论文 thesis，不再继续增加平级概念：

- **研究问题：** foundation model 的 mask 会 split、merge、miss 和跳变，机器人怎样仍然维护正确的物理对象历史？
- **核心假设：** object 不是一个 mask 或 UUID，而是一个持久、可预测、也可被后续物理证据推翻的 worldline hypothesis。
- **唯一方法闭环：** `Conserve → Test → Repair`。普通观测下守恒实体；机器人交互检验实体对运动、接触和依附关系的预测；矛盾结果触发对过去 world topology 的联合修订。
- **两种时间尺度：** 帧钟只更新几何、可见性和 observation support；世界钟只在抓取、接触、释放、重新显露等事件窗口修改实体数量、attachment、site ownership 和 event participant。
- **系统输出：** 一个可版本化的 entity–site–event world interpretation，而不是每帧 object list。原始图像、mask、点云和动作日志始终不可变。
- **唯一主场景：** 两个堆叠方块最初被一个 mask 合并；抓走上层方块后，系统回溯确认原 mask 覆盖两个实体，并同步修正底层实体的持续存在、grasp participant 与 support relation。
- **候选核心差异：** 不是只改进当前 segmentation，也不是只平滑历史 association，而是用交互结果联合修订历史实体数量、observation coverage、site ownership 与 event participation。
- **暂不纳入主贡献：** TCIO、主动信息动作、通用 VLA memory、conformal guard 和完整 scene-graph reasoning。它们只作为下游评测或后续工作。

如果后续算法或实验不能直接支撑这七点，就不进入主论文主线。

### 当前修订的正确性判据

一次 retrospective repair 只有同时满足以下四点才算成功：

- **历史一致：** 修订后的实体、site 和 observation coverage 能共同解释已经保留的历史证据；遮挡表示暂时不可见，而不是无证据地删除实体。
- **交互一致：** 修订后的 world topology 能解释抓取、接触、共同运动和重新显露的结果。
- **地址一致：** 历史 event、task site 和 target/reference 查询仍指向同一物理语义；内部 UUID 改变不能造成静默换指。
- **最小修订：** 只有被新证据迫使的实体数量、归属或关系才改变，且每次改变都能追溯到具体 evidence。

因此 repair 不是“把当前 ID 调得更像”，而是寻找一个能够闭合历史、物理交互和任务引用的最小世界解释。

如果多个解释都满足这些条件，系统保留它们的 posterior 或身份等价类，不强行制造唯一 UUID；repair 的目标是排除不一致解释，而不是假装信息已经足够。

### Repair 只允许四类拓扑编辑

| 编辑类型 | 修改的对象 | 典型问题 |
| --- | --- | --- |
| `coverage` | 哪些 observation fragments 支持哪些 entity | O4/O10 被错误拆成两个 ID，或一个 mask 覆盖多个实体 |
| `cardinality` | 一个还是多个持久实体的解释 | 堆叠方块先 merge、后续交互再 split |
| `relation` | carrier、site、attachment 和 event participant 的归属 | 孔的 parent 错误、抓取参与者错误、support 关系跳变 |
| `visibility/lifecycle` | 当前不可见与真实不存在的区分 | 夹爪遮挡、离开视野、重新显露；不能把 miss 直接当 death |

`target/reference` 不属于拓扑编辑，而是对修订后 world belief 的查询。内部 UUID 也不属于物理世界编辑；UUID 重排不能改变语义地址。

### 主论文的硬性证伪条件

在“早期 merged mask、后续交互拆分”的场景中，如果一个固定实体数量的普通 smoother 在相同 evidence、hypothesis budget 和计算量下，能够同样准确地恢复历史 cardinality、event participant 和 site ownership，那么 `Retrospective World Topology Repair` 不构成方法创新，应收缩为 benchmark/system work。

## 1. 问题本质

当前 SAM3 → 多视角融合 → 时域 tracking → Qwen/VLA 的默认逻辑是：

> 检测到一个区域 → 创建一个对象 → 分配一个 ID → 后续继续匹配这个 ID。

但视觉模型输出的是短暂且有缺陷的 observations，不是物理实体：

- 同一物体可以被切成多个 mask 和多个 ID；
- 多个堆叠物体可以被合成一个 mask；
- 被夹爪遮挡、放入容器或移出视野后，物体仍然存在；
- 孔、把手等关键部位可能暂时不可见，但任务仍然需要它；
- 相同外观物体在信息不足时，本来就无法恢复唯一历史身份。

因此核心问题不是“如何把 tracker 调得更稳”，而是：

> 如何让机器人从不稳定视觉观测中维护一个持续、可修订、面向任务的物理世界？

### 1.1 单一统一思想：观测拓扑不等于世界拓扑

一帧图像中有几个 mask、哪些 mask 相连，属于 **observation topology**；真实场景中有几个物理实体、哪些部位依附于哪个实体，属于 **world topology**。

当前 pipeline 的根本错误是把两者直接绑定：mask 分裂就产生新对象，mask 合并就丢失对象，mask 消失就删除对象。Conserve3D 的核心原则是：

> **观测拓扑可以快速变化；物理世界拓扑默认守恒，只能由持续证据或真实物理事件修订。**

因此多视角融合、时域 tracking、遮挡记忆、堆叠拆分和孔位持续性不再是五类独立补丁，而是同一个 observation-to-world inference 问题。

## 2. 统一世界模型

Conserve3D 将系统中的信息分成三类：

- **Physical entity / carrier**：持续存在、可运动、可被抓取的物理实体；
- **Interaction site**：依附于实体的孔、把手、按钮、支撑面等任务部位；
- **Event**：抓取、释放、接触、移动、堆叠等改变或揭示世界状态的交互。

相机中的 mask、box、point cloud 和 VLM 描述都只是这些物理变量的证据，而不是世界节点本身。

### 2.1 物理实体从哪里来

物理实体是一个 latent explanatory variable，不由某一帧 detector 直接“生成”：

1. 多视角 proposals 先形成带 provenance 的 observation fragments，作为过完备证据集合；
2. 几何连续性、跨帧持续性、共同运动、接触和语义部位证据共同产生一个或多个 entity hypotheses；
3. 普通帧只传播已有 hypotheses，交互或重新显露结果才支持 split、merge 或关系修订；
4. 若证据不足以区分两个解释，就保留多个 hypotheses，而不是把 top-1 假设伪装成物理真值。

因此 \(z\) 的含义不是“candidate 的永久 ID”，而是“哪个物理世界解释能够覆盖并预测这组 evidence”。插孔任务中的孔也同理：它先作为依附于底座 carrier 的 site hypothesis 存在，暂时不可见时由 parent pose 和接触证据传播，而不是重新等待一个新 mask。

### 2.2 Candidate→entity 不是局部 matching，而是解释选择

系统不为每个 candidate 单独寻找最近的 ID，而是比较能够解释整个 episode 的 world hypotheses。一个候选可以支持多个实体，多个候选也可以共同支持一个实体；选择依据是：

- 能否解释跨视角、跨时间的 observation evidence；
- 能否预测交互后的运动、接触和重新显露；
- 是否只引入必要的 split、merge、relation 或 lifecycle edits；
- 在证据不足时是否诚实保留多个解释。

因此 `candidate → entity` 是一个全局 posterior/incidence 查询，而不是一次不可逆的标签分配。

这一表示统一覆盖：

- 抓取后物体被夹爪遮挡；
- 堆叠物体发生 mask merge/split；
- 插孔过程中孔被机械臂遮挡；
- 相同外观物体发生身份混淆；
- 后续接触或重新显露修正之前的错误判断。

## 3. 整体闭环

```text
视觉观测
   ↓
生成多个物理解释，而不是立即确定 ID
   ↓
维护持续的 entity–site–event world belief
   ↓
利用机器人运动、抓取和接触更新或回滚解释
   ↓
根据当前任务判断剩余身份歧义是否影响动作
   ↓
继续执行 / 使用公共安全动作 / 主动消歧 / 拒绝承诺
```

这里 SAM/Qwen 的角色也随之改变：

- SAM3 是 observation proposal generator，不是实体管理器；
- Qwen 是语义和任务约束提供者，不负责维持物理身份；
- 持续世界 belief 是两者之间独立、可审计的机器人记忆层。

## 4. 核心创新主线：Conserve–Test–Repair

整体 idea 可以收敛为一个闭环，而不是多个松散模块：**先守恒物理假设，再用交互检验假设，最后根据矛盾证据回溯修复世界解释。** 帧级连续状态和事件级离散拓扑分属两种时间尺度，避免每个 noisy frame 都触发身份重写。

### 4.1 Conserve：物理假设不随检测结果任意出生和死亡

系统维护的是持久但可证伪的物理实体假设。视觉 miss、split、merge 或重复检测不会直接导致实体出生、死亡或改名。

### 4.2 Test：每个 object hypothesis 都必须预测交互结果

物体不是 SAM 画出的区域，而是一个会对抓取、推动、接触和释放结果作出预测的物理假设：哪些几何会随夹爪运动、哪些部位保持刚性依附、哪些对象仍留在原处。交互结果因此是对 objecthood 的物理检验，而不只是 action log。

### 4.3 Repair：矛盾证据必须能够改变对过去的解释

如果真实交互结果与某个假设不一致，系统不能只在当前帧换一个 ID，而要回头修订当时有几个实体、历史 mask 覆盖了谁、site 属于谁、event 的参与者是谁。原始 evidence 保持不可变，world interpretation 可以产生新版本。

三条原则共同形成一个更集中的研究命题：

> **A persistent object is a falsifiable physical hypothesis; robot interactions test it, and contradictory outcomes retrospectively repair the world model.**

对应的整体研究假设是：相比直接延长 track、保存 object token 或增加视觉 matching 特征，这个闭环能够在 observation topology 失真时恢复更一致的实体历史，并让一次后续交互同时纠正过去与现在的世界解释。

“身份歧义是否影响任务”仍然重要，但它是 repaired world belief 的下游使用方式，而不是主论文的第四个核心机制。若多个解释支持相同安全动作，TCIO 可以允许系统继续执行；这用于证明 world repair 的操作价值，不与主贡献并列。

### 4.4 证据不是等权 matching feature

- **视觉/几何证据**主要提供支持：颜色、纹理、形状和点云相似只能增加某个解释的概率，不能单独创建永久身份；
- **持续性/可见性证据**提供约束：解释必须允许实体在遮挡、单相机观测或稀疏采样期间继续存在；
- **交互/接触/重新显露证据**提供区分和证伪：不同实体假设对运动、attachment 和 reappearance 的预测不同，后续结果可以排除错误解释。

因此 interaction 的价值不是给 appearance score 再加一个 feature，而是改变哪些 world hypotheses 仍然可行。

### 4.5 Target/reference 是 typed world query，不是实体属性

`target`、`reference`、`support` 和 `previously_placed` 不应由 Qwen 直接写入某个当前帧 ID，而应作为带类型的查询：它可以约束 carrier 的类别和可操作性，也可以引用 task site、支撑关系或某个 event role。比如“stack 4 rose blocks”中的 target 是满足类别/可搬运谓词的 block 集合，reference 可以是当前支撑关系或已放置实体的地址，不必被强制命名为另一个 rose block。

查询返回的是持久地址、候选集合或 unresolved posterior；当 world topology revision 发生时，查询重新解析，但同一语义 role 不应静默跳到另一个物理实体。

## 5. 与现有方向的关系

- MOT 重点是输出稳定轨迹和 ID；本工作允许身份暂时不确定，并关心这种不确定性是否影响机器人任务。
- Object permanence 重点是不可见物体仍然存在；本工作还要求在 split/merge、错误关联和交互后修订实体结构。
- 3D scene graph 重点是保存对象和关系；本工作中的节点与事件参与关系本身可以是不确定、可回滚的。
- Object-centric VLA 重点是向 policy 提供对象 token；本工作关注这些 token 所指向的物理实体是否可靠，以及何时足以支持动作。
- POMDP 和 conformal decision methods 提供通用不确定决策工具；本工作要证明的是机器人 entity/site/event identity 的具体表示与审计机制，而不是重新命名通用理论。

因此不应把“3D memory”“scene graph”“多假设 tracking”或“conformal prediction”单独作为贡献。真正需要证明的是：统一的物理实体 belief 是否能同时减少错误身份承诺、保持任务执行率，并在身份不可恢复时作出正确决策。

## 6. Scope 取舍：一个主创新，而不是两篇论文叠在一起

[Object-composition POMDP](https://arxiv.org/abs/2010.13565) 已经维护多种对象分割/组成假设、利用机器人动作获取组成信息，并根据任务效用规划。因此“多假设 + 交互 + task-aware decision”不能作为本工作的宽泛新意。

更清晰的论文结构是：

- **唯一主创新：Retrospective World Topology Repair。** 在长时域、多视角和机器人交互中，维护并回溯修订 persistent entity–site topology，使它不随 foundation masks 的 split/merge/miss 任意变化。
- **下游验证接口：Task-Conditional Identity。** 用“身份不确定时是否仍能安全完成任务”证明修复后的世界 belief 有操作价值，而不把通用 POMDP 或 decision theory 重新包装成第二个主贡献。
- **未来独立方向：TCIO。** 只有在它显著优于 action disagreement、robust POMDP 和 decision-aware uncertainty baselines 时，才值得发展成单独理论工作。

这种取舍让论文的核心问题更具体：

> **Can a robot maintain and retrospectively repair the topology of its physical world when foundation observations split, merge, disappear, and reappear?**

### 6.1 不是“记得更久”，而是“能够改变对过去的解释”

普通 persistent memory 保存已经创建的 object nodes；scene-graph systems 通常继续向图中累积信息；multi-scan trackers 可以修正过去的 measurement association。Conserve3D 更强的目标是联合修订过去的物理 ontology：

- 当时究竟存在一个还是多个实体；
- 一个历史 mask 覆盖了哪些实体；
- hole/handle site 当时属于哪个 carrier；
- 某次 grasp/contact event 的真实参与者是谁。

原始图像、mask、点云和机器人日志保持不可变；改变的是系统根据全部后续证据导出的 world interpretation。这样既能回滚错误，又不会篡改观测记录。

[RoboEXP](https://arxiv.org/abs/2402.15487) 已通过交互增量构建 action-conditioned scene graph，[multi-scan trajectory PMBM](https://arxiv.org/abs/1912.01748) 等方法也能修正历史关联。因此不能把“interaction graph”或“smoothing”本身当贡献；候选差异必须是针对 fallible foundation proposal topology，对 cardinality、association、site ownership 和 event participation 的联合 retrospective repair。

### 6.2 North-star 场景

最能说明整篇论文的一个例子是：

1. 初始时，两个上下堆叠方块被 SAM 合成一个 mask；
2. 系统不立即断言场景只有一个实体，而保留一个 mask 对应一个或两个实体的解释；
3. 机器人抓走上层方块，运动和接触结果表明只有部分几何随夹爪移动；
4. 系统回头修订初始解释：历史 mask 覆盖两个实体，底层方块从未消失，grasp event 的参与者是上层实体；
5. 所有后续 target/reference、support 和 site 关系随 world revision 一致更新。

这一个场景同时包含 merge、遮挡、交互、重新显露、身份保持和历史修订，应作为论文主图和核心定性实验。

### 6.3 物理直觉：为什么 object hypothesis 是可证伪的

上述方法依赖一个直观的物理判据：

> **Object 不是视觉上被圈在一起的区域，而是在可行机器人交互下呈现一致物理响应的最小持久单元。**

能够被分别抓起的上下方块应属于两条 physical worldlines；始终共同运动的刚性几何暂时可以属于同一个 carrier；孔和把手会随 carrier 运动、却承担局部交互功能，因此表示为 site 而不是独立可运输实体。这使 entity、part 和 site 的区分由 embodied physics 约束，而不是完全交给 VLM 命名。

但不能将“用运动或交互发现对象”本身作为新意：[Almeida et al.](https://merl.com/publications/docs/TR2019-119.pdf) 已用交互验证并拆分错误 object maps，[RISeg](https://arxiv.org/abs/2403.01731) 已利用刚体运动特征修复 under-segmentation。该抽象只有在它进一步支持**跨遮挡的 worldline persistence，以及对历史 coverage、site ownership、event participant 和任务地址的联合 revision**时，才可能形成足够区别。

还要避免一个逻辑错误：没有观察到“可分离运动”不等于两个部分一定是同一物体，因为机器人可能尚未执行有效交互。因此这个物理判据只用于更新 hypothesis belief，而不是一次动作后的硬分割规则。

## 7. 最小论文故事

论文只需围绕三个代表性场景：

1. **Grasp and occlusion**：同色物体被抓取并被夹爪完全遮挡；
2. **Stack and topology change**：堆叠物体在一个 mask 与多个 mask 之间变化；
3. **Insertion and persistent site**：孔最初可见，后续被遮挡但在三维世界中继续存在。

两个主问题：

1. 系统能否在视觉 observation topology 错误时保持正确的物理实体 belief？
2. 机器人交互结果能否修正过去的 world topology，而不只是平滑 association 或当前 ID？

另有一个不参与主方法成立与否的下游问题：系统能否区分“身份不确定但任务可继续”和“身份不确定且会导致错误动作”？如果 TCIO 不能显著优于 entropy、action disagreement 或普通 POMDP/decision-aware uncertainty baseline，就移出当前论文，不否定 world-topology repair 主线。

### 7.1 三条主张与最小证据

| 主张 | 最小场景 | 必须观察到的结果 | 关键反例 |
| --- | --- | --- | --- |
| **Persistent world**：观测 miss/遮挡不应制造实体死亡 | grasp + full occlusion | entity existence、轨迹和 site persistence 优于 TTL/hard tracking | 只是延长 track 也能达到同样结果 |
| **Retrospective topology repair**：后续交互可以改写过去的 ontology | merged stack → grasp top block | 历史 cardinality、coverage、event participant 和 support/site ownership 被共同修正 | fixed-cardinality smoother 在同等预算下同样有效 |
| **Stable world address**：拓扑修订不应造成语义地址跳变 | insertion + hidden hole/site | role/site 查询与动作 grounding 在 revision 前后保持一致 | 仍然依赖当前帧 O-ID，或地址稳定但不提升任务结果 |

只有第一、第二条主张成立，主论文的方法贡献才成立；第三条用于证明 repair 对机器人任务确实有用。

### 7.2 防止 hindsight cheating：实时决策与回溯修订分离

- 在时刻 \(t\) 执行动作时，只能读取版本 \(W_t\)，其证据范围不超过 \(t\)；
- 后续交互或重新显露到达后，系统生成新版本 \(W_{t+1}\)，可以重新解释历史，但不能改变已经执行的动作；
- 原始 evidence ledger 始终不可变，所有历史拓扑修改都记录版本和 provenance；
- 实验分别报告在线执行结果与 retrospective history accuracy，不能用离线修订后的标签替代实时决策性能。

如果 repair 只能提高离线历史重建，却不能改善后续动作、site 查询或错误承诺率，它只能算分析工具，不能算 manipulation world model 的主贡献。

### 7.3 三段式评测协议

1. **Online belief：** 在每个动作决策点只使用当时可见 evidence，测实体存在、可见性和当前查询是否安全；
2. **Retrospective revision：** 释放后续抓取/接触/重新显露事件，允许修订固定历史窗口，测 cardinality、observation coverage、site ownership 和 event participant；
3. **Future query：** 用修订后的 world belief 进行下一步 target/reference/site 查询和动作 grounding，测地址稳定性、错误承诺率与任务成功率。

因此主指标不是单一 IDF1，而是“历史修订增益 + 地址稳定性 + 在线操作收益”；普通 MOT 只作为底层对照。

## 8. 推荐论文定位

**主问题：** 如何在失真的 foundation observations 和机器人交互过程中，持续维护并回溯修订物理世界拓扑？

**最小接口：** immutable evidence ledger → versioned entity–site–event belief → `Conserve–Test–Repair` update → world-query addresses。所有角色决策和可视化都从最后一个接口读取，不再把当前帧 O-ID 当作物理真值。

**对当前 pipeline 的最小映射：** Stage 1 保留为 evidence proposal generator；Stage 2 不再把 `frame_fused_candidates.json` 当最终对象真值，而是生成/更新 versioned world belief；Stage 3/5/6 只负责显示 belief、uncertainty 和 revision provenance；Stage 4 的 target/reference 决策改为查询 world addresses。这样不需要先重写 SAM3 或 Qwen，只需要在 fusion 与 role decision 之间加入可回溯的 world layer。

**主方法：** Conserve–Test–Repair world model：原始 evidence 不可变；系统维护可版本化的 entity–site–event world interpretation，交互结果触发对当前和过去 world topology 的联合 revision。

**主结果：** 相比 hard tracking、persistent token、scene graph fusion 和 object-composition baselines，在 split/merge/miss、全遮挡和重新显露下更准确地保持实体数量、轨迹连续性与 task-site persistence，并提高操作成功率。

推荐标题：

> **Conserve3D: Retrospective World Topology Repair from Robot Interactions**

如果后续 TCIO 形成独立理论工作，可使用：

> **Conserve3D: When Does Object Identity Matter for Robot Manipulation?**

## 9. 当前判断

最有价值的不是继续为每种失败增加规则，而是建立一个统一原则：

> **Conserve physical hypotheses, test them through interaction, and repair the world model retrospectively.**

后续所有算法、数据结构和实验都应服务于这句话。不能直接提升这条主张可信度的模块，应从主论文中删除或降为实现细节。
