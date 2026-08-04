# Conserve3D：一页 Thesis Card

> 用于组会、开题和论文 framing 的高层版本。完整研究方案见 [`RESEARCH_PROPOSAL_CONSERVE3D_CN.md`](RESEARCH_PROPOSAL_CONSERVE3D_CN.md)，整体叙事见 [`CONSERVE3D_IDEA_OVERVIEW_CN.md`](CONSERVE3D_IDEA_OVERVIEW_CN.md)。

## 一句话

**机器人不应把每帧 mask 或 O-ID 当作物理真值，而应维护一个能解释并预测交互结果的持久 world belief；当后续证据推翻旧解释时，回溯修订世界拓扑，而不改写原始观测。**

## 研究问题

Foundation perception 会产生 split、merge、miss、遮挡和跳变。当前 pipeline 却把 observation topology 直接固化成 object list，导致同一物体多个 ID、堆叠物体一个 ID、抓取后目标消失、孔位丢失和 target/reference 跳变。

问题不是“如何把 tracker 调得更稳”，而是：

> 在观测拓扑持续失真的情况下，机器人如何维护并回溯修订物理世界拓扑？

## 唯一核心闭环：Conserve → Test → Repair

1. **Conserve**：普通帧不因 mask 数量变化而创建、删除或重命名实体；
2. **Test**：抓取、接触、共同运动和重新显露检验 entity/site hypothesis 的物理预测；
3. **Repair**：矛盾证据到达后，在有限历史窗口内修订 cardinality、observation coverage、site ownership 和 event participant。

帧钟更新几何和可见性；世界钟只在物理事件窗口修改离散拓扑。

## 最小世界表示

- `entity/carrier`：持续存在、可运动和可被操作的物理实体；
- `task site`：依附于 carrier 的孔、把手、按钮或支撑面；
- `event`：抓取、释放、接触、移动和重新显露等物理事件；
- `evidence ledger`：不可变的 mask、点云、相机、机器人状态和动作证据；
- `world belief`：可版本化、可回滚的 entity–site–event 解释。

实体不是 detector 输出，而是能够共同解释整段 evidence 的 latent hypothesis。证据不足时保留多个解释，不强制生成唯一 UUID。

## 唯一主场景

两个上下堆叠的方块最初被 SAM 合成一个 mask。机器人抓走上层方块后，运动和接触结果表明只有部分几何随夹爪移动。系统回溯确认历史 mask 覆盖两个实体，底层实体从未消失，grasp event 的参与者是上层实体，并同步更新 support/site/role 查询。

## 和已有工作的边界

主贡献不是首次使用 object permanence、交互分割、scene graph、RFS smoothing 或 persistent object token。候选差异是：**交互结果联合修订过去的实体数量、观测覆盖、site ownership 和 event participation，并保持后续 world address 稳定。**

## 论文成立的三道检查

- 遮挡/漏检下，实体持续性优于 hard tracking/TTL；
- merged mask 后，拓扑修订优于固定 cardinality 的普通 smoother；
- revision 后，target/reference/site address 稳定，并改善后续操作。

若只能提高离线历史重建、不能改善后续查询或动作，方法不成立为 manipulation world model。

## 工程落点

Stage 1 继续生成 observation evidence；Stage 2 与 Stage 4 之间加入 versioned world layer；Stage 3/5/6 负责 belief、uncertainty 和 revision provenance 可视化；Stage 4 通过 typed world query，而不是当前帧 O-ID，解析 target/reference。

## 暂不纳入主贡献

TCIO、主动信息动作、通用 VLA memory、conformal guard 和完整 policy learning 只作为下游评测或后续方向。

> **Conserve physical hypotheses, test them through interaction, and repair the world model retrospectively.**
