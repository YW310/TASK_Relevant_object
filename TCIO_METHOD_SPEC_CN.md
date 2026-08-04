# TCIO 方法规格：什么时候物体身份真的重要？

> 文档定位：Conserve3D 的下游 identity-decision 接口与潜在后续理论；不再作为主论文的并列核心  
> 版本：v0.4，2026-08-04  
> 完整背景与文献审查见 [`RESEARCH_PROPOSAL_CONSERVE3D_CN.md`](RESEARCH_PROPOSAL_CONSERVE3D_CN.md)

## 0. 一页结论

当前 pipeline 的根本错误不是 tracker 参数不够好，而是默认每个物体始终存在一个可由视觉恢复的唯一 ID。对同外观物体、完全遮挡、稀疏采样和交换运动，这个假设并不成立。

TCIO 不要求系统在所有时刻强行恢复 UUID，而回答三个独立问题：

1. `identity_resolution`：历史身份是否已被证据区分；
2. `task_sensitivity`：剩余身份置换是否会改变当前任务的答案、动作或安全性；
3. `resolvability`：若身份重要，能否通过自然后续观测或低代价交互恢复。

核心判据是：

> 观测仍允许的身份置换，是否全部属于任务不敏感的置换？

若答案为是，即便 ID 未解析也继续任务；若答案为否，才计算身份信息价值并选择重观测、信息动作或 abstain。

### 0.1 创新主线

必须区分主论文与后续方向：**Conserve3D 主论文研究 Retrospective World Topology Repair；TCIO 是读取 repaired belief 的 identity-decision 接口及潜在后续方法。** TCIO 不输出一个被迫唯一的 ID，而先把 identity/event addresses 按 safe-action signature 取商，再检查校准集合是否具有非空的公共安全动作。

该主创新只依赖两个必要机制，而不是一组并列贡献：

1. **Equivariant event/site address**：以事件角色和依附于 carrier 的 interaction site 指代实体，内部 UUID 重命名不改变查询语义；
2. **Support-aware conformal guard**：显式保留 `out_of_beam`，防止真实 association 已被剪枝时产生假确定。

carrier–site belief、FOT likelihood、fixed-lag smoothing、接触因子和 VLM 语义约束都是实现该 certificate 的底座或证据源，不单独作为论文创新。

### 0.2 Identity-gauge contract 与公共动作命题

内部 UUID 只是 latent entities 的坐标选择。对没有 symmetry-breaking evidence 的置换 \(\pi\)，\(b_t\) 与 \(\pi\!\cdot b_t\) 表示同一个物理 belief。任何外部 event/site address 必须满足：

\[
\operatorname{Resolve}(A,\pi\!\cdot b_t)
=\pi\!\cdot\operatorname{Resolve}(A,b_t).
\]

令地址 \(a\) 下满足可行性与风险预算 \(\rho\) 的技能集合为：

\[
\mathcal K_\rho(q,a)=
\{k\in\mathcal K:\text{feasible}(k;q,a),
\ R(k;q,a)\le\rho\}.
\]

对 conformal address set \(\Gamma_\alpha\)，定义公共安全动作核：

\[
\mathcal K_\cap(q,\Gamma_\alpha)=
\bigcap_{a\in\Gamma_\alpha}
\mathcal K_\rho(q,a).
\]

执行条件改为：`out_of_beam` 不在集合中，且 \(\mathcal K_\cap\neq\varnothing\)。若 conformal set 以至少 \(1-\alpha\) 覆盖真实地址，且 skill feasibility/risk oracle 正确，则任选 \(k\in\mathcal K_\cap\)，由地址漏覆盖导致的 identity-only unsafe commitment 概率不超过 \(\alpha\)。这是 coverage 的直接推论，不把 generic conformal decision theory 本身当作创新；候选新意只在于如何从 fallible carrier–site–event belief 构造 gauge-consistent address universe，并以非平凡执行覆盖率验证该 certificate。

## 1. 输入与输出契约

输入不是单帧框，而是 Stage 2 保留的 top-K 物理世界 hypotheses：

```text
H_t = {
  carrier/site trajectories,
  proposal-to-entity incidence,
  support/attachment/contact relations,
  event lineage,
  posterior weight
}
```

任务输入必须先转成 typed query，例如：

```text
any(type=rose_block)
same_as(event=grasp_12)
site_of(parent=socket_base, type=insertion_hole)
supported_by(entity=target)
```

自然语言到 typed query 的错误必须独立评测，不能混进 identity inference 的主结果。

### 1.1 Event-anchored address

`same_as(grasp_12)` 不能存成 `grasp_12 -> carrier_3` 的确定映射。事件可能夹空、夹错或发生在 merged proposal 下，因此事件角色的参与者也是随机变量：

\[
Z_{e,r}\in\{C_1,\ldots,C_N,\varnothing,\text{multi}\}.
\]

一个外部地址写成 `event_id + participant_role + optional_site_path`，例如：

```text
event(grasp_12).patient
event(probe_07).patient/site(insertion_hole)
event(place_03).destination
```

解析结果是当前 carriers/sites 上的 posterior，不是永久 UUID。后续共同运动、接触或重新显露可以回滚 event-participant association，而查询本身保持不变。

事件 anchor strength 为 participant posterior 的归一化负熵。低强度事件只能缩小 identity orbit，不能强行打破 symmetry。

## 2. 最小数学定义

令 belief \(b_t(h)=p(h\mid\mathcal D_t)\)，\(\mathfrak S_N\) 为 carrier labels 的置换群。

观测仍无法排除的近似置换集合：

\[
G_{\mathrm{obs}}(b_t)=
\{\pi\in\mathfrak S_N:
d(b_t,\pi\!\cdot b_t)\le\epsilon_{\mathrm{obs}}\}.
\]

任务查询不敏感的置换集合：

\[
G_Q=
\{\pi:Q(\pi\!\cdot h)=Q(h),
\ \forall h\in\operatorname{supp}(b_t)\}.
\]

安全未解析的结构条件为：

\[
G_{\mathrm{obs}}(b_t)\subseteq G_Q.
\]

由于实际系统只有 top-K hypotheses，第一版不宣称恢复精确群，只提取累计 posterior mass 至少为 \(1-\delta\) 的 witness permutations 和 identity orbits。

任务敏感时计算完美身份信息价值：

\[
\operatorname{VoI}_{\mathrm{id}}(b_t)=
\mathbb E_h[\max_a U(a,h)]-
\max_a\mathbb E_h[U(a,h)].
\]

这比 label entropy 更符合任务：大量身份不确定性可能完全不影响动作；很小的错误概率也可能在碰撞或插入任务中产生高风险。

### 2.1 Conformal out-of-beam guard

top-K posterior 可能遗漏真实 hypothesis，因此证书还必须输出 conformal address set：

\[
\Gamma_\alpha(q)=
\{a\in\mathcal A_t\cup\{\texttt{out\_of\_beam}\}:
s(q,a)\le\hat q_{1-\alpha}\}.
\]

- calibration label 使用 RLBench simulator handle 对应的真实 identity orbit/event participant；
- nonconformity score 第一版使用 APS rank 或 `-log posterior`；
- calibration unit 是完整 episode，score 为该 episode 所有 critical queries 的最大值；
- `out_of_beam` 进入集合时，禁止输出 `resolved`；
- 只有 `out_of_beam` 不在集合中，且所有地址的安全技能集合存在非空交集时才允许执行。

该 coverage 只在新 episode 与 calibration episodes 可交换时成立。在线更新只能宣称 long-run coverage；sim-to-real、policy、camera、SAM/Qwen 版本变化均需重新校准或明确报告失效。

## 3. Certificate schema

```json
{
  "query_id": "same_as:grasp_event_12",
  "identity_resolution": "unresolved",
  "task_sensitivity": "sensitive",
  "resolvability": "active",
  "observation_orbits": [["carrier_3", "carrier_7"]],
  "witness_permutation": {
    "carrier_3": "carrier_7",
    "carrier_7": "carrier_3"
  },
  "identity_value": 0.31,
  "false_commitment_risk": 0.22,
  "posterior_mass_covered": 0.95,
  "conformal": {
    "alpha": 0.05,
    "address_set": ["event(grasp_12).patient", "out_of_beam"],
    "calibration_unit": "episode",
    "calibration_id": "rlbench_tcio_v1"
  },
  "common_safe_skills": [],
  "execution_authorized": false,
  "recommended_action": "reobserve:left_shoulder"
}
```

三个轴的允许值：

- `identity_resolution`: `resolved | unresolved`；
- `task_sensitivity`: `invariant | sensitive`；
- `resolvability`: `passive | active | structural`。

### 3.1 Event participant record

```json
{
  "event_id": "grasp_12",
  "event_type": "grasp",
  "roles": {
    "patient": [
      {"entity_id": "carrier_3", "probability": 0.55},
      {"entity_id": "carrier_7", "probability": 0.40},
      {"entity_id": null, "probability": 0.05}
    ]
  },
  "anchor_strength": 0.19,
  "revision": 0,
  "evidence_refs": ["gripper_width:120", "motion:120-140"]
}
```

事件记录采用 append-only revisions；query 引用 `event_id + role`，不引用某个 revision 的 top-1 entity。

## 4. 在线算法

```text
Algorithm: Symmetry-Audited Identity Decision

Input: top-K world hypotheses H, typed query Q, skill library K

1. Canonicalize each hypothesis without UUID labels.
2. Match hypotheses with the same unlabeled carrier/site/event graph.
3. Extract label mappings as witness permutations.
4. Marginalize event-role participants Z_e,r over hypotheses.
5. Conformalize identity/event addresses, including out_of_beam.
6. If out_of_beam is in the conformal set:
       expand hypotheses or return unresolved; never claim resolved
7. Form posterior-covered identity orbits G_obs.
8. Resolve event-anchored addresses under every witness.
9. Compute K_rho(Q, a) for every conformal address.
10. Compute the common safe-action core K_cap by set intersection.
11. If out_of_beam is absent and K_cap is non-empty:
       execute the highest-utility skill in K_cap;
       report task sensitivity separately
12. Else estimate VoI_id and false-commitment risk.
13. Score feasible information actions by expected risk reduction / cost.
14. If ordinary future observation is sufficient:
       return unresolved × sensitive × passive
15. If a positive-net-value information action exists:
       return unresolved × sensitive × active; execute best action
16. Otherwise:
       return unresolved × sensitive × structural; request_help/abstain
```

推荐动作必须引用 typed skill library，禁止由 VLM 自由生成不可验证动作。

| 证书 | 默认行为 |
| --- | --- |
| `resolved` | 正常执行 |
| `unresolved × invariant` | 保留 identity class，继续执行 |
| `unresolved × sensitive` 且 \(\mathcal K_\cap\neq\varnothing\) | 执行公共安全技能，但不谎称身份已解析 |
| `unresolved × sensitive × passive` | 等待或利用自然下一观测 |
| `unresolved × sensitive × active` | 执行最低代价信息动作 |
| `unresolved × sensitive × structural` | 请求帮助、放宽任务或 abstain |

## 5. 最干净的主实验

每个 ambiguous posterior 使用相同图像、点云、动作历史和 hypothesis weights，只替换 typed query：

| 场景 | invariant query | sensitive query |
| --- | --- | --- |
| 两个相同 rose blocks | `any(rose_block)` | `same_as(grasp_12)` |
| 两个堆叠同类方块 | `argmax(height)` | `same_as(place_03)` |
| 两个兼容孔位 | `any(compatible_hole)` | `same_as(probe_07)` |

该 paired protocol 固定 perception difficulty。统一 entropy threshold 必须对两条查询给出相同结果；TCIO 应在左列继续任务，在右列识别风险并选择消歧。

### 5.1 Hypothesis-dropout protocol

从原始 top-K beam 中受控删除包含 simulator GT association 的 hypothesis，构造 posterior-collapse stress test。方法不能看到 dropout 标记，只能依据其 score 和 calibration data 判断是否应包含 `out_of_beam`。

Calibration/test 必须按完整 episode 和 random seed 分离，禁止同一 episode 的不同帧跨 split。分别报告 IID simulator、未见 task variation、camera corruption、SAM/Qwen 版本变化和 sim-to-real；coverage guarantee 只适用于满足所声明 exchangeability 的 split。

### 5.2 Gauge-metamorphic protocol

对同一 posterior 的所有 carrier UUID、event-participant support、site parent 和 witness mapping 施加一致随机置换，typed query 与物理几何保持不变。变换前后必须满足：

- `execution_authorized`、公共安全动作集合和最终物理 skill 不变；
- address posterior 与 witness 只按同一置换重命名；
- event/site query 不得因内部数组顺序或 canonical representative 改变语义。

再构造负对照：只置换某一层或删除 event-role evidence，确认审计器能够检出 contract violation。报告 gauge-consistency violation rate，而不是只看 IDF1。

## 6. 必须击败的 baselines

1. MAP/硬 ID tracker；
2. label entropy threshold；
3. posterior action-disagreement threshold；
4. labeling-uncertainty posterior；
5. full FOT posterior，不做 gauge audit 或 common-action certificate；
6. object-composition POMDP；
7. deterministic event graph：事件直接连接 top-1 UUID；
8. event graph + oracle re-ID：给出事件寻址性能上界；
9. 小规模 exact POMDP oracle。
10. temperature/isotonic calibration：只校准 probability，不输出集合；
11. split conformal、structured conformal 与 online conformal variants。

核心指标：

- false identity commitment rate at fixed coverage；
- task-sensitive recall 与 task-invariant accuracy；
- witness validity 与 posterior mass coverage；
- identity-value calibration；
- information-action regret/cost；
- event-participant NLL/Brier、anchor-strength calibration 与 revision accuracy；
- identity/event set coverage、average set size 与 coverage–efficiency curve；
- common-action availability、robust-action regret 与 gauge-consistency violation rate；
- `out_of_beam` recall、false alarm 和 hypothesis expansion cost；
- episode-level simultaneous coverage 与 per-step/long-run coverage 分开报告；
- downstream success 与 risk–coverage curve。

## 7. Novelty 与 No-Go

不能主张首次提出 permutation symmetry、label uncertainty、belief equivalence、value of information、active perception、event memory、event grounding、conformal prediction 或 POMDP planning。Symmetric-group MOT 已维护 permutation distribution；Event-Grounding Graph 已支持基于事件历史的对象查询，但假设对象是唯一实体并使用 ground-truth re-identification；MOT-CUP、Conformal Structured Prediction、Perceive with Confidence、Utility-Directed Conformal Prediction 和 Conformal Decision Theory 已覆盖 tracking uncertainty、structured sets、robot planning 与 decision-aware calibration。TCIO 是问题定义；公共动作风险命题是 coverage 的推论，都不是算法贡献。

唯一候选方法贡献是：

> **Gauge-consistent common-action certificate over persistent entity belief**：在会 split/merge/miss 的 foundation observations 下，构造对内部 UUID 重命名等变、含 `out_of_beam` 的 calibrated event/site address set；求各地址安全技能集合的交集，交集非空才执行，否则返回 witness、信息动作或 abstention。它必须比相同 belief 上的 entropy、action disagreement、普通 decision-aware conformal 和 POMDP baselines，在相同 risk/coverage 下保留更高的有效执行率。

equivariant event/site address 与 support-aware conformal guard 是该 quotient 的两个必要机制；其余模块都是 posterior 底座、证据源或评测协议。

以下任一结果成立即 No-Go：

- posterior entropy 或 action disagreement 可复现全部收益；
- false commitment 下降只因系统总是 abstain；
- witness permutation 与真实 ambiguity 不一致；
- event anchor 退化为把 top-1 UUID 写入日志，或其校准/修订不优于 deterministic event graph；
- nominal 95% identity/event set 在有效 IID split 上覆盖率显著不足；
- coverage 只能靠几乎总是包含 `out_of_beam` 或全部 entities 获得；
- typed query 换成 counterfactual pair 后，输出不随 task sensitivity 改变；
- exact POMDP 在相同算力下显著更好且 TCIO 没有可解释性/效率优势。

## 8. 最小代码边界

```text
identity_symmetry.py
  canonicalize_unlabeled_graph()
  extract_witness_permutations()
  build_identity_orbits()

task_query_dsl.py
  parse_typed_query()
  evaluate_query_under_hypothesis()

event_addressing.py
  infer_participant_posterior()
  resolve_event_address()
  append_event_revision()

identity_observability.py
  compare_task_stabilizer()
  estimate_identity_value()
  rank_information_actions()
  build_certificate()

conformal_identity.py
  fit_episode_quantile()
  build_address_set()
  detect_out_of_beam()
  audit_coverage_shift()

evaluate_identity_observability.py
  paired_query_metrics()
  false_commitment_curve()
  witness_audit()
```

第一阶段只读取离线 top-K hypothesis JSONL，不修改现有 Stage 2。只有离线 paired protocol 通过 Go/No-Go 后，才接入在线 pipeline。
