不错！整体思路和符号都很接近 HILP / successor features 的常见写法，不过有几处要点我建议你改一改或标注清楚。我按你的小节逐条给出“✅正确/⚠️建议修改/📝补充建议”。

# 1. HILP

## 1.1 Formulation

### (a) Q 备份

* ✅ 公式结构对（是 SARSA/期望备份范式）：
  $Q(s,a,z)=r+\gamma Q(s',a',z)$
* ⚠️ 建议写成**带策略期望**或**max**的通用形式，并把“奖励依赖 z”的事实写明确：

  $$
  Q^\pi(s,a;z)=r_z(s,a,s')+\gamma\,\mathbb E_{a'\sim\pi(\cdot|s')}[Q^\pi(s',a';z)]
  $$

  或

  $$
  Q^*(s,a;z)=r_z(s,a,s')+\gamma\max_{a'}Q^*(s',a';z)
  $$

  这样读者能一眼看出你是 on-policy 还是 off-policy。

### (b) Successor Features（SF）

* ⚠️ **最重要的修正**：**SF 不应依赖 $z$**。
  标准 SF 定义与 $z$ 解耦：$\psi(s,a)=\mathbb E[\sum_{t\ge0}\gamma^t\,\phi(\cdot)_t]$，然后

  $$
  Q(s,a;z)=\psi(s,a)^\top z
  $$

  你当前写的是 $SF(s,a,\mathbf z)$ 和递推 $SF(s,a,z)=R_{sf}+\gamma SF(s',a',z)$，虽然数值上仍能训练，但会在概念上把“任务权重 $z$”和“动力学累积（SF）”混在一起，**建议统一改成**：

  $$
  \psi(s,a)=\underbrace{c(s,a,s')}_{\text{cumulant}}+\gamma\,\psi(s',a')
  $$

  其中 $c$ 就是你定义的 $R_{sf}$（见下一条）。
* ✅ cumulant（$R_{sf}$）两种设定写法没问题：

  * state-type：$c_{state}=\phi(s')-\bar\phi$
  * diff-type：$c_{diff}=\phi(s')-\phi(s)$
* ✅ 用 SF 得到 Q：

  $$
  Q(s,a;z)=\psi(s,a)^\top z
  $$

  （等价于你写的 $SF @ z^\top$，但建议把 $SF$ 统一记作 $\psi$，并去掉 $z$ 下标）
* 📝 终止状态：实现里要有 done-mask：
  $\psi(s,a)=c+\gamma(1-\text{done})\psi(s',a')$
* 📝 目标与损失：你现在的

  $$
  TD_{sf}=R_{sf}+\gamma\,SF(s',a',z)
  $$

  再配

  $$
  L_{sf}=\mathrm{MSE}(TD_{sf},SF(s,a,z))
  $$

  语义正确（把左边当 target），但为了不让读者误会“TD=目标”，建议直接写：

  $$
  \text{target}_{sf}=c+\gamma\,\psi(s',a'),\quad
  L_{sf}=\mathrm{MSE}(\text{target}_{sf},\psi(s,a))
  $$
* 📝 “use\_q\_loss”变体：把 $\psi$ 映射到标量 $Q=\psi^\top z$ 后做 MSE 也行：

  $$
  \text{target}_Q=(c+\gamma\,\psi(s',a') )^\top z,\quad
  L=\mathrm{MSE}(\text{target}_Q,\ \psi(s,a)^\top z)
  $$

  但要意识到：**只按某个 $z$ 的投影训练，会丢失正交方向的信息**。若要学“通用”$\psi$，建议主损失放在向量级 $\psi$ 上，必要时再加上若干随机 $z$ 的投影损失做正则。

## 1.2 训练目标（固定 $z$ 最大化回报）

* ✅ 两种 reward 的定义与关系式基本正确：

  $$
  r_{diff,t}=(\phi(s_t)-\phi(s_{t-1}))^\top z,\quad
  r_{state,t}=(\phi(s_t)-\bar\phi)^\top z
  $$

  $$
  r_{diff,t}=r_{state,t}+(\bar\phi-\phi(s_{t-1}))^\top z
  $$

  $$
  \Rightarrow\ \mathrm{Ret}_{diff}
  =\mathrm{Ret}_{state}
  +\sum_t\gamma^t(\bar\phi-\phi(s_{t-1}))^\top z
  $$
* 📝 更严谨的小修饰：

  * 把时间索引统一（比如从 $t=1$ 起，且 $s_{t-1}$ 定义清楚）。
  * 说明 $\bar\phi$ 是**数据分布下的特征均值**（经验均值 or EMA），否则“center”的来源不清楚。
  * 如果想讨论与“潜在势能塑形（potential-based shaping）”的关系，要特别注意 $\gamma$ 的系数；当前关系式不是标准的 $\gamma F(s')-F(s)$ 形式，仅是一个可解释的**中心化**差异项。

## 1.3 Evaluation（从任务奖励反推出 $z$）

* ✅ 用数据集的 $(s,s')$ 和任务奖励 $r_{task}(s,s')$ 做回归的思路正确。
* ⚠️ **目标函数应为二次/最小二乘**，你现在写的是线性的 $\arg\min \mathbb E[r_{task}-\dots]$，会出现“无限制地把 $z$ 放大/反号”的退化。建议改成：

  $$
  z^*=\arg\min_{z}\ \mathbb E_{(s,s')\sim D}\Big[\big(r_{task}(s,s')-(\phi(s')-\phi(s))^\top z\big)^2\Big]
  $$

  并可加 $L_2$ 正则 $\lambda\|z\|_2^2$。
* 📝 闭式解 & 规范化：

  * 闭式（无正则）：$z^*=(X^\top X)^{-1}X^\top r$，其中 $X=\phi(s')-\phi(s)$，$r=r_{task}$。
  * 若你希望把 $z$ 放在希尔伯特单位球上再比较方向，可在得到 $z^*$ 后做
    $\hat z=\frac{z^*}{\|z^*\|_2}$（这一步你已经写了，✅）。
* 📝 若训练时用的是 **state-type cumulant**，评估阶段的线性模型也要匹配：

  $$
  r_z(s,s')=(\phi(s')-\bar\phi)^\top z
  $$

  否则会有系统性偏差（与训练目标不一致）。

---

# 小结（一页版改写）

* 用**与 $z$ 无关**的 $\psi(s,a)$：

  $$
  \psi(s,a)=c(s,a,s')+\gamma\,\psi(s',a'),\quad
  Q(s,a;z)=\psi(s,a)^\top z
  $$
* cumulant $c$：

  $$
  c_{diff}=\phi(s')-\phi(s)\quad\text{或}\quad c_{state}=\phi(s')-\bar\phi
  $$
* 训练损失：

  $$
  L_{sf}=\mathrm{MSE}\big(c+\gamma\,\psi(s',a'),\ \psi(s,a)\big)
  $$

  （可选再加若干随机 $z$ 的投影损失）
* 评估/拟合 $z$（与训练的 cumulant 一致）：

  $$
  z^*=\arg\min_z \mathbb E[(r_{task}-Xz)^2]+\lambda\|z\|_2^2,\quad
  X=
  \begin{cases}
    \phi(s')-\phi(s) & \text{diff}\\
    \phi(s')-\bar\phi & \text{state}
  \end{cases}
  $$

如果你愿意，我可以把上面这套改成你代码/草稿里的精确符号，并顺手给出带 done-mask、target network、随机 $z$ 正则的 PyTorch 版伪代码。
