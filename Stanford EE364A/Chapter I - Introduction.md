# Chapter I: Introduction
第一章介绍了本书的主要内容、核心问题的前置知识以及变量标记方法。

## 1.1 Mathematical optimization
一个标准的优化问题应当具备以下形式：
$$
    \begin{align}
        \notag \min \ \ &f_0 (\pmb{x}) \\
        \notag s.t. \ \ &f_i(\pmb{x}) \leqslant b_i, \ i=1,2,\cdots,m
    \end{align}
    \tag{1}
$$
其中行（或列）向量 $\pmb{x} \in \mathbb{R}^n$ 称为优化问题的**优化变量**，映射函数 $f_0: \mathbb{R}^n \rightarrow \mathbb{R}$ 称为**目标函数**，映射函数 $f_i: \mathbb{R}^n \rightarrow \mathbb{R}$ 称为**约束函数**，$b_i$ 为 $f_i$ 的**约束/边界条件**。若式 (1) 存在最优解，则记为 $\pmb{x}^*$，$\pmb{x}^*$ 在满足全体约束函数 $f_i$ 的前提下还将取得目标函数的最小值，即有：
$$
    \forall \pmb{z}, \ \ 
    \begin{cases}
        f_1(\pmb{z}) \leqslant b_1,\\
        f_2(\pmb{z}) \leqslant b_2,\\
        \cdots \\
        f_m(\pmb{z}) \leqslant b_m,\\
    \end{cases}, \ \ f_0(\pmb{z}) \geqslant f_0(\pmb{x}^*)
    \tag{2}
$$
优化问题通常按照目标函数、约束函数的种类进行划分。如常见的线性优化问题，即目标/约束函数满足线性条件：
$$
    f_i(\alpha \pmb{x} + \beta \pmb{y}) = \alpha f_i(\pmb{x}) + \beta f_i(\pmb{y}), \ \ \forall \pmb{x},\pmb{y} \in \mathbb{R}^n, \ \alpha,\beta \in \mathbb{R}
    \tag{3}
$$
本教程主要研究凸优化问题，其目标函数与约束函数需满足：
$$
    \begin{cases}
        f_i(\alpha \pmb{x} + \beta \pmb{y}) \leqslant \alpha f_i(\pmb{x}) + \beta f_i(\pmb{y}), \ \ \forall \pmb{x},\pmb{y} \in \mathbb{R}^n\\
        \ \\
        \alpha + \beta = 1, \ \ \alpha,\beta \geqslant 0,
    \end{cases}
    \tag{4}
$$
对比式 (3) 可见，凸性比线性更具有普遍意义：首先，不等式的可规划范围大于等式，后者的限制性过强以至于缺乏现实应用场景；其次，不等条件仅对特定的函数系数 $\alpha$、$\beta$ 成立，即凸问题可视为线性问题的推广。
***

## 1.2 Least-squares and linear programming
最小二乘问题是一种求解方法相对成熟的无约束优化问题，其目标函数是误差项的平方和：
$$
    \min \ \ f_0(\pmb{x}) = \left\| \pmb{Ax} - \pmb{b} \right\|_2^2 = \sum_{i=1}^m \left( \pmb{a}_i \pmb{x} - b_i \right)^2, \ \ \pmb{A} \in \mathbb{R}^{m \times n}, \pmb{x} \in \mathbb{R}^n
    \tag{5}
$$
式 (5) 等价于求解线性方程组 ${\pmb{A}}^T \pmb{Ax} = {\pmb{A}}^T \pmb{b}$，其解析解为 $\pmb{x} = {\left( {\pmb{A}}^T \pmb{A} \right)}^{-1} {\pmb{A}}^T \pmb{b}$。式中 $\pmb{a}_i$ 为 $\pmb{A}$ 的行向量，$b_i$ 为向量 $\pmb{b}$ 的元素。最小二乘问题的求解具有较高精度与可靠性，其理论求解时间与 $mn^2$ 近似成比例。当 $\pmb{A}$ 为稀疏矩阵时，求解速度将进一步提高。当然，若变量数 $n$ 多达几百万时，通过上述方法解决最小二乘问题倒是可能面临硬件方面的困难。

判定一个优化问题是否为最小二乘问题，首先需要确认目标是否为二次函数，之后确认二次型是否为半正定的。最小二乘问题主要有两种变体应用：加权最小二乘与正则化。在加权最小二乘问题中，权重系数 $w_i$ 反映了误差项的重要程度，亦或者是对解向量 $\pmb{x}$ 的修正，其目标函数被替换为：
$$
    \min \ \ f_0(\pmb{x}) = \left\| \pmb{WAx} - \pmb{Wb} \right\|_2^2 = \sum_{i=1}^m w_i \left( \pmb{a}_i \pmb{x} - b_i \right)^2, \ \ w_i > 0, \ \pmb{W} = diag(w_1,w_2,\cdots,w_m)
    \tag{6}
$$
正则化问题是在式 (5) 的基础上增添了额外的惩罚项，通常用来约束解向量 $\pmb{x}$ 的某些性质，如数值之和（能量大小）、与某固定值的偏差等等，因此正则化项的具体设计是多种多样的。一种常见的正则化形式如下所示：
$$
    \min \ \ f_0(\pmb{x}) = \left\| \pmb{Ax} - \pmb{b} \right\|_2^2 + \rho \left\| \pmb{x} \right\|_2^2, \ \ \rho > 0
    \tag{7}
$$
值得注意的是，加权最小二乘与正则化最小二乘都能通过一些数学方法转化为标准的最小二乘问题加以解决，具体方法将在后续章节详细介绍。

线性规划是另一类重要的优化问题，其目标函数与约束函数均满足线性要求：
$$
    \begin{align}
        \notag \min \ \ &\pmb{c}^T \pmb{x}, \ \ \pmb{c} \in \mathbb{R}^n\\
        \notag s.t. \ \ &{\pmb{a}_i}^T \pmb{x} \leqslant b_i, \ \ i=1,2,\cdots,m, \ \pmb{a}_i \in \mathbb{R}^n, \ b_i \in \mathbb{R}
    \end{align}
    \tag{8}
$$
与最小二乘问题不同，线性规划问题通常不存在解析解，同时也难以精确估计求解时间，但我们依旧可以使用各种行之有效的方法来对其求解。同样地，若问题本身是稀疏的，求解速度能够进一步加快。

举一个形式简单的例子作为说明，考虑切比雪夫近似问题（Chebyshev approximation problem）：
$$
    \min \ \ \underset{i=1,2,\cdots,m} \max \left| {\pmb{a}_i}^T \pmb{x} - b_i \right|, \ \ \pmb{x} \in \mathbb{R}^n, \ \pmb{a}_i \in \mathbb{R}^n, \ b_i \in \mathbb{R}
    \tag{9}
$$
在最小二乘问题中，目标函数是误差项的平方和，而此处目标函数是误差项的最大绝对值。前者的目标是二次的、可微的，后者不可微。形如式 (9) 的问题可以转化为如下所示的线性规划问题，具体原因及解法将在后续章节介绍：
$$
    \begin{align}
        \notag \min \ \ &t, \ \ t \in \mathbb{R}^+\\
        \notag s.t \ \ &
            \begin{cases}
                {\pmb{a}_i}^T\pmb{x} - t \leqslant b_i\\
                \ \\
                -{\pmb{a}_i}^T\pmb{x} - t \leqslant -b_i\\
            \end{cases}, \ \ i=1,2,\cdots,m
    \end{align}
    \tag{10}
$$
***

## 1.3 Convex optimization
凸优化问题的表现形式在 1.1 节已经给出。同样地，凸优化问题通常没有解析解，但存在诸多其它的有效求解方法。令人宽慰的是，从概念上可以确定，一旦某个优化问题被转化为凸优化问题，则该问题就能够获得明确的最优解，其中唯一的困难在于，相比最小二乘与线性规划问题，凸优化问题的识别与转化往往需要更多艰深精妙的技巧。
***

## 1.4 Nonlinear optimization
非线性优化在本教程中不会涉及太多，但是我们有必要在 Introduction 部分简单认识它们与凸优化问题的联系。非线性优化与线性优化是对立的，但是它与凸优化并不互斥。**局部优化**与**全局优化**是两种主要的非线性优化方法，而遗憾的是，目前没有能够解决一般非线性规划的通用有效方法，上述两者都存在性能或结果上的牺牲。

非线性优化的大部分研究都落脚于局部优化。在工程设计中，局部优化通常用于改进手工方案或其它预设方案，这种方法放弃了寻找 $\pmb{x}^*$，转而在预先设定的可行域上寻找局部最优点。由于局部优化的目标函数和约束函数是可微的，因此具有运算速度快、处理规模大、应用广泛等优点。局部优化的缺点自然落在“局部”二字上，对初始变量（条件）的重复多次尝试并不能保证迭代结果的一致性，同时局部解离全局最优点的效能难以确定，且部分方法对于算法参数设置非常敏感。教材原文中对局部优化有一个非常有趣的描述：

>Roughly speaking, local optimization methods are more art than technology. Local optimization is a well developed art, and often very effective, but it is nevertheless an art.

关于局部优化与凸优化的关系，文中也有一个精妙的对比：

（1）大多数局部优化方法只需要目标/约束函数是连续可微的，因此把实际问题视为非线性优化问题是很简单的，真正的艺术（难点）在于如何解决问题；
（2）在凸优化问题中情况则完全相反，将实际问题识别并提炼转化为凸优化问题是相对困难的，此举一旦完成，问题距离解决仅剩一步之遥。。

全局优化意味着找到优化问题的真全局解，相应的代价自然是效率，毕竟遍历能解决一切问题，只要等得起。全局优化通常用于少变量离散优化（遍历成本相对较低）或者系统安全性能分析（真全局解至关重要）等场景。例如在航天系统的可靠性分析问题中，局部最优也许能够快速找到一组乃至多组风险参数，但无法保证面面俱到。此时没有什么比安全顺利地完成任务更加重要，因此工程师往往会选择全局最优方法，以确保系统设计能够抵抗预想范围内最恶劣的意外情况。

本节第一段已有说明，即非线性优化不一定是凸优化。对于部分非凸优化问题，凸优化依然有用武之地。我们可以尝试用一个凸优化问题近似表述原始问题，并获得近似凸问题的精确最优解。在此基础上进行原始问题的局部优化，能够显著降低求解时长与局部解误差。同时，许多全局优化方法都要求对原始非凸问题的最优值求解存在易行可计算下界（埋头干活之前总得估计一下寻优成本吧），对原始问题的约束条件进行松弛以至于达到凸优化标准是常用的方法之一。
***