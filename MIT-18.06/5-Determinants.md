---
html:
    toc: true
print_background: true
---

# 5. Determinants
当我学习线代时，这一部分的内容在同济版线代课本中是第一节（天津大学自用的课本也一样），试想一下对于一个纯新手而言，开场白如同天书一般难以消化，往后的日子大概率也是遍布阴霾。而在 MIT-18.06 这里，行列式被安排在了第五章。此时我们已经对矩阵的初等变换、LU 分解、可逆性等诸多性质都有了充分的感性与理性理解，关于行列式的各种性质推导也都变得顺理成章。

我相信作为教师，国内高校老师的水平与国外没有显著差别，甚至本土老师更了解国情与学生的个性，本应提供质量更高的课堂。然而受限于违反教育规律的各种教材，最终质量往往不如人意，这一现象值得每一位教育从业者以及有意向投身教育事业的人反思。

## 5.1 The Properties of Determinants
方阵的行列式是一个简单的数字，但是这个数字包含了非常丰富的矩阵信息。其中最基本、最重要的信息是 $det (\pmb{A}) = 0$，则矩阵奇异，即不可逆；反之 $det (\pmb{A}) \ne 0$，则矩阵非奇异、可逆。对于二阶方阵而言，行列式的具体计算公式为：
$$
    \pmb{A} = 
    \begin{bmatrix}
        a & b\\ c & d\\
    \end{bmatrix}, \ \det (\pmb{A}) = 
    \begin{vmatrix}
        a & b\\ c & d\\
    \end{vmatrix} = ad - bc
    \tag{5-1-1}
$$

在此基础上，我们接下来依次介绍行列式的十条基本性质。

**（1）单位阵的行列式为 1**：
$$
    \det (\pmb{I}) = 1
    \tag{5-1-2}
$$
**（2）每进行一次行交换，行列式变号一次**。关于这一条的延申是：任何置换矩阵的行列式不是 1 就是 -1，符号取决于它由单位阵经过了多少次行变换得来。

**（3-A）矩阵单行乘上常系数，其行列式的值也乘上相同系数**：
$$
    \begin{vmatrix}
        ta & tb\\ c & d\\
    \end{vmatrix} = t
    \begin{vmatrix}
        a & b\\ c & d\\
    \end{vmatrix}
    \tag{5-1-3}
$$
**（3-B）矩阵单行对应的线性组合同样存在于行列式中**：
$$
    \begin{vmatrix}
        a+a_0 & b+b_0\\ c & d\\
    \end{vmatrix} = 
    \begin{vmatrix}
        a & b\\ c & d\\
    \end{vmatrix} + 
    \begin{vmatrix}
        a_0 & b_0\\ c & d\\
    \end{vmatrix}
    \tag{5-1-4}
$$
前三条性质是基础中的基础，根据它们以及一些学过的矩阵知识（无需（5-1-1））即可以推导出剩下的七条性质！

**（4）矩阵中存在相同行时，行列式为 0**：
$$
    \begin{vmatrix}
        a & b\\ a & b\\
    \end{vmatrix} = 0
    \tag{5-1-5}
$$
这一条性质可由（2）证明，交换两相同行不会改变矩阵的任何元素，所以行列式应当相等；而进行了一次行交换意味着行列式应当变号。唯一能同时满足以上两个条件的情况即行列式为 0。

**（5）线性组合矩阵中的行向量不会改变矩阵的行列式**：
$$
    \begin{vmatrix}
        a & b\\ c+ka & d+kb
    \end{vmatrix} = 
    \begin{vmatrix}
        a & b\\ c & d\\
    \end{vmatrix}
    \tag{5-1-6}
$$
这一条可由（3）（4）联合证明：
$$
    \begin{vmatrix}
        a & b\\ c+ka & d+kb
    \end{vmatrix} = \begin{vmatrix}
        a & b\\ c & d\\
    \end{vmatrix} + k
    \underbrace{
        \begin{vmatrix}
            a & b\\ a & b\\
        \end{vmatrix}}_0 = 
    \begin{vmatrix}
        a & b\\ c & d\\
    \end{vmatrix}
$$
**（6）矩阵存在全零行时，行列式为 0**：
$$
    \begin{vmatrix}
        a & b\\ 0 & 0\\
    \end{vmatrix} = 0
    \tag{5-1-7}
$$
这一条可以通过（3-B）证明：
$$
    \begin{vmatrix}
        a & b\\ 0 & 0\\
    \end{vmatrix} = 
    \begin{vmatrix}
        a & b\\ 0 & 0\\
    \end{vmatrix} + 
    \begin{vmatrix}
        a & b\\ 0 & 0\\
    \end{vmatrix} \ \Longrightarrow \ 
    \begin{vmatrix}
        a & b\\ 0 & 0\\
    \end{vmatrix} = 0
$$
**（7）三角矩阵（或对角阵）的行列式为主元乘积**：
$$
    \begin{align}
        \notag \pmb{A} &= \pmb{LU} \in \mathbb{R}^{n \times n}, \ \det(\pmb{U}) = \prod_{i=1}^n \pmb{U}(i,i)\\
        \notag \ \\
        \notag \pmb{D} &= diag(d_1,d_2,\cdots,d_n) \in \mathbb{R}^{n \times n}, \ \det(\pmb{D}) = \prod_{i=1}^n d_i
    \end{align}
    \tag{5-1-8}
$$
首先需要明确，经过 LU 分解得到的上三角矩阵 $\pmb{U}$ 可以进一步行变换得到对角阵 $\pmb{D}$。接下来对 $det(\pmb{D})$ 反复应用（3-A）提取每一行的系数：
$$
    \det(\pmb{D}) = d_1
    \begin{vmatrix}
        1 & 0 & \cdots & 0\\
        0 & d_2 & \cdots & 0\\
        \vdots & \vdots & \ddots & \vdots\\
        0 & 0 & \cdots & d_n
    \end{vmatrix} = \cdots = 
    (d_1 d_2 \cdots d_n) \times \det(\pmb{I}_n) = \prod_{i=1}^n d_i
$$
**（8）$det(\pmb{A})=0$ 则 $\pmb{A}$ 是奇异矩阵，反之 $det(\pmb{A}) \ne 0$ 则 $\pmb{A}$ 可逆**。由于后半句与前半句严格对立，因此我们着重证明前半句：

$\Leftarrow$：奇异矩阵通过行变换能化简出全零行，所以行列式为 0；

$\Rightarrow$：以二阶方阵行列式为例：
$$
    \det(\pmb{A}) = 
    \begin{vmatrix}
        a & b\\ c & d\\
    \end{vmatrix} = ad-bc = 0
$$
假设 $abcd \ne 0$（若其中存在某元素为 0，则必存在全零行或全零列），现证明 $\pmb{A}$ 的两行向量存在线性关系：
$$
    ad = \left(a \times \dfrac{b}{a} \right) \times \left(d \times \dfrac{a}{b} \right) = b \times \dfrac{ad}{b} = bc
$$
因为四个元素均不为零，则必有 $\dfrac{ad}{b}=c$，即有：
$$
    \pmb{A} = 
    \begin{bmatrix}
        a & b\\
        c & d\\
    \end{bmatrix} = 
    \begin{bmatrix}
        a & b\\
        a \times \dfrac{d}{b} & b \times \dfrac{d}{b}\\
    \end{bmatrix}
$$
即行列式为零的二阶方阵中，行向量组存在线性关系，由此可以推断出矩阵奇异。

**（9）矩阵乘积的行列式等于行列式的乘积**：
$$
    \det(\pmb{AB}) = \det(\pmb{A}) \times \det(\pmb{B})
    \tag{5-1-9}
$$
这一条可结合 LU 分解的性质来加以证明。已知在**无需行交换**的情况下，LU 分解的结果是一个下三角矩阵 $\pmb{L}$（对角线元素全为 1）以及上三角矩阵 $\pmb{U}$，因此 $\det(\pmb{L})=1$。考虑 $\pmb{A}$ 与 $\pmb{U}$ 的关系：$\pmb{U}$ 由 $\pmb{A}$ 在不经过行交换（**行列式不变号**）的情况下，仅通过行向量的线性组合（**行列式不变**）得来，因此必有 $\det(\pmb{U})=\det(\pmb{A})$，即：
$$
    \det(\pmb{A}) = \det(\pmb{LU}) = \underbrace{\det(\pmb{L})}_{1} \times \det(\pmb{U})
$$
这是无需行交换的情况。那么当需要行交换时，其本质是在 $\pmb{L}$ 的基础上新增一些置换矩阵，假设它们的乘积是 $\pmb{P}$ 且 $\det(\pmb{P}) = (-1)^n$，$n$ 表示行交换次数。考虑 $\pmb{A}$ 与 $\pmb{U}$ 的关系：$\pmb{U}$ 由 $\pmb{A}$ 经过 $n$ 次行交换（**行列式变号 $n$ 次**）、行向量线性组合（**行列式不变**）得来，因此有 $\det(\pmb{U}) \times (-1)^n = \det(\pmb{A})$，即：
$$
    \det(\pmb{A}) = \det\left(\pmb{P}^{-1}\pmb{LU}\right) = \underbrace{\dfrac{1}{\det(\pmb{P})}}_{(-1)^n} \times \underbrace{\det(\pmb{L})}_{1} \times \det(\pmb{U})
$$
综上所述，不论是否需要行交换，我们始终可以通过 LU 分解证明性质（9）的成立。

**（10）矩阵转置后行列式不变**：
$$
    \det\left(\pmb{A}^T\right) = \det(\pmb{A})
    \tag{5-1-10}
$$
在（9）的基础上，最后一条其实很好证明了。假设有 $\pmb{A} = \pmb{P}^{-1}\pmb{LU} \in \mathbb{R}^{n \times n}$，则 $\pmb{A}^T = \pmb{U}^T \pmb{L}^T \pmb{P}$（置换矩阵的转置即为其逆矩阵），故有：
$$
    \begin{align}
        \notag \det(\pmb{A}) &= \det\left(\pmb{P}^T\right) \times \det(\pmb{L}) \times \det(\pmb{U})\\
        \notag \ \\
        \notag \det\left(\pmb{A}^T \right) &= \det\left(\pmb{U}^T \right) \times \det\left(\pmb{L}^T \right) \times \det(\pmb{P})
    \end{align}
$$
由于 $\pmb{L}$、$\pmb{U}$ 都是三角阵，因此它们的转置不改变行列式的值。现在唯一的问题在于 $\det\left(\pmb{P}^T\right)$ 是否等于 $\det(\pmb{P})$。

首先毫无疑问的是 $\left|\det(\pmb{P})\right|=1$，即只存在符号差别；

其次单位阵以及部分 $\pmb{P}$ 本身即为 *Hermitte* 矩阵（$\pmb{P}^T=\pmb{P}$），这类 $\pmb{P}$ 显然有 $\det\left(\pmb{P}^T\right) = \det(\pmb{P})$。需要强调一下什么时候会产生对称置换矩阵：**变更位置的行向量有且仅有一次交换**。

最后是部分非对称的 $\pmb{P}$，假设它们主对角线上有 $m$ 个 1（$m<n$），说明进行了 $n-m-1$ 次**不回头**的行交换（即不考虑交换以后又换回去的情况），当它转置后，偏离主对角线的元素数量不变，还是相当于进行了 $n-m-1$ 次**不回头**的行交换，所以行列式的值不会改变，同样有 $\det\left(\pmb{P}^T\right) = \det(\pmb{P})$。

综上可知（5-1-10）成立。综上可知（5-1-10）成立。（5-1-10）一个更重要的意义在于，之前的（2）至（6）条性质可以拓展至列：所有列变换对行列式带来的影响等价于矩阵转置后的行变换。因此有：**一次列交换使得行列式变号一次**、**矩阵单列对应的线性组合同样存在于行列式中**、**矩阵存在相同列时行列式为 0**、**线性组合矩阵的列向量不改变行列式**、**矩阵存在全零列时行列式为零**。

## 5.2 Permutations and Cofactors
### 5.2.1 行列式计算公式
5.1 节我们讲解了行列式的基本性质，以二阶方阵为例加以证明。这一节我们将彻底解决 $n$ 阶方阵行列式的计算问题。在此之前，我们有必要先回顾一下二阶行列式：
$$
    \begin{vmatrix}
        a & b\\ c & d\\
    \end{vmatrix} = ad-bc
$$
这个公式可以通过基本性质（1）至（3）推导出来，学习这个过程有利于我们理解 $n$ 阶行列式的情况：
$$
    \begin{vmatrix}
        a & b\\ c & d\\
    \end{vmatrix} = 
    \begin{vmatrix}
        a & 0\\ c & d\\
    \end{vmatrix} + 
    \begin{vmatrix}
        0 & b\\ c & d\\
    \end{vmatrix} = \left(
    \underbrace{
        \begin{vmatrix}
            a & 0\\ c & 0\\
        \end{vmatrix}}_{0} + 
    \underbrace{
        \begin{vmatrix}
            a & 0\\ 0 & d\\
        \end{vmatrix}}_{ad} \right) + \left(
    \underbrace{
        \begin{vmatrix}
            0 & b\\ c & 0\\
        \end{vmatrix}}_{-bc} + 
    \underbrace{
        \begin{vmatrix}
            0 & b\\ 0 & d\\
        \end{vmatrix}}_{0} \right)
$$
接下来把维度拓展至 $\mathbb{R}^{3 \times 3}$：
$$
    \det(\pmb{A}) = 
    \begin{vmatrix}
        a_{11} & a_{12} & a_{13}\\
        a_{21} & a_{22} & a_{23}\\
        a_{31} & a_{32} & a_{33}\\
    \end{vmatrix} = 
    \begin{vmatrix}
        a_{11} & 0 & 0\\
        0 & a_{22} & 0\\
        0 & 0 & a_{33}\\
    \end{vmatrix} + 
    \begin{vmatrix}
        a_{11} & 0 & 0\\
        0 & 0 & a_{23}\\
        0 & a_{32} & 0\\
    \end{vmatrix}\\
    \ \\ + 
    \begin{vmatrix}
        0 & a_{12} & 0\\
        a_{21} & 0 & 0\\
        0 & 0 & a_{33}\\
    \end{vmatrix} + 
    \begin{vmatrix}
        0 & a_{12} & 0\\
        0 & 0 & a_{23}\\
        a_{31} & 0 & 0\\
    \end{vmatrix} + 
    \begin{vmatrix}
        0 & 0 & a_{13}\\
        a_{21} & 0 & 0\\
        0 & a_{32} & 0\\
    \end{vmatrix} + 
    \begin{vmatrix}
        0 & 0 & a_{13}\\
        0 & a_{22} & 0\\
        a_{31} & 0 & 0\\
    \end{vmatrix}
$$
当我们排除掉行列式为 0 的组合时，就只剩下六种情况了。不难发现，如上所示的组合都有一个共同特点：**各行各列有且仅有一个默认非零的元素**，且组合总个数为 $n!$。进一步地，我们把 $\det(\pmb{A})$ 的公式完全展开：
$$
    \det(\pmb{A}) = a_{11}a_{22}a_{33} - a_{11}a_{23}a_{32} - a_{12}a_{21}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31}
    \tag{5-2-1}
$$
（5-2-1）我们先留着，后边还能用上。据此给出 $\pmb{A} \in \mathbb{R}^{n \times n}$ 时的行列式：
$$
    \det(\pmb{A}) = \sum_{i=1}^{n!} \pm {a_{1,i_1} a_{2,i_2} \cdots a_{n,i_n}}
    \tag{5-2-2}
$$
这个公式其实不太完整，至少我们不知道行列式的取值是负是正。在这里我并不想提到国内教材里逆序数的概念，that's rubbish。用一个抽象的数学概念掩盖直观可视的过程，这种看似简约的做法在我看来反而是愚蠢的。因此，我们需要通过另一种途径——代数余子式来达到最后的终点。

### 5.2.2 代数余子式
对（5-2-1）提取公因式：
$$
    \det(\pmb{A}) = a_{11}(a_{22}a_{33}-a_{23}a_{32}) + a_{12}(-a_{21}a_{33}+a_{23}a_{31}) + a_{13}(a_{21}a_{32}-a_{22}a_{31})\\
    \ \\
    = a_{11} 
    \begin{vmatrix}
        a_{22} & a_{23}\\
        a_{32} & a_{33}\\
    \end{vmatrix} - a_{12}
    \begin{vmatrix}
        a_{21} & a_{23}\\
        a_{31} & a_{33}\\
    \end{vmatrix} + a_{13}
    \begin{vmatrix}
        a_{21} & a_{22}\\
        a_{31} & a_{32}\\
    \end{vmatrix}
$$
其中 $a_{11}$ 后面的行列式称为 $a_{11}$ 的余子式，它是从 $\pmb{A}$ 中划去 $a_{11}$ 所在行、列之后剩下的矩阵部分，同理可知 $a_{12}$、$a_{13}$ 的余子式。值得注意的是，余子式的选定并非必须基于第一行展开，任何一行都是可以的。

但是我们还有一个关键的问题没有解决，即余子式前的符号。这个正负号与余子式对应的 $a_{ij}$ 有关，考虑符号之后的余子式称为代数余子式：
$$
    C_{ij} = (-1)^{i+j}\det\binom{\mathbb{R}^{(n-1) \times (n-1)} \ with}{Row \ i \ \& \ Column \ j \ erased}
    \tag{5-2-3}
$$
或者我们可以想象一个类似棋盘的结构：
$$
    \begin{vmatrix}
        + & - & + & - & \cdots\\
        - & + & - & + & \cdots\\
        \vdots & \vdots & \vdots & \vdots & \ddots
    \end{vmatrix}
$$
其中正负号的位置对应于余子式前元素的位置。至此，我们总算来到了终点：
$$
    \det(\pmb{A}) = \sum_j^n {C_{ij}}, \ \forall i \in \{1,2,\cdots,n\}
    \tag{5-2-4}
$$
最后以一类特殊形式的矩阵结束这一部分：
$$
    \pmb{A}_4 = 
    \begin{bmatrix}
        1 & 1 & 0 & 0\\
        1 & 1 & 1 & 0\\
        0 & 1 & 1 & 1\\
        0 & 0 & 1 & 1\\
    \end{bmatrix}
$$
这一类矩阵的特点是主对角线与上下两线上的元素值均为 1。除了 $\pmb{A}_4$，我们当然还能列出更大尺寸的矩阵，这里我们暂时以 $\pmb{A}_4$ 及以下尺寸的矩阵进行说明。不难发现，$\det(\pmb{A}_1)=1$（单元素矩阵），$\det(\pmb{A}_2)=0$（全 1 矩阵），$\det(\pmb{A}_3)$ 稍微麻烦一点：
$$
    \det(\pmb{A}_3) = 
    \begin{vmatrix}
        1 & 1 & 0\\
        1 & 1 & 1\\
        0 & 1 & 1\\
    \end{vmatrix} = 1 \times 
    \begin{vmatrix}
        1 & 1\\
        1 & 1\\
    \end{vmatrix} - 1 \times 
    \begin{vmatrix}
        1 & 1\\
        0 & 1\\
    \end{vmatrix} = \det(\pmb{A}_2) - \det(\pmb{A}_1)
$$
对于 $\pmb{A}_4$，依次选定 $\pmb{A}_4(0,0)$、$\pmb{A}_4(0,1)$ 划取余子式，得到 $C_{00}=\det(\pmb{A}_3)$，且：
$$
    C_{01} = 
    \begin{vmatrix}
        1 & 1 & 0\\
        0 & 1 & 1\\
        0 & 1 & 1\\
    \end{vmatrix} = 1 \times 
    \begin{vmatrix}
        1 & 1\\
        1 & 1\\
    \end{vmatrix} - 1 \times 
    \begin{vmatrix}
        0 & 1\\
        0 & 1\\
    \end{vmatrix} = \det(\pmb{A}_2)
$$
故有：
$$
    \det(\pmb{A}_4) = \det(\pmb{A}_3) - \det(\pmb{A}_2)
$$
这个规律可以推广至 $n$ 维矩阵：
$$
    \det(\pmb{A}_n) = \det(\pmb{A}_{n-1}) - \det(\pmb{A}_{n-2})
    \tag{5-2-5}
$$
如果我们进一步观察：
$$
    \det(\pmb{A}_1) = 1, \ \det(\pmb{A}_2) = 0, \ \det(\pmb{A}_3) = -1\\
    \ \\
    \det(\pmb{A}_4) = -1, \ \det(\pmb{A}_5) = 0, \ \det(\pmb{A}_6) = 1, \ \det(\pmb{A}_7) = 1 \cdots\\
$$
换句话说，这些矩阵的行列式组成了一个以 6 为周期的序列。*Gilbert* 在课上说希望大家喜欢这个案例，我很喜欢。