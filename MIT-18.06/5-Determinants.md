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

## 5.3 Cramer's Rule, Inverse and Volumes
### 5.3.1 逆矩阵公式
在接触行列式的概念之后，我们终于可以获得一个通用的矩阵求逆公式（尽管这个公式操作起来未必就比 LU 分解或者行变换简单），首先以二阶方阵为例：
$$
    \pmb{A}^{-1} = 
    \begin{bmatrix}
        a & b\\ c & d\\
    \end{bmatrix}^{-1} = \dfrac{1}{ad-bc}
    \begin{bmatrix}
        d & -b\\ -c & a\\
    \end{bmatrix} = \dfrac{1}{\det(\pmb{A})}
    \begin{bmatrix}
        C_{1,1} & C_{1,2}\\ C_{2,1} & C_{2,2}
    \end{bmatrix}^T
$$
推广到 $\pmb{A} \in \mathbb{R}^{n \times n}$：
$$
    \pmb{A}^{-1} = \dfrac{1}{\det(\pmb{A})}\pmb{C}^T, \ \pmb{C} = 
    \begin{bmatrix}
        C_{1,1} & C_{1,2} & \cdots & C_{1,n}\\
        C_{2,1} & C_{2,2} & \cdots & C_{2,n}\\
        \vdots & \vdots & \ddots & \vdots\\
        C_{n,1} & C_{n,2} & \cdots & C_{n,n}\\
    \end{bmatrix}
    \tag{5-3-1}
$$
接下来证明（5-3-1），即证明：
$$
    \pmb{A}\pmb{C}^T = \det(\pmb{A}) \pmb{I}
$$
$$
    \underbrace{
        \begin{bmatrix}
            a_{1,1} & a_{1,2} & \cdots & a_{1,n}\\
            a_{2,1} & a_{2,2} & \cdots & a_{2,n}\\
            \vdots & \vdots & \ddots & \vdots\\
            a_{n,1} & a_{n,2} & \cdots & a_{n,n}\\
        \end{bmatrix}}_{\pmb{A}}
    \underbrace{
        \begin{bmatrix}
            C_{1,1} & C_{2,1} & \cdots & C_{n,1}\\
            C_{1,2} & C_{2,2} & \cdots & C_{n,2}\\
            \vdots & \vdots & \ddots & \vdots\\
            C_{1,n} & C_{2,n} & \cdots & C_{n,n}\\
        \end{bmatrix}}_{\pmb{C}^T} = 
    \begin{bmatrix}
        \det{\pmb{A}} & & & \\
        & \det{\pmb{A}} & & \\
        & & \ddots & \\
        & & & \det{\pmb{A}}\\
    \end{bmatrix}
$$
我们先观察等式右边矩阵的第一个元素来源 $a_{1,1}C_{1,1} + a_{1,2}C_{1,2} + \cdots + a_{1,n}C_{1,n}$，不难发现这就是 $\det(\pmb{A})$ 的计算公式，同理右矩阵主对角线上的其它元素分别对应于依照 $\pmb{A}$ 的**不同行**展开的行列式计算公式。现在仅剩的问题是，矩阵其它部分都是 0 吗？换句话说，**当展开行与余子式对应行不匹配时，矩阵元素与代数余子式的乘积是否为 0**？即是否有：
$$
    a_{i,1}C_{j,1} + a_{i,2}C_{j,2} + \cdots + a_{i,n}C_{j,n} = \sum_{k=1,i \ne j}^{n} a_{i,k}C_{j,k} = 0
    \tag{5-3-2}
$$
当然我都给编号了，所以它是成立的。想直接证明这个等式非常困难，但是（5-3-2）的形式与行列式计算很相似，因此我们通过构造特殊矩阵的方式加以说明。

我们具体一点，取 $i=1$，$j=n$ 的情况进行分析，即第一行展开，但是用的是最后一行各元素对应的代数余子式。我们设想这样一种矩阵 $\pmb{X}$：第一行与最后一行相等。显然 $\pmb{X}$ 是奇异矩阵，即 $\det(\pmb{X})=0$。我们分别按第一行和最后一行展开行列式：
$$
    \begin{align}
    \notag \det(\pmb{A}) =
    \begin{vmatrix}
        a_{1,1} & a_{1,2} & \cdots & a_{1,n}\\
        a_{2,1} & a_{2,2} & \cdots & a_{2,n}\\
        \vdots & \vdots & \ddots & \vdots\\
        a_{n-1,1} & a_{n-1,2} & \cdots & a_{n-1,n}\\
        a_{n,1} & a_{n,2} & \cdots & a_{n,n}\\
    \end{vmatrix} &= a_{1,1}C_{1,1} - a_{1,2}C_{1,2} + \cdots + (-1)^{n+1}a_{1,n}C_{1,n} = 0\\
    \notag \ \\
    \notag &= a_{n,1}C_{n,1} - a_{n,2}C_{n,2} + \cdots + (-1)^{n+1}a_{n,n}C_{n,n} = 0
    \end{align}
$$
由于 $\pmb{A}$ 的第一行与最后一行相等，因此有：
$$
    C_{1,1} = (-1)^m C_{n,1}, C_{1,2} = (-1)^m C_{n,2},\ \cdots, \ C_{1,n} = (-1)^m C_{n,n}, \ (-1)^m \sum_{k=n}^n a_{n,k} C_{1,k}=0
$$
当我们构建 $\pmb{A}$ 的第 $i$、$j$ 行相等时，对应于（5-3-2）中 $i$、$j$ 取相应值的情况。至此，我们已经得到并证明了面向一般方阵的逆矩阵求解公式。

### 5.3.2 Cramer 法则
首先需要声明，*Cramer* 法则在线性方程组求解过程中并不实用，计算量大、数值不稳定等等都是它的缺点。据说在微分几何中有它的用武之地，不过这也不是我研究的方向。

在普通线代领域，*Cramer* 法则的理论价值在于方程组有解性的判断：对于方阵系数的方程组（方程个数与未知量个数相等），当系数行列式不等于 0 时，方程组具有唯一解；若系数行列式为 0，则方程组无解或有无数解。

接下来给出基本说明。对于有唯一解的 $n$ 阶线性方程组 $\pmb{Ax}=\pmb{b}$，其解为 $\pmb{x}=\pmb{A}^{-1} \pmb{b} = \dfrac{1}{\det{\pmb{A}}}\pmb{C}^T \pmb{b}$。*Cramer* 法则认为方程解可以由 $n$ 个分量构成：
$$
    \pmb{x}_1 = \dfrac{\det(\pmb{B}_1)}{\det(\pmb{A})}, \ \pmb{x}_2 = \dfrac{\det(\pmb{B}_2)}{\det(\pmb{A})}, \ \cdots \ , \ \pmb{x}_n = \dfrac{\det(\pmb{B}_n)}{\det(\pmb{A})}
    \tag{5-3-3}
$$
其中方阵 $\pmb{B}_1$、$\pmb{B}_2$、$\cdots$、$\pmb{B}_n$ 分别表示 $\pmb{A}$ 中第 $1$、$2$、$\cdots$、$n$ 列被 $\pmb{b}$ 替换后构成的方阵：
$$
    \pmb{B}_1 = 
    \begin{bmatrix}
        b_1 & a_{1,2} & \cdots & a_{1,n}\\
        b_2 & a_{2,2} & \cdots & a_{2,n}\\
        \vdots & \vdots & \ddots & \vdots\\
        b_n & a_{n,2} & \cdots & a_{n,n}\\
    \end{bmatrix}, \ \pmb{B}_2 = 
    \begin{bmatrix}
        a_{1,1} & b_1 & \cdots & a_{1,n}\\
        a_{2,1} & b_2 & \cdots & a_{2,n}\\
        \vdots & \vdots & \ddots & \vdots\\
        a_{n,1} & b_n & \cdots & a_{n,n}\\
    \end{bmatrix}, \ \cdots \ \pmb{B}_n = 
    \begin{bmatrix}
        a_{1,1} & a_{1,2} & \cdots & b_1\\
        a_{2,1} & a_{2,2} & \cdots & b_2\\
        \vdots & \vdots & \ddots & \vdots\\
        a_{n,1} & a_{n,2} & \cdots & b_n\\
    \end{bmatrix}
$$

### 5.3.3 行列式与几何
在阶数较低的时候，方阵行列式与欧式几何之间存在紧密联系（高维的时候相关规律也许依然存在，只是没那么好理解和运用了）。以二阶（平面）为例：

![三角形面积示意图](/figures/triangle.png)

如果不给定坐标系，我们求三角形面积的常用办法是“底乘高除以二”。如今我们在坐标系中，手拥线代这一把利器，一切都变得不一样了。如图所示的三角形，其中一个顶点在原点上，剩余两点的坐标分别为 $(a,b)$、$(c,d)$。我们先按照纯几何的角度分析一下问题，在完善辅助线后，不难发现目标区域面积 $S$ 为矩形减去周边三个三角形的面积：
$$
    S_{\triangle OAB} = bc - \dfrac{1}{2}ab - \dfrac{1}{2}cd - \dfrac{1}{2}(b-d)(c-a) = \dfrac{bd-ac}{2}
$$ 
这个形式与二阶矩阵的行列式非常相似：
$$
    \pmb{A} = 
    \begin{vmatrix}
        a & b\\ c & d\\
    \end{vmatrix}, \ S_{\triangle OAB} = \dfrac{1}{2} \left|\det{(\pmb{A})}\right|
$$
之所以要取绝对值，是因为行列式本身并不像面积一样满足非负性。数值的正负与图形的方向（或立体的手型）有关。

由这个案例可以看出，线性代数和向量对于几何分析带来的巨大帮助。当三角形顶点不在原点时，以向量的角度，无非是 $\vec{OA}$、$\vec{OB}$ 的起点发生了平移，三边之间的位置关系是不变的，因此只需将对应坐标修正至以原点为顶点的情况即可继续套用公式。例如三点坐标分别为 $A(x_1,y_1)$、$B(x_2,y_2)$ 以及 $C(x_3,y_3)$ 时，将 $A$ 平移至原点：
$$
    \begin{align}
        \notag
        \begin{cases}
            A(x_1,y_1) \rightarrow A^{'}(0,0)\\
            B(x_2,y_2) \rightarrow B^{'}(x_2-x_1,y_2-y_1)\\
            C(x_3,y_3) \rightarrow C^{'}(x_3-x_1,y_3-y_1)\\
        \end{cases}, \ S_{\triangle ABC} &= \dfrac{1}{2}
        abs\left(
            \begin{vmatrix}
                x_2-x_1 & y_2-y_1\\
                x_3-x_1 & y_3-y_1\\
            \end{vmatrix}
        \right)\\
        \notag &= \left|x_1(y_1-y_3) + x_2(y_3-y_1) + x_3(y_1-y_2)\right|
    \end{align}
    \tag{5-3-4}
$$
或者我们也可以用另外一个公式：
$$
    \pmb{S} = 
    \begin{bmatrix}
        x_1 & y_1 & 1\\
        x_2 & y_2 & 1\\
        x_3 & y_3 & 1\\
    \end{bmatrix}, \ S_{\triangle ABC}=\dfrac{1}{2}\left|\det(\pmb{S})\right|
    \tag{5-3-5}
$$
把（5-3-5）的 $\det(\pmb{S})$ 按第一列展开，结果与（5-3-4）是一样的。类似的结论可以沿用至平行四边形。因为在二维坐标系中决定一个平行四边形其实仅需要三个点即可，相应区域的面积即为三角形面积的两倍，故不再赘述。

除了二维空间中的面积，行列式（绝对值）还能表示空间平行多面体的体积。以平行六面体为例，两个 $\mathbb{R}^3$ 中的向量 $\pmb{x}_1$、$\pmb{x}_2$ 可确定空间中的一个平行四边形（平移至同一起点），加上第三个不与该平面平行的向量 $\pmb{x}_3$ 可确定一个平行六面体。三个向量（从图论的角度即为边）组成的矩阵为：
$$
    \pmb{A} = 
    \begin{bmatrix}
        x_1 & y_1 & z_1\\
        x_2 & y_2 & z_2\\
        x_3 & y_3 & z_3\\
    \end{bmatrix}
$$
$|\det(\pmb{A})|$ 即为该立体的体积。我们从行列式的几条基本性质印证这个结论：

（1）$\det(\pmb{I})=1$ $\rightarrow$ 单位正方体的体积为 $1$；

（2）行交换不改变行列式结果的绝对值 $\rightarrow$ 六面体同一顶点上的三条边没有变化，体积不变；

（3a）以常数倍乘矩阵某一行，行列式值也乘上相同倍数 $\rightarrow$ 倍增平行六面体的某一行，其体积也相应倍增；

（3b）矩阵单行对应的线性组合同样存在于行列式中 $\rightarrow$ 从二维的角度，将某一边的终点平移至另一点，两边构成的平行四边形将产生“拉伸”（假设 $A$ 在原点）：

![3b性质几何表现示意图](/figures/3b_in_volume.png)

$$
    \underbrace{
        \begin{vmatrix}
            x_0 & y_0 & z_0\\
            x_1+x^{'} & y_1+y^{'} & z_1+z^{'}\\
        \end{vmatrix}}_{S_{ABC^{'}D^{'}}} = 
    \underbrace{
        \begin{vmatrix}
            x_0 & y_0 & z_0\\
            x_1 & y_1 & z_1\\
        \end{vmatrix}}_{S_{ABCD}} + 
    \underbrace{
        \begin{vmatrix}
            x_0 & y_0 & z_0\\
            x^{'} & y^{'} & z^{'}\\
        \end{vmatrix}}_{S_{ABC^{''}D^{''}}=S_{CDD^{'}C^{'}}}
$$
二维平面中这样的等式关系成立，三维立体空间中自然也成立。

### 5.3.4 叉积
三维方阵的行列式还存在一种特殊应用，即叉积（*Cross product*）。对于两个三维向量 $\pmb{u}$、$\pmb{v}$，叉积 $\pmb{u} \times \pmb{v}$ 的数学公式为：
$$
    \begin{align}
        \notag
        \pmb{u} \times \pmb{v} = 
        \begin{vmatrix}
            \pmb{i} & \pmb{j} & \pmb{k}\\
            u_1 & u_2 & u_3\\
            v_1 & v_2 & v_3\\
        \end{vmatrix} &= 
        \begin{vmatrix}
            u_2 & u_3\\
            v_2 & v_3\\
        \end{vmatrix} \pmb{i} - 
        \begin{vmatrix}
            u_1 & u_3\\
            v_1 & v_3\\
        \end{vmatrix} \pmb{j} + 
        \begin{vmatrix}
            u_1 & u_2\\
            v_1 & v_2\\
        \end{vmatrix} \pmb{k}\
        \notag \ \\
        \notag &= (u_2v_3-u_3v_2)\pmb{i} + (u_3v_1-u_1v_3)\pmb{j} + (u_1v_2-u_2v_1)\pmb{k}
    \end{align}
    \tag{5-3-6}
$$
显然叉积的结果仍然是三维向量。上述写法虽然略有不规范（后两行是数字元素，而第一行是向量元素），但是对于记忆大有裨益。接下来展示叉积的一些性质：

（1）转换叉积的顺序意味着行列式进行了一次行交换，因此最终结果在数值上是相反的。即 $\pmb{u} \times \pmb{v} = -(\pmb{v} \times \pmb{u})$；

（2）**叉积与 $\pmb{u}$、$\pmb{v}$ 都是正交的**。这一点可以通过向量内积（点乘）判断：
$$
    \pmb{u} \cdot (\pmb{u} \times \pmb{v}) = u_1(u_2v_3-u_3v_2) + u_2(u_3v_1-u_1v_3) + u_3(u_1v_2-u_2v_1) = 
    \begin{vmatrix}
        u_1 & u_2 & u_3\\
        u_1 & u_2 & u_3\\
        v_1 & v_2 & v_3\\
    \end{vmatrix} = 0
$$

（3）**向量与自身的叉积是零向量**，即 $\pmb{u} \times \pmb{u} = \pmb{0}$。由（2）可知，叉积必须与两向量始终保持正交，而此时参与叉积运算的向量实质上只有一个，与之正交的方向有无数个，因此只有零向量能够始终满足正交条件；

（4）叉积与点乘具有数学形式上的联系。对于 $\pmb{u}$、$\pmb{v}$，当二者平行时，$\pmb{u} \times \pmb{v} = \pmb{0}$；当二者垂直（正交）时，$\pmb{u} \cdot \pmb{v} = 0$。此外，叉积与点乘的数值之间存在三角函数的转换关系（$\theta$ 为向量夹角）：
$$
    \|\pmb{u} \times \pmb{v}\| = \|\pmb{u}\| \|\pmb{v}\| \left|\sin \theta \right|, \ 
    \left|\pmb{u} \cdot \pmb{v}\right| = \|\pmb{u}\| \|\pmb{v}\| \left|\cos \theta \right|
    \tag{5-3-7}
$$
换句话说，叉积的结果是一个长度为 $\|\pmb{u}\| \|\pmb{v}\| \left|\sin \theta \right|$ 的向量，方向与 $\pmb{u}$、$\pmb{v}$ 垂直；点乘的结果是一个标量，其大小为 $\|\pmb{u}\| \|\pmb{v}\| \left|\cos \theta \right|$（绝对值）；

（5）**叉积向量（$\pmb{u} \times \pmb{v}$）的长度等于以 $\pmb{u}$ 和 $\pmb{v}$ 为边的平行四边形的面积**。想证明这一条性质略有些复杂，我们可以先考虑一个平行六面体，其底面为以 $\pmb{u}$ 和 $\pmb{v}$ 为边的平行四边形 $\text{▱}uv$ ，另一条边向量 $\pmb{h}$ 垂直该底面。则该体块的体积 $V=h \times S_{\text{▱}uv}$。不难发现 $\pmb{u} \times \pmb{v}$ 就是一种 $\pmb{h}$，根据 5.3.3 的结论，共顶点三向量矩阵的行列式绝对值为相应平行六面体的体积：
$$
    \begin{align}
        \notag
        V = \left|\det(\pmb{A}) \right| &= abs \left(
        \begin{vmatrix}
            u_1 & u_2 & u_3\\
            v_1 & v_2 & v_3\\
            u_2v_3-u_3v_2 & u_3v_1-u_1v_3 & u_1v_2-u_2v_1\\
        \end{vmatrix} \right)\\
        \notag \ \\
        \notag \det(\pmb{A}) &= \left(u_1^2v_2^2 - u_1u_2v_1v_2 - u_1u_3v_1v_3 + u_1^2v_3^2 \right)\\
        \notag \ \\
        \notag &- \left(u_1u_2v_1v_2 + u_2^2v_1^2 + u_2^2v_3^2 - u_2u_3v_2v_3 \right)\\
        \notag \ \\
        \notag &+ \left(u_3^2v_1^2 - u_1u_3v_1v_3 + u_2u_3v_2v_3 + u_3^2v_2^2 \right)
    \end{align}
$$
由于 $\pmb{h}$ 既是基本边，又是高，所以平行四边形的面积等于体积除以高。在数值上，$\dfrac{V}{\|\pmb{h}\|_2}$ 与 $S_{\text{▱}uv}$ 是相等的，即有：
$$
    \dfrac{\left|\det(\pmb{A}) \right|}{\|\pmb{u} \times \pmb{v}\|_2} = S_{\text{▱}uv}
$$
字母过多我们就不展开写了，通过分析多项式组成不难发现 $\dfrac{\left|\det(\pmb{A}) \right|}{\|\pmb{u} \times \pmb{v}\|_2}$ 与 $\|\pmb{u} \times \pmb{v}\|_2$ 相等，即结论（5）成立。

### 5.3.5 标量三重积
标量三重积（*Scalar triple product*），又叫混合积，其主要表现形式为 $(\pmb{u} \times \pmb{v}) \cdot \pmb{w}$，结果为标量。其计算公式可表示为行列式：
$$
    (\pmb{u} \times \pmb{v}) \cdot \pmb{w} = 
    \begin{vmatrix}
        w_1 & w_2 & w_3\\
        u_1 & u_2 & u_3\\
        v_1 & v_2 & v_3\\
    \end{vmatrix} = 
    \begin{vmatrix}
        u_1 & u_2 & u_3\\
        v_1 & v_2 & v_3\\
        w_1 & w_2 & w_3\\
    \end{vmatrix}
    \tag{5-3-8}
$$
这里仅需关注一种特殊情况，即 $(\pmb{u} \times \pmb{v}) \cdot \pmb{w} = 0$，当且仅当 $\pmb{u}$、$\pmb{v}$ 和 $\pmb{w}$ 位于同一平面时该式成立，其理由有三条：

（1-a）若 $\pmb{u}$ 与 $\pmb{v}$ 共线，则 $\pmb{u} \times \pmb{v} = \pmb{0}$，不论 $\pmb{w}$ 为何，混合积均为 0，且三向量必位于同一平面；

（1-b）若 $\pmb{u}$ 与 $\pmb{v}$ 不共线，则 $\pmb{u} \times \pmb{v}$ 垂直于二者构成的平面，当 $\pmb{w} \ne \pmb{0}$，则有 $\pmb{w} \perp (\pmb{u} \times \pmb{v})$，$\pmb{w}$ 与 $\pmb{u}$、$\pmb{v}$ 共平面；当 $\pmb{w}=\pmb{0}$，混合积满足为 0，此时 $\pmb{w}$ 可视为任意方向，因此也与 $\pmb{u}$、$\pmb{v}$ 共平面；

（2）行列式为 0 意味着原矩阵是奇异的，即三向量存在线性关系，当且仅当三者位于同一平面时才满足非独立条件；

（3）行列式可视为平行六面体的体积，该体块体积为 0，意味着共顶点的三边在同一平面上（没有高度信息），即三向量共平面。

由上述理由（3）可知，标量三重积的一个重要应用就是计算三维空间中平行六面体的体积，该平行六面体可由共顶点的三条边向量（$\pmb{u}$、$\pmb{v}$、$\pmb{w}$）决定。