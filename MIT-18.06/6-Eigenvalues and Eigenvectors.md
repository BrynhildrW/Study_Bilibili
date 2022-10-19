---
html:
    toc: true
print_background: true
---

# 6. Eigenvalues and Eigenvectors
终于来到了线代部分最关键的环节，特征值与特征向量。不夸张地说，特征向量在 EEG 信号分类算法设计中占据了半壁江山，在空间滤波器设计中更是尤为重要。

## 6.1 Introduction to Eigenvalues
### 6.1.1 特征值与特征向量的意义
对于给定矩阵 $\pmb{A}$ 与未知列向量 $\pmb{x}$ 的乘法，我们既可以将其视为一种投影（$\pmb{A}$ 为实对称矩阵），抑或是一种函数 $f(\pmb{x})$。总之，向量 $\pmb{Ax}$ 的结果往往是多种多样的，其中存在一部分特殊的 $\pmb{x}$，使得 $\pmb{Ax}$ 作用前后的方向平行，即：
$$
    \pmb{Ax} = \lambda \pmb{x}
    \tag{6-1-1}
$$


满足该方程的向量 $x$ 即为 $\pmb{A}$ 的特征向量，$\lambda$ 为特征值。值得注意的是，“方向平行”并不限定方向完全一致，即有可能出现方向相反的情况。为何我们需要关注 $\pmb{Ax} = \lambda\pmb{x}$ 这种特殊情况呢，此处我们通过一个 *Markov* 矩阵幂乘运算的示例来进行说明：
$$
    \pmb{A} = 
    \begin{bmatrix}
        0.8 & 0.3\\ 0.2 & 0.7\\
    \end{bmatrix}, \ 
    \pmb{A}^2 = 
    \begin{bmatrix}
        0.70 & 0.30\\ 0.45 & 0.55\\
    \end{bmatrix}, \ \cdots\ 
    \pmb{A}^{100} \approx 
    \begin{bmatrix}
        0.6 & 0.6\\ 0.4 & 0.4\\
    \end{bmatrix}
$$
根据 $\pmb{A}$ 求 $\pmb{A}^{100}$ 这种题目有点考试出题的意思了。显然我们不需要真的进行 100 遍矩阵乘法才能知道答案，根据 $\pmb{A}$ 的特征值与特征向量可以很方便地获取答案。这里我们直接给出相应数据：
$$
    \begin{align}
        \notag \pmb{x}_1 = 
        \begin{bmatrix}
            0.6\\ 0.4\\
        \end{bmatrix}, \ \pmb{Ax}_1 = \pmb{x}_1 \ &\Longrightarrow \ \lambda_1 = 1\\
        \notag \pmb{x}_2 = 
        \begin{bmatrix}
            1\\ -1\\
        \end{bmatrix}, \ \pmb{Ax}_2 = \dfrac{1}{2}\pmb{x}_2 \ &\Longrightarrow \ \lambda_2 = \dfrac{1}{2}\\
    \end{align}
$$
根据特征值不难发现，$\pmb{A}^n\pmb{x}_1 = 1^n\pmb{x}_1$、$\pmb{A}^n\pmb{x}_2 = \dfrac{1}{2^n}\pmb{x}_2$。当矩阵 $\pmb{A}$ 进行幂乘运算时，**特征向量保持不变，特征值进行相应的幂乘操作**。换句话说，特征向量的方向始终不变，尺度随运算次数的增加逐步衰减（除了特征值为 1 的主特征向量）。事实上，**$\pmb{A}$ 的每个列向量均可由特征向量的线性组合表示**：
$$
    \pmb{A}(:,i) = \alpha_i\pmb{x}_1 + \beta_i\pmb{x}_2 \ \Longrightarrow \ 
    \begin{bmatrix}
        0.8\\ 0.2\\
    \end{bmatrix} = \pmb{x}_1 + 0.2\pmb{x}_2, \ 
    \begin{bmatrix}
        0.7\\ 0.3\\
    \end{bmatrix} = \pmb{x}_1 + 0.1\pmb{x}_2\\
$$
换句话说，特征向量可以作为原矩阵对应空间的一组基。矩阵幂乘对于向量的影响如下所示：
$$
    \begin{align}
        \notag \pmb{A}^2 &= 
        \begin{bmatrix}
            \pmb{A}\pmb{A}(:,0) & \pmb{A}\pmb{A}(:,1)\\
        \end{bmatrix}, \ 
        \begin{cases}
            \pmb{A}\pmb{A}(:,0) = \alpha_0\lambda_1\pmb{x}_1 + \beta_0\lambda_2\pmb{x}_2\\
            \pmb{A}\pmb{A}(:,1) = \alpha_1\lambda_1\pmb{x}_1 + \beta_1\lambda_2\pmb{x}_2\\
        \end{cases}\\
        \notag \vdots\\
        \notag \pmb{A}^n &= 
        \begin{bmatrix}
            \pmb{A}^{n-1}\pmb{A}(:,0) & \pmb{A}^{n-1}\pmb{A}(:,1)\\
        \end{bmatrix}, \ 
        \begin{cases}
            \pmb{A}^{n-1}\pmb{A}(:,0) = \alpha_0\lambda_1^{n-1}\pmb{x}_1 + \beta_0\lambda_2^{n-1}\pmb{x}_2\\
            \pmb{A}^{n-1}\pmb{A}(:,1) = \alpha_1\lambda_1^{n-1}\pmb{x}_1 + \beta_1\lambda_2^{n-1}\pmb{x}_2\\
        \end{cases}
    \end{align}
$$
根据上述公式，我们绕开了**矩阵乘法**，直奔主题——**向量数乘与加法**。后者在运算上的优势不言而喻。最终 $\pmb{A}^n$ 的诸多特征向量中，仅有主特征向量能够保持原尺度，其余特征向量均衰减至可以忽略不计。所以当 $n$ 大到一定程度后，$\pmb{A}^n$ 的每个列向量均接近主特征向量。

以上是特征分解的一个简单应用。尽管它在数学上拥有一些非常奇妙的性质，实际场景下却并不常见（至少在我研究的领域）。在 EEG 信号分析中，更常见的应用是对某投影矩阵（协方差矩阵） $\pmb{P}$ 进行特征分解。那么此时应当如何理解特征值、特征向量以及某些特殊情况下二者的取值？

以 $\mathbb{R}^3$ 中的平面 $\pmb{P}$（$rank=2$）为例，当且仅当非零向量 $\pmb{x}$ 位于平面 $\pmb{P}$ 上时，投影向量 $\pmb{Px}$ 才会与 $\pmb{x}$ 平行（投影没有实际作用），即：
$$
    \pmb{Px}_1 = \pmb{x}_1, \ \Rightarrow \ \lambda = 1
$$
当 $\pmb{x} \perp \pmb{P}$ 时，投影结果为一个没有长度的点，即：
$$
    \pmb{Px}_2 = \pmb{0}, \ \Rightarrow \ \lambda = 0
$$
换句话说，当我们求解投影矩阵 $\pmb{P}$（*Hermitte* 矩阵） 的特征向量时，本质上是在寻找某个位于 $\pmb{C}(\pmb{A})$ 的 $\pmb{x}$，每一个 $\pmb{x}$ 都对应于一个特征值 $\lambda$。特征值的大小（归一化后）在一定程度上表示着该投影向量与投影空间的“契合”程度。当 $\pmb{x}$ 就位于投影空间内时，特征值最大；反之当 $\pmb{x}$ 与投影空间毫无关系（即正交）时，特征值最小。

### 6.1.2 特征值问题
形如（6-1-1）的方程称为特征值方程（问题），显然这样的方程已经无法使用简单的消元法求解了，需要使用一点小技巧将其转化为线性方程组：
$$
    \pmb{Ax} = \lambda\pmb{x} \ \Longrightarrow \ (\pmb{A}-\lambda\pmb{I})\pmb{x} = \pmb{0}, \ \pmb{A} \in \mathbb{R}^{n \times n}
    \tag{6-1-2}
$$
根据第三章的相关知识，我们知道（6-1-2）存在非零解的前提是 $\pmb{A}-\lambda\pmb{I}$ 为奇异矩阵，即 $\det\left(\pmb{A}-\lambda\pmb{I}\right)=0$。这样就将一个双边未知向量方程转化为了一个多项式方程。不出意外地，我们能解出 $n$ 个特征值（包括重根、虚根）。将其代入（6-1-2）的线性方程组，可以求解出 $n$ 个彼此正交的特征向量。同时与幂乘操作类似，$\pmb{A}-\lambda\pmb{I}$ 并没有改变 $\pmb{A}$ 的特征向量，方阵特征值减少了 $\lambda$。

值得注意的是，并非 $\pmb{A}$ 加上任意同维度方阵都有上述结论，即以下所示的推导往往是**不成立**的：
$$
    \begin{cases}
        \pmb{Ax} = \lambda\pmb{x}\\
        \pmb{Bx} = \theta\pmb{x}\\
    \end{cases} \ \nRightarrow \ (\pmb{A}+\pmb{B})\pmb{x} = (\lambda+\theta)\pmb{x}
$$
这主要是因为不同矩阵的特征向量一般不同，所以实际情况是这样的：
$$
    \pmb{Ax}_1 + \pmb{Bx}_2 = \lambda\pmb{x}_1 + \theta\pmb{x}_2 = \ ?
$$
啥也不是。特征分解不满足线性关系。由 6.1.1 的示例可知，通常特征值都是各不相同的实数，它们表示原矩阵对应的线性变换**在相应特征向量方向上的拉伸/压缩**。接下来看看一些特殊情况：

首先是**复数特征值**，以旋转矩阵 $\pmb{Q}$ 为例（旋转 90°）：
$$
    \pmb{Q} = 
    \begin{bmatrix}
        0 & -1\\ 1 & 0\\
    \end{bmatrix}
$$
从几何意义上看，旋转矩阵的作用就是将原向量旋转相应的角度，因此在实数范围内不存在非零向量经过非特殊角度旋转后还能与原向量平行的情况。事实上：
$$
    \det\left(\pmb{Q}-\lambda\pmb{I}\right) = \lambda^2+1 = 0 \ \Longrightarrow \
    \begin{cases}
        \lambda_1 = i\\ \lambda_2 = -i 
    \end{cases}
$$
当实数方阵 $\pmb{Q}$ 非对称时，特征值可能出现复数。关于复数特征值，我们有一段形式不太简略、但是内容非常直观的解释：复数特征值代表**旋转与放缩的结合变换**，且有：

**（1）特征值的模代表长度放缩的倍数**；

**（2）幅角代表旋转角度**；

**（3）复特征向量组的实部向量与虚部向量代表实现上述变换的一组基**。

以二维空间为例，复特征向量必然是共轭出现的，因此假设 $\pmb{A} \in \mathbb{R}^{2 \times 2}$ 有复数特征值以及对应的特征向量：
$$
    \lambda_1,\lambda_2=a\pm bi = r(\cos\theta \pm i\sin\theta), \
    \begin{cases}
        r=\sqrt{a^2+b^2}\\
        a = r\cos\theta\\
        b = r\sin\theta\\
    \end{cases}\\
    \ \\
    \pmb{x}_1,\pmb{x}_2 = 
    \begin{bmatrix}
        \alpha_1\\ \alpha_2\\
    \end{bmatrix} \pm i
    \begin{bmatrix}
        \beta_1\\ \beta_2\\
    \end{bmatrix}
$$
若我们选定特征向量组的实部与虚部分别构成一个基向量，这组基形成的线性变换等价于：
$$
    \pmb{B} = r
    \begin{bmatrix}
        \cos\theta & -\sin\theta\\
        \sin\theta & \cos\theta\\
    \end{bmatrix}
$$
在特征向量的实部与虚部构成的基底下，这个线性变换对坐标轴的作用是逆时针旋转 $\theta$ 角度、同时拉伸 $r$ 倍。对于高维空间，可能会出现多对共轭复特征值以及数个实特征值的组合情况，此时特征向量基构成的线性变换可视为“不同维度的旋转、放缩” + “不同维度的放缩”。

除了复数特征值，更严重的异常情况是**重根特征值**。以如下矩阵为例：
$$
    \pmb{A} = 
    \begin{bmatrix}
        3 & 1\\ 0 & 3\\
    \end{bmatrix}, \ \lambda_1=\lambda_2=3
$$
$\pmb{A}$ 是一个对角阵，不难发现特征值存在以下两条规律：

（1）**$tr(\pmb{A})$ 是主对角线元素之和，也是特征值之和**；

（2）**特征值的乘积是方阵的行列式 $\det(\pmb{A})$**。

上述结论不仅在对角阵中存在，在普通方阵中也一样存在。言归正传，重根特征值会导致特征向量的缺失：
$$
    (\pmb{A}-\lambda\pmb{I})\pmb{x} = \pmb{0} \ \Longrightarrow \ 
    \begin{bmatrix}
        0 & 1\\ 0 & 0\\
    \end{bmatrix}
    \begin{bmatrix}
        x_1\\ x_2\\
    \end{bmatrix} = \pmb{0}
$$
$\pmb{A}-\lambda\pmb{I}$ 是一个秩 1 矩阵，即该线性方程组的有效方程只有 1 个，只能解出一个特征向量。这一点从直观角度也很好理解，相同特征值不会对应不同的、独立的特征向量。