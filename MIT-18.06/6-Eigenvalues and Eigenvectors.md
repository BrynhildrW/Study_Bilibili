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

## 6.2 Diagonalizing a Matrix
### 6.2.1 方阵对角化
方阵对角化是目前为止我们接触的第三种矩阵分解方法（前两种分别为：基于初等变换的 LU 分解、基于 Gram-Schmidt 正交化的 QR 分解），对角化的重要理论依据就是特征方程：$\pmb{Ax}=\lambda\pmb{x}$，它能将矩阵乘法转化为简单的向量数乘。

取 $\pmb{A} \in \mathbb{R}^{n \times n}$ 的 $n$ 个特征向量（**假设互异、独立**），按照对应特征值大小排序组合成特征矩阵 $\pmb{S}$：
$$
    \pmb{S} = 
    \begin{bmatrix}
        \pmb{x}_1 & \pmb{x}_2 & \cdots & \pmb{x}_n
    \end{bmatrix} \in \mathbb{R}^{n \times n}
    \tag{6-2-1}
$$
观察矩阵乘法 $\pmb{AS}$，不难发现：
$$
    \begin{align}
        \notag \pmb{AS} &= 
        \begin{bmatrix}
            \pmb{Ax}_1 & \pmb{Ax}_2 & \cdots & \pmb{Ax}_n
        \end{bmatrix} \xrightarrow{\pmb{Ax}=\lambda\pmb{x}}
        \begin{bmatrix}
            \lambda_1\pmb{x}_1 & \lambda_2\pmb{x}_2 & \cdots & \lambda_n\pmb{x}_n
        \end{bmatrix}\\
        \notag\ \\
        \notag &= 
        \begin{bmatrix}
            \pmb{x}_1 & \pmb{x}_2 & \cdots & \pmb{x}_n
        \end{bmatrix}
        \begin{bmatrix}
            \lambda_1 & 0 & \cdots & 0\\
            0 & \lambda_2 & \cdots & 0\\
            \vdots & \vdots & \ddots & \vdots\\
            0 & 0 & \cdots & \lambda_n\\
        \end{bmatrix} = \pmb{S\Lambda}
    \end{align}
    \tag{6-2-2}
$$
由此得到方阵对角化表达式的两种形式：
$$
    \pmb{AS} = \pmb{S\Lambda} \ \Longrightarrow \ \pmb{S}^{-1}\pmb{AS} = \pmb{\Lambda} \ or \ \pmb{A} = \pmb{S\Lambda}\pmb{S}^{-1}
    \tag{6-2-3}
$$
需要注意的是，这种分解方法需要 $\pmb{S}$ 可逆，即**方阵 $\pmb{A}$ 拥有 $n$ 个互相独立的特征向量**，即 **$\pmb{A}$ 的特征值互不相等**，即 **$\pmb{A}$ 满秩**。在此基础上，我们可以从两个角度审视 6.1.1 中设想的问题，即 $\pmb{A}^n$ 的特征值与特征向量如何变化：
$$
    \pmb{Ax} = \lambda\pmb{x} \Longrightarrow \ \pmb{A}^n\pmb{x} = \lambda\pmb{A}^{n-1}\pmb{x} = \lambda^2\pmb{A}^{n-2}\pmb{x} = \cdots = \lambda^n\pmb{x}
    \tag{6-2-4}
$$
$$
    \pmb{A}^n = \left(\pmb{S\Lambda}\pmb{S}^{-1}\right) \left(\pmb{S\Lambda}\pmb{S}^{-1}\right) \cdots \left(\pmb{S\Lambda}\pmb{S}^{-1}\right) = \pmb{S} \pmb{\Lambda}^n \pmb{S}^{-1}
    \tag{6-2-5}
$$
即 $\pmb{A}^n$ 的特征值是 $\pmb{A}$ 的 $n$ 次幂，特征向量（矩阵）不变。显然当 $|\lambda_{max}|<1$ 时，随着指数 $n$ 的增加，$\pmb{\Lambda}^n \rightarrow \pmb{0}$，导致 $\pmb{A}^n \rightarrow \pmb{0}$。

在这一部分，有两（大）点值得花点时间额外注意：

（1）尽管特征值和特征向量是绑定的概念，但是它们分别决定了矩阵的不同性质。特征值决定了矩阵是否可逆（具有 $0$ 特征值的矩阵满足 $\pmb{Ax}=\pmb{0}$ 有非零解，因此不可逆），特征向量决定了矩阵是否可以对角化（特征向量缺失时矩阵不可对角化）。互异特征值能够保证特征向量互相独立且足量，**但是特征值重根未必就意味着缺失特征向量**，单位阵 $\pmb{I}_N$ 就是一个典型示例。$\pmb{I}_N$ 的特征值全为 $1$，而它的特征向量依旧是完整独立的。而对于更一般的对角阵 $\pmb{D}$，显然不再需要什么画蛇添足的操作，它已经对角化了，即 $\pmb{D} = \pmb{S\Lambda}\pmb{S}^{-1} = \pmb{\Lambda}$。

（2）关于 $\pmb{Ax}=\lambda\pmb{x}$ 的过程我们有了充分的了解，自然地我们期待着在 $\pmb{AB}$ 以及 $\pmb{A} + \pmb{B}$ 中找到一些熟悉的结论。可惜它们是不存在的。即：
$$
    \begin{cases}
        \pmb{Ax}=\lambda\pmb{x}\\
        \pmb{Bx}=\theta\pmb{x}\\
    \end{cases} \ \nRightarrow \
    \begin{cases}
        (\pmb{A}+\pmb{B})\pmb{x}=(\lambda+\theta)\pmb{x}\\
        \pmb{ABx} = \lambda\theta\pmb{x}
    \end{cases}
$$
这是因为一般情况下，$\pmb{A}$ 和 $\pmb{B}$ 的特征向量是不同的。因此乘法分配律失效，也就没有后续的各种“性质”。不过存在一种特殊情况，$\pmb{A}$ 和 $\pmb{B}$ 共享同一套特征向量组，即可交换矩阵（*Commuting matrix*）：$\pmb{AB}=\pmb{BA}$，此时上述性质成立。

### 6.2.2 特征值方程与差分系统
对于一个一阶差分系统，我们希望将其写成矩阵乘法的形式：
$$
    \pmb{u}_{k+1} = \pmb{A}\pmb{u}_k
    \tag{6-2-6}
$$
其中 $\pmb{A} \in \mathbb{R}^{n \times n}$ 是一个方阵，具有 $n$ 个独立特征向量。由此，在给定系统初值 $\pmb{u}_0$ 的情况下，我们可以精确求解或准确估计系统在每一个状态下的值：
$$
    \begin{align}
        \notag \pmb{u}_1 &= \pmb{Au}_0\\
        \notag \pmb{u}_2 &= \pmb{Au}_1 = \pmb{A}^2\pmb{u}_0\\
        \notag \pmb{u}_3 &= \pmb{Au}_2 = \pmb{A}^2\pmb{u}_1 = \pmb{A}^3\pmb{u}_0\\
        \notag \vdots \\
        \notag \pmb{u}_k &= \pmb{Au}_{k-1} = \cdots = \pmb{A}^k\pmb{u}_0
    \end{align}
$$
由前述知识可知，$\pmb{A}$ 的特征向量可以构成一组基。为了利用 $\pmb{A}$ 的特征信息，我们需要将初值 $\pmb{u}_0$ 分解为 $\pmb{A}$ 的特征向量的线性组合：
$$
    \begin{align}
        \notag \pmb{S} &= 
        \begin{bmatrix}
            \pmb{x}_1 & \pmb{x}_2 & \cdots & \pmb{x}_n
        \end{bmatrix} \in \mathbb{R}^{n \times n}, \ \pmb{c} = 
        \begin{bmatrix}
            c_1\\ c_2\\ \vdots\\ c_n
        \end{bmatrix} \in \mathbb{R}^{n \times 1}\\
        \notag \ \\
        \notag \pmb{u}_0 &= c_1\pmb{x}_1 + c_2\pmb{x}_2 + \cdots + c_n\pmb{x}_n = \pmb{Sc}
    \end{align}
    \tag{6-2-7}
$$
在此基础上，对差分系统的各阶段输出进行改写：
$$
    \pmb{A}^k\pmb{u}_0 = c_1\lambda_1^k\pmb{x}_1 + c_2\lambda_2^k\pmb{x}_2 + \cdots + c_n\lambda_n^k\pmb{x}_n = \pmb{S}\pmb{\Lambda}^k\pmb{c}
    \tag{6-2-8}
$$
根据特征值数值大小，可以预估差分系统在未来时刻的输出，**系统输出的增（减）速主要由最大特征值决定**。例如当仅有 $|\lambda_1|>1$ 时，随着 $k \rightarrow \infty$，系统输出主要为第一分量 $c_1\lambda_1^k\pmb{x}_1$，其余分量可忽略不计。在现实问题中，$\pmb{A}$ 及其各特征向量可能具有对应的物理意义，因此（6-2-8）在某种程度上还能帮助我们总结一些问题的物理规律。

### 6.2.3 二阶差分系统的转化
接下来我们以一个经典二阶差分系统为例，说明如何将其转化为矩阵（向量）形式的一阶差分系统，以及如何应用上述方法求解系统状态。*Fibonacci* 数列是大家熟知的经典数学问题，其数学表述形式为：
$$
    \begin{cases}
        F_{k+2} = F_{k+1} + F_{k}\\
        F_0 = 0, \ F_1 = 1\\
    \end{cases}
    \tag{6-2-9}
$$
这里需要用到一个小 trick，我们为（6-2-9）添加一个方程 $F_{k+1}=F_{k+1}$，并令状态值为向量 $\pmb{u}_k = \left[F_{k+1},F_k\right]^T$，从而获得一阶差分方程 $\pmb{u}_{k+1} = \pmb{A}\pmb{u}_k$：
$$
    \underbrace{
        \begin{bmatrix}
            F_{k+2}\\ \ \\F_{k+1}
        \end{bmatrix}}_{\pmb{u}_{k+1}} = 
    \underbrace{
        \begin{bmatrix}
            1 & 1\\
            \ \\
            1 & 0\\
        \end{bmatrix}}_{\pmb{A}} \ 
    \underbrace{
        \begin{bmatrix}
            F_{k+1}\\ \ \\F_k
        \end{bmatrix}}_{\pmb{u}_k}
$$
对 $\pmb{A}$ 进行特征值分解，不难得出：
$$
    \lambda_1, \lambda_2 = \dfrac{1 \pm \sqrt{5}}{2}, \  \pmb{x}_1,\pmb{x}_2 = 
    \begin{bmatrix}
        \dfrac{1 \pm \sqrt{5}}{2}\\ \ \\ 1\\
    \end{bmatrix}
$$
第一特征值 $\lambda_1 \approx 1.618$，第二特征值 $|\lambda_2|<1$，由此可知 *Fibonaccci* 数列的增速约为 $1.618$：
$$
    \begin{align}
        \notag \dfrac{F(5)}{F(4)} &= \dfrac{3}{2} = 1.5, \ \dfrac{F(6)}{F(5)} = \dfrac{5}{3} \approx 1.667, \ \dfrac{F(7)}{F(6)} = \dfrac{8}{5} = 1.6, \\
        \notag \ \\
        \notag \dfrac{F(8)}{F(7)} &= \dfrac{13}{8} = 1.625, \ \dfrac{F(9)}{F(8)} = \dfrac{21}{13} \approx 1.615,\ \dfrac{F(10)}{F(9)} = \dfrac{34}{21} \approx 1.619 \ \cdots
    \end{align}
$$
若将数列比值作为一个新系统来观察，可发现其具有阻尼性质，即随着时间推进，系统状态值在 $1.618$ 两侧交替演进并逐渐稳定。将 $\pmb{x}_1$、$\pmb{x}_2$ 代入初值 $\pmb{u}_0$ 求解系数向量 $\pmb{c}$：
$$
    \begin{cases}
        \dfrac{1+\sqrt{5}}{2}c_1 + \dfrac{1-\sqrt{5}}{2}c_2 = 1\\
        \ \\
        c_1 + c_2 = 0
    \end{cases} \ \Longrightarrow \ \pmb{c} = 
    \begin{bmatrix}
        \dfrac{\sqrt{5}}{5}\\
        \ \\
        \dfrac{-\sqrt{5}}{5}\\
    \end{bmatrix}
$$
因此，*Fibonacci* 数列的一阶差分系统方程（或者说通项公式）为：
$$
    \begin{align}
    \notag
    \begin{bmatrix}
        F_{k+1}\\ \ \\ F_{k}\\
    \end{bmatrix} &= 
    \begin{bmatrix}
        \dfrac{1+\sqrt{5}}{2} & \dfrac{1-\sqrt{5}}{2}\\
        \ \\
        1 & 1\\
    \end{bmatrix}
    \begin{bmatrix}
        \left(\dfrac{1+\sqrt{5}}{2}\right)^k & 0\\
        \ \\
        0 & \left(\dfrac{1-\sqrt{5}}{2}\right)^k\\
    \end{bmatrix}
    \begin{bmatrix}
        \dfrac{\sqrt{5}}{5}\\
        \ \\
        \dfrac{-\sqrt{5}}{5}\\
    \end{bmatrix}\\
    \notag \ \\
    \notag &= 
    \begin{bmatrix}
        \left(\dfrac{1+\sqrt{5}}{2}\right)^{k+1} & \left(\dfrac{1-\sqrt{5}}{2}\right)^{k+1}\\
        \ \\
        \left(\dfrac{1+\sqrt{5}}{2}\right)^k & \left(\dfrac{1-\sqrt{5}}{2}\right)^k\\
    \end{bmatrix}
    \begin{bmatrix}
        \dfrac{\sqrt{5}}{5}\\
        \ \\
        \dfrac{-\sqrt{5}}{5}\\
    \end{bmatrix}\\
    \notag \ \\
    \notag F_k &= 
    \dfrac{\sqrt{5}}{5} \left(\dfrac{1+\sqrt{5}}{2}\right)^k - \dfrac{\sqrt{5}}{5} \left(\dfrac{1-\sqrt{5}}{2}\right)^k
    \end{align}
$$
我们通过代码实际验证一下递归函数与通项公式函数在运行速度上的差异：
```python
import numpy as np
from time import perf_counter

# eigenvalues & eigenvectors method
x1 = (1+np.sqrt(5))/5
x2 = (1-np.sqrt(5))/5

def fibonacci_1(k):
    c = np.sqrt(5)/5
    return c*x1**k - c*x2**k

# recursion method | please don't make k>40, just trust me
def fibonacci_2(k):
    if k==0:
        return 0
    elif k==1:
        return 1
    return fibonacci_2(k-1) + fibonacci_2(k-2)

k = 20

st1 = perf_counter()
f1 = fibonacci_1(k)
et1 = perf_counter()
print("Time usage: " + str(et1-st1) + "s")

st2 = perf_counter()
f2 = fibonacci_2(k)
et2 = perf_counter()
print("Time usage: " + str(et2-st2) + "s")
```
当 $k=5$ 时，递归比通项公式快约 $1$ 个数量级；当 $k=10$ 时，递归与通项公式基本持平；当 $k=20$ 时，通项公式所需的时间基本不变（$10^{-5}$ 秒量级），此时递归所需的时间已达 $10^{-3}$ 秒量级，被拉开了整整两个数量级；当 $k=40$ 时，递归需要耗费接近 $40$ 秒，通项公式依然在 $10^{-5}$ 秒量级。当 $k$ 继续增大时，递归不仅面临时间过长的缺陷，甚至可能直接堆栈溢出，毕竟 Python 没有进行尾栈优化。这个示例充分展现了线性代数在工程计算、科学计算优化中的巨大威力。

## 6.3 Applications to Differential Equations
### 6.3.1 一阶线性齐次微分方程
根据微积分知识可知，对于一元函数 $y=f(x)$ 的常系数微分方程 $\dfrac{dy}{dx}=\lambda y$，其通解具有 $y=ce^{\lambda x}$ 的指数函数形式，其中 $c$ 为常数，其值由初始值（初始条件、边界条件等）确定。对于向量形式的微分方程：
$$
    \begin{cases}
        \dfrac{d\pmb{u}}{dt} = \pmb{Au}, \ \pmb{A} \in \mathbb{R}^{n \times n}\\
        \ \\
        \pmb{u}(0)=
            \begin{bmatrix}
                u_1 & u_2 & \cdots & u_n\\
            \end{bmatrix}^T\\
    \end{cases}
    \tag{6-3-1}
$$
其中 $\pmb{A}$ 是常数矩阵，不随时间变量 $t$ 改变（其它线性微分方程），也不随系统输入 $\pmb{u}$ 改变（非线性微分方程）。有且仅有满足以上条件的微分方程才能直接转化为线性代数形式。类似 $y=ce^{\lambda x}$，（6-3-1）的通解具有如下形式：
$$
    \pmb{u}=c_1 e^{\lambda_1 t}\pmb{x}_1 + c_2 e^{\lambda_2 t}\pmb{x}_2 + \cdots + c_n e^{\lambda_n t}\pmb{x}_n = \sum_{i=1}^{n} c_i e^{\lambda_i t}\pmb{x}_i = \pmb{S} e^{\pmb{\Lambda}t} \pmb{c}
    \tag{6-3-2}
$$
其中 $\pmb{x}_i$ 为 $\pmb{A}$ 的对应于特征值 $\lambda_i$ 的特征向量，$\pmb{x}_i$ 拼接组成特征向量矩阵 $\pmb{S}$，常系数 $c_i$ 由初值 $\pmb{u}(0)$ 决定：
$$
    \dfrac{d\pmb{u}}{dt} = c_1 \lambda_1 e^{\lambda_1 t}\pmb{x}_1 + c_2 \lambda_2 e^{\lambda_2 t}\pmb{x}_2 + \cdots + c_n \lambda_n e^{\lambda_n t}\pmb{x}_n = \sum_{i=1}^{n} c_i \lambda_i e^{\lambda_i t}\pmb{x}_i\\
    \pmb{Au} = c_1 e^{\lambda_1 t}\pmb{Ax}_1 + c_2 e^{\lambda_2 t}\pmb{Ax}_2 + \cdots + c_n e^{\lambda_n t}\pmb{Ax}_n = \sum_{i=1}^{n} c_i \lambda_i e^{\lambda_i t}\pmb{x}_i = \dfrac{d\pmb{u}}{dt}
$$
由上可验证通解（6-3-2）的正确性，在之后我们还会发现，这一通解（$\pmb{u} = \pmb{S} e^{\pmb{\Lambda}t} \pmb{c}$）与差分方程有相似之处（$\pmb{u}_k=\pmb{S\Lambda}^k \pmb{c}$）。接下来以一个实例说明上述过程的操作步骤：
$$
    \begin{cases}
        \dfrac{du_1}{dt} = -u_1 + 2u_2\\
        \ \\
        \dfrac{du_2}{dt} = u_1 -2u_2\\
    \end{cases} \ \Rightarrow \pmb{A} = 
    \begin{bmatrix}
        -1 & 2\\
        1 & -2\\
    \end{bmatrix}, \ \pmb{u}(0) = 
    \begin{bmatrix}
        1\\ 0\\
    \end{bmatrix}
$$
对 $\pmb{A}$ 进行特征值分解：
$$
    \lambda_1=0, \ \lambda_2=-3 \ \Longrightarrow \ \pmb{x}_1 = 
    \begin{bmatrix}
        2\\ 1\\
    \end{bmatrix}, \ \pmb{x}_2 = 
    \begin{bmatrix}
        -1\\ 1\\
    \end{bmatrix}
$$
将特征值与特征向量代回通解公式，结合初值条件可得：
$$
    \pmb{u}(t) = c_1 \times 1 \times
    \begin{bmatrix}
        2\\ 1\\
    \end{bmatrix} + c_2 \times e^{-3t} \times
    \begin{bmatrix}
        -1\\ 1\\
    \end{bmatrix}\\
    \ \\
    \pmb{u}(0) =
    \begin{bmatrix}
        2c_1-c_2\\ c_1+c_2\\
    \end{bmatrix} = 
    \begin{bmatrix}
        1\\ 0\\
    \end{bmatrix} \ \Longrightarrow \ 
    \begin{cases}
        c_1=\dfrac{1}{3}\\ c_2=-\dfrac{1}{3}
    \end{cases}\\
    \ \\
    \therefore \pmb{u}(t)=\dfrac{1}{3}
    \begin{bmatrix}
        2\\ 1\\
    \end{bmatrix} + \dfrac{1}{3}e^{-3t}
    \begin{bmatrix}
        1\\ -1\\
    \end{bmatrix} \xrightarrow{t\rightarrow \infty} \dfrac{1}{3}
    \begin{bmatrix}
        2\\ 1\\
    \end{bmatrix}
$$
根据通解（6-3-2）可以讨论一阶线性常系数微分方程系统的相关特性：

（1）稳定性（收敛性）：当 $Re(\lambda_i)<0, \forall i \leqslant n$ 时，系统随时间推移终将收敛，至于是欠阻尼、临界阻尼还是过阻尼系统则视情况（初值，或系统输入）而定。这里重点解释虚数特征值情况。假设 $\lambda=a \pm bi$：
$$
    \left|e^{(a \pm bi)t}\right|=\left|e^{at}\right| \left|e^{bti}\right| = e^{at} \left|\cos(bt)+i\sin(bt)\right| = e^{at} \sqrt{\cos^2(bt)+\sin^2(bt)} = e^{at}
    \tag{6-3-3}
$$
因此只有虚数特征值的实部才能影响系统的收敛性；

（2）非零稳态：根据（1）可知，只要全体特征值的实部均小于零，对应系统必将收敛。更进一步地，若 $\lambda_1=0$（默认降序排列特征值）且 $c_1 \ne 0$ 则系统稳态必不为 $0$（此时稳态取决于零特征值项的常系数）；

（3）发散性：若存在 $Re(\lambda_i)>0$，则该系统发散。

### 6.3.2 矩阵指数 & 矩阵级数
回到刚才的案例：
$$
    \dfrac{d\pmb{u}}{dt} = \pmb{Au}, \ \pmb{A} \in \mathbb{R}^{n \times n}
$$
当 $\pmb{A}$ 为非对角阵时，向量 $\pmb{u}$ 的各元素（分量）之间存在耦合作用，即 $\dfrac{du_1}{dt}$ 中不仅包含常数项、$u_1$ 项，还包括 $u_2$ 项。此时令 $\pmb{u}=\pmb{Sv}$ 并代回：
$$
    \pmb{S} \dfrac{d\pmb{v}}{dt} = \pmb{ASv} \ \Longrightarrow \ \pmb{S}^{-1}\pmb{S}\dfrac{d\pmb{v}}{dt} = \pmb{S}^{-1}\pmb{ASv} \ \Longrightarrow \ \dfrac{d\pmb{v}}{dt} = \Lambda \pmb{v}
    \tag{6-3-4}
$$
其中 $\pmb{S}$ 由 $\pmb{A}$ 的特征向量拼接组成。假设 $\pmb{A}$ 没有重根特征值、特征向量足量且互相正交，即 $\pmb{A}$ 是可对角化的。此时 $\pmb{S}$ 满秩且列向量构成 $\mathbb{R}^{n \times n}$ 的一组正交基，因此形如 $\pmb{u}=\pmb{Sv}$ 的变量代换是必然存在的。此时微分方程由向量 $\pmb{v}$ 构成，且 $\pmb{v}$ 的各分量之间不再存在耦合现象：
$$
    \dfrac{d\pmb{v}_1}{dt} = \lambda_1 v_1, \ \dfrac{d\pmb{v}_2}{dt} = \lambda_2 v_2, \ \cdots \ \dfrac{d\pmb{v}_n}{dt} = \lambda_n v_n
    \tag{6-3-5}
$$
因此有
$$
    \pmb{v}(t) = \pmb{I} e^{\pmb{\Lambda}t} \pmb{v}(0) \ \xrightarrow{\pmb{v} = \pmb{S}^{-1}\pmb{u}} \ \pmb{u}(t) = \pmb{S} e^{\pmb{\Lambda} t} \pmb{S}^{-1} \pmb{u}(0) \ \Longrightarrow \ e^{\pmb{A}t} = \pmb{S} e^{\pmb{\Lambda} t} \pmb{S}^{-1}
    \tag{6-3-6}
$$
其中 $\pmb{e}^{A}t$ 是一类典型的矩阵（方阵）指数，类似方阵对角化 $\pmb{A} = \pmb{S\Lambda} \pmb{S}^{-1}$，方阵指数也可以对角化为 $\pmb{e}^{A}t = \pmb{S} e^{\Lambda t} \pmb{S}^{-1}$。接下来我们通过泰勒级数证明这一结论。已知：
$$
    e^x = 1 + x + \dfrac{1}{2}x^2 + \dfrac{1}{6}x^3 + \cdots + \dfrac{1}{n!}x^n = \sum_{n=0}^{\infty}\dfrac{x^n}{n!}
    \tag{6-3-7}
$$
类比地有：
$$
    \begin{align}
        \notag
        e^{\pmb{A}t} &= \pmb{I} + \pmb{A}t + \dfrac{1}{2}\left(\pmb{A}t\right)^2 + \dfrac{1}{6}\left(\pmb{A}t\right)^3 + \cdots + \dfrac{1}{n!}\left(\pmb{A}t\right)^n\\
        \notag \ \\
        \notag &= \pmb{I} + \pmb{S\Lambda}\pmb{S}^{-1}t + \dfrac{1}{2}\pmb{S}\pmb{\Lambda}^2\pmb{S}^{-1}t^2 + \dfrac{1}{6}\pmb{S}\pmb{\Lambda}^3\pmb{S}^{-1}t^3 + \cdots + \dfrac{1}{n!}\pmb{S}\pmb{\Lambda}^n\pmb{S}^{-1}t^n\\
        \notag \ \\
        \notag &= \pmb{S}\left(\pmb{I} + \pmb{\Lambda} t + \dfrac{1}{2}\pmb{\Lambda}^2t^2 + \dfrac{1}{6}\pmb{\Lambda}^3t^3 + \cdots + \dfrac{1}{n!}\pmb{\Lambda}^nt^n\right) \pmb{S}^{-1} = \pmb{S}e^{\pmb{\Lambda}t}\pmb{S}^{-1}
    \end{align}
    \tag{6-3-7}
$$
特别地，$e^{\pmb{\Lambda}t}$ 与 $\pmb{\Lambda}$ 一样具有对角性质：
$$
    e^{\pmb{\Lambda}t} = 
    \begin{bmatrix}
        e^{\lambda_1t} & 0 & \cdots & 0\\
        0 & e^{\lambda_2t} & \cdots & 0\\
        \vdots & \vdots & \ddots & \vdots\\
        0 & 0 & \cdots & e^{\lambda_nt}\\
    \end{bmatrix}
    \tag{6-3-8}
$$
除了 $e$ 级数（$e^x=\sum_{n=0}^{\infty}\dfrac{x^n}{n!}$），几何级数（$\dfrac{1}{x-1}=\sum_{n=0}^{\infty}x^n$）也有类似的变换。利用等比数列求和公式可知：
$$
    \sum_{n=0}^{\infty}x^n = \lim_{n \rightarrow \infty}\dfrac{1 \times (1-x^n)}{1-x} \xrightarrow{|x|<1} \dfrac{1}{1-x}
    \tag{6-3-9}
$$
类似地，当 $Re\left(\lambda_i(\pmb{A}t)\right)<1$ 时，有：
$$
    \left(\pmb{I}-\pmb{A}t\right)^{-1} = \pmb{I} + \pmb{A}t + (\pmb{A}t)^2 + \cdots
    \tag{6-3-10}
$$
至此，我们可在 $\pmb{A}$ 的特征值构成的复数域平面（纵轴为虚部，横轴为实部）上总结目前我们遇到的两类问题：

（1）当特征值均位于左半面时，$Re(\lambda_i)<0$，此时微分方程 $\dfrac{d\pmb{u}}{dt}=\pmb{Au}$ 具有稳定解，即一阶常系数线性齐次微分系统时域收敛；

（2）当特征值均位于单位圆内（含边界）时，$|\lambda_i| \leqslant 1$，此时矩阵的幂收敛，即一阶差分系统 $\pmb{u}_{k+1}=\pmb{Au}_k$ 时域收敛。

### 6.3.3 高阶常系数线性齐次微分方程的转化
接下来我们以一个普通的二阶常系数线性齐次微分方程为例，说明如何将其转化为矩阵（向量）形式的一阶系统，并延申至更高阶次系统：
$$
    y^{''} + k_1 y^{'} + k_2 y = 0
    \tag{6-3-11}
$$
与 6.2.3 一样，需要使用一个小 trick，为（6-3-11）添加方程 $y^{'}=y^{'}$，构建向量 $\pmb{u} = \left[y^{'},y\right]^T$，从而获得方程 $\dfrac{d\pmb{u}}{dt} = \pmb{Au}$：
$$
    \underbrace{
        \begin{bmatrix}
            y^{''}\\ \ \\ y^{'}\\
        \end{bmatrix}}_{\dfrac{d\pmb{u}}{dt}} = 
    \underbrace{
        \begin{bmatrix}
            -k_1 & -k_2\\
            \ \\
            1 & 0\\
        \end{bmatrix}}_{\pmb{A}}
    \underbrace{
        \begin{bmatrix}
            y^{'}\\ \ \\ y\\
        \end{bmatrix}}_{\pmb{u}}
    \tag{6-3-12}
$$
对于 $n$ 阶常系数线性齐次微分方程：
$$
    y^{(n)} + k_1 y^{(n-1)} + k_2 y^{(n-2)} + \cdots + k_n y = 0
    \tag{6-3-13}
$$
通过添加 $n-1$ 个方程 $y^{(i)} = y^{(i)}$、构建向量 $\pmb{u} = \left[y^{(n-1)},y^{(n-2)},\cdots,y\right]^T$，可将其化为矩阵形式：
$$
    \begin{bmatrix}
        y^{(n)}\\ y^{(n-1)}\\ y^{(n-2)}\\ \vdots \\ y^{(1)}
    \end{bmatrix} = 
    \begin{bmatrix}
        -k_1 & -k_2 & \cdots & -k_n\\
        1 & 0 & \cdots & 0\\
        0 & 1 & \cdots & 0\\
        \vdots & \vdots & \ddots & \vdots\\
        0 & 0 & \cdots & 1\\
    \end{bmatrix}
    \begin{bmatrix}
        y^{(n-1)}\\ y^{(n-2)}\\ y^{(n-3)}\\ \vdots \\ y\\
    \end{bmatrix}
    \tag{6-3-14}
$$

## 6.4 Applications of Eigenvalues
### 6.4.1 Markov 矩阵
*Markov* 矩阵是一类与概率统计紧密相关的矩阵，它有两个主要特征：

（1）矩阵内所有元素均为非负数；

（2）所有的列向量元素之和均为1。

事实上，*Markov* 矩阵的列向量对应于某一类事件中不同结果的发生概率，由此可以解释上述两条特征。例如 $\pmb{A}$:
$$
    \pmb{A} = 
    \begin{bmatrix}
        0.1 & 0.01 & 0.3\\
        0.2 & 0.99 & 0.3\\
        0.7 & 0 & 0.4\\
    \end{bmatrix}
$$
*Markov* 矩阵固定具有一个等于 $1$ 的特征值 $\lambda_1$（至少一个），其余特征值实部的绝对值均小于 $1$。此外，$\lambda_1$ 对应的特征向量 $\pmb{x}_1$ 中全体元素均为正值。对于一阶差分系统 $\pmb{u}_k = \pmb{S}\pmb{\Lambda}^k\pmb{c}$ 而言，若 $c_1>0$，则系统的稳态响应也为正值。

现简单证明单位特征值的存在。对于 $\pmb{B}$:
$$
    \pmb{B} = 
    \begin{bmatrix}
        b_{11} & b_{12} & b_{13}\\
        b_{21} & b_{22} & b_{23}\\
        b_{31} & b_{32} & b_{33}\\
    \end{bmatrix}, \ 
    \begin{cases}
        b_{11} + b_{21} + b_{31} = 1\\
        b_{12} + b_{22} + b_{32} = 1\\
        b_{13} + b_{23} + b_{33} = 1\\
    \end{cases}
$$
若 $\pmb{B}$ 存在单位特征值，则 $\pmb{B}-\pmb{I}$ 为奇异矩阵，即：
$$
    \begin{align}
        \notag \pmb{Y} &= \pmb{B} - \pmb{I} = 
        \begin{bmatrix}
            -(a_{21}+a_{31}) & a_{12} & a_{13}\\
            a_{21} & -(a_{12}+a_{32}) & a_{23}\\
            a_{21} & a_{32} & -(a_{13}+a_{23})\\
        \end{bmatrix}\\
        \notag \ \\
        \notag \pmb{Yx} &= \pmb{0} \ \Longrightarrow \ \pmb{x} = 
        \begin{bmatrix}
            1\\ 1\\ 1\\
        \end{bmatrix} \ne \pmb{0}
    \end{align}
$$
方程 $(\pmb{B} - \pmb{I})\pmb{x} = \pmb{0}$ 存在非零解，因此 $\pmb{B}-\pmb{I}$ 为奇异矩阵。

接下来通过一个实际应用介绍 *Markov* 矩阵与相应一阶差分方程的实际应用。假设有 $A$、$B$ 两个地区，两地的初始人口分别为 $400$、$900$，每经过一段固定时间（达到新状态），两地之间存在人口流通。令 $P_{AB}$ 表示居民从 $A$ 地迁移至 $B$ 的概率，$P_{AA}$ 表示留在 $A$ 地的概率。则有：
$$
    \begin{bmatrix}
        u_A\\ \ \\ u_B\\
    \end{bmatrix}_{k+1} = 
    \begin{bmatrix}
        P_{AA} & P_{BA}\\
        \ \\
        P_{AB} & P_{BB}\\
    \end{bmatrix}
    \begin{bmatrix}
        u_A\\ \ \\ u_B\\
    \end{bmatrix}_k, \ 
    \begin{cases}
        P_{AA} + P_{AB} = 1\\
        \ \\
        P_{BA} + P_{BB} = 1\\
    \end{cases}
    \tag{6-4-1}
$$
具体而言，以：
$$
    \pmb{P} = 
    \begin{bmatrix}
        0.9 & 0.2\\
        \ \\
        0.1 & 0.8\\
    \end{bmatrix}
$$
为例。根据 *Markov* 矩阵的先验知识可知 $\lambda_1 = 1$，$tr\left(\pmb{P}\right) = 0.9 + 0.8$，故 $\lambda_2 = 0.7$，进而可解两个特征向量：
$$
    (\pmb{P} - \pmb{I})\pmb{x} = \pmb{0} \ \Longrightarrow \ \pmb{x}_1 = 
    \begin{bmatrix}
        2\\ 1\\
    \end{bmatrix}\\
    \ \\
    (\pmb{P} - 0.7\pmb{I})\pmb{x} = \pmb{0} \ \Longrightarrow \ \pmb{x}_2 = 
    \begin{bmatrix}
        -1\\ 1\\
    \end{bmatrix}
$$
差分系统方程为：
$$
    \pmb{u}_k = \pmb{S}\pmb{\Lambda}^k\pmb{c} = 
    \begin{bmatrix}
        2 & -1\\
        1 & 1\\
    \end{bmatrix}
    \begin{bmatrix}
        1 & 0\\
        0 & 0.7^k\\
    \end{bmatrix}
    \begin{bmatrix}
        c_1\\ c_2\\
    \end{bmatrix} = c_1
    \begin{bmatrix}
        2\\ 1\\
    \end{bmatrix} + c_2 \times 0.7^k
    \begin{bmatrix}
        -1\\ 1\\
    \end{bmatrix}
$$
代入系统初值可完整求解：
$$
    \begin{cases}
        2c_1 - c_2 = 400\\
        \ \\
        c_1 + c_2 = 800\\
    \end{cases} \ \Longrightarrow \ 
    \begin{cases}
        c_1 = 400\\
        \ \\
        c_2 = 400\\
    \end{cases}
$$
显然 $\lambda_2$ 对应的分量将随着时间推移逐渐衰减，最终系统会稳定于第一分量。即两地人口分布（假设没有新生儿与人口死亡）将稳定于 $800$、$400$。

### 6.4.2 Fourier 变换分解
根据信号系统的相关知识可知，任意连续信号 $f(t)$ 都可分解为直流分量与不同频率的正弦交流分量的线性组合，即 *Fourier* 变换：
$$
    f(t) = a_0 + a_1\cos(t) + b_1\sin(t) + a_2\cos(2t) + b_2\sin(2t) + \cdots
    \tag{6-4-2}
$$
*Fourier* 变换是信号分析领域所有概念的基础。它之所以如此重要、如此管用的关键原因就是它是一种正交分解，且正交基是单一频率的正弦信号，即各频率的正余弦信号分量是彼此正交的，即：
$$
    \int_{0}^{2\pi}\sin(f_1t+\phi_1)\sin(f_2t+\phi_2)dt = 
    \begin{cases}
        0, \ \ f_1 \ne f_2 \ or \ \phi_1 \ne \phi_2\\
        c(c\ne0), \ \ f_1=f_2 \ and \ \phi_1=\phi_2
    \end{cases}
    \tag{6-4-3}
$$

