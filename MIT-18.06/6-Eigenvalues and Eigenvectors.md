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

---
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

---
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

---
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

---
## 6.5 Symmetric Matrices & Positive Definite Matrices
### 6.5.1 对称矩阵
对称矩阵（Symmetric Matrix）是最重要的矩阵之一，它们不光在数理性质上拥有一些奇妙的特性，同时还在工程实践中具有广泛的应用场景。在之前我们提到了矩阵的特征值分解，此处我们首先通过对称矩阵的特征值问题 $\pmb{Ax} = \lambda \pmb{x}$ 来认识这类矩阵的性质。

对于某对称矩阵 $\pmb{A} = {\pmb{A}}^T \in \mathbb{R}^{n \times n}$，其特征值对角化形式为 $\pmb{A} = \pmb{S} \pmb{\Lambda} {\pmb{S}}^{-1}$。考虑 ${\pmb{A}}^T = \left( {\pmb{S}}^{-1} \right)^T \pmb{\Lambda} {\pmb{S}}^T = \pmb{S} \pmb{\Lambda} {\pmb{S}}^{-1}$，不难想到当 $\pmb{S}^{-1} = \pmb{S}^T$ 时上式必然成立，即有 $\pmb{S}^T \pmb{S} = \pmb{I}$。由此得出对称矩阵的两条重要特性：

**1. 对称矩阵的特征向量具有正交性。**

**2. 对称矩阵的特征值均为实数。**

因此，我们换一个字母，用 $\pmb{Q}$ 来表示满足列向量正交性质的特征向量矩阵，从而得到谱定理（Spectral Theorem）或主轴定理（Principal Axis Theorem）：

任意对称矩阵 $\pmb{A}$ 均具有如下分解形式：$\pmb{A} = \pmb{Q} \pmb{\Lambda} {\pmb{Q}}^{-1} = \pmb{Q} \pmb{\Lambda} {\pmb{Q}}^T$，其中 $\pmb{Q}^{-1} = \pmb{Q}^T$，$\pmb{\Lambda}$ 是由实数特征值构成的对角阵。

该定理的前半部分刚才已经证明了，但是想证明后半部分中的“全实数特征值”还是有些难度。首先我们已知特征值方程 $\pmb{Ax} = \lambda \pmb{x}$，假设特征值为复数，则对应特征向量亦为复数向量，则二者存在共轭成分 $\bar{\lambda}$、$\bar{\pmb{x}}$，且共轭成分依旧满足原特征值方程，即有：
$$
\pmb{Ax} = \lambda \pmb{x} \ \Longrightarrow \ \pmb{A} \bar{\pmb{x}} = \bar{\lambda} \bar{\pmb{x}}
\tag{6-5-1}
$$
对上式两边取转置后，等式两边右乘向量 $\pmb{x}$：
$$
{\bar{\pmb{x}}}^T {\pmb{A}}^T = {\bar{\pmb{x}}}^T \pmb{A} = \bar{\lambda} {\bar{\pmb{x}}}^T, \ \ {\bar{\pmb{x}}}^T \pmb{A} \pmb{x} = \bar{\lambda} {\bar{\pmb{x}}}^T \pmb{x} = \lambda {\bar{\pmb{x}}}^T \pmb{x}
\tag{6-5-2}
$$
由此可得 $\lambda = \bar{\lambda}$，即特征值为实数（不含虚部）。

我们对谱定理的理解还可以更深入一些：
$$
\begin{align}
\notag
\pmb{A} &= \pmb{Q} \pmb{\Lambda} {\pmb{Q}}^T = 
\begin{bmatrix}
\ & \ & \ \\
\pmb{q}_1 & \cdots & \pmb{q}_n\\
\ & \ & \ \\
\end{bmatrix}
\begin{bmatrix}
\lambda_1 & \ & \ \\
\ & \ddots & \ \\
\ & \ & \lambda_n \\
\end{bmatrix}
\begin{bmatrix}
\ & \pmb{q}_1^T & \ \\
\ & \vdots & \ \\
\ & \pmb{q}_n^T & \ \\
\end{bmatrix}\\
\notag \ \\
\notag
&= \lambda_1 \pmb{q}_1 \pmb{q}_1^T + \lambda_2 \pmb{q}_2 \pmb{q}_2^T + \cdots + \lambda_n \pmb{q}_n \pmb{q}_n^T = \sum_{i=1}^{n} \lambda_i \pmb{q}_i \pmb{q}_i^T
\end{align}
\tag{6-5-3}
$$
不难发现，$\pmb{q}_i \pmb{q}_i^T \in \mathbb{R}^{n \times n}$ 是一个秩 1 矩阵，或者说是投影矩阵。每个对称矩阵都可以分解为多个正交投影矩阵的组合。

对称矩阵的最后一个性质是关于主元符号的。在计算（手算）特征值时，我们需要求解多项式方程 ${\rm det}(\pmb{A} - \lambda \pmb{I}) = 0$；在计算主元时，我们需要对矩阵进行初等变换。二者本不相干，只因它们都联系上了一个概念：行列式。主元的乘积与特征值的乘积是相同的。二者存在如下关系：对称矩阵的正值特征值个数等于正值主元个数。

想要证明这一性质，需要回顾一下之前学过的 LU 分解、LDU 分解：任意方阵 $\pmb{A}$ 均可通过初等行变换分解为主元为 1 的下三角矩阵 $\pmb{L}$ 与主元为 $\pmb{A}$ 主元的上三角矩阵 $\pmb{U}$ 的乘积：$\pmb{A} = \pmb{LU}$。更进一步地，对 $\pmb{U}$ 进行初等列变换，可将 $\pmb{A}$ 彻底分解为下三角矩阵 $\pmb{L}$、对角阵 $\pmb{D}$ 与 上三角矩阵 $\pmb{U}^{'}$ 的乘积：$\pmb{A} = \pmb{LD} \pmb{U}^{'}$。当 $\pmb{A}$ 为对称矩阵时，$\pmb{U}^{'} = \pmb{L}^T$，即 LDU 分解可转变为 LDLT 分解：$\pmb{A} = \pmb{LD} \pmb{L}^T$，这是一种对角化分解。

类似式（6-5-3），LDLT 分解也可以表示为二次型形式。根据代数学中的 Sylvester's Law of Inertia（西尔维斯特惯性定理），实数域中的二次型可以通过线性变换化简为唯一的标准型，其中正项数（正惯性系数）、负项数（负惯性系数）以及零项的数目可唯一确定。因此 $\pmb{LD} \pmb{L}^T$ 与 $\pmb{Q} \pmb{\Lambda} \pmb{Q}^T$ 将指向某个唯一确定的标准型，两者的惯性系数符号情况必然相等。尽管比较抽象，但这已经是我目前能找到的比较明确的证明过程，Gilbert 教材原本上的证明是通过一个二阶方阵的实例说明的，类似的案例我们当然可以无穷无尽地举例验证，在此不再赘述。

### 6.5.2 复矩阵与傅里叶矩阵
对称矩阵（Symmetric Matrix）是限定于实数域的概念，在复数域中也存在类似的概念。对于复向量 $\pmb{z} = \left[z_1,z_2,\cdots,z_n \right]^T \in \mathbb{C}^{n}$，由于模长必须为非负实数，因此 $\pmb{z}$ 的向量模长定义为 $\bar{\pmb{z}}^T \pmb{z} = \pmb{z}^H \pmb{z} = \left| z_1 \right|^2 + \left| z_2 \right|^2 + \cdots + \left| z_n \right|^2$；复向量 $\pmb{x}$、$\pmb{y}$ 的内积定义为 $\pmb{y}^H \pmb{x}$；复“对称”矩阵，即 Hermitian 矩阵定义为：$\pmb{A} = \pmb{A}^H \in \mathbb{C}^{n \times n}$。

尽管在 18.06 中无需过多探讨复矩阵的知识，但有一个案例应用还是需要着重介绍一下，即傅里叶矩阵（Fourier Matrix）。Fourier 矩阵严格来说是一系列符合某些规则的矩阵，以 $n$ 阶 Fourier 矩阵 $\pmb{F}_n \in \mathbb{C}^{n \times n}$ 为例：
$$
\pmb{F}_n = 
\begin{bmatrix}
1 & 1 & 1 & \cdots & 1\\
1 & w_n & w_n^2 & \cdots & w_n^{n-1}\\
1 & w_n^2 & w_n^4 & \cdots & w_n^{2(n-1)}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
1 & w_n^{n-1} & w_n^{2(n-1)} & \cdots & w_n^{(n-1)^2}\\
\end{bmatrix}, \ \ [F_n]_{ij} = w_n^{i \times j}, \ \ w_n = e^{i 2 \pi/n} = \cos (\dfrac{2\pi}{n}) + i \sin (\dfrac{2\pi}{n})
\tag{6-5-4}
$$
其中复数 $w_n$ 位于复平面单位圆上的 $n$ 等分点处（以 $Re$ 轴为起点，逆时针旋转），并满足 $w_n^n = 1$、$\left| w_n^n \right|=1$。需要注意的是，并非所有阶次的 Fourier 矩阵都是 Hermitian 矩阵，这与主对角线上元素是否包含虚数有关，即与 $w_n^{(n-1)^2}$ 有关。例如 $\pmb{F}_4$ 的主对角线元素为 $\left\{ 1,w_4,w_4^4,w_4^9 \right\}$，即 $\left\{ 1,i,1,-i \right\}$，因此 $\pmb{F}_4^T = \pmb{F}_4$，但 $\pmb{F}_4^H \ne \pmb{F}_4$。因此 $\pmb{F}_4$ 只是 Symmetric 矩阵。Fourier 矩阵可转变为酉矩阵（Unitary Matrix），具有标准正交列向量：
$$
\left( \dfrac{1}{2} \pmb{F}_n^H \right) \left( \dfrac{1}{2} \pmb{F}_n \right) = \pmb{I}_n, \ \ {\pmb{F}_n}^{-1} = \dfrac{1}{4} \pmb{F}_n^H
\tag{6-5-5}
$$
Fourier 矩阵还具有另一个重要性质，它可被分解为大量稀疏矩阵的乘积，从而极大程度地减少矩阵乘法的运算量，基于该理念发展而来的快速 Fourier 变换（Fast Fourier Transform, FFT）是信号与系统领域的技术基石，FFT 的核心要义就是将 $\pmb{F}_n$ 与 $\pmb{F}_{n/2}$ 联系起来。以 $\pmb{F}_4$ 为例，使用普通矩阵运算进行一次方阵乘积操作，需要执行 16 次乘法（加法忽略不计）。而 $\pmb{F}_4$ 可转化为：
$$
\pmb{F}_4 = 
\begin{bmatrix}
\pmb{I}_2 & \pmb{D}_{4,2}\\
\ & \ \\
\pmb{I}_2 & -\pmb{D}_{4,2}\\
\end{bmatrix}
\begin{bmatrix}
\pmb{F}_2 & \pmb{0}_2\\
\ & \ \\
\pmb{0}_2 & \pmb{F}_2\\
\end{bmatrix} \pmb{P}_4
$$
其中， $\pmb{I}_2$ 表示 2 阶单位阵，$\pmb{D}_2$ 为对角阵，$\pmb{0}_2$ 表示 2 阶零矩阵，$\pmb{P}_4$ 为置换矩阵（Permutation Matrix）：
$$
\pmb{F}_4 = 
\begin{bmatrix}
1 & \ & 1 & \ \\
\ & 1 & \ & i\\
1 & \ & -1 & \ \\
\ & 1 & \ & -i\\
\end{bmatrix}
\begin{bmatrix}
1 & 1 & \ & \ \\
1 & -1 & \ & \ \\
\ & \ & 1 & 1\\
\ & \ & 1 & -1\\
\end{bmatrix}
\begin{bmatrix}
1 & \ & \ & \ \\
\ & \ & 1 & \ \\
\ & 1 & \ & \ \\
\ & \ & \ & 1\\
\end{bmatrix}
$$
置换矩阵 $\pmb{P}_4$ 的主要作用是把偶数列（下标从 0 计数）提前归拢，把奇数列滞后放置（即 $\left\{ c_0,c_2,c_1,c_3 \right\}$）；$\pmb{D}_{4,2}$ 是以 $w_4^n$ 为组成元素的对角阵：$\pmb{D}_{4,2} = {\rm diag} \left( 1,w_4^{(2-1)} \right)$。由于矩阵稀疏性可知，实际需要参与运算的部分大幅减少（$\pmb{P}_4$、$\pmb{I}_2$ 均不视为参与乘法运算）。当 $n$ 的阶次进一步提高（如 10），$\pmb{F}_n$ 能够反复进行上述分解：$\pmb{F}_{1024}$、$\pmb{F}_{512}$、$\pmb{F}_{256}$、$\pmb{F}_{128}$、$\cdots$。最终可将乘法运算次数从 $n^2$ 降低至 $\dfrac{1}{2} \log_2(n)$。

### 6.5.3 正定矩阵
在课程的开始，Gilbert 给出了三个问题：
（1）如何判断一个矩阵是正定矩阵（Positive Definite Matrix）？
（2）正定矩阵与几何学（直观表现）有哪些联系？
（3）为什么我们要对正定矩阵如此关注？

当然可能 Gilbert 讲 high 了，在视频课程中这些问题并没有系统性地予以回答，因此本节的笔记主要来源于教材。

从定义来看，正定矩阵是特征值全为正值的**对称矩阵**。以简单的 2 阶实对称矩阵为例：
$$
\pmb{A} =
\begin{bmatrix}
a & b\\
b & c\\
\end{bmatrix}
$$
**问题 1：**

判断 $\pmb{A}$ 的正定性有如下几条方法（性质），每条性质都可以独立证明正定性，当然彼此之间也可以互相证明：

（1）特征值（定义）：$\lambda_1,\lambda_2>0$

（2）行列式：$a>0$，$ac-b^2>0$

（3）主元：$a>0$，$c-\dfrac{b^2}{a}>0$

（4）二次型：$\forall \pmb{x} \in \mathbb{R}^{2}$，$\pmb{x}^T \pmb{Ax}>0$

接下来，我们从条件（1）出发，依次验证一下（2）至（4）是否与（1）等价。

（1）$\rightarrow$（2）：
$$
\lambda_1,\lambda_2>0 \ \ \Longleftrightarrow \ \ \lambda_1 \lambda_2 > 0, \ \lambda_1+\lambda_2 > 0 \\
\begin{align}
\notag \\
\notag &\therefore \det (\pmb{A}) = \lambda_1 \lambda_2 = ac-b^2 > 0\\
\notag \ \\
\notag &\therefore {\rm tr} (\pmb{A}) = \lambda_1 + \lambda_2 = a + c > 0 \ \ \Rightarrow \ \ a, c >0
\end{align}
$$
**知识回顾**：对于 n 阶矩阵（假设满秩） $\pmb{X} \in \mathbb{R}^{n \times n}$，有 $\det (\pmb{X}) = \prod_{i=1}^{n} \lambda_i$，${\rm tr}(\pmb{X}) = \sum_{i=1}^{n} x_{ii} = \sum_{i=1}^{n} \lambda_i$。

（2）$\rightarrow$（1）：
由（1）的证明过程可知，根据条件（2）可得 $\det(\pmb{A})>0$、${\rm tr}(\pmb{A})>0$，从而（1）得证。

（1）$\Leftrightarrow$（3）：
因为（1）与（2）等价，而（2）至（3）易证，反之亦然，故命题得证。

（1）$\Leftrightarrow$（4）：
首先观察二次型 $\pmb{x}^T \pmb{Ax}$ 的展开形式 $f(x,y)$：
$$
f(x,y) = \pmb{x}^T \pmb{Ax} = 
\begin{bmatrix}
x & y
\end{bmatrix}
\begin{bmatrix}
a & b\\
b & c\\
\end{bmatrix}
\begin{bmatrix}
x \\ y
\end{bmatrix} = ax^2 + 2bxy + cy^2
\tag{6-5-6}
$$
根据上式可见，之所以 $f(x,y)$ 被称为“二次型”，是因为其所有元素（与 $\pmb{A}$ 相关）都是二次非线性的，展开形式 $f(x,y)$ 中无常数项、无高次项。这一特点在高阶正定矩阵中依然存在。对 $f(x,y)$ 进行因式分解，不难得到：
$$
f(x,y) = a \left( x+\dfrac{b}{a}y \right)^2 + \left( c-\dfrac{b^2}{a} \right) y^2, \ \ a \ne 0
\tag{6-5-7}
$$
根据（1）与（2）的等价关系，不难得知当（2）成立时，$f(x,y)>0$ 成立；反之亦然，故命题得证。

把相关结论拓展到 $n$ 阶矩阵 $\pmb{A} \in \mathbb{R}^{n \times n}$，有：

（1）特征值（定义）：$\lambda_1, \lambda_2, \cdots , \lambda_n > 0$

（2）行列式：全体左上行列式均为正值，即：
$$
\begin{vmatrix}
x_{11}
\end{vmatrix}, \ 
\begin{vmatrix}
x_{11} & x_{12}\\
x_{21} & x_{22}\\
\end{vmatrix}, \ 
\begin{vmatrix}
x_{11} & x_{12} & x_{13}\\
x_{21} & x_{22} & x_{23}\\
x_{31} & x_{23} & x_{33}\\
\end{vmatrix}, \ \cdots , \ \det (\pmb{A}) > 0
$$
（3）主元：全体主元均为正值；

（4）二次型：$\forall \pmb{x} \in \mathbb{R}^n, \ \pmb{x} \ne \pmb{0}, \ \pmb{x}^T \pmb{Ax} > 0$

**问题 2：**

如果我们令 $z=f(x,y)$，即可在三维笛卡尔坐标系中绘制曲面图形。为方便起见，这里我们固定 $a$ 和 $b$ 的数值及属性（$a,b>0$），仅着眼于 $c$。当 $c > \dfrac{b^2}{a}$ 时（满足正定条件），曲面 $z$ 为椭圆抛物面。标准的椭圆抛物面方程为：
$$
z = \dfrac{x^2}{a^2} + \dfrac{y^2}{b^2}
\tag{6-5-8}
$$
本节中 $f(x,y)$ 的交叉项 $2bxy$ 决定了抛物面倾斜于 $x$、$y$ 轴的角度。当交叉项不存在时，二次曲面垂直于上述两坐标轴。

当 $c = \dfrac{b^2}{a}$ 时（半正定），$f(x,y)$ 表现为抛物线沿空间直线 $y=-\dfrac{b}{a}$ 无限延长生成的不闭合抛物面。

当 $c < \dfrac{b^2}{a}$ 时（非正定），注意此时不能直接判定为负定矩阵，$f(x,y)$ 表现为双曲抛物面（马鞍面），马鞍面的标准方程为：
$$
\pm 2z = \dfrac{x^2}{a^2} - \dfrac{y^2}{b^2}
\tag{6-5-9}
$$
同式（6-5-8）一样，交叉项决定了标准马鞍面相对 $x$、$y$ 轴的偏转量。

除了描述抛物面，正定矩阵（2 阶）还与圆锥曲线（尤其是椭圆）以及曲线变换有着紧密关联。见原书中示例：

![fig_6_4](/Study_Bilibili/figures/fig_6_4.png)

左图所示的椭圆曲线方程 $f_1$ 为 $5x^2+8xy+5y^2 = 1$，将其旋转（去偏转）之后得到右图 $9x^2+y^2 = 1$（记为标准椭圆 $f_0$）。椭圆 $f_1$ 对应于正定矩阵 $\pmb{A}$，曲线方程为 $\pmb{x}^T \pmb{Ax} = 1$:
$$
\pmb{A} =
\begin{bmatrix}
5 & 4\\ 4 & 5
\end{bmatrix}
$$
这里不妨对 $\pmb{A}$ 进行特征分解 $\pmb{A} = \pmb{Q} \pmb{\Lambda} \pmb{Q}^T$，不难得到：
$$
\pmb{A} = \underbrace{
\left( \dfrac{1}{\sqrt{2}}
\begin{bmatrix}
1 & -1\\ 1 & 1
\end{bmatrix} \right)}_{\pmb{Q}} \ \
\underbrace{
\begin{bmatrix}
9 & 0\\ 0 & 1
\end{bmatrix}}_{\pmb{\Lambda}} \ \
\underbrace{
\left( \dfrac{1}{\sqrt{2}}
\begin{bmatrix}
    1 & 1\\ -1 & 1
\end{bmatrix} \right)}_{\pmb{Q}^T}
$$
在矩阵 $\pmb{Q}$ 前增加系数是为了将特征向量单位化。观察 $\pmb{A}$ 的特征值可以发现，$\pmb{\Lambda}$ 即为标准椭圆对应的正定矩阵，其曲线方程为 $\pmb{x}^T \pmb{\Lambda} \pmb{x} = 1$。换句话说，不论椭圆 $f_0$ 如何旋转，只要满足旋转中心为原点且没有拉伸或放缩变换，则任意偏转椭圆 $f_i$ 均可通过相同的特征值矩阵 $\pmb{\Lambda}$ 与标准椭圆 $f_0$ 建立联系。而这种“联系”的具体方式则隐藏在特征向量矩阵 $\pmb{Q}$ 中。我们首先回到二次型 $\pmb{x}^T \pmb{Ax}$ 的展开式：
$$
\pmb{x}^T \pmb{Ax} = 5x^2+8xy+5y^2 = 9 \left( \dfrac{x+y}{\sqrt{2}} \right)^2 + 1 \left( \dfrac{-x+y}{\sqrt{2}} \right)^2
$$
在这种形式下，平方项的系数不再是式（6-5-7）中的主元，而是特征值。平方项中包含了 $\pmb{A}$ 的特征向量 $(1,1)/\sqrt{2}$、$(-1,1)/\sqrt{2}$。观察图像可知，这些特征向量正是偏转椭圆 $f_1$ 的长短轴方向。更进一步地，由于 $\pmb{Q}^T \pmb{Q} = \pmb{Q} \pmb{Q}^T = \pmb{I}$，所以将 $f_1$ 标准化为 $f_0$ 的方法是经过旋转矩阵 $\pmb{Q}$ 的处理将 $\pmb{A}$ 特征对角化：$\pmb{\Lambda} = \pmb{Q}^T \pmb{A} \pmb{Q}$。这种去偏转（或偏转）的过程与平面几何中心旋转的本质是一样的，即 $\pmb{A}$ 经过 $\pmb{Q}$ 处理后得到 $\pmb{\Lambda}$，若再经过 $\pmb{Q}$ 处理就会“继续旋转”到与 $\pmb{A}$ 关于 $y$ 轴对称的 $\pmb{A}^{'}$：
$$
\pmb{A}^{'} = \pmb{Q}^T \left( \pmb{Q}^T \pmb{A} \pmb{Q} \right) \pmb{Q} = \pmb{Q}^T \pmb{\Lambda} \pmb{Q} = \begin{bmatrix}
a & -b\\
-b & c\\
\end{bmatrix}, \ \ \pmb{x}^T \pmb{A}^{'} \pmb{x} = ax^2 - 2bxy + cy^2
\tag{6-5-10}
$$
总结一下椭圆的旋转过程，对于正定矩阵 $\pmb{A} = \pmb{Q} \pmb{\Lambda} \pmb{Q}^T$，其标准化椭圆 $f_0: \pmb{X}^T \pmb{\Lambda} \pmb{X} = 1$ 与偏转椭圆 $f_1: \pmb{x}^T \pmb{Ax} = 1$ 存在以下关系：
$$
\begin{bmatrix}
x & y
\end{bmatrix} \pmb{Q} \pmb{\Lambda} \pmb{Q}^T
\begin{bmatrix}
x\\ y\\
\end{bmatrix} =
\begin{bmatrix}
X & Y
\end{bmatrix} \pmb{\Lambda}
\begin{bmatrix}
X\\ Y\\
\end{bmatrix} = \lambda_1 X^2 + \lambda_2 Y^2 = 1
\tag{6-5-11}
$$
根据解析几何的相关知识，标准椭圆 $f_0$ 的半轴长分别为 $1/\sqrt {\lambda_1}$、$1/\sqrt {\lambda_2}$。

在实际应用中，我们通常的思考（或研究）顺序是从左向右的，即从大量具有分布特征 $\pmb{A}$ 的训练数据（有偏转）中提取共同的特征矩阵 $\pmb{\Lambda}$（无偏转），利用特征向量 $\pmb{Q}$ 将原始数据进行旋转投影（去偏转），从而获得分布相对更为均一（优化后）的数据集。

**对问题 2 的一点补充：**

我们利用 LDLT 分解处理一下矩阵 $\pmb{A}$：
$$
\pmb{A} =
\begin{bmatrix}
a & b\\
b & c\\
\end{bmatrix} = \underbrace{
\begin{bmatrix}
1 & 0\\
b/a & 1
\end{bmatrix}
}_{\pmb{L}} \underbrace{
\begin{bmatrix}
a & 0\\
0 & c-b^2/a
\end{bmatrix}
}_{\pmb{D}} \underbrace{
\begin{bmatrix}
1 & b/a\\
0 & 1
\end{bmatrix}
}_{\pmb{L}^T}
$$
根据正定条件可知，对角阵中的主对角线元素均为正值。这一点与教材上的一个批注有关系：Cholesky 分解（Cholesky Factorization），又称平方根法。对于正定矩阵 $\pmb{A}$，Cholesky 分解可将 $\pmb{A}$ 表示为下三角矩阵 $\pmb{L^{'}}$ 与其转置的乘积。其原理与上述 LDLT 分解有关：考虑到 $\pmb{D}$ 中主对角线元素均为正值，则有 $\pmb{D} = \sqrt{\pmb{D}} \sqrt{\pmb{D}} = \sqrt{\pmb{D}} {\sqrt{\pmb{D}}}^T$，因此有 $\pmb{A} = \pmb{L^{'}} {\pmb{L^{'}}}^T = \left( \pmb{L} \sqrt{\pmb{D}} \right) {\left( \pmb{L} \sqrt{\pmb{D}} \right)}^T$。

**问题 3：**

我们对于正定矩阵的亲睐并不直接来源于其定义或某些衍生性质，而是在解决某些问题的过程中，我们发现了正定矩阵在其中扮演的重要角色，因此对正定矩阵的性质、来源进行了充分研究。这类问题就是最小二乘问题：
$$
\hat{\pmb{x}} = \underset{\pmb{x}} \argmin \left\| \pmb{Ax} - b \right\|_2^2
$$
实际应用中的最小二乘问题，其矩阵 $\pmb{A} \in \mathbb{R}^{m \times n}$ 往往是“长方形”的（$m<n$），因此 $\pmb{A}^{-1}$ 不存在，必须采用投影方法求解。当 $\pmb{A}$ 满秩，即 ${\rm rank}(\pmb{A}) = n$ 时，上述问题有精确最优解 $\hat{\pmb{x}} = \left( \pmb{A}^T \pmb{A} \right)^{-1} \pmb{A}^T \pmb{b}$。注意到其中 $\pmb{A}^T \pmb{A} \in \mathbb{R}^{n \times n}$ 是一个方阵、实对称矩阵，同时还是正定矩阵！我们观察二次型可知：
$$
\pmb{x}^T \left( \pmb{A}^T \pmb{A} \right) \pmb{x} = (\pmb{Ax})^T \pmb{Ax} = \| \pmb{Ax} \|^2 > 0, \ \ \pmb{x} \ne \pmb{0}
\tag{6-5-12}
$$
式（6-5-12）经常出现在各种空间滤波器的目标函数中，其物理含义的解释则往往从能量（范数平方项）入手。

除此之外，我们知道正定矩阵可以对角化表示为 $\pmb{A} = \pmb{Q} \pmb{\Lambda} \pmb{Q}^T$，因此 $\pmb{A}$ 的逆也变得非常简单明了：$\pmb{A}^{-1} = \pmb{Q} \pmb{\Lambda}^{-1} \pmb{Q}^T$。
$$
\pmb{A} \pmb{A}^{-1} = \pmb{Q} \pmb{\Lambda} \pmb{Q}^T \pmb{Q} \pmb{\Lambda}^{-1} \pmb{Q}^T = \pmb{I}, \ \ \Lambda^{-1} = {\rm diag} \left( \dfrac{1}{\lambda_1},\dfrac{1}{\lambda_2},\cdots,\dfrac{1}{\lambda_n} \right)
\tag{6-5-13}
$$
由于正定矩阵性质决定了 $\lambda_1$、$\lambda_2$、$\cdots$、$\lambda_n$ 均大于 0，因此 $\Lambda^{-1}$ 的主对角线元素也大于 0，即 $\pmb{A}^{-1}$ 也是正定矩阵。

最后，当 $\pmb{A}$、$\pmb{B} \in \mathbb{R}^{n \times n}$ 同为正定矩阵时，$\pmb{A} + \pmb{B}$ 依然为正定矩阵。注意这并不是因为 $\pmb{A}+\pmb{B}$ 的特征值为两者特征值之和（特征值不存在这样的线性加和特性），而是根据二次型得来的：
$$
\pmb{x}^T \left( \pmb{A} + \pmb{B} \right) \pmb{x} = \pmb{x}^T \pmb{Ax} + \pmb{x}^T \pmb{Bx} > 0
\tag{6-5-14}
$$
所以，二次型不仅仅是正定特性的一种“曲折性”表达，它还是我们联系线性代数与几何、研究正定矩阵拓展性质的优秀工具。

---

## 6.6 Similar Matrices
### 6.6.1 相似矩阵的特征
相似矩阵的概念源于矩阵对角化 $\pmb{\Lambda} = \pmb{S}^{-1} \pmb{AS}$，其中 $\pmb{\Lambda}$ 为特征值矩阵，$\pmb{S}$ 为特征向量矩阵。不过一个显而易见的问题是，并非所有方阵 $\pmb{A}$ 都能进行如上分解。这些无法特征值对角化的矩阵 $\pmb{A}$ 并没有那么特殊，我们依然可以找到一类矩阵与之构建变换联系。比如 $\pmb{M}^{-1} \pmb{AM}$，其中 $\pmb{M}$ 为任意可逆矩阵，基于如上规则构建的一系列矩阵是**共享特征值**的，即 $\pmb{B} = \pmb{M}^{-1} \pmb{AM}$ 是 $\pmb{A}$ 的相似矩阵。因为 $\pmb{A} = \pmb{MB} \pmb{M}^{-1}$，所以这种相似关系是双向的：
$$
\pmb{Ax} = \lambda \pmb{x} \ \Longrightarrow \ \pmb{MB} \pmb{M}^{-1} \pmb{x} = \lambda \pmb{x} \ \Longrightarrow \ \pmb{B} \left( \pmb{M}^{-1} \pmb{x} \right) = \lambda \left( \pmb{M}^{-1} \pmb{x} \right)
\tag{6-6-1}
$$

对于可对角化的矩阵 $\pmb{X}$ 而言，其最简化相似矩阵即为分解后的对角阵 $\Lambda$，此时 $\pmb{M}$ 即为特征向量矩阵。显然 $\pmb{X}$ 与 $\pmb{\Lambda}$ 将共享特征值。这种通过寻找相似矩阵来简化原矩阵的过程源于微分方程的变量代换：
$$
\dfrac{d\pmb{u}}{dt} = \pmb{Au} \ \overset{\pmb{u} = \pmb{Mv}} \Longrightarrow \ \pmb{M} \dfrac{d\pmb{v}}{dt} = \pmb{AMv} \ \Longrightarrow \ \dfrac{d\pmb{v}}{dt} = \pmb{M}^{-1} \pmb{AMv}
\tag{6-6-2}
$$
通过合适的 $\pmb{M}$，原系统有望转变成另一个数值上更易于求解的新系统。在这种相似变换过程中，矩阵的部分参数发生了改变：特征向量、零空间、列空间、行空间、左零空间，奇异值；

不过，既然二者“相似”，也有一些参数保持不变：特征值、迹、行列式、秩、独立特征向量个数、Jordan 标准型。


### 6.6.2 Jordan 相似形
以如下非零二阶方阵 $\pmb{A}$ 为例：
$$
\pmb{A} =
\begin{bmatrix}
cd & d^2\\
-c^2 & -cd\\
\end{bmatrix}, \ \ \det \left(\pmb{A} \right) = 0, \ \ {\rm rank} \left( \pmb{A} \right) = 1
$$
选择如下形式的矩阵 $\pmb{M}$：
$$
\pmb{M} = 
\begin{bmatrix}
a & b\\
c & d\\
\end{bmatrix}, \ \ \det \left( \pmb{M} \right) = 1
$$
则相似矩阵 $\pmb{B}$ 为：
$$
\pmb{B} = \pmb{M}^{-1} \pmb{A} \pmb{M} = 
\begin{bmatrix}
d & -b\\
-c & a\\
\end{bmatrix}
\begin{bmatrix}
cd & d^2\\
-c^2 & -cd\\
\end{bmatrix}
\begin{bmatrix}
a & b\\
c & d\\
\end{bmatrix} = 
\begin{bmatrix}
0 & 1\\
0 & 0\\
\end{bmatrix}
$$
我们知道，$\pmb{A}$ 是无法对角化的，形式上也不够简洁。而 $\pmb{B}$ 虽然也不是对角阵，但形式上已经非常简单了。事实上，在 $\pmb{A}$ 的所有相似矩阵中，$\pmb{B}$ 是最简单、最接近对角阵的形式，即 Jordan 标准型。严格来说，Jordan 标准型是一种分块对角阵。假设一未标准化的原始矩阵 $\pmb{A} \in \mathbb{R}^{n \times n}$ 共有 $s$ 个独立的特征向量，则 $\pmb{A}$ 的 Jordan 标准型 $\pmb{J} \in \mathbb{R}^{n \times n}$ 由 $s$ 个 Jordan 块组成：
$$
\pmb{J} = \pmb{M}^{-1} \pmb{AM} = 
\begin{bmatrix}
\pmb{J}_1 & \ & \ & \ \\
\ & \pmb{J}_2 & \ & \ \\
\ & \ & \ddots & \ \\
\ & \ & \ & \pmb{J}_s\\
\end{bmatrix}
\tag{6-6-3}
$$
每个 Jordan 块的大小不完全一致，其对角线元素为相等的特征值 $\lambda_i$（重根），每个主对角线元素正上方的元素为 1：
$$
\pmb{J}_i = 
\begin{bmatrix}
\lambda_i & 1 & \ & \ & \ \\
\ & \lambda_i & 1 & \ & \ \\
\ & \ & \ddots & \ddots & \ \\
\ & \ & \ & \lambda_i & 1 \\
\ & \ & \ & \ & \lambda_i \\
\end{bmatrix}
\tag{6-6-4}
$$
对于任意选择的可逆矩阵 $\pmb{M}$，其产生的各种与 $\pmb{A}$ 相似的矩阵，彼此之间都共享同一个 Jordan 标准型。即如果 $\pmb{A}$ 与 $\pmb{B}$ 相似，则两个矩阵的 Jordan 标准型相同，否则二者不相似。

之所以说 Jordan 标准型能够简化运算，是因为 Jordan 矩阵大多为稀疏矩阵，且元素数值相对简单（包含了大量的 1）。例如基于三阶方阵 $\pmb{X}$ 构建的微分方程（~~其实我是先有了 $\pmb{J}$ 才去找 $\pmb{M}$ 的，鬼知道都是整数的可逆矩阵多难找~~），系统初始值为 $[1,2,3]^T$：
$$
\pmb{X} = 
\begin{bmatrix}
0 & 8 & -25\\
-9 & 19 & -43\\
-2 & 3 & -4\\
\end{bmatrix}, \ \ \dfrac{d\pmb{u}}{dt} = \pmb{Xt} = 
\begin{cases}
\dfrac{dx_u}{dt} = 8y_u - 25z_u\\
\ \\
\dfrac{dy_u}{dt} = -9x_u + 19y_u - 43z_u\\
\ \\
\dfrac{dz_u}{dt} = -2x_u + 3y_u - 4z_u\\
\end{cases}
$$
如何化简得到 Jordan 标准型算是一项技术活，在这里不加详细说明地给出这样的神奇矩阵 $\pmb{M}$：
$$
\pmb{J} = \pmb{M}^{-1} \pmb{X} \pmb{M} = 
\begin{bmatrix}
-1 & 2 & -6\\
-1 & 2 & -7\\
1 & -1 & 2\\
\end{bmatrix}
\begin{bmatrix}
0 & 8 & -25\\
-9 & 19 & -43\\
-2 & 3 & -4\\
\end{bmatrix}
\begin{bmatrix}
3 & -2 & 2\\
5 & -4 & 1\\
1 & -1 & 0\\
\end{bmatrix} = 
\begin{bmatrix}
5 & 1 & 0\\
0 & 5 & 1\\
0 & 0 & 5\\
\end{bmatrix}
$$
因此进行变量代换 $\pmb{u} = \pmb{Mv}$，则原方程变为 $\dfrac{d\pmb{v}}{dt} = \pmb{Jv}$：
$$
\begin{cases}
\dfrac{dx_v}{dt} = 5x_v + y_v\\
\ \\
\dfrac{dy_v}{dt} = 5y_v + z_v\\
\ \\
\dfrac{dz_v}{dt} = 5z_v\\
\end{cases} \ \Longrightarrow \ 
\pmb{v} = 
\begin{bmatrix}
x_v(0) & y_v(0) & z_v(0)\\
y_v(0) & z_v(0) & 0\\
z_v(0) & 0 & 0\\
\end{bmatrix}
\begin{bmatrix}
e^{5t}\\
te^{5t}\\
0.5t^2 e^{5t}\\
\end{bmatrix}
$$
因为 $\pmb{J}$ 缺失了 2 个特征向量，所以微分方程的解中产生了 $te^{5t}$、$t^2 e^{5t}$ 两项；因为特征值 5 是三重根，所以存在 $t$、$t^2$ 两个未知量系数。代回原始变量，首先得到变换后的系统初始值与系统解：
$$
\pmb{v}(0) = \pmb{M}^{-1} \pmb{u}(0) = 
\begin{bmatrix}
-1 & 2 & -6\\
-1 & 2 & -7\\
1 & -1 & 2\\
\end{bmatrix}
\begin{bmatrix}
1\\ 2\\ 3\\
\end{bmatrix} = 
\begin{bmatrix}
-15\\ -18\\ 5\\
\end{bmatrix}, \ \ 
\pmb{v} = 
\begin{bmatrix}
-15 & -18 & 5\\
-18 & 5 & 0\\
5 & 0 & 0\\
\end{bmatrix}
\begin{bmatrix}
e^{5t}\\
te^{5t}\\
0.5t^2 e^{5t}\\
\end{bmatrix}
$$
最后得到原始系统解
$$
\pmb{u} = 
\begin{bmatrix}
3 & -2 & 2\\
5 & -4 & 1\\
1 & -1 & 0\\
\end{bmatrix}\pmb{v} = 
\begin{bmatrix}
e^{5t} - 64 te^{5t} + 7.5 t^2 e^{5t}\\
\ \\
2e^{5t} - 110 te^{5t} + 12.5 t^2 e^{5t}\\
\ \\
3e^{5t} - 23 te^{5t} + 2.5 t^2 e^{5t}\\
\end{bmatrix}
$$
除了简化求解微分方程的部分关键步骤，Jordan 标准型还可以像 $\pmb{\Lambda}$ 一样用来求解矩阵幂次的问题。例如对于可对角化方阵 $\pmb{A}$，$\pmb{A}^{100} = \pmb{S} \pmb{\Lambda}^{100} \pmb{S}^{-1}$。类似地，对于不可对角化、但存在 Jordan 标准型 $\pmb{J} = \pmb{M}^{-1} \pmb{AM}$ 的方阵 $\pmb{A}$ 而言，$\pmb{A}^2 = \pmb{MJ} \pmb{M}^{-1} \pmb{MJ} \pmb{M}^{-1} = \pmb{M} \pmb{J}^2 \pmb{M}^{-1}$，同理 $\pmb{A}^{100} = \pmb{M} \pmb{J}^{100} \pmb{M}^{-1}$。或许这才是 Jordan 标准型最重要的意义所在，即使不可对角化，也能通过近似手段达到相近的目的。

---
## 6.7 Singular Value Decomposition (SVD)
### 6.7.1 奇异值分解的概念
SVD 分解是 MIT 18.06 这门课的核心知识点，也是线性代数应用的关键技术之一，在数据加密、图像压缩、信号解码等诸多场景都有重要应用。简单来说，对于一个任意形状的矩阵 $\pmb{A} \in \mathbb{R}^{m \times n}$，${\rm rank}(\pmb{A}) = r$，SVD 将其分解为列向量正交基 $\pmb{u}_i$、奇异值系数 $\pmb{\sigma}_i$ 与行向量正交基 $\pmb{v}_i$ 的乘积：$\pmb{A} \pmb{v}_i = \sigma_i \pmb{u}_i$，其矩阵形式为 $\pmb{A} = \pmb{U \Sigma} \pmb{V}^T$（或 $\pmb{A} = \pmb{U \Sigma} \pmb{V}^{-1}$）。

我们首先以一个非对称矩阵 $\pmb{A}$ 为例：
$$
\pmb{A} = 
\begin{bmatrix}
2 & 2\\
-1 & 1\\
\end{bmatrix}
$$
$$
\pmb{A} 
\begin{bmatrix}
\pmb{v}_1 & \pmb{v}_2
\end{bmatrix} =
\begin{bmatrix}
\sigma_1 \pmb{u}_1 & \sigma_2 \pmb{u}_2
\end{bmatrix} = 
\begin{bmatrix}
\pmb{u}_1 & \pmb{u}_2
\end{bmatrix}
\begin{bmatrix}
\sigma_1 & 0\\
0 & \sigma_2\\
\end{bmatrix}
\tag{6-7-1}
$$
对比 SVD 分解与特征分解可以发现，奇异值对角阵 $\pmb{\Sigma}$ 与特征值对角阵 $\pmb{\Lambda}$ 比较类似，前者主对角线元素为奇异值，后者为特征值。两种分解的关键差别在于 $\pmb{U}$、$\pmb{V}$ 与特征向量矩阵 $S$。由于 ${\rm rank} (\pmb{A}) = 2$ 满秩，所以对角化分解 $\pmb{A} = \pmb{S\Lambda} \pmb{S}^{-1}$ 肯定是存在的，但是 $\pmb{A}$ 并非 Symmetric 矩阵，所以 $\pmb{S}^T \pmb{S} \ne \pmb{I}$。而 SVD 分解要求 $\pmb{U}^T \pmb{U} = \pmb{I}$、$\pmb{V}^T \pmb{V} = \pmb{I}$。我们在列向量层面进行观察，可知有 $\pmb{A} \pmb{v}_i = \sigma_i \pmb{u}_i$，这种表达式的形式也非常接近常规特征值问题。那么有没有办法解决等号两边向量不相同的问题呢？我们不妨考察一下 $\pmb{A}$ 的协方差矩阵：
$$
\begin{align}
\notag
\pmb{A}^T \pmb{A} = \left( \pmb{U} \pmb{\Sigma} \pmb{V}^T \right)^T \pmb{U} \pmb{\Sigma} \pmb{V}^T = \pmb{V} \pmb{\Sigma} \pmb{U}^T \pmb{U} \pmb{\Sigma} \pmb{V}^T = \pmb{V} \pmb{\Sigma}^2 \pmb{V}^T\\
\notag \ \\
\notag
\pmb{A} \pmb{A}^T = \pmb{U} \pmb{\Sigma} \pmb{V}^T \left( \pmb{U} \pmb{\Sigma} \pmb{V}^T \right)^T = \pmb{U} \pmb{\Sigma} \pmb{V}^T \pmb{V} \pmb{\Sigma} \pmb{U}^T = \pmb{U} \pmb{\Sigma}^2 \pmb{U}^T
\end{align}
\tag{6-7-2}
$$
上式与对称矩阵的 $\pmb{Q} \pmb{\Lambda} \pmb{Q}^T$ 分解如出一辙，其本质也是将非对称矩阵 $\pmb{A}$ 替换成了对称矩阵 $\pmb{AA}^T$ 或 $\pmb{A}^T \pmb{A}$。注意到 $\pmb{A}^T \pmb{A}$ 为（半）正定矩阵这一结论几乎是不存在限制条件的（式（6-5-12）），因此这也从侧面体现了 SVD 分解的广泛适用性。

我们回到具体案例 $\pmb{A}$，计算 $\pmb{A}^T \pmb{A}$：
$$
\pmb{A}^T \pmb{A} = 
\begin{bmatrix}
5 & 3\\ 3 & 5\\
\end{bmatrix} = \dfrac{1}{\sqrt{2}}
\begin{bmatrix}
1 & -1\\ 1 & 1\\
\end{bmatrix}
\begin{bmatrix}
8 & 0\\ 0 & 2\\
\end{bmatrix} \dfrac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1\\ -1 & 1\\
\end{bmatrix}
$$
可知奇异值分别为 $\sqrt{8} = 2\sqrt{2}$、$\sqrt{2}$。之后提取特征向量 $\pmb{v}_1$、$\pmb{v}_2$ 并根据方程 $\pmb{A}\pmb{v}_i = \sigma_i \pmb{u}_i$ 计算左奇异矩阵 $\pmb{U}$：
$$
\pmb{A} \pmb{v}_1 = 
\begin{bmatrix}
2 & 2\\ -1 & 1\\
\end{bmatrix}
\begin{bmatrix}
\dfrac{1}{\sqrt{2}}\\
\ \\
\dfrac{1}{\sqrt{2}}\\
\end{bmatrix} = 
\begin{bmatrix}
2\sqrt{2}\\
\ \\
0\\
\end{bmatrix}, \ \
\pmb{u}_1 = 
\begin{bmatrix}
1\\ 0\\
\end{bmatrix}\\
\ \\
\pmb{A} \pmb{v}_2 = 
\begin{bmatrix}
2 & 2\\ -1 & 1\\
\end{bmatrix}
\begin{bmatrix}
\dfrac{-1}{\sqrt{2}}\\
\ \\
\dfrac{1}{\sqrt{2}}\\
\end{bmatrix} = 
\begin{bmatrix}
0\\
\ \\
\sqrt{2}\\
\end{bmatrix}, \ \
\pmb{u}_1 = 
\begin{bmatrix}
0\\ 1\\
\end{bmatrix}
$$
整合已有结果，$\pmb{A}$ 的 SVD 分解形式为：
$$
\pmb{A} = 
\begin{bmatrix}
2 & 2\\
-1 & 1\\
\end{bmatrix} =
\underbrace{
\begin{bmatrix}
    1 & 0\\ 0 & 1\\
\end{bmatrix}}_{\pmb{U}} \ \ 
\underbrace{
\begin{bmatrix}
    2\sqrt{2} & 0\\ 0 & \sqrt{2}\\
\end{bmatrix}}_{\pmb{\Sigma}} \ \
\underbrace{
\begin{bmatrix}
    1/\sqrt{2} & 1/\sqrt{2}\\ -1/\sqrt{2} & 1/\sqrt{2}\\
\end{bmatrix}}_{\pmb{V}^T}
$$

### 6.7.2 SVD 分解的应用
首先我们像 6.5.3 节一样，看看 $\mathbb{R}^2$ 中 SVD 分解对圆锥曲线的作用：

![SVD对圆锥曲线作用](/Study_Bilibili/figures/fig_6_7.png)

矩阵 $\pmb{U}$、$\pmb{V}^T$ 的作用是旋转（坐标轴、正交基旋转）：由于 $\pmb{U}$、$\pmb{V}^T$ 是正交标准化的，所以它们的列向量构成标准正交基（互相垂直且无拉伸），因此几何表现为单纯的旋转；奇异值矩阵 $\pmb{\Sigma}$ 的作用是拉伸（正交基数乘线性变换）：由于 $\pmb{\Sigma}$ 是对角阵，因此不涉及旋转变换，对角线元素的数值即为对应坐标轴放缩变换的尺度。

除了几何解释，SVD 在现实世界中也用相当丰富的应用。当我们在对 $\pmb{A}$ 的 SVD 分解中保留全部的奇异值 $\sigma_i$ 时，这种分解是“无损”的，即我们可以通过 $\pmb{U}$、$\pmb{\Sigma}$ 与 $\pmb{V}^T$ 完全恢复 $\pmb{A}$ 的每一个元素。类似特征值，奇异值的大小在某种程度上也能代表矩阵信息的占比大小：选取前 $n$ 个奇异值以及对应的正交基 $\pmb{u}_i$、$\pmb{v}_i$ 能够在尽可能少地损失原有矩阵数据信息的前提下，得到对 $\pmb{A}$ 的一种同维度近似（或对 $\pmb{A}$ 进行降噪）：
$$
\pmb{A}^{'} = \pmb{U}(:,k) \pmb{\Sigma}(:k,:k) \pmb{V}(:,:k)^T = 
\begin{bmatrix}
\pmb{u}_1 & \pmb{u}_2 & \cdots & \pmb{u}_k
\end{bmatrix}
\begin{bmatrix}
\sigma_1 & \ & \ & \ \\
\ & \sigma_2 & \ & \ \\
\ & \ & \ddots & \ \\
\ & \ & \ & \sigma_k \\
\end{bmatrix}
\begin{bmatrix}
\pmb{v}_1^T \\
\pmb{v}_2^T \\
\vdots \\
\pmb{v}_k^T \\
\end{bmatrix}, \ \ \pmb{A}^{'} \in \mathbb{R}^{m \times n}
\tag{6-7-3}
$$
奇异值有一个诱人的特性：降序衰减幅度非常大。通常前 10% 甚至 1% 的奇异值，其数值之和占比即可超过整体的 99%。当数据信息“规律性”越强，所需的奇异值数目越少。因此在文字索引编排、规则图像压缩等应用场景下，使用 SVD 技术能够达到更好的效果。例如一张大小为 512 $\times$ 256 像素的灰度图片，完整存储/传输这张图片需要记录 $2^{17}$ 个浮点数，以 Python 对浮点数的内存占用标准来看，约 3 MB。看上去似乎不是很大？（其实已经很大了，一般分辨率为 600 ppi 的彩色图片，其大小才会达到 MB 级别）当我们考虑一部时长 1 小时的 30 帧率视频（108000 张图片，316.41 GB）的时候，你还会觉得这种存储方式是合理的吗？事实上，如果现在真有一份这样的视频文件（分辨率约 360p），我估计它的大小可能也就几百甚至几十 MB 吧。

现在考虑两种不同的压缩技术：

（1）第一种俗称“马赛克”，即对 2 $\times$ 2 范围内的 4 个像素点进行平均，相当于使用了更大的像素单元。直观表现就是图片变“糊”了，因为人眼很容易感知到这种新像素单元的变化。让我们再激进一点，对 4 $\times$ 4 范围内的 16 个像素点进行平均，此时的数据压缩比为 16 : 1，这样我们就成功地把一部 316.41 GB 的低清视频压缩成了 18.78 GB 的像素风视频。

（2）第二种即为 SVD 分解。根据式（6-7-3），我们选择 $k=5$，这样 $\pmb{U}(:,:5) \in \mathbb{R}^{256 \times 5}$，$\pmb{V}(:,:5)^T \in \mathbb{R}^{5 \times 512}$，而 $\pmb{\Sigma}$ 就存成 5 个浮点数。在需要读取或展示图像的时候，进行一次 $\pmb{U}(:,5) {\rm diag}(\sigma_1,\cdots,\sigma_5) \pmb{V}(:,:5)^T$ 的矩阵乘法运算即可。注意到此时图片的像素密度完全没有改变（当然图案可能存在一定的畸变），但存储图片所需记录的浮点数由 $2^{17}$ 减少为 3845 个（$< 2^{12}$），数据压缩比约为 34 : 1，不仅在存储容量层面更加友好，在画质上更是遥遥领先普通的马赛克技术。

当然，SVD 分解也不是万用万能的。现在常见的图片格式，如 jpeg、JPEG2000 等，使用的压缩技术都不是 SVD（分别为快速 Fourier 变换、小波变换）。一个重要的原因是 SVD 分解得到的奇异值缺乏明确的物理意义对应，对于一些成分复杂的数据而言，不加人工干预地使用 SVD 分解容易损失重要信息。另一个更为关键的原因是它的数据压缩比还不是足够高。以上述视频文件为例，即使我们令 $k=1$，数据压缩比达到 170 : 1，压缩后的视频大小依然超过了 1 GB，注意这还只是灰度视频！当我们考虑 RGB 编码的彩色视频、考虑增加同步音频轨道、同步字幕输入等等情况下，这份普普通通的低清视频文件的体积还会进一步膨胀，这显然与我们的日常生活经验是相违背的。不过综合来看，SVD 分解集合了几乎所有前述的关键知识点和技巧（矩阵乘法、特征值分解、正定矩阵等），在 MIT-18.06 这门课程中的重要地位依旧是不言而喻的。