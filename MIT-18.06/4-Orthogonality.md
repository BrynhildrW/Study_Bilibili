# 4 Orthogonality
## 4.1 Orthogonality of the Four Subspaces
### 4.1.1 正交向量
在大多数情况下，“正交”是“垂直”的另一种说法。在 $\mathbb{R}^n$ 空间中，两个列向量 $\pmb{x}$、$\pmb{y}$ 正交意味着：
$$
\pmb{x}^T \pmb{y} = 0 \ or \ \left< \pmb{y}, \pmb{x} \right> = 0
\tag{4-1-1}
$$
这一点可以在二维空间中通过**勾股定理**加以验证。对于 $\mathbb{R}^2$ 中由向量 $\pmb{x}$、$\pmb{y}$ 以及 $\pmb{x}+\pmb{y}$ 围成的**直角三角形**，勾股定理告诉我们：
$$
\|\pmb{x}\|^2 + \|\pmb{y}\|^2 = \|\pmb{x}+\pmb{y}\|^2
\tag{4-1-2}
$$
对于向量而言，“长度”的概念可以从几何的角度理解：例如 $\pmb{x}=[x_1,x_2,\cdots,x_n]^T$，从空间原点到坐标点 $(x_1,x_2,\cdots,x_n)$ 的距离为 $\sqrt{x_1^2+x_2^2+\cdots+x_n^2}$，即：
$$
\|\pmb{x}\|^2 = \pmb{x}^T \pmb{x}
\tag{4-1-3}
$$
在目前的知识体系下，这样表示是没有问题的。当然等以后我们接触了矩阵范数的概念时，会对（4-1-3）有一个更为严谨的理解。把上述结果代回（4-1-2）可得：
$$
\pmb{x}^T \pmb{x} + \pmb{y}^T \pmb{y} = (\pmb{x}+\pmb{y})^T(\pmb{x}+\pmb{y}) \ \Longrightarrow \ \pmb{y}^T \pmb{x} + \pmb{x}^T \pmb{y} = 0\\
\ \\
\because \ \pmb{y}^T \pmb{x}=\left(\pmb{x}^T \pmb{y}\right)^T \in \mathbb{R}\\
\ \\
\therefore \ \pmb{y}^T \pmb{x}=\pmb{x}^T \pmb{y} \ \Longrightarrow \ \pmb{x}^T \pmb{y} = 0
$$

### 4.1.2 正交子空间
下图展示了四种子空间的基本信息以及相互关系：

![子空间关系图](/figures/4ss.png)

这张图还描述了两组正交关系，在说明它们之前，我们有必要先了解一下空间正交的含义，对于两个 $n$ 维子空间 $\pmb{S}$、$\pmb{T}$，当我们说“**子空间 $\pmb{S}$、$\pmb{T}$ 二者正交**”时，其含义为：
$$
\pmb{S} \perp \pmb{T} \ \Longrightarrow \ \forall \ \pmb{s} \in \pmb{S}, \pmb{t} \in \pmb{T}, \ \pmb{s}^T \pmb{t} \ or \ \pmb{t}^T \pmb{s} = 0
\tag{4-1-4}
$$
向量空间的“正交”与我们在欧式几何中谈论的“垂直”并不完全一致。譬如三维笛卡尔坐标系中的 $xOy$ 平面与 $yOz$ 平面。这两个平面在立体几何的概念里是**垂直**的；在线代领域里，它们都是 $\mathbb{R}^3$ 的子空间。然而根据（4-1-4）的定义，两者并不**正交**。很显然位于 $y$ 轴上的向量同时属于两个平面，却相互平行而非垂直。需要注意的是，问题的根源并不是二者有交集，而是它们的交集包含了非零元素。换个例子，我们知道 $x$ 轴和 $y$ 轴是垂直且正交的，二者的唯一交集就是零元（原点）。更一般地，**如果子空间 $\pmb{S}$ 与 $\pmb{T}$ 存在非零元的交集，则二者必不可能正交**。

接下来我们说明图中的正交关系。首先是行空间与零空间，二者不仅正交，还将母空间 $\mathbb{R}^n$ 划分成了两个**互补**空间，即：
$$
\pmb{C}\left(\pmb{A}^T\right), \pmb{N}\left(\pmb{A}\right) \subseteq \mathbb{R}^n, \ \pmb{C}\left(\pmb{A}^T\right)=\pmb{N}\left(\pmb{A}\right)^{\perp}
\tag{4-1-5}
$$
我们先从零空间入手，看看它们正交的原因。对于线性方程组 $\pmb{Ax}=\pmb{0}$ ：
$$
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n}\\
a_{21} & a_{22} & \cdots & a_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
a_{m1} & a_{m2} & \cdots & a_{mn}\\
\end{bmatrix}
\begin{bmatrix}
x_1\\ x_2\\ \vdots\\ x_n\\
\end{bmatrix} = 
\begin{bmatrix}
0\\ 0\\ \vdots\\ 0\\
\end{bmatrix} \ \Longrightarrow \
\begin{cases}
\pmb{a}_1^T \pmb{x} = 0\\
\pmb{a}_2^T \pmb{x} = 0\\
\vdots\\
\pmb{a}_n^T \pmb{x} = 0\\
\end{cases}
$$
其中 $\pmb{a}_i^T$ 表示 $\pmb{A}$ 中的第 $i$ 行向量。这个方程组说明零空间中的**任意向量** $\pmb{x}$ 都与 $\pmb{A}$ 的**每个行向量**正交，进而满足（4-1-4）提出的定义，即有 $\pmb{C}\left(\pmb{A}^T\right) \perp \pmb{N}\left(\pmb{A}\right)$。至于正交补，是因为零空间与列空间的维度之和为母空间维度，且二者的交集只包含零元。举例来说，对于 $\mathbb{R}^3$ 中的直线簇 $\pmb{L}$：
$$
\pmb{L} = 
\begin{bmatrix}
1 & 2 & 5\\
2 & 4 & 10\\
\end{bmatrix}
$$
$\pmb{C}(\pmb{L}^T)$ 表示经过原点与 $(1,2,5)$ 的直线，$dim\left[\pmb{C}(\pmb{L}^T)\right]=rank(\pmb{L}^T)=1$，$dim\left[\pmb{N}(\pmb{L})\right]=2$。更具体地：
$$
\pmb{N}(\pmb{L}) = k_1 
\begin{bmatrix}
-2\\ 1\\ 0\\
\end{bmatrix} + k_2
\begin{bmatrix}
-5\\ 0\\ 1\\
\end{bmatrix} = k_1 \pmb{x}_1 + k_2 \pmb{x_2}, \ k_1,k_2 \in \mathbb{R}
$$
可见 $\pmb{N}(\pmb{L})$ 的几何表现为一个平面。根据立体几何的相关知识，$\pmb{x}_1$ 与 $\pmb{x}_2$ 是平面 $\pmb{N}(\pmb{L})$ 上的非共线向量。而方向向量 $[1,2,5]^T$ 经计算验证可知与 $\pmb{x}_1$、$\pmb{x}_2$ 均垂直，因此 $\pmb{C}(\pmb{L}^T) \perp \pmb{N}(\pmb{L})$。更进一步地，$\pmb{C}(\pmb{L}^T)$ 与 $\pmb{N}(\pmb{L})$ 相交于原点，二者空间相加 $\pmb{C}(\pmb{L}^T) + \pmb{N}(\pmb{L})$ 的几何含义是：以 $\pmb{C}(\pmb{L}^T)$ 为中心轴，平移平面 $\pmb{N}(\pmb{L})$ 所形成的立体空间。根据（3.7.1）的结论，相加空间依然是线性子空间。显然这个子空间与 $\pmb{R}^3$ 是等价的，即 $\pmb{C}(\pmb{L}^T)$ 是 $\pmb{N}(\pmb{L})$ 的正交补，反之亦然。

以上分析过程同样适用于 $\pmb{C}(\pmb{A})$ 与 $\pmb{N}\left(\pmb{A}^T\right)$，即 $\pmb{C}(\pmb{A}) = \pmb{N}\left(\pmb{A}^T\right)^{\perp}$，证明过程在此不再赘述。

## 4.2 Projections
### 4.2.1 平面投影
在二维笛卡尔坐标系中，假设有如下图所示的点 $A$、$B$，现需要在射线 $OA$ 寻找一点 $P$ 使得 $BP$ 长度最短。即寻找向量 $\vec{OB}$ 在 $\vec{OA}$ 方向上的投影 $\vec{OP}$：

![二维投影示意图](/figures/2-d%20projection.png)

在中学数学中，这顶多只能算是课堂开场白难度的问题，我们可以很熟练地用三角函数把它从里到外分析得清清楚楚。但是线性代数追求的就是公式上的极简风格，哪怕能少写几个字母都是快乐的。这种极简主义看似抽象，但在从低维向高维延申拓展时会有奇效。

从向量的视角来看：
$$
\vec{OB} + \vec{BP} = \vec{OP} \ \Longrightarrow \ \vec{BP} = \vec{OP} - \vec{OB}
$$
简单起见，我们令 $\vec{OA}=\pmb{a}$、 $\vec{OB}=\pmb{b}$、$\vec{OP}=\pmb{p}$、$\vec{BP} = \pmb{e}$，则有 $\pmb{e} \perp \pmb{a}$。根据上一节的定义，有：
$$
\pmb{a}^T \pmb{e}=0 \ \Rightarrow \ \pmb{a}^T(\pmb{b}-\pmb{p}) = 0
$$
由于 $\pmb{p}$ 与 $\pmb{a}$ 共线，则二者必然存在倍数关系 $x$（$x \in \mathbb{R}$）使得 $\pmb{p} = \pmb{a}x$，进而上述正交关系可以表示为：
$$
\pmb{a}^T (\pmb{b} - \pmb{a} x) = 0
\tag{4-2-1}
$$
由于 $\pmb{a}$、$\pmb{b}$ 都是一维向量（序列），系数 $x$ 很容易化简求解：
$$
x = \dfrac{\pmb{a}^T \pmb{b}}{\pmb{a}^T \pmb{a}}
\tag{4-2-2}
$$
同时投影过程可以表示为：
$$
\pmb{p}=\pmb{a}x \ \Longrightarrow \ \pmb{p} = \dfrac{\pmb{a}\pmb{a}^T}{\pmb{a}^T\pmb{a}} \pmb{b}
$$
至此，我们得以走向最终解决问题所需的“总抓手”——投影矩阵 $\pmb{P}=\dfrac{\pmb{a}\pmb{a}^T}{\pmb{a}^T\pmb{a}}$。之所以说它是“抓手”，是因为我们在已知投影方向 $\pmb{a}$ 和待投影目标 $\pmb{b}$ 的情况下，最需要的不是投影结果 $\pmb{p}$，而是一个能够帮助我们稳定获取 $\pmb{p}$ 的工具，即投影矩阵 $\pmb{P}$。之后不论遇到怎么样的 $\pmb{b}$，都可以方便快捷地获取投影结果 $\pmb{p}=\pmb{Pb}$。

投影结果 $\pmb{p}$ 与待投影向量 $\pmb{b}$ 是同步变化的，然而不论投影方向向量 $\pmb{a}$ 在长度上作何变化（非零），投影结果 $\pmb{p}$ 都不会改变，也就是说投影矩阵 $\pmb{P}$ 没有发生变化。因此我们可以作一个合理推测：尽管 $\pmb{P}$ 的计算公式由 $\pmb{a}$ 组成，但其本质上是由 $\pmb{a}$ 的单位方向向量 $\pmb{i}_{\pmb{a}}$ 决定的。

将上述结果从 $\mathbb{R}^2$ 稍稍拓展至 $\mathbb{R}^n$，可以得到类似的结果，即 $\pmb{P}=\dfrac{\pmb{a}\pmb{a}^T}{\pmb{a}^T\pmb{a}} \in \mathbb{R}^{n \times n}$。除此之外，我们还能得出以下结论：

（1）由于 $\pmb{a}$ 可视为秩 1 矩阵，因此 $rank(\pmb{a}\pmb{a}^T)=1$，即 $\pmb{P}$ 也是秩 1 矩阵，$\pmb{C}(\pmb{P})$ 是 $\pmb{a}$ 方向的射线簇；

（2）$\pmb{P}$ 是 *Hermitte* 矩阵，在实数范围内有 $\pmb{P} = \pmb{P}^T$，进一步地，**投影矩阵都是对称矩阵**；

（3）对投影结果 $\pmb{p}$（或 $\pmb{Pb}$）再投影一次，结果不变。因此 $\pmb{P}$ 是幂等矩阵，即 $\pmb{PP}=\pmb{P}$：
$$
\pmb{PP} = \dfrac{\pmb{a}\left(\pmb{a}^T\pmb{a}\right)\pmb{a}^T} {\left(\pmb{a}^T\pmb{a}\right)\pmb{a}^T\pmb{a}} = \dfrac{\pmb{a}\pmb{a}^T}{\pmb{a}^T\pmb{a}} = \pmb{P}
$$
在了解了投影的过程之后，我们有必要从一个更深层次的角度思考投影的本质。$\pmb{Pb}$ 与 $\pmb{b}$ 的关系不仅仅是直角边与斜边、$cos\theta$、光学投射等等表象性质的关系，**$\pmb{Pb}$ 是 $\pmb{b}$ 在 $\pmb{a}$ 方向上的最佳逼近之一**。

说“之一”，是因为在不同误差标准的评判尺度下，最佳结果可能不止一个；说“最佳”，是因为我们当前对于误差的定义是误差向量的 2 范数平方 $\|\pmb{Pb}-\pmb{b}\|_2^2$，通俗地说即为向量长度的平方。而在 $\pmb{a}$ 上取其它任意投影点 $P_i$ 构成的投影向量 $\pmb{p}_i$，均有 $\|\pmb{p}_i-\pmb{b}\|_2^2 > \|\pmb{Pb}-\pmb{b}\|_2^2$。从平面几何的角度很好理解：$\|\pmb{Pb}-\pmb{b}\|_2$ 是垂线段 $BP$ 的长度，$\|\pmb{p}_i-\pmb{b}\|_2$ 是直角三角形 $P_iBP$ 的斜边，斜边自然是长于直角边的。

### 4.2.2 空间投影
当我们把问题的视角从平面向量升格至多维空间时，遵循 $\mathbb{R}^2$ 中的分析方法可以获得形式上相似的结果。我们先从稍微简单的情景入手，例如 $\mathbb{R}^3$ 中的向量 $\pmb{b}$ 至平面 $A$ 上的投影：

![三维投影示意图](/figures/3-d%20projection.png)

类似二维平面，我们同样可以在三维空间中寻找 $\pmb{b}$ 的末端点至 $\pmb{A}$ 的垂线，连接 $\pmb{b}$ 的起点与垂足形成投影向量 $\pmb{p}$ 以及误差向量 $\pmb{e}$。尽管我个人非常喜欢这种几何学的脑内风暴，但是我们还是要了解线代的描述语言。本例与上一节案例的最大差别在于目标向量 $\pmb{a}$ 变成了平面 $A$，所以我们需要换一种线代语言来描述一个平面，基向量组（矩阵）就是一个很好的方法。

假设 $A$ 的基（列）向量为 $\pmb{a}_1$、$\pmb{a}_2$，构造矩阵 $\pmb{A}=[\pmb{a}_1,\pmb{a}_2]$，则 $\pmb{A}$ 的列向量生成空间（或列空间）即为平面 $A$、$A$ 上的任意向量都可以用 $\pmb{a}_1$、$\pmb{a}_2$ 的线性组合表示：
$$
\begin{align}
\notag
\pmb{p} &= \pmb{a}_1x_1+\pmb{a}_2x_2 =
\begin{bmatrix}
    \pmb{a}_1 & \pmb{a}_2\\
\end{bmatrix}
\begin{bmatrix}
    x_1\\ x_2\\
\end{bmatrix} = \pmb{Ax}\\
\notag
\pmb{e} &= \pmb{b}-\pmb{p}=\pmb{b}-\pmb{Ax}
\end{align}
$$ 
误差向量 $\pmb{e}$ 垂直于平面 $A$，所以 $\pmb{e}$ 正交于平面上任意向量，即正交于平面的两个基向量：
$$
\begin{cases}
\pmb{a}_1^T \left(\pmb{b}-\pmb{Ax} \right) = 0\\
\pmb{a}_2^T \left(\pmb{b}-\pmb{Ax}  \right) = 0\\
\end{cases} \ \Rightarrow \
\begin{bmatrix}
\pmb{a}_1^T\\ \pmb{a}_2^T
\end{bmatrix} \left(\pmb{b}-\pmb{Ax} \right) = 
\begin{bmatrix}
0\\ 0
\end{bmatrix} \ \Rightarrow \
\pmb{A}^T \left(\pmb{b}-\pmb{Ax} \right) = \pmb{0}
\tag{4-2-3}
$$
（4-2-3）与（4-2-1）简直如出一辙对吧，这就是极简风格的魅力所在。同时，我们还能发现一个小彩蛋：$\pmb{b}-\pmb{Ax} \in \pmb{N}\left(\pmb{A}^T\right)$，而误差向量 $\pmb{b}-\pmb{Ax}$ 与平面垂直，即正交于平面列空间 $\pmb{C}(\pmb{A})$。这一点与 4.1.2 图片中的空间正交关系是对应的。类似地，我们可以解出投影矩阵 $\pmb{P}$：
$$
\begin{align}
\notag
\pmb{A}^T \left(\pmb{b}-\pmb{Ax} \right) = \pmb{0} \ &\Longrightarrow \ \pmb{x} = \left(\pmb{A}^T \pmb{A} \right)^{-1} \pmb{A}^T \pmb{b}\\
\notag \ \\
\notag \pmb{p} = \pmb{Ax} = \pmb{Pb} \ &\Longrightarrow \ \pmb{P} = \pmb{A} \left(\pmb{A}^T \pmb{A} \right)^{-1} \pmb{A}^T
\end{align}
\tag{4-2-4}
$$
如果我们进一步拓展维度，将 $\pmb{b}$ 升格为矩阵空间 $\mathbb{R}^{n \times n}$ 中的矩阵 $\pmb{B} \in \mathbb{R}^{b_1 \times b_2}$（$b_1,b_2 \leqslant n$），平面 $A$ 升格为子空间 $\pmb{A} \in \mathbb{R}^{a_1 \times a_2}$（$a_1,a_2 \leqslant n$），投影矩阵的形式依然为（4-2-4）所示，投影结果升格为 $\pmb{PB}$。同理可以证明空间投影中的矩阵 $\pmb{P}$ 依然满足 *Hermitte* 矩阵、幂等矩阵的性质，且 **$\pmb{PB}$ 是 $\pmb{B}$ 在 $\pmb{A}$ 空间上的最佳逼近之一**。

最后我们不得不面临的一个问题就是：为什么（或者说什么时候）需要投影？这个问题的答案其实与线性方程组的近似解有关。根据列空间的相关知识我们知道，$\pmb{Ax}=\pmb{b}$ 并非总是有解的：对于超定系统，大多数情况下线性方程组无解，可超定系统却是现实物理世界中遍布各地的存在。例如描述某个物理过程的理论模型设计了 5 个参数，而由于测量误差等干扰因素的存在，实际测试往往需要重复成百上千次，此时方程数量远多于未知数。对于随机系统，方程系数矩阵往往还是满秩的。除了删减优化样本（方程个数），最常用、也是最合理的办法就是**投影**，即“最小二乘逼近法”，相关内容将在下一节展开。

## 4.3 Least Squares Approximations
### 4.3.1 最小二乘回归
继续上一节末尾的问题，投影是解决超定系统方程组的重要手段之一。对于 $\pmb{Ax}=\pmb{b}$，我们知道 $\pmb{Ax}$ 总在 $\pmb{C}(\pmb{A})$ 中，而 $\pmb{b}$ 不一定。只要我们对 $\pmb{b}$ 进行微调，使得 $\hat{\pmb{b}} \approx \pmb{b}$ 且 $\hat{\pmb{b}} \in \pmb{C}(\pmb{A})$，则方程组 $\pmb{A}\hat{\pmb{x}}=\hat{\pmb{b}}$ 有解。微调的结果就是将 $\pmb{b}$ 替换成其在列空间 $\pmb{C}(\pmb{A})$ 上的投影，即最小二乘法。这里我们以一个简单的直线回归问题作为案例：

![最小二乘拟合直线示意图](/figures/OLS.png)

假设 $\{B_1,B_2,B_3\}$ 为数据采样点集：
$$
\begin{cases}
B_1: (1,1)\\ B_2: (2,2)\\ B_3: (3,2)
\end{cases} \ \Longrightarrow \
y = kx + b, \ 
\begin{cases}
k + b = 1\\
2k + b = 2\\
3k + b = 2\\
\end{cases}\\
\ \\
\begin{bmatrix}
1 & 1\\
2 & 1\\
3 & 1\\
\end{bmatrix}
\begin{bmatrix}
k\\ b\\
\end{bmatrix} = 
\begin{bmatrix}
1\\ 2\\ 2\\
\end{bmatrix} \ \Longrightarrow \ \pmb{Ax}=\pmb{b}
$$
显然没有任何一条直线能够恰好同时经过这三个点，即 $\pmb{Ax}=\pmb{b}$ 无解。这里尤其要注意 $\pmb{x}$ 与 $x$、$\pmb{b}$ 与 $b$没有关联，单纯只是用惯了这些字母而已。我们需要拟合一条直线，其表达式为 $y = \hat{k}x+ \hat{b}$（虚线所示），使得在对应横坐标下的采样点 $\{P_1,P_2,P_3\}$ 与原始数据的**误差平方和** $e$ 最小：
$$
e = e_1^2 + e_2^2 + e_3^2 = \left(\hat{k} + \hat{b} - 1 \right)^2 + \left(2\hat{k} + \hat{b} - 2 \right)^2 + \left(3\hat{k} + \hat{b} - 2 \right)^2
$$
这里我们规定向上为正，向下为负（即拟合点减去实际点的偏差）。将误差表示为向量，并将一切都用线代语言翻译一下：
$$
\begin{bmatrix}
\hat{k}\\ \hat{b}\\
\end{bmatrix} = 
\hat{\pmb{x}} = \underset{\pmb{x}} \argmin \|\pmb{e}\|^2
= \underset{\pmb{x}} \argmin \|\pmb{Ax}-\pmb{b}\|^2
\tag{4-3-1}
$$
我们不妨先通过我们熟悉的微积分来考虑总体误差 $e$：
$$
f(\hat{k},\hat{b}) = e = 14\hat{k}^2 + 12\hat{k}\hat{b} + 3\hat{b}^2 - 22\hat{k} -10\hat{b} + 9\\
\ \\
\begin{cases}
\dfrac{\partial f}{\partial \hat{k}} = 28\hat{k} + 12\hat{b} - 22 = 0\\
\ \\
\dfrac{\partial f}{\partial \hat{b}} = 6\hat{b} + 12\hat{k} - 10 = 0\\
\end{cases} \ \Longrightarrow \
\begin{cases}
14\hat{k} + 6\hat{b} = 11\\
\ \\
6\hat{k} + 3\hat{b} = 5\\
\end{cases} \ \Longrightarrow \
\begin{cases}
\hat{k} = \dfrac{1}{2}\\
\ \\
\hat{b} = \dfrac{2}{3}\\
\end{cases}
$$
因此拟合直线为 $y = \dfrac{1}{2} x + \dfrac{2}{3}$，拟合采样点的纵坐标依次为 $\dfrac{7}{6}$、$\dfrac{5}{3}$、$\dfrac{13}{6}$，对应误差依次为 $-\dfrac{1}{6}$、$\dfrac{2}{6}$、$-\dfrac{1}{6}$。
接下来看看线代如何解决以上问题：
$$
\begin{align}
\notag \pmb{A}\hat{\pmb{x}} &= \pmb{b}\\
\notag \ \\
\notag \pmb{A}^T\pmb{A} \hat{\pmb{x}} &= \pmb{A}^T \pmb{b}\\
\end{align} \ \Longrightarrow \
\begin{bmatrix}
14 & 6\\ 6 & 3\\
\end{bmatrix}
\begin{bmatrix}
\hat{k}\\ \hat{b}
\end{bmatrix} = 
\begin{bmatrix}
11\\ 5\\
\end{bmatrix}
$$
至此我们已经得到了与微积分方法相同的二元一次方程组，我们甚至可以一步得到 $\hat{\pmb{x}} = \left(\pmb{A}^T \pmb{A} \right)^{-1} \pmb{A}^T \pmb{b}$。根据上一节的知识，我们知道 $\pmb{A} \hat{\pmb{x}}$ 是 $\pmb{b}$ 在空间 $\pmb{A}$ 上的投影。这里如果你深想了，但没有深透的话，会发现一件非常“诡异”的事：

我们知道 $\pmb{Ax}=\pmb{b}$ 是无解的，事实上 $\pmb{A}\hat{\pmb{x}} \ne \pmb{b}$。但我们是从 $\pmb{A}^T\pmb{A} \hat{\pmb{x}} = \pmb{A}^T \pmb{b}$ 求解得到的 $\hat{\pmb{x}}$，为什么从这一步逆向回到 $\pmb{A}\hat{\pmb{x}}=\pmb{b}$ 就行不通了呢？

（~~我强烈建议各位读者不要急着浏览接下来的部分。尽管这段推理本身无足轻重，但是费尽心机解开一个 bug 之后，我的多巴胺释放水平显著提升了，本着公平起见的原则，大伙也痛苦一下吧。~~）

这个问题可以从两种角度解释。

（1）首先是比较本质、抽象一点的方法：好比 $\pmb{DAB}=\pmb{DC}$，得到 $\pmb{D}(\pmb{AB}-\pmb{C})=\pmb{0}$ 是没问题的，但不能进一步等价于 $\pmb{AB}-\pmb{C} = \pmb{0}$，更不能等价于 $\pmb{AB}=\pmb{C}$。原因很简单，各位难道没见过 $\pmb{Ax}=\pmb{0}$ 有多个解的情况吗？$\pmb{AB}-\pmb{C} \in \pmb{N}(\pmb{D})$，只要矩阵 $\pmb{D}$ 不满秩，零空间就必然包含除零元以外的其它向量元素，换句话说 $\pmb{AB}-\pmb{C}$ 有无数个也毫不奇怪。由已知条件有：
$$
\pmb{A}^T (\pmb{A}\hat{\pmb{x}}-\pmb{b})=\pmb{0} \ \Longrightarrow \ \pmb{A}\hat{\pmb{x}}-\pmb{b} \in \pmb{N} \left(\pmb{A}^T \right)
$$
而 $dim \left[\pmb{N} \left(\pmb{A}^T \right) \right]=1$，即左零空间包含无限多个元素，因此存在不同的 $\pmb{b}$ 使得上式成立是正常情况。自然地，不同的 $\pmb{b}$ 可能对应不同的解 $\hat{\pmb{x}}$（或者无解）。

（2）其次是更表层、直观一点的方法：正因为 $\pmb{Ax}=\pmb{b}$ 无解，我们才需要把 $\pmb{b}$ 微调成 $\hat{\pmb{b}}$，在这个过程中：
$$
\begin{align}
\notag
\pmb{A}^T\pmb{b} &= 
\begin{bmatrix}
    1 & 2 & 3\\
    1 & 1 & 1\\
\end{bmatrix}
\begin{bmatrix}
    1\\ 2\\ 2\\
\end{bmatrix} = 
\begin{bmatrix}
    11\\ 5\\
\end{bmatrix}\\
\notag \ \\
\notag
\pmb{A}^T\hat{\pmb{b}} &= 
\begin{bmatrix}
    1 & 2 & 3\\
    1 & 1 & 1\\
\end{bmatrix}
\begin{bmatrix}
    \dfrac{7}{6}\\ \ \\ \dfrac{5}{3}\\ \ \\ \dfrac{13}{6}\\
\end{bmatrix} = 
\begin{bmatrix}
    11\\ 5\\
\end{bmatrix}\\
\end{align}
$$
什么叫微调？这就叫微调。尽管 $\pmb{Ax}=\pmb{b}$ 确实无解，但还是可以通过 $\pmb{A}^T\pmb{Ax}=\pmb{A}^T\pmb{b}=\pmb{A}^T\hat{\pmb{b}}$ 抢救一下，在等式维持成立的情况下对右侧进行调整，最终 $\pmb{A}\hat{\pmb{x}}=\hat{\pmb{b}}$ 成功化险为夷。
最后总结一下，当超定系统线性方程组 $\pmb{Ax}=\pmb{b}$ 无解时，可以尝试将 $\pmb{b}$ 投影至 $\pmb{A}$ 的列空间中，从而使得原方程有解。特别地，投影矩阵为 $\pmb{P} = \pmb{A} \left(\pmb{A}^T\pmb{A} \right)^{-1}\pmb{A}^T$。

### 4.3.2 需要补充的一些特性
对于 $\pmb{Ax}=\pmb{b}$，有投影矩阵 $\pmb{P} = \pmb{A} \left(\pmb{A}^T\pmb{A} \right)^{-1} \pmb{A}^T$。通过对 $\pmb{b}$ 投影（$\pmb{Pb}$）可将其投射至 $\pmb{A}$ 的列空间 $\pmb{C}(\pmb{A})$，这里存在两种特殊情况：

（1）若 $\pmb{b} \in \pmb{C}(\pmb{A})$，则原方程有解，投影没有实际意义，即有 $\pmb{Pb}=\pmb{b}$：

$$
\pmb{b} = \pmb{Ay}, \ \Longrightarrow \ \pmb{Pb} = \pmb{A} \left(\pmb{A}^T\pmb{A} \right)^{-1} \pmb{A}^T \pmb{Ay} = \pmb{Ay} = \pmb{b}
$$

根据上一节的分析，我们需要注意 $\pmb{Pb}=\pmb{b}$ 并不意味着 $\pmb{P}=\pmb{I}$；

（2）若 $\pmb{b} \in \pmb{N}\left(\pmb{A}^T\right)$（或 $\pmb{b} \perp \pmb{C}(\pmb{A})$），则有 $\pmb{Pb}=\pmb{0}$，即投影过程会使得 $\pmb{b}$ 完全衰减：

$$
\pmb{A}^T\pmb{b} = \pmb{0} \ \Longrightarrow \ \pmb{Pb} = \pmb{A} \left(\pmb{A}^T\pmb{A} \right)^{-1} \pmb{A}^T \pmb{b} = \pmb{0}
$$

通常情况下，$\pmb{b}$ 中既有属于列空间的成分，也有属于左零空间的成分。因此投影过程能够削减一部分无关分量，保留有效成分，这一点在信号处理过程中经常使用。

最后是一点关于误差向量的内容。对于原始向量 $\pmb{b}$，通过投影我们将其分解为 $\pmb{p} + \pmb{e}$。其中误差（属于左零空间）与投影分量（属于列空间）是正交的，即 $\pmb{e} \perp \pmb{p}$。换个角度看，其实误差向量也是某种投影，其目标空间是左零空间，即与目标空间正交的子空间。因此误差向量也存在投影矩阵 $\pmb{I}-\pmb{P}$，且该矩阵同样是 *Hermitte* 矩阵，满足幂等性质。

## 4.4  Orthogonal Basis and Gram-Schmidt
### 4.4.1 正交基与正交矩阵
正交基与正交向量组的概念很接近，其差异仅在于正交基是单位化的。即对于一组正交基 $\pmb{q}_1$、$\pmb{q}_2$、$\cdots$、$\pmb{q}_n$，它们满足以下数学关系：
$$
\pmb{q}_i^T \pmb{q}_j = 
\begin{cases}
0, \ i \ne j\\
1, \ i = j
\end{cases}
\tag{4-4-1}
$$
由正交基向量组成的矩阵满足 $\pmb{Q}^T \pmb{Q} = \pmb{I}$。当 $\pmb{Q}$ 为方阵时，还有  $\pmb{Q}^T = \pmb{Q}^{-1}$，这样的 $\pmb{Q}$ 称为正交矩阵。

接下来展示一些典型正交矩阵的示例，如 $n$ 阶方阵的置换矩阵、如下所示的 $\pmb{Q}_1$、$\pmb{Q}_2$：
$$
\pmb{Q}_1 = 
\begin{bmatrix}
cos \theta & -sin \theta\\
sin \theta & cos \theta\\
\end{bmatrix}, \ \pmb{Q}_2 = \dfrac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1\\ 1 & -1\\
\end{bmatrix}
$$
这两个矩阵的示例可不是随便举的：$\pmb{Q}_1$ 又称旋转矩阵，对于任何二维向量 $\pmb{x}$，$\pmb{Q}_1 \pmb{x}$ 的几何表现为将其**逆时针旋转** $\theta$ 角度。事实上，$\pmb{Q}_1$ 的列向量就是由 $\pmb{i}$、$\pmb{j}$ 旋转而来。$\pmb{Q}_1^T=\pmb{Q}_1^{-1}$ 与之相反，其作用是将向量**顺时针旋转** $\theta$ 角度。其原理很简单，从数形结合的角度来看，$\pmb{Q}_1^{-1}$ 的作用与 $\pmb{Q}_1$ 应当是相互抵消的。

而关于 $\pmb{Q}_2$，*Hadamard* 基于分块矩阵的思想对其进行了扩展：
$$
\begin{align}
\notag
\pmb{Q}_3 &= \dfrac{1}{2}
\begin{bmatrix}
    1 & 1 & 1 & 1\\
    1 & -1 & 1 & -1\\
    1 & 1 & -1 & -1\\
    1 & -1 & -1 & 1\\
\end{bmatrix} = \left(\dfrac{1}{\sqrt{2}} \right)^2
\begin{bmatrix}
    \pmb{Q}_2 & \pmb{Q}_2\\
    \pmb{Q}_2 & -\pmb{Q}_2\\
\end{bmatrix}\\
\notag \ \\
\notag
\pmb{Q}_4 &= \dfrac{1}{2 \sqrt{2}}
\begin{bmatrix}
    \pmb{Q}_2 & \pmb{Q}_2 & \pmb{Q}_2 & \pmb{Q}_2\\
    \pmb{Q}_2 & -\pmb{Q}_2 & \pmb{Q}_2 & -\pmb{Q}_2\\
    \pmb{Q}_2 & \pmb{Q}_2 & -\pmb{Q}_2 & -\pmb{Q}_2\\
    \pmb{Q}_2 & -\pmb{Q}_2 & -\pmb{Q}_2 & \pmb{Q}_2\\
    \end{bmatrix} = \left(\dfrac{1}{\sqrt{2}} \right)^3
    \begin{bmatrix}
    \pmb{Q}_3 & \pmb{Q}_3\\
    \pmb{Q}_3 & -\pmb{Q}_3\\
\end{bmatrix}\\
\end{align}
$$
诸如此类的方阵都是正交矩阵。*Hadamard* 矩阵的数学定义是“一个仅由 1 和 -1 组成的正交矩阵序列”，且序列中的相邻矩阵存在递归关系。*Hadamard* 矩阵常用于纠错码，如 *Reed-Muller* 码等。

正交矩阵的神奇性质之一表现在投影上，即正交投影。假设我们需要将向量 $\pmb{x} \in \mathbb{R}^{n \times 1}$ 投影至**具有正交列向量组**的矩阵 $\pmb{Q}$ 的列空间 $\pmb{C}(\pmb{Q})$ 中，根据已有知识我们知道投影矩阵为 $\pmb{P} = \pmb{Q} \left(\pmb{Q}^T\pmb{Q} \right)^{-1}\pmb{Q}^T$。而 $\pmb{Q}$ 满足 $\pmb{Q}^T\pmb{Q}=\pmb{I}$，即正交投影矩阵的形式可简化为 $\pmb{Q}\pmb{Q}^T$。

注意这一结论的前提是 $\pmb{Q}$ 仅需满足具有正交列向量组即可，当 $\pmb{Q}$ 为方阵（正交矩阵）时，$\pmb{Q}\pmb{Q}^T=\pmb{I}$，即正交投影没有实质性作用。这是因为 $\pmb{x}$ 的维度与 $\pmb{C}(\pmb{Q})$ 的维度相同，而 $\pmb{C}(\pmb{Q})$ 本质上等价于 $\mathbb{R}^n$，把向量投影至原空间当然不需要额外的任何操作，单位阵足矣。

正交矩阵的第二性质是范数不变性，严格来说是向量的长度不变，即 $\|\pmb{Qx}\|_2^2 = \|\pmb{x}\|_2^2$：
$$
\|\pmb{Qx}\|_2^2 = \left(\pmb{Qx} \right)^T \left(\pmb{Qx} \right) = \pmb{x}\pmb{Q}\pmb{Q}^T\pmb{x}^T = \pmb{x}\pmb{x}^T = \|\pmb{x}\|_2^2
\tag{4-4-2}
$$

### 4.4.2 Gram-Schmidt 正交化
除了上一节举例的那些具有特殊几何或物理意义的正交矩阵，当然还存在很多相对比较“平凡”的正交矩阵。事实上，我们完全可以对一个非正交方阵进行某种变换获得正交矩阵，*Gram-Schmidt* 正交化就是其中一种非常经典的方法。对于 $\mathbb{R}^n$ 中的非正交向量 $\pmb{a}$ 与 $\pmb{b}$，现演示通过 *Gram-Schmidt* 正交化获得正交向量 $\pmb{\alpha}$、$\pmb{\beta}$ 的过程：

（1）固定初始向量 $\pmb{a}$ 的方向，并将其标准化：
$$
\pmb{\alpha} = \dfrac{\pmb{a}}{\|\pmb{a}\|}
\tag{4-4-3}
$$
（2）将第二向量 $\pmb{b}$ 分解为平行与垂直于 $\pmb{\alpha}$（或 $\pmb{a}$）的两个部分，并将后者标准化。平行部分即为 $\pmb{b}$ 在该方向上的投影 $\pmb{p}_{\pmb{b},\pmb{a}}$，垂直部分表示误差向量，即 $\pmb{e}_{\pmb{b}} = \pmb{b} - \pmb{p}_{\pmb{b},\pmb{a}}$：
$$
\pmb{e}_{\pmb{b}} = \pmb{b} - \dfrac{\pmb{a}\pmb{a}^T}{\pmb{a}^T\pmb{a}}\pmb{b}, \ \pmb{\beta} = \dfrac{\pmb{e}_{\pmb{b}}}{\|\pmb{b}\|}
\tag{4-4-4}
$$
上述过程当然可以继续拓展，例如此时我们再引入第三个向量 $\pmb{c}$：

（3）将第三向量 $\pmb{c}$ 依次面向 $\pmb{b}$ 与 $\pmb{a}$ 进行如（2）所示的步骤，最终可得到同时与 $\pmb{b}$ 与 $\pmb{a}$ 正交的标准向量 $\pmb{\theta}$：
$$
\pmb{e}_{\pmb{c}} = \pmb{c} - \dfrac{\pmb{a}\pmb{a}^T}{\pmb{a}^T\pmb{a}}\pmb{c} - \dfrac{\pmb{b}\pmb{b}^T}{\pmb{b}^T\pmb{b}}\pmb{c}, \ \pmb{\theta} = \dfrac{\pmb{e}_{\pmb{c}}}{\|\pmb{c}\|}
\tag{4-4-5}
$$
以此类推，正交化可以面向更多数目的向量组。在（2）中提到了一个名词即“正交分解”，这提醒我们 *Gram-Schmidt* 正交化本质上与 LU 分解一样，可以将普通矩阵分解为正交向量组矩阵与另一个矩阵的乘积，事实上这就是正交三角分解，它更装逼的名字叫约化 QR 分解（*QR decomposition*）。对于矩阵 $\pmb{A} \in \mathbb{R}^{m \times n}$，约化 QR 分解将其列向量转化为标准正交向量（基）并进行重构 $\pmb{A} = \pmb{QR}$：
$$
\pmb{A} = 
\underbrace{
\begin{bmatrix}
    \pmb{q}_1 & \pmb{q}_2 & \cdots & \pmb{q}_n
\end{bmatrix}}_{\pmb{Q}}
\underbrace{\begin{bmatrix}
\pmb{a}_1^T\pmb{q}_1 & \pmb{a}_2^T\pmb{q}_1 & \cdots & \pmb{a}_n^T\pmb{q}_1\\
0 & \pmb{a}_2^T\pmb{q}_2 & \cdots & \pmb{a}_n^T\pmb{q}_2\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & \pmb{a}_n^T\pmb{q}_n\\
\end{bmatrix}}_{\pmb{R}}
\tag{4-4-6}
$$
其中 $\pmb{R}$ 原本的形态应该是：
$$
\pmb{R} = 
\begin{bmatrix}
\pmb{a}_1^T\pmb{q}_1 & \pmb{a}_2^T\pmb{q}_1 & \cdots & \pmb{a}_n^T\pmb{q}_1\\
\pmb{a}_1^T\pmb{q}_2 & \pmb{a}_2^T\pmb{q}_2 & \cdots & \pmb{a}_n^T\pmb{q}_2\\
\vdots & \vdots & \ddots & \vdots\\
\pmb{a}_1^T\pmb{q}_n & \pmb{a}_1^T\pmb{q}_n & \cdots & \pmb{a}_n^T\pmb{q}_n\\
\end{bmatrix}
$$
由于正交化的步骤中保证了新向量必然正交于原有的全部向量，因此 $\pmb{a}_1^T\pmb{q}_2 = \pmb{a}_1^T\pmb{q}_3 = \cdots = \pmb{a}_{n-1}^T\pmb{q}_n = 0$。接下来以具体矩阵 $\pmb{A}$ 为例：
$$
\pmb{A} = 
\begin{bmatrix}
1 & 1\\
1 & 0\\
1 & 2\\
\end{bmatrix}, \ \pmb{a}_1 = 
\begin{bmatrix}
1\\ 1\\ 1\\
\end{bmatrix}, \ \pmb{a}_2 = 
\begin{bmatrix}
1\\ 0\\ 2\\
\end{bmatrix}
$$
$$
\pmb{e}_{\pmb{a}_2} = \pmb{a}_2 - \dfrac{\pmb{a}_1\pmb{a}_1^T}{\pmb{a}_1^T\pmb{a}_1}\pmb{a}_2 = 
\begin{bmatrix}
1\\ 0\\ 2\\
\end{bmatrix} - \dfrac{1}{3}
\begin{bmatrix}
1 & 1 & 1\\
1 & 1 & 1\\
1 & 1 & 1\\
\end{bmatrix}
\begin{bmatrix}
1\\ 0\\ 2\\
\end{bmatrix} = 
\begin{bmatrix}
0\\ -1\\ 1\\
\end{bmatrix}
$$
$$
\pmb{q}_1 = \dfrac{\pmb{a}_1}{\|\pmb{a}_1\|} = \dfrac{1}{\sqrt{3}}
\begin{bmatrix}
1\\ 1\\ 1\\
\end{bmatrix}, \ 
\pmb{q}_2 = \dfrac{\pmb{a}_2}{\|\pmb{a}_2\|} = \dfrac{1}{\sqrt{2}}
\begin{bmatrix}
0\\ -1\\ 1\\
\end{bmatrix}, \ 
\pmb{Q} = 
\begin{bmatrix}
\dfrac{1}{\sqrt{3}} & 0\\
\ \\
\dfrac{1}{\sqrt{3}} & \dfrac{-1}{\sqrt{2}}\\
\ \\
\dfrac{1}{\sqrt{3}} & \dfrac{1}{\sqrt{2}}\\
\end{bmatrix}
$$