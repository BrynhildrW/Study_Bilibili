### 4.1 Orthogonality of the Four Subspaces
#### 4.1.1 正交向量
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

#### 4.1.2 正交子空间
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

