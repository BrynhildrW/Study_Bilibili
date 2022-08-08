### 5.3 列空间 & 零空间
以矩阵 $\pmb{X} \in \mathbb{R}^{3 \times 2}$ 为例。$\pmb{X}$ 的各列都属于 $\mathbb{R}^3$，因此列向量 $\pmb{X}(:,i)$ 的全体线性组合可以构成一个 $\mathbb{R}^3$ 的线性子空间，称为 $\pmb{X}$ 的**列空间**，记为 $\pmb{C}(\pmb{X})$。从欧氏几何的角度来看，$\pmb{C}(\pmb{X})$ 是一个过原点的平面（列向量不共线），$\pmb{X}$ 的两个列向量位于平面上。列空间有什么作用呢？我们通过一个实例来说明。

给定系数矩阵 $\pmb{A}$（这个例子将会持续存在一段时间）以及线性方程组 $\pmb{A} \pmb{x}^T = \pmb{b}^T$：
$$
    \pmb{A} = 
    \begin{bmatrix}
        1 & 1 & 2\\
        2 & 1 & 3\\
        3 & 1 & 4\\
        4 & 1 & 5\\
    \end{bmatrix} \in \mathbb{R}^{4 \times 3}
$$
这种方程数量超过未知数数量的情况称为“**超定**”（*over-determined*）系统。超定系统在大多数情况下是没有精确解的。从感性认识的角度考虑，如果我们通过正定系统获得一个精确解后，再额外添加一个或多个方程，则增添方程必须满足**某些特定条件**才能使得等式依旧成立，接下来我们将从 $\pmb{b}$ 入手，深入研究 $\pmb{A} \pmb{x}^T = \pmb{b}^T$ 有解的条件。
$$
    \begin{bmatrix}
        1 & 1 & 2\\
        2 & 1 & 3\\
        3 & 1 & 4\\
        4 & 1 & 5\\
    \end{bmatrix}
    \begin{bmatrix}
        x_1\\ x_2\\ x_3\\
    \end{bmatrix} = 
    \begin{bmatrix}
        b_1\\ b_2\\ b_3\\
    \end{bmatrix}
$$
显然 $\pmb{x} = \pmb{b} = \pmb{0}$ 对于任意数值、任意维度的系数矩阵 $\pmb{A}$ 都成立，$\pmb{0}$ 通常称为“**零解**”或“**平凡解**”。除此之外，结合矩阵的列视图，我们还能很快地举出一些其它例子：
$$
    \begin{bmatrix}
        b_1\\ b_2\\ b_3\\
    \end{bmatrix} = x_1 
    \begin{bmatrix}
        1\\ 2\\ 3\\ 4\\
    \end{bmatrix} + x_2 
    \begin{bmatrix}
        1\\ 1\\ 1\\ 1\\
    \end{bmatrix} + x_3 
    \begin{bmatrix}
        2\\ 3\\ 4\\ 5\\
    \end{bmatrix}
$$
以上这个拆解形式很容易让人联想到矩阵的列空间 $\pmb{C} (\pmb{A})$，因此有以下结论：当且仅当 $\pmb{b} \in \pmb{C} (\pmb{A})$ 时，方程组 $\pmb{A} \pmb{x}^T = \pmb{b}^T$ 确定有解。

说完了列空间，我们再来谈谈零空间。还是以 $\pmb{A}$ 为例，满足 $\pmb{A}\pmb{x}^T=\pmb{0}$ 的全体 $\pmb{x}$ 构成的空间 $\pmb{N}(\pmb{A})$ 称为 $\pmb{A}$ 的零空间。注意区分，对于 $\pmb{A} \pmb{x}^T = \pmb{b}^T$，$\pmb{C} (\pmb{A})$ 关心的是 $\pmb{b}$，而 $\pmb{N}(\pmb{A})$ 关心的是 $\pmb{x}$；对于 $\pmb{A} \in \mathbb{R}^{m \times n}$，$\pmb{C} (\pmb{A}) \in \mathbb{R}^m$ 而 $\pmb{N} (\pmb{A}) \in \mathbb{R}^n$。
$$
    \begin{bmatrix}
        1 & 1 & 2\\
        2 & 1 & 3\\
        3 & 1 & 4\\
        4 & 1 & 5\\
    \end{bmatrix}
    \begin{bmatrix}
        x_1\\ x_2\\ x_3\\
    \end{bmatrix} = 
    \begin{bmatrix}
        0\\ 0\\ 0\\
    \end{bmatrix}
$$
显然，$\pmb{0}$ 再一次满足 $\pmb{N}(\pmb{A})$ 的要求。事实上 $[1,1,-1]^T$ 的所有线性组合都满足要求。可见此时 $\pmb{N}(\pmb{A})$ 表现为 $\mathbb{R}^3$ 中的一条直线。

需要指出，当 $\pmb{b} \ne \pmb{0}$ 时，同样可能存在很多 $\pmb{x}$ 使得方程等式平衡。但是这些 $\pmb{x}$ 无法构成线性向量空间。很简单，因为其中一定没有零元 $\pmb{0}$。所以当我们站在线性方程组的角度上回顾子空间的概念时应当明确：

（1）根据一个给出的矩阵 $\pmb{X} \in \mathbb{R}^{m \times n}$，可以从中构建两种线性空间：列空间 $\pmb{C}(\pmb{X}) \in \mathbb{R}^m$ 与零空间 $\pmb{N}(\pmb{X}) \in \mathbb{R}^n$，而不论哪种空间都必须满足“**零元存在**”；

（2）列空间的构建方法是对已有列向量进行线性组合，零空间的构建方法是在矩阵基础上寻找满足条件的解集。二者都是非常重要的子空间构建方法。