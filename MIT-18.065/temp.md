## I.1 Multiplication $\pmb{Ax}$ Using Columns of $\pmb{A}$
老爷子还是那么喜欢举例，那我们也先看一个 $\mathbb{R}^{3 \times 3}$ 的矩阵 $\pmb{A}$ 以及相关的乘法 $\pmb{Ax}$：
$$
    \pmb{Ax} = 
    \begin{bmatrix}
        2 & 1 & 3\\
        3 & 1 & 4\\
        5 & 7 & 12\\
    \end{bmatrix}
    \begin{bmatrix}
        x_1\\ x_2\\ x_3\\
    \end{bmatrix} =
    \underbrace{
        \begin{bmatrix}
            2x_1 + x_2 + 3x_3\\
            3x_1 + x_2 + 4x_3\\
            5x_1 + 7x_2 + 12x_3\\
        \end{bmatrix}}_{By \ rows: \ dot(inner) \ product} =
    \underbrace{
        \begin{bmatrix}
            2\\ 3\\ 5\\
        \end{bmatrix}x_1 + 
        \begin{bmatrix}
            1\\ 1\\ 7\\
        \end{bmatrix}x_2 + 
        \begin{bmatrix}
            3\\ 4\\ 12\\
        \end{bmatrix}x_3}_{By \ columns: \ vectors \ approach}
$$
对于接触过 **MIT-18.06** 课程的朋友而言，上述两种矩阵-向量乘法的表现形式并不陌生。前者浅显，易于机械计算；后者深刻，适合理解本质：**$\pmb{Ax}$ 是矩阵 $\pmb{A}$ 的列向量的线性组合**。对于上述案例，$\pmb{Ax}$ 是 $\mathbb{R}^3$ 中的一个向量；在不限定 $\pmb{x}$ 的情况下，全体 $\pmb{Ax}$（即列空间 $\pmb{C(A)}$）构成 $\mathbb{R}^3$ 的一个子空间。更具体地，$\pmb{C(A)}$ 是一个**平面**，因为 $rank(\pmb{A})=2$。结合该案例可有总结：

对于 $\mathbb{R}^3$ 的子空间，根据维度不同，其对应的场景（矩阵 $\pmb{A} \in \mathbb{R}^{3 \times n}$）也各不相同：
（1）**0 维**：零向量 $\pmb{0}=[0,0,0]^T$;
（2）**1 维**：直线簇 $x_1\pmb{a}_1$，$\pmb{A}$ 的秩为 1；
（3）**2 维**：平面 $x_1\pmb{a}_1 + x_2\pmb{a}_2$，$\pmb{A}$ 的秩为 2；
（4）**3 维**：完整 $\mathbb{R}^3$ 空间 $x_1\pmb{a}_1 + x_2\pmb{a}_2 + x_3\pmb{a}_3$，$\pmb{A}$ 满秩。

另外，当我们考虑向量 $\pmb{b}$ 是否位于 $\pmb{C(A)}$ 中时，实际考察的问题为：$\pmb{Ax}=\pmb{b}$ 是否有解。而该问题的答案又与对应其次方程组的解息息相关：当且仅当 $\pmb{Ax}=\pmb{0}$ 只有全零解 $\pmb{x}=[0,0,0]^T$ 时，非齐次方程组 $\pmb{Ax}=\pmb{b}$ 有唯一解 $\pmb{x}=\pmb{A}^{-1}\pmb{b}$。

进一步地，我们来回忆矩阵的 CR 分解，仍然以方阵 $\pmb{A}$ 为例：
$$
    \begin{align}
    \notag \pmb{A} &=
        \begin{bmatrix}
            2 & 1\\
            3 & 1\\
            5 & 7\\
        \end{bmatrix}
        \begin{bmatrix}
            1 & 0 & 1\\
            0 & 1 & 1\\
        \end{bmatrix} = \underbrace{
        \begin{bmatrix}
            2 & 1\\
            3 & 1\\
            5 & 7\\
        \end{bmatrix}}_{\pmb{C} \in \mathbb{R}^{3 \times 2}} \underbrace{
            \begin{bmatrix}
                x_{11} & x_{12} & x_{13}\\
                x_{21} & x_{22} & x_{23}\\
            \end{bmatrix}}_{\pmb{R} \in \mathbb{R}^{2 \times 3}}
    \notag \ \\
    \notag \ \\
    \notag &= \underbrace{\left(
        \begin{bmatrix}
            2\\ 3\\ 5\\
        \end{bmatrix}x_{11} + 
        \begin{bmatrix}
            1\\ 1\\ 7\\
        \end{bmatrix}x_{21}\right)}_{
        \begin{bmatrix}
            2\\ 3\\ 5\\
        \end{bmatrix} \Rightarrow
            \begin{cases}
                x_{11} = 1\\
                \ \\
                x_{21} = 0\\
            \end{cases}} \oplus \underbrace{\left(
        \begin{bmatrix}
            2\\ 3\\ 5\\
        \end{bmatrix}x_{12} + 
        \begin{bmatrix}
            1\\ 1\\ 7\\
        \end{bmatrix}x_{22}\right)}_{
            \begin{bmatrix}
                1\\ 1\\ 7\\
            \end{bmatrix} \Rightarrow
            \begin{cases}
                x_{12} = 0\\
                \ \\
                x_{22} = 1\\
            \end{cases}} \oplus \underbrace{\left(
        \begin{bmatrix}
            2\\ 3\\ 5\\
        \end{bmatrix}x_{13} + 
        \begin{bmatrix}
            1\\ 1\\ 7\\
        \end{bmatrix}x_{23}\right)}_{
            \begin{bmatrix}
                3\\ 4\\ 12\\
            \end{bmatrix} \Rightarrow
            \begin{cases}
                x_{13} = 1\\
                \ \\
                x_{23} = 1\\
            \end{cases}}
    \end{align}
$$
$\pmb{C}$ 由 $\pmb{A}$ 的独立列向量组合而成，其列向量个数为 $\pmb{A}$ 的秩，即**矩阵的秩等于其列空间的维度**。此外，$\pmb{R} = rref(\pmb{A})$ 为 $\pmb{A}$ 的**行缩减梯形**（*row-reduced echelon form*），同时 $rank(\pmb{A})$ 与 $\pmb{R}$ 的行向量个数相等。CR 分解对于我们理解奇异值分解（*Singular Value Decomposition, SVD*）具有重要意义，事实上 SVD 分解是 CR 分解的一种特殊形式，其中 $\pmb{C}$ 的 $r$ 个列向量彼此正交、$\pmb{R}$ 的 $r$ 个行向量彼此正交。