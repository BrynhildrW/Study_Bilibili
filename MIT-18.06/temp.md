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

### 6.5.2 正定矩阵
正定矩阵（Positive Definite Matrix）是一类特殊的对称矩阵。
