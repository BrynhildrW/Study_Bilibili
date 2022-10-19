## 6.2 Diagonalizing a Matrix
### 6.2.1 方阵对角化
方阵对角化是目前为止我们接触的第三种矩阵分解方法（前两种分别为：基于初等变换的 LU 分解、基于 Gram-Schmidt 正交化的 QR 分解），对角化的重要理论依据就是特征方程：$\pmb{Ax}=\lambda\pmb{x}$，它能将矩阵乘法转化为简单的向量数乘。

取 $\pmb{A} \in \mathbb{R}^{n \times n}$ 的 $n$ 个特征向量（假设互异、独立），按照对应特征值大小排序组合成特征矩阵 $\pmb{S}$：
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
即 $\pmb{A}^n$ 的特征值是 $\pmb{A}$ 的 $n$ 次幂，特征向量（矩阵）不变。显然当 $max \ \lambda$
