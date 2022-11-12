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

