---
html:
    toc: true
print_background: true
---

# MIT-18.06：Linear Algebra | 线性代数
[课程视频][total_lesson] 和 [配套教材][document]（提取码：whs9）在这里！

如果我大二的时候学的是这套线代教程，我的人生轨迹就会发生变化：可能绩点会有大的改观，可能拿到保研资格，可能不会因为考研不考数学而跟医工院深度绑定，可能离开天津前往上海，可能硕士毕业就参加工作，可能不会在一个不喜欢的行业继续读博苟延残喘，之后的人生理应处处发生改变吧。

这份文档并非是面面俱到的笔记，而是我在学习相关课程中发现的重要结论与新的收获，因此部分章节目录存在跳跃。

若无特殊说明，本文中大写字母表示二维矩阵，小写字母表示向量。

[total_lesson]: https://www.bilibili.com/video/BV16Z4y1U7oU?p=1&vd_source=16d5baed24e8ef63bea14bf1532b4e0c
[document]: https://pan.baidu.com/s/139zZkqWUxa-sHpKJzU5vdg

***

## 1 Introduction to Vectors
### 1.2 Length and Dot Products
**点乘**是一个需要与矩阵乘法相区分的运算操作，在泛函里一般称为**内积**。在向量这个低级维度，可以简单理解为对应元素相乘并相加，即对于 $\pmb{x},\pmb{y} \in \mathbb{R}^{m \times 1}$：
$$
    \pmb{x} \cdot \pmb{y} = \left< \pmb{x}, \pmb{y} \right> = \sum_{i=1}^m x_i y_i \ \Longleftrightarrow \ \pmb{y}^T \pmb{x}
    \tag{1-2-1}
$$
虽然说点积与矩阵乘法不一样，但矩阵乘法里处处都是点积。结果矩阵中的每一个元素其实都来源于一个行向量与列向量的点积。

**向量长度**，通常又称为向量的2范数，表示向量空间中的零元至该点的空间距离，其中 $|*|$ 表示取模：
$$
    \|\pmb{x}\|_2 = \sqrt{\sum_{i=1}^m |x_i|^2}
    \tag{1-2-2}
$$

#### 1.2.1 向量单位化
在此基础上，我们可以定义单位向量 $\pmb{u}$ 以及将一个任意非零向量 $\pmb{v}$ 单位化的过程：
$$
    \|\pmb{u}\|_2 = 1, \ \pmb{u}_{\pmb{v}} = \dfrac{\pmb{v}}{\|\pmb{v}\|_2}
    \tag{1-2-3}
$$

#### 1.2.2 正交向量
对于点乘结果为 0 的特殊向量组合，将其称为正交向量组，即二者相互垂直：
$$
    \left<\pmb{x}, \pmb{y} \right> = 0 \ \Longleftrightarrow \ \pmb{x} \perp \pmb{y}
    \tag{1-2-4}
$$

#### 1.2.3 向量夹角
点乘并非只能判断是否正交，还能进一步量化向量间的夹角。对于单位向量 $\pmb{u}$、$\pmb{v}$：
$$
    \pmb{u} \cdot \pmb{v} = cos \theta
    \tag{1-2-5}
$$

#### 1.2.4 余弦公式 & 柯西不等式
由（1-2-5）进一步可得面向两个非单位向量（如 $\pmb{x}$、$\pmb{y}$）夹角的余弦公式：
$$
    cos \theta = \dfrac{\pmb{x} \cdot \pmb{y}}{\|\pmb{x}\| \ \|\pmb{y}\|}
    \tag{1-2-6}
$$
这一公式也常用于高中数学立体几何部分的平面关系判断。说来惭愧，落笔之日距我高考已达7年有余，根本想不起来当初郑永盛是怎么教我们理解这个公式的了（~~可能单纯就是硬记吧~~），回首往日甚是怀念。
最后，根据三角函数数值大小的规律可得 *Cauchy-Schwarz-Buniakowsky* 不等式，严格来说，是柯西不等式的其中一种形式：
$$
    |\pmb{x} \cdot \pmb{y}| \leqslant \|\pmb{x}\| \ \|\pmb{y}\|
    \tag{1-2-7}
$$

## 2 Solving Linear Equations
### 2.1 Vectors and Linear Equations
#### 2.1.1 矩阵的“列视图”
对于多元一次线性方程组：
$$
    \begin{cases}
        a_{11}x_1 + a_{12}x_2 + a_{13}x_3 = b_{1}\\
        a_{21}x_1 + a_{22}x_2 + a_{23}x_3 = b_{2}\\
        a_{31}x_1 + a_{32}x_2 + a_{33}x_3 = b_{3}\\
    \end{cases}
$$
该方程组可以很容易地改写为矩阵 $\pmb{A} \in \mathbb{R}^{3 \times 3}$ 与向量 $\pmb{x} \in \mathbb{R}^{3 \times 1}$ 的乘法：
$$
    \begin{bmatrix}
        a_{11} & a_{12} & a_{13}\\
        a_{21} & a_{22} & a_{23}\\
        a_{31} & a_{32} & a_{33}\\
    \end{bmatrix}
    \begin{bmatrix}
        x_1\\ x_2\\ x_3\\
    \end{bmatrix} = 
    \begin{bmatrix}
        b_1\\ b_2\\ b_3\\
    \end{bmatrix} \ \Longrightarrow \ \pmb{A} \pmb{x} = \pmb{b}
    \tag {2-1-1}
$$
我们通常习惯以“**元素的加权和**”这一视角来描述这种“数群”至“数字”的运算过程，即存在这样的思维习惯：$b_1 = a_{11}x_1 + a_{12}x_2 + a_{13}x_3$。此时我们接受系数矩阵 $\pmb{A}$ 的方式是逐行进行的，是一种略显复杂的动态过程。

**列视图**是一种全新视角，其本质为矩阵列的线性组合，我们可以把式（2-1-1）改写为：
$$
    x_1
    \begin{bmatrix}
        a_{11}\\ a_{21}\\ a_{31}\\
    \end{bmatrix} + x_2
    \begin{bmatrix}
        a_{12}\\ a_{22}\\ a_{32}\\
    \end{bmatrix} + x_3
    \begin{bmatrix}
        a_{13}\\ a_{23}\\ a_{33}\\
    \end{bmatrix} = 
    \begin{bmatrix}
        b_1\\ b_2\\ b_3\\
    \end{bmatrix}
    \tag {2-1-2}
$$
在面对形如“矩阵右乘向量”之类的运算时，使用**列视图**能够更好地简化思路，在实际应用中帮助我们明确信息流传递的真实意义。

#### 2.1.2 矩阵的“行视图”
对于行向量 $\pmb{a}$ & 矩阵 $\pmb{X}$ 的乘法：
$$
    \pmb{aX}=\pmb{b} \ \Longleftrightarrow \ 
    \begin{bmatrix}
        a_1 & a_2 & a_3\\
    \end{bmatrix}
    \begin{bmatrix}
        x_{11} & x_{12} & x_{13}\\
        x_{21} & x_{22} & x_{23}\\
        x_{31} & x_{32} & x_{33}\\
    \end{bmatrix} = 
    \begin{bmatrix}
        b_1 & b_2 & b_3\\
    \end{bmatrix}
    \tag {2-1-3}
$$
对于“矩阵左乘向量”的运算形式，可以使用**行视图**对其进行观察与分析：
$$
    a_1
    \begin{bmatrix}
        x_{11} & x_{12} & x_{13}\\
    \end{bmatrix} + a_2
    \begin{bmatrix}
        x_{21} & x_{22} & x_{23}\\
    \end{bmatrix} + a_3
    \begin{bmatrix}
        x_{31} & x_{32} & x_{33}\\
    \end{bmatrix} = 
    \begin{bmatrix}
        b_1 & b_2 & b_3\\
    \end{bmatrix}
    \tag{2-1-4}
$$
掌握矩阵的**行视图**与**列视图**有助于我们进一步理解矩阵初等变换以及逆矩阵的本质。

#### 2.1.3 矩阵乘法
矩阵之间的乘法可视为**左矩阵**与多个**列向量**的乘法，抑或为多个**行向量**与**右矩阵**的乘法。例如 $\pmb{AB}=\pmb{C}$，除了最基础、最抽象的**运算型**（*regular way*）表示：
> The regular way: a row of $\pmb{A}$ times a column of $\pmb{B}$ to get a number of $\pmb{C}$

$$
    \begin{bmatrix}
        a_{11} & a_{12}\\
        a_{21} & a_{22}\\
        a_{31} & a_{32}\\
    \end{bmatrix}
    \begin{bmatrix}
        b_{11} & b_{12} & b_{13}\\
        b_{21} & b_{22} & b_{23}\\
    \end{bmatrix} = 
    \begin{bmatrix}
        \sum_{i} a_{1i} b_{i1} & \sum_{i} a_{1i} b_{i2} & \sum_{i} a_{1i} b_{i3}\\
        \sum_{i} a_{2i} b_{i1} & \sum_{i} a_{2i} b_{i2} & \sum_{i} a_{2i} b_{i3}\\
        \sum_{i} a_{3i} b_{i1} & \sum_{i} a_{3i} b_{i2} & \sum_{i} a_{3i} b_{i3}\\
    \end{bmatrix}
    \tag{2-1-5}
$$
还可以表示为**列型**（*column way*）和**行型**（*row way*）：
> The column way: matrix $\pmb{A}$ times a column of $\pmb{B}$ to get a column of $\pmb{C}$
> The row way: a row of $\pmb{A}$ times matrix $\pmb{B}$ to get a row of $\pmb{C}$

$$
    \pmb{AB} = \pmb{A}
    \begin{bmatrix}
        b_{11}\\ b_{21}\\
    \end{bmatrix} \oplus \pmb{A}
    \begin{bmatrix}
        b_{21}\\ b_{22}\\
    \end{bmatrix} \oplus \pmb{A}
    \begin{bmatrix}
        b_{31}\\ b_{32}\\
    \end{bmatrix}
    \tag{2-1-6}
$$
$$
    \pmb{AB} = 
    \begin{bmatrix}
        a_{11} & a_{12}\\
    \end{bmatrix} \pmb{B} \oplus 
    \begin{bmatrix}
        a_{21} & a_{22}\\
    \end{bmatrix} \pmb{B} \oplus 
    \begin{bmatrix}
        a_{31} & a_{32}\\
    \end{bmatrix} \pmb{B}
    \tag{2-1-7}
$$
此时运算符 “ $\oplus$ ” 视情况分别表示**列向量在横向**或**行向量在纵向**的拼接。除了上述三种运算角度，其实还有**行视图**、**列视图**混用的第四种形式：
> The 4th way: a column of $\pmb{A}$ times a row of $\pmb{B}$ to get a partial matrix of $\pmb{C}$

$$
    \pmb{AB} = 
    \begin{bmatrix}
        a_{11}\\ a_{21}\\ a_{31}\\
    \end{bmatrix}
    \begin{bmatrix}
        b_{11} & b_{12} & b_{13}\\
    \end{bmatrix} + 
    \begin{bmatrix}
        a_{12}\\ a_{22}\\ a_{32}\\
    \end{bmatrix}
    \begin{bmatrix}
        b_{21} & b_{22} & b_{23}\\
    \end{bmatrix} = \sum_i \pmb{A}(:,i) \pmb{B}(i,:)
    \tag{2-1-8}
$$
这种混合视图隐藏的信息非常重要：

（1）$\pmb{AB}$ 结果中的**每一列**，都是左矩阵 $\pmb{A}$ 中**各列**的**线性组合**

（2）$\pmb{AB}$ 结果中的**每一行**，都是右矩阵 $\pmb{B}$ 中**各行**的**线性组合**

上述结论通常与行（列）空间结合，帮助我们判断矩阵是否可逆（满秩）。
***

### 2.3 Elimination Using Matrices
对矩阵 $\pmb{X}$ 进行单次初等行变换，相当于 $\pmb{X}$ 左乘一个变换矩阵 $\pmb{E}$，例如矩阵**第一行不变**、**第二行减去第一行的两倍**：
$$
    \underbrace{
        \begin{bmatrix}
            x_{11} & x_{12}\\
            x_{21}-2x_{11} & x_{22}-2x_{12}\\
        \end{bmatrix}}_{\pmb{Y}} \Longleftrightarrow 
    \underbrace{
        \begin{bmatrix}
            1 & 0\\ -2 & 1\\
        \end{bmatrix}}_{\pmb{E}}
    \underbrace{
        \begin{bmatrix}
            x_{11} & x_{12}\\
            x_{21} & x_{22}\\
        \end{bmatrix}}_{\pmb{X}}
$$
上述过程太抽象？想想矩阵的**行视图**、矩阵乘法的**行型**：
$$
    \begin{bmatrix}
        1 & 0\\ -2 & 1\\
    \end{bmatrix}
    \begin{bmatrix}
        x_{11} & x_{12}\\
        x_{21} & x_{22}\\
    \end{bmatrix} = 
    \underbrace{
        \begin{bmatrix}
            1 & 0\\
        \end{bmatrix}
        \begin{bmatrix}
            x_{11} & x_{12}\\
            x_{21} & x_{22}\\
        \end{bmatrix}}_{1st \ row \ of \ \pmb{Y}} \oplus 
    \underbrace{
        \begin{bmatrix}
            -2 & 1\\
        \end{bmatrix}
        \begin{bmatrix}
            x_{11} & x_{12}\\
            x_{21} & x_{22}\\
        \end{bmatrix}}_{2nd \ row \ of \ \pmb{Y}}
$$
$$
    \begin{cases}
        \begin{bmatrix}
            1 & 0\\
        \end{bmatrix}
        \begin{bmatrix}
            x_{11} & x_{12}\\
            x_{21} & x_{22}\\
        \end{bmatrix} = 1 \times
        \begin{bmatrix}
            x_{11} & x_{12}\\
        \end{bmatrix} + 0 \times
        \begin{bmatrix}
            x_{21} & x_{22}\\
        \end{bmatrix}\\
        \\
        \begin{bmatrix}
            -2 & 1\\
        \end{bmatrix}
        \begin{bmatrix}
            x_{11} & x_{12}\\
            x_{21} & x_{22}\\
        \end{bmatrix} = -2 \times
        \begin{bmatrix}
            x_{11} & x_{12}\\
        \end{bmatrix} + 1 \times
        \begin{bmatrix}
            x_{21} & x_{22}\\
        \end{bmatrix}\\
    \end{cases}
$$
对应地，单次初等列变换等价于 $\pmb{X}$ 右乘一个操作矩阵 $\pmb{C}$，例如矩阵**第一列除以 2**、**第二列乘 3 以后加上第一列变换后的 5 倍**：
$$
    \begin{bmatrix}
        \dfrac{x_{11}}{2} & 3x_{12}+\dfrac{5 x_{11}}{2}\\
        \\
        \dfrac{x_{21}}{2} & 3x_{22}+\dfrac{5 x_{21}}{2}\\
    \end{bmatrix} \Leftrightarrow
    \underbrace{
        \begin{bmatrix}
            x_{11} & x_{12}\\ x_{21} & x_{22}\\
        \end{bmatrix}}_{\pmb{X}}
    \underbrace{
        \begin{bmatrix}
            \dfrac{1}{2} & \dfrac{5}{2}\\
            \\
            0 & 3\\
        \end{bmatrix}}_{\pmb{C}}
$$
朝闻道，夕死可矣。


### 2.5 Inverse Matrices
#### 2.5.1 判断矩阵是否可逆
书接上回，例如病态矩阵 $\pmb{A}$：
$$
    \pmb{A} = 
    \begin{bmatrix}
        1 & 4\\
        2 & 8\\
    \end{bmatrix}
$$
要判断 $\pmb{A}$ 是否可逆，即寻找一个矩阵 $\pmb{A}^{-1}$ 使得 $\pmb{AA}^{-1} = \pmb{I}$。根据上一节末尾的结论，$\pmb{AA}^{-1}$ 的每一列都是 $\pmb{A}$ 中各列的线性组合。但是 $\pmb{A}$ 的各列是**共线**的，而 $[1,0]^T$ 所在直线并不在方向向量为 $[1,2]^T$ 的线簇中，因此在本例中我们永远无法组合出单位向量，换句话说 $\pmb{A}$ 不可逆。

更一般地，如果我们能找到一个非零列向量 $\pmb{x} \ne \pmb{0}$ 使得方程 $\pmb{Ax} = \pmb{0}$ 成立，则 $\pmb{A}$ 不可逆。这一结论可以从两方面理解：

（1）$\pmb{A}$ 中存在某一列（行）可以由其它列（行）的线性组合表示，即该列（行）对于填充矩阵必要信息没有起到任何作用，因此这个矩阵就是“病态”的，是不完整的，也就不可逆；

（2）若 $\pmb{A}$ 可逆，则有 $\pmb{A}^{-1} \pmb{Ax} = \pmb{0}$，从而 $\pmb{x} = \pmb{0}$，这与非零列向量的前提是矛盾的，所以 $\pmb{A}$ 不可逆。

#### 2.5.2 求解方阵的逆矩阵
简单起见，我们先具体到一个非奇异的二阶方阵 $\pmb{B}$：
$$
    \pmb{B} = 
    \begin{bmatrix}
        1 & 3\\ 2 & 7\\
    \end{bmatrix}
$$
关于如何求解逆矩阵 $\pmb{B}^{-1}$，有两种主要思路，**第一种**是 *Gauss's way*，即依次求解二元一次线性方程组：
$$
    \pmb{B} \pmb{x}_1 = 
    \begin{bmatrix}
        1\\ 0\\
    \end{bmatrix}, \ 
    \pmb{B} \pmb{x}_2 = 
    \begin{bmatrix}
        0\\ 1\\
    \end{bmatrix} \ \Longrightarrow \ 
    \pmb{B}^{-1} = 
    \begin{bmatrix}
        \pmb{x}_1 & \pmb{x}_2
    \end{bmatrix}
$$
需要注意的是，*Gauss* 通过增广矩阵（*augmented matrix*）与初等行变换来实现方程组求解，以 $\pmb{Bx}_1$ 为例。如果我们得到了 $\pmb{x}_1$ ，以此类推可解出 $\pmb{x}_2$：
$$
    \hat{\pmb{B}} = 
    \begin{bmatrix}
        1 & 3| & 1\\
        2 & 7| & 0\\
    \end{bmatrix} \to 
    \begin{bmatrix}
        1 & 3| & 1\\
        0 & 1| & -3\\
    \end{bmatrix} \to
    \begin{bmatrix}
        1 & 0| & 7\\
        0 & 1| & -2\\
    \end{bmatrix}
$$
**第二种**是 *Gauss-Jordan's way*，即一次性求解多个多元一次线性方程组，具体实施方法是在 *Gauss* 的基础上进一步拓展增广矩阵 $\hat{\pmb{B}}$，并通过彻底的初等行变换将增广矩阵的左半部分化简为单位阵，则增广部分即为逆矩阵：
$$
    \hat{\pmb{B}} = 
    \begin{bmatrix}
        1 & 3| & 1 & 0\\
        2 & 7| & 0 & 1\\
    \end{bmatrix} \to 
    \begin{bmatrix}
        1 & 3| & 1 & 0\\
        0 & 1| & -2 & 1\\
    \end{bmatrix} \to
    \begin{bmatrix}
        1 & 0| & 7 & -3\\
        0 & 1| & -2 & 1\\
    \end{bmatrix}
$$
这一点很好理解，首先对于 $\pmb{B}$ 以及 $\hat{\pmb{B}}$ 的增广部分（ $\pmb{I}$ ）而言，二者进行了同步初等行变换，即左乘了同一个变换矩阵（ $\pmb{E}$ ）。 $\pmb{B}$ 经过变换后成为了 $\pmb{I}$，这意味着 $\pmb{E} = \pmb{B}^T$，那自然也有 $\pmb{RI}=\pmb{E}$，所以增广部分就会转换成逆矩阵。即：
$$
    \pmb{B}^{-1} = 
    \begin{bmatrix}
        7 & -3\\ -2 & 1\\
    \end{bmatrix}
$$
***

### 2.6 Elimination = Factorization
#### 2.6.1 LU分解的概念
以三阶方阵 $\pmb{A} \in \mathbb{R}^{3 \times 3}$ 为例，假设线性方程组 $\pmb{Ax}=\pmb{b}$ 为一个具有唯一解的三元一次方程组，且方程顺序排列合适（无需进行行交换），则可以通过至多**三次**初等行变换，将系数矩阵 $\pmb{A}$ 转变成一个上三角矩阵 $\pmb{U}$（*Upper triangle matrix*），其对角线元素称为“主元”（*pivot*）：
$$
    \pmb{E}_{32} \pmb{E}_{31} \pmb{E}_{21} \pmb{A} = \pmb{U} \
    \Longrightarrow \ \hat{\pmb{E}} \pmb{A} = \pmb{U}
    \tag{2-6-1}
$$
显然，$\hat{\pmb{E}}$ 主对角线上的元素均为 1。稍作变换即可得到 $\pmb{A}$ 的 LU 分解：
$$
    \pmb{A} = {\hat{\pmb{E}}}^{-1} \pmb{U} = \pmb{LU}
    \tag{2-6-2}
$$
其中 $\pmb{L}$ 是一个下三角矩阵（*Lower triangle matrix*）。值得注意的是，$\pmb{U}$ 不一定是对角阵，但是 LU 分解的结果已经非常接近对角化了。我们只需对 $\pmb{U}$ 再进行至多**三次**初等列变换，即可将其彻底转化为一个对角阵 $\pmb{D}$（*Diagonal matrix*），同时列变换操作矩阵的逆将是一个上三角矩阵。因此 LU 分解可以进一步变成 LDU 分解 $\pmb{A} = \pmb{LDU}$。

说回到 LU 分解，我们自然而然地会产生一个问题：为什么我们倾向于展示 $\pmb{A} = \pmb{LU}$，而不是 $\hat{\pmb{E}} \pmb{A} = \pmb{U}$？这里首先给出一个结论：
> For $\pmb{A}=\pmb{LU}$, if there are no row exchanges, those numbers that we multiplied rows by and subtracted during an elimination step go directly into $\pmb{L}$.

接下来我们以一个 4 阶方阵为例，先假设出一些行变换矩阵：
$$
    \hat{\pmb{E}} = 
    \underbrace{
        {\begin{bmatrix}
            1 & 0 & 0 & 0\\
            0 & 1 & 0 & 0\\
            0 & 0 & 1 & 0\\
            0 & 0 & -4 & 1\\
        \end{bmatrix}}}_{\pmb{E}_{to3}} 
    \underbrace{
        \begin{bmatrix}
            1 & 0 & 0 & 0\\
            0 & 1 & 0 & 0\\
            0 & 2 & 1 & 0\\
            0 & 3 & 0 & 1\\
        \end{bmatrix}}_{\pmb{E}_{to2}}
    \underbrace{
        \begin{bmatrix}
            1 & 0 & 0 & 0\\
            -2 & 1 & 0 & 0\\
            1 & 0 & 1 & 0\\
            -3 & 0 & 0 & 1\\
        \end{bmatrix}}_{\pmb{E}_{to1}} = 
    \begin{bmatrix}
        1 & 0 & 0 & 0\\
        -2 & 1 & 0 & 0\\
        -3 & 2 & 1 & 0\\
        3 & -5 & -4 & 1\\
    \end{bmatrix}
$$
我们其实不需要严格执行两次矩阵乘法才能获得 $\hat{\pmb{E}}$。以第一列为例：

（1）$\hat{\pmb{E}} (0,0) = 1$ 是毫无疑问的（我习惯使用 *pythonic* 风格记录数据下标），之前也有提到对角线上元素都应该是 1；

（2）$\hat{\pmb{E}} (1,0) = \pmb{E}_{to1} (1,0) = -2$；

（3）$\hat{\pmb{E}} (2,0) = \pmb{E}_{to1} (2,0) + \pmb{E}_{to2} (2,1) \hat{\pmb{E}} (1,0) = -3$；

（4）$\hat{\pmb{E}} (3,0) = \pmb{E}_{to1} (3,0) + \pmb{E}_{to2} (3,1) \hat{\pmb{E}} (1,0) + \pmb{E}_{to3} (3,2) \hat{\pmb{E}} (2,0) = 3$。

同理可依次获得第二列、第三列下三角部分的元素值。

不论如何，在 $\pmb{E}_{to3}$、$\pmb{E}_{to2}$ 以及 $\pmb{E}_{to1}$ 的基础上想获取 $\hat{\pmb{E}}$，多多少少需要一些运算。当数字并非示例中如此友好时，这个过程会变得非常痛苦。而如果我们尝试写出 $\pmb{L}$，则无需任何运算，甚至比你敲完这段代码的时间还要快得多：
```python
import numpy as np
from numpy import linalg as nLA

E3 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,-4,1]])
E2 = np.array([[1,0,0,0],[0,1,0,0],[0,2,1,0],[0,3,0,1]])
E1 = np.array([[1,0,0,0],[-2,1,0,0],[1,0,1,0],[-3,0,0,1]])
hatE = E3 @ E2 @ E1
L = nLA.inv(hatE)
```
$$
    \pmb{L} = 
    \begin{bmatrix}
        1 & 0 & 0 & 0\\
        2 & 1 & 0 & 0\\
        -1 & -2 & 1 & 0\\
        3 & -3 & 4 & 1\\
    \end{bmatrix}
$$
回顾上面那段英文结论，结合这个示例，我们不难发现：

（1）$\pmb{L} (1,0) = -\pmb{E}_{to1} (1,0)$；

（2）$\pmb{L} (2,0) = -\pmb{E}_{to1} (2,0), \ \pmb{L} (2,1) = -\pmb{E}_{to2} (2,1)$；

（3）$\pmb{L} (3,0) = -\pmb{E}_{to1} (3,0), \ \pmb{L} (3,1) = -\pmb{E}_{to2} (3,1), \ \pmb{L} (3,2) = -\pmb{E}_{to3} (3,2)$。

综上所述，在已知所需的行变换操作与 $\pmb{U}$ 时，$\pmb{A} = \pmb{LU}$ 的效率远高于 $\hat{\pmb{E}} \pmb{A} = \pmb{U}$。这是因为我们可以在进行初等行变换的过程中，同步计算变换结果 $\pmb{U}$、轻松获取 $\pmb{L}$ 的元素。

#### 2.6.2 方阵消元的时间复杂度
我们把单个元素的一次乘法与加法设为基本操作单元（除法等同于乘法，不乘等同于乘 1 ），在无需行交换的情况下，将 $\pmb{A} \in \mathbb{R}^{n \times n}$ 通过初等行变换进行 LU 分解所需的操作次数为：
$$
    O(n) = O \left(\pmb{E}_{to1} \right) + O \left(\pmb{E}_{to2} \right)+ \cdots + O \left(\pmb{E}_{to(n-1)} \right)\\
    \ \\
    = n(n-1) + (n-1)(n-2) + \cdots + 2 \times 1 = \sum_{i=2}^{n}{i^2} - \sum_{i=2}^{n}{i} = \sum_{i=1}^{n}{i^2} - \sum_{i=1}^{n}{i} \\
    \ \\
    = \dfrac{i(i+1)(2i+1)}{6} - \dfrac{(i+1)i}{2} = \dfrac{n^3-n}{3} \approx \dfrac{1}{3} n^3
    \tag{2-6-3}
$$
以防万一，这里给出一些自然数 n 项 p 次方的求和公式。需要注意的是，我们可以用微积分的思路来估计级数的和，但是这个估计是有偏差的：
$$
    \begin{cases}
        \sum_{i=1}^{n}{i} = C_{n+1}^{2} =  \dfrac{n(n+1)}{2}\\
        \\
        \sum_{i=1}^{n}{i^2} = C_{n+1}^{2} + 2C_{n+1}^{3} = \dfrac{n(n+1)(2n+1)}{6}\\
        \\
        \sum_{i=1}^{n}{i^3} = C_{n+1}^{2} + 6C_{n+1}^{3} + 6C_{n+1}^{4} = \dfrac{n^2(n+1)^2}{4}\\
        \\
        \sum_{i=1}^{n}{i^4} = C_{n+1}^{2} + 14C_{n+1}^{3} + 36C_{n+1}^{4} + 24C_{n+1}^{4} = \dfrac{n(n+1)(6n^3+9n^2+n-1)}{30}\\
        \\
        \sum_{i=1}^{n}{i^5} = C_{n+1}^{2} + 30C_{n+1}^{3} + 150C_{n+1}^{4} + 240C_{n+1}^{4} + 120C_{n+1}^{6} = \dfrac{n^2(n+1)(2n^3+4n^2+n-1)}{12}\\
    \end{cases}
    \tag{2-6-4}
$$
数学真是有意思，以后没准还会开个微积分的专栏（听说是 **MIT-18.01**？回头看看能不能搜到视频）。言归正传，由于矩阵消元通常发生在线性方程组求解过程（$\pmb{A}\pmb{x}^T=\pmb{b}^T$）中，即我们还有一个系数列向量 $\pmb{b}^T$ 需要同步进行行变换以及变量代回处理。在这个阶段所需的运算次数（包含在 $\pmb{b}^T$ 以及 $\pmb{U}$ 上的运算，后者的运算仅限于计算乘法所需的系数）约为：
$$
    (n-1) + 2 \times (1+2+3+\cdots+n) = n^2+2n-1 \approx n^2
    \tag{2-6-4}
$$

### 2.7 Transposes and Permutations: $\pmb{A}=\pmb{LU}$
之前提到，对矩阵进行初等行变换，等价于左乘一个变换矩阵。当时我们考虑的情况都是不包含行交换的，其实行交换也等价于一次矩阵乘法，即置换（*Permutation*）矩阵。置换矩阵的本质是**行重新排列的单位矩阵**。对于 $n$ 阶方阵，其对应的置换矩阵共有 $n!$ 种。以三阶方阵为例，三阶置换矩阵可以表示为 6 个矩阵组成的群：
$$
    \pmb{P}_{123} =
    \begin{bmatrix}
        1 & 0 & 0\\
        0 & 1 & 0\\
        0 & 0 & 1\\
    \end{bmatrix}, \ \pmb{P}_{132} =
    \begin{bmatrix}
        1 & 0 & 0\\
        0 & 0 & 1\\
        0 & 1 & 0\\
    \end{bmatrix}, \ \pmb{P}_{312} =
    \begin{bmatrix}
        0 & 1 & 0\\
        1 & 0 & 0\\
        0 & 0 & 1\\
    \end{bmatrix}\\
    \ \\
    \pmb{P}_{321} =
    \begin{bmatrix}
        0 & 0 & 1\\
        0 & 1 & 0\\
        1 & 0 & 0\\
    \end{bmatrix}, \ \pmb{P}_{231} =
    \begin{bmatrix}
        0 & 1 & 0\\
        0 & 0 & 1\\
        1 & 0 & 0\\
    \end{bmatrix}, \ \pmb{P}_{213} =
    \begin{bmatrix}
        0 & 1 & 0\\
        1 & 0 & 0\\
        0 & 0 & 1\\
    \end{bmatrix}
$$
之所以要划定矩阵群的概念，是因为**群内所有矩阵的乘积、逆矩阵依然在群内**，这是一个非常有趣的特性。除此之外，**置换矩阵的逆还等于该矩阵的转置**，即：
$$
    \pmb{P}^T = {\pmb{P}}^{-1}, \ \pmb{P} \pmb{P}^T = \pmb{P}^T \pmb{P} = \pmb{I}
    \tag{2-7-1}
$$
当 $\pmb{A}$ 并非我们先前假设的那么“完美”时，通过置换矩阵预先处理 $\pmb{A}$，将主元位置调整合适之后，可以继续进行 LU 分解：
$$
    \pmb{PA} = \pmb{LU} \ \Longrightarrow \ \pmb{A} = {\pmb{P}}^T \pmb{LU}
    \tag{2-7-2}
$$

## 3 Vector Spaces and Subspaces
### 3.1 Spaces of Vectors
#### 3.1.1 线性空间与子空间
在我国高等教育体系中（非数学系），关于这一部分的数学知识通常在应用泛函分析这门课程里才会得到真正重视，此时大伙多半已经是研究生了。而年级越高，讲课的、听课的往往都越混，尤以前者为甚。再度惋惜自己没能在初见线代的时候学习 **MIT-18.06**，后边用得多了发现线代根本没有考试时那么可怕。

对于任意一个线性空间 $\pmb{M}$ ，属于该空间的元素 $m_1$、$m_2$ 等均需满足对**数乘**与**加法**运算的封闭性，即有 $a \times m_1 + b \times m_2 \in \pmb{M}$（$\forall  \ a,b \in \mathbb{R}$），且空间内必须存在一个零元。具体到加法、数乘的方式，可以脱离常规定义，只要满足封闭性即可。同理，零元的定义也与数字 0 有所区别。具体定义在此就不详述了，感兴趣的可以参考泛函的相关知识。

上述结论同样适用于线性空间的线性子空间。这里我们需要研究一下常见空间的线性子空间，以 $\mathbb{R}^2$ 为例。以下三类都是 $\mathbb{R}^2$ 的线性子空间：

（1）$\mathbb{R}^2$ 自己。换句话说任意线性空间都是自身的子空间；

（2）过原点的直线。这里需要区分的是，$\mathbb{R}$ 虽然也是一种直线空间，但是它并不是 $\mathbb{R}^2$ 的子空间。显然我们只需一个维度的标量即可定义 $\mathbb{R}$ 中的元素，而描述任意 $\mathbb{R}^2$ 中的元素（即使是原点）都需要两个维度的标量。

（3）$\pmb{0}$。即 $\mathbb{R}^2$ 的零元。任意线性空间的零元可单独构成该空间的线性子空间。

对于 $\mathbb{R}^3$ 呢？我们很容易想到两个极端：（1）$\mathbb{R}^3$ 本身；（2）$\pmb{0}$。结合 $\mathbb{R}^2$ 的经验我们知道还有（3）任意穿过原点的平面；以及最后（4）穿过原点的直线。

#### 3.1.2 如何构成子空间
当谈论到子空间时，我们的基础语境是已经存在了一个线性空间作为“母空间”。这里给出一种面向一般情况的构建方法，以 $\mathbb{R}^3$ 为例。假设 $\pmb{P}$、$\pmb{l}$ 分别是 $\mathbb{R}^3$ 中经过原点的某平面、某直线，显然后两者都是前者的子空间。现在我们考虑集合运算“并”（$\cup$）与“交”（$\cap$），即 $\pmb{P} \cup \pmb{l}$ 与 $\pmb{P} \cap \pmb{l}$：

（1）$\pmb{P} \cup \pmb{l}$ **不是子空间**。因为在 $\pmb{P}$ 上取某向量 $\pmb{p}$ 与 $\pmb{l}$ 中的某向量相加，其结果很可能既不在平面 $\pmb{P}$ 上，也不在直线 $\pmb{l}$ 上，因此该空间对加法运算不封闭，所以不是 $\mathbb{R}^3$ 的子空间。

（2）$\pmb{P} \cap \pmb{l}$ **是子空间**。事实上 $\pmb{P} \cap \pmb{l} = \pmb{0}$。

更一般地，对于同属某一线性空间的两个子空间 $\pmb{S}$ 与 $\pmb{T}$，$\pmb{S} \cap \pmb{T}$ 仍然是子空间，但 $\pmb{S} \cup \pmb{T}$ **通常**不满足子空间条件。简单证明一下，$\forall \ \pmb{v},\pmb{w} \in \pmb{S} \cap \pmb{T}$，均有：
$$
    (a\pmb{v}+b\pmb{w}) \in \pmb{S}, \ (a\pmb{v}+b\pmb{w}) \in \pmb{T}
    \Longrightarrow \ (a\pmb{v}+b\pmb{w}) \in \pmb{S} \cap \pmb{T}, \ \forall a,b \in \mathbb{R}
    \tag{5-2-1}
$$
除了上述抽象方法，我们当然可以从一个具体的矩阵入手来构造一系列子空间。这些内容将在下一节详细展开。

### 3.2 The Nullspace of $\pmb{A}$: Solving $\pmb{Ax}=\pmb{0}$
#### 3.2.1 列空间与零空间
以矩阵 $\pmb{X} \in \mathbb{R}^{3 \times 2}$ 为例。$\pmb{X}$ 的各列都属于 $\mathbb{R}^3$，因此列向量 $\pmb{X}(:,i)$ 的全体线性组合可以构成一个 $\mathbb{R}^3$ 的线性子空间，称为 $\pmb{X}$ 的**列空间**，记为 $C(\pmb{X})$。从欧氏几何的角度来看，$C(\pmb{X})$ 是一个过原点的平面（若列向量不共线），$\pmb{X}$ 的两个列向量位于平面上。列空间有什么作用呢？我们通过一个实例来说明。

给定系数矩阵 $\pmb{A}$（这个例子将会持续存在一段时间）以及线性方程组 $\pmb{A} \pmb{x} = \pmb{b}$：
$$
    \pmb{A} = 
    \begin{bmatrix}
        1 & 1 & 2\\
        2 & 1 & 3\\
        3 & 1 & 4\\
        4 & 1 & 5\\
    \end{bmatrix} \in \mathbb{R}^{4 \times 3}
$$
这种方程数量超过未知数数量的情况称为“**超定**”（*over-determined*）系统。超定系统在大多数情况下是没有精确解的。从感性认识的角度考虑，如果我们通过正定系统获得一个精确解后，再额外添加一个或多个方程，则增添方程必须满足**某些特定条件**才能使得等式依旧成立。相应地，并非所有的系数向量都能使得方程组有解。接下来我们将从 $\pmb{b}$ 入手，深入研究 $\pmb{A} \pmb{x} = \pmb{b}$ 有解的条件。
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
以上这个拆解形式很容易让人联想到矩阵的列空间 $C(\pmb{A})$，因此有以下结论：当且仅当 $\pmb{b} \in C(\pmb{A})$ 时，方程组 $\pmb{A} \pmb{x} = \pmb{b}$ 确定有解。

说完了列空间，我们再来谈谈零空间。还是以 $\pmb{A}$ 为例，满足 $\pmb{A}\pmb{x}=\pmb{0}$ 的全体 $\pmb{x}$ 构成的空间 $N(\pmb{A})$ 称为 $\pmb{A}$ 的零空间。注意区分，对于 $\pmb{A} \pmb{x} = \pmb{b}$，$C(\pmb{A})$ 关心的是 $\pmb{b}$，而 $N(\pmb{A})$ 关心的是 $\pmb{x}$；对于 $\pmb{A} \in \mathbb{R}^{m \times n}$，$C(\pmb{A}) \in \mathbb{R}^m$ 而 $N(\pmb{A}) \in \mathbb{R}^n$。
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
显然，$\pmb{0}$ 再一次满足 $N(\pmb{A})$ 的要求。事实上 $[1,1,-1]^T$ 的所有线性组合都满足要求。可见此时 $N(\pmb{A})$ 表现为 $\mathbb{R}^3$ 中的一条直线。

需要指出，当 $\pmb{b} \ne \pmb{0}$ 时，同样可能存在很多 $\pmb{x}$ 使得方程等式平衡。但是这些 $\pmb{x}$ 无法构成线性向量空间。很简单，因为其中一定没有零元 $\pmb{0}$。所以当我们站在线性方程组的角度上回顾子空间的概念时应当明确：

（1）根据一个给出的矩阵 $\pmb{X} \in \mathbb{R}^{m \times n}$，可以从中构建两种线性空间：列空间 $C(\pmb{X}) \in \mathbb{R}^m$ 与零空间 $N(\pmb{X}) \in \mathbb{R}^n$，而不论哪种空间都必须满足“**零元存在**”；

（2）列空间的构建方法是对已有列向量进行线性组合，零空间的构建方法是在矩阵基础上寻找满足条件的解集。二者都是非常重要的子空间构建方法。

#### 3.2.2 零空间求解算法
以矩阵 $\pmb{A}$ 为例：
$$
    \pmb{A} = 
    \begin{bmatrix}
        1 & 2 & 2 & 2\\
        2 & 4 & 6 & 8\\
        3 & 6 & 8 & 10\\
    \end{bmatrix}
$$
我们来研究求解 $\pmb{A}\pmb{x}=\pmb{0}$ 的过程。由于右侧向量已经是 $\pmb{0}$ 了，是否对 $\pmb{A}$ 增广没有区别，因此我们可以直接对 $\pmb{A}$ 进行 LU 分解：
$$
    \begin{bmatrix}
        1 & 2 & 2 & 2\\
        2 & 4 & 6 & 8\\
        3 & 6 & 8 & 10\\
    \end{bmatrix} = 
    \begin{bmatrix}
        1 & 0 & 0\\
        2 & 1 & 0\\
        3 & 1 & 1\\
    \end{bmatrix}
    \begin{bmatrix}
        1 & 2 & 2 & 2\\
        0 & 0 & 2 & 4\\
        0 & 0 & 0 & 0\\
    \end{bmatrix}
$$
在这里先停一下。可以看出：（1）$\pmb{U}$ 的主元有 2 个，称该矩阵的秩（*rank*）为 2，即存在 **2 个有效方程**；（2）**未知数有 4 个**，而根据数学常识，两个方程只能求解 2 个未知数，因此 $\pmb{A}\pmb{x}=\pmb{0}$ 不受约束的自由变量还剩 4-2=2 个。

关于自由变量，我们可以从更具体的角度理解。比如本例中 $\pmb{U}(1,:)$ 的对应方程为 $2x_3+4x_4=0$，当 $x_3$ 固定时，$x_4$ 也随之固定，因此两者只有一个为“自由变量”；再接着看 $\pmb{U}(0,:)$，对应方程为 $x_1+2x_2+2x_3+2x_4=0$，由于 $x_3$、$x_4$ 均已固定，因此又进入了 $x_1$、$x_2$ 二选一的情况，也是只剩一个自由变量，因此该方程组共有 2 个自由变量。**自由变量的数量即为特解的数量**。这一点从反面很好理解：当方程组正定且有解时，系数矩阵满秩，此时没有自由变量，也就没有所谓的特解。

由于 $\pmb{U}$ 的主元分别对应于 $x_1$ 与 $x_3$，我们不妨将剩下的 $x_2$ 与 $x_4$ 定为自由变量（这样做能与自由变量的概念相对应，避免出错），并设为 1、0 或 0、1。对应的特解 $\pmb{x}_1$、$\pmb{x}_2$ 以及方程组通解 $\pmb{x}$ 为：
$$
    \pmb{x}_1 = 
    \begin{bmatrix}
        -2\\ 1\\ 0\\ 0\\
    \end{bmatrix}, \ 
    \pmb{x}_2 = 
    \begin{bmatrix}
        2\\ 0\\ -2\\ 1\\
    \end{bmatrix} \ \Longrightarrow \
    \pmb{x} = a\pmb{x}_1+b\pmb{x}_2, \ a,b \in \mathbb{R}
$$

### 3.3 The Rank and the Row Reduced Form
#### 3.3.1 行简化形式
行简化矩阵 $\pmb{R}$ 在上三角矩阵 $\pmb{U}$ 的基础上进一步化简，使得（1）所有主元均为 1；（2）主元上下位置的元素均为 0。依旧以 3.2.2 的矩阵 $\pmb{A}$ 为例：
$$
    \pmb{R} = 
    \begin{bmatrix}
        1 & -1 & 0\\
        0 & 0.5 & 0\\
        0 & 0 & 0\\
    \end{bmatrix} \pmb{U} = 
    \begin{bmatrix}
        1 & 2 & 0 & -2\\
        0 & 0 & 1 & 2\\
        0 & 0 & 0 & 0\\
    \end{bmatrix}
$$
首先需要指出，$\pmb{Ax}=\pmb{0}$、$\pmb{Ux}=\pmb{0}$ 以及 $\pmb{Rx}=\pmb{0}$ 这三个方程的解是相同的，因此初等行变换并不会导致线性方程组的解产生变化。其次，通过上述三者获取解的难度显然是逐次降低的。最后，为了说明行简化矩阵求解的优越性，请注意观察 $\pmb{R}$ 以及 $\pmb{x}_1$、$\pmb{x}_2$，是否发现其中的元素高度相似？自由变量位置分别为 0 或 1，而主元位置是自由变量系数的相反数？

这个“奇技淫巧”的原理如下：一个理想状态下的 $\pmb{R}_i^{m \times n}$ 可以表示为分块矩阵，其中 $\pmb{I}$ 表示主元矩阵（单位阵），$\pmb{F}$ 表示自由变量矩阵：
$$
    \pmb{R}_i = 
    \begin{bmatrix}
        \pmb{I}_{r \times r} & \pmb{F}_{r \times (n-r)}\\
        \pmb{0}_{(m-r) \times r} & \pmb{0}_{(m-r) \times (n-r)}\\
    \end{bmatrix}
    \tag{3-3-1}
$$
不失一般性地，我令 $\pmb{F}$ 为一个非方阵，它的底部甚至可以是一些纯 0 行，只要保证 $\pmb{R}_i$ 与 $\pmb{F}$ 的行数相等即可。接下来我们考察方程 $\pmb{R}_i\pmb{x}=\pmb{0}$ 的解。由于方程是欠定的，因此存在一些自由变量和秩次数个特解。换句话说我们可以构建一个零空间矩阵 $\pmb{N} \in \mathbb{R}^{n \times (n-r)}$，使得 $\pmb{R}_i\pmb{N}=\pmb{0} \in \mathbb{R}^{m \times (n-r)}$，$\pmb{N}$ 中的每一列构成线性方程组的一个特解。这一点同时照应了上一节提到的“特解数量为自由变量个数”的结论：
$$
    \pmb{N} = 
    \begin{bmatrix}
        -\pmb{F}_{r \times (n-r)}\\ \pmb{I}_{(n-r) \times (n-r)}\\
    \end{bmatrix}
    \tag{3-3-2}
$$
我们不妨再换个角度来审视这个过程，$\pmb{R}_i\pmb{x}=\pmb{0}$ 可以展开为：
$$
    \begin{bmatrix}
        \pmb{I} & \pmb{F}\\
        \pmb{0} & \pmb{0}\\
    \end{bmatrix}
    \begin{bmatrix}
        \pmb{x}_{pivot}\\ \pmb{x}_{free}
    \end{bmatrix} = \pmb{0}
    \tag{3-3-3}
$$
显然该等式成立的条件是：主元变量为自由变量的相反数，自由变量构成单位阵（即挨个等于1，剩下为0）。

上述步骤说明了行简化矩阵与特解之间的对应关系，但前提是理想情况。对于需要列交换的情况还成立吗？显然成立，只不过我们需要从一个更具体的视角入手。以本节初给出的 $\pmb{R}$ 为例，将非主元的非零元素用字母替代，方便我们看清本质：
$$
    \pmb{R} = 
    \begin{bmatrix}
        1 & r_1 & 0 & r_2\\
        0 & 0 & 1 & r_3\\
        0 & 0 & 0 & 0\\
    \end{bmatrix}
$$
自由变量分别为第 2、第 4 变量，因此提前写出两个特解：
$$
    \pmb{x}_1 = 
    \begin{bmatrix}
        ?\\ 1\\ ?\\ 0\\
    \end{bmatrix}, \ 
    \pmb{x}_2 = 
    \begin{bmatrix}
        ?\\ 0\\ ?\\ 1\\
    \end{bmatrix}
$$
第一自由变量所在列为 $\pmb{R}(:,1)$，且仅 $\pmb{R}(0,1)$ 不为 0，则在特解 $\pmb{x}_1$ 中由上至下依次填入 $-r_1$ 与 $0$；第二自由变量所在列为 $\pmb{R}(:,3)$，在特解 $\pmb{x}_2$ 中由上至下依次填入 $-r_2$、$-r_3$。以由于我们已经把数值变量化了，因此通过表达式可以轻松证明以上操作步骤的成立。

#### 3.3.2 矩阵的秩
矩阵的形状可以千奇百怪，但是其中包含的有效信息往往并不像表面看上去那么丰富，真正决定有效信息量的参数是秩。对于矩阵 $\pmb{M} \in \mathbb{R}^{m \times n}$，$r(\pmb{M})$ 表示主元的个数，它是一个不大于 $m$ 或 $n$ 的正整数。$r=m$、$r=n$ 分别对应于“行满秩”与“列满秩”两种情况：

（1）行满秩：把 $\pmb{M}$ 化简为 $\pmb{R}$ 后，每一行都有一个主元。即**不存在全零行**。仅适用于 $m \leqslant n$ 的情况（正定或欠定，矩阵形如横着的长方形）:

（2）列满秩：$\pmb{R}$ 的每一列均有一个主元。即**不存在自由变量**。仅适用于 $m \geqslant n$ 的情况（正定或超定，矩阵形如竖着的长方形），事实上 $\pmb{R}$ 的有效部分为单位阵。

（3）**矩阵的秩次数不随矩阵的转置而变化**。即 $r(\pmb{M})=r\left(\pmb{M}^T\right)$。

目前想要证明结论（3）有些困难，我们当然可以用字母表达式的方式在 $\mathbb{R}^{3 \times 3}$、$\mathbb{R}^{3 \times 4}$ 等矩阵上进行验证，无非是表达式比较冗长罢了。未来我们一定会用更成熟的理论来解释这个问题。

### 3.4 The Complete Solution to $\pmb{Ax}$=$\pmb{b}$
#### 3.4.1 可解性与解的结构
先前我们讨论了线性方程组 $\pmb{Ax}=\pmb{b}$ 有解的条件，即 $\pmb{b} \in C(\pmb{A})$，之后以 $\pmb{Ax}=\pmb{0}$ 为例讲解了**齐次线性方程组**的解法。现在来看看非齐次方程组的求解流程。例如：
$$
    \pmb{A} = 
    \begin{bmatrix}
        1 & 2 & 2 & 2\\
        2 & 4 & 6 & 8\\
        3 & 6 & 8 & 10\\
    \end{bmatrix}, \ \pmb{b} = 
    \begin{bmatrix}
        1\\ 5\\ 6\\
    \end{bmatrix}
$$
（1）通过 LU 分解化简增广系数矩阵 $\tilde{\pmb{A}}$：
$$
    \pmb{E}_{to2} \pmb{E}_{to1} \pmb{A} = 
    \begin{bmatrix}
        1 & 0 & 0\\
        0 & 1 & 0\\
        0 & -1 & 0\\
    \end{bmatrix}
    \begin{bmatrix}
        1 & 0 & 0\\
        -2 & 1 & 0\\
        -3 & 0 & 1\\
    \end{bmatrix} \pmb{A} = 
    \begin{bmatrix}
        1 & 2 & 2 & 2\\
        0 & 0 & 2 & 4\\
        0 & 0 & 0 & 0\\
    \end{bmatrix} = \pmb{U}\\
    \ \\
    \tilde{\pmb{A}} = 
    \begin{bmatrix}
        1 & 0 & 0\\
        2 & 1 & 0\\
        3 & 1 & 1\\
    \end{bmatrix}
    \begin{bmatrix}
        1 & 2 & 2 & 2 \ | \ 1\\
        0 & 0 & 2 & 4 \ | \ 3\\
        0 & 0 & 0 & 0 \ | \ 0\\
    \end{bmatrix}
$$
（2）将所有自由变量（本例中为 $x_2$、$x_4$）设为 0，求出特解 $\pmb{x}_p$：
$$
    \begin{cases}
        x_1+2x_3=1\\
        2x_3=3\\
    \end{cases} \ \Longrightarrow \ \pmb{x}_p =
    \begin{bmatrix}
        -2\\ 0\\ 1.5\\ 0\\
    \end{bmatrix}
$$
（3）求出 $\pmb{A}$ 的零空间 $N(\pmb{A})$ 作为通解 $\pmb{x}_n$：
$$
    \pmb{Ax}=\pmb{0} \ \Longrightarrow \ \pmb{Ux}=\pmb{0} \ \Longrightarrow \ \pmb{Rx}=\pmb{0}, \ \pmb{R} = 
    \begin{bmatrix}
        1 & 2 & 0 & -2\\
        0 & 0 & 1 & 2\\
        0 & 0 & 0 & 0\\
    \end{bmatrix}\\
    \ \\
    \pmb{x}_n = k_1 
    \begin{bmatrix}
        -2\\ 1\\ 0\\ 0\\
    \end{bmatrix} + k_2 
    \begin{bmatrix}
        2\\ 0\\ -2\\ 1\\
    \end{bmatrix}, \ k_1,k_2 \in \mathbb{R}
$$
（4）合并通解与特解，得到全解 $\pmb{x} = \pmb{x}_p + \pmb{x}_n$：
$$
    \pmb{x} = 
    \begin{bmatrix}
        -2\\ 0\\ 1.5\\ 0\\
    \end{bmatrix} + k_1 
    \begin{bmatrix}
        -2\\ 1\\ 0\\ 0\\
    \end{bmatrix} + k_2 
    \begin{bmatrix}
        2\\ 0\\ -2\\ 1\\
    \end{bmatrix}, \ k_1,k_2 \in \mathbb{R}
$$
$$
    \pmb{A}(\pmb{x}_p+\pmb{x}_n) = \pmb{b} + \pmb{0} = \pmb{b}
    \tag{3-4-1}
$$
在本例中，全解表现为 $\mathbb{R}^4$ 空间中的一个二维平面，注意它并非子空间。我们可以说 $N(\pmb{A})$ 是 $\mathbb{R}^4$ 的一个子空间，它在欧式几何上表现为过零元的一个平面。而 $\pmb{x}$ 是一个经过特解点 $\pmb{x}_p$ 的平面，相当于将 $N(\pmb{A})$ 进行平移后的结果。因为这个空间内没有零元，所以它不是子空间。

#### 3.4.2 矩阵秩次对于解的影响
我们知道对于一个系数矩阵 $\pmb{A} \in \mathbb{R}^{m \times n}$，其秩次数 $r(\pmb{A})$ 可决定矩阵不满秩、行满秩或列满秩。这三种情况分别对应了线性方程组 $\pmb{Ax}=\pmb{b}$ 的不同解情况。

首先是列满秩（$r(\pmb{A}) = n$）：结合 3.3.2 节末尾的结论，此时不存在自由变量，$N(\pmb{M})$ 仅有 $\pmb{0}$ 一个向量，即不存在通解 $\pmb{x}_n$。线性方程组 $\pmb{Ax}=\pmb{b}$ 只有 1 个特解 $\pmb{x}_p$ 或无解（$\pmb{b} \notin C(\pmb{A})$）;

其次是行满秩（$r(\pmb{A}) = m$）：行满秩时行简化矩阵没有全零行。与列满秩不同，此时方程组的可解性对于 $\pmb{b}$ **没有任何要求**（不论超定或正定）。自由变量有 $n-m$ 个，即通解 $\pmb{x}_n$ 由 $n-m$ 项组成，全解要么只有 1 个（特解，$m=n$），要么有无数个（特解 + 通解）；

这里有必要强调一种特殊情况：满秩（$r(\pmb{A}) = m = n$）。此时方程组的系数矩阵是一个可逆的方阵，其行简化矩阵为单位阵。由于它是列满秩，因此 $N(\pmb{A})$ 依然只有 $\pmb{0}$。而它又是行满秩，故 $\pmb{Ax}=\pmb{b}$ 必有解，且解为 $\pmb{A}^{-1} \pmb{b}$；

最后谈谈不满秩的情况（$r(\pmb{A}) < m, \ r(\pmb{A}) < n$）：此时行简化矩阵有全零行（当且仅当 $\pmb{b}\in C(\pmb{A})$ 时有解），也有自由变量（通解有无数个）。即要么无解，要么有无数个解。