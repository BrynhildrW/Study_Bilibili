---
html:
toc: true
print_background: true
---

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
\begin{align}
\notag
\begin{bmatrix}
1 & 0\\ -2 & 1\\
\end{bmatrix}
\begin{bmatrix}
x_{11} & x_{12}\\
x_{21} & x_{22}\\
\end{bmatrix} &= 
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
\end{bmatrix}}_{2nd \ row \ of \ \pmb{Y}}\\
\notag
\ \\
\notag
\begin{bmatrix}
1 & 0\\
\end{bmatrix}
\begin{bmatrix}
x_{11} & x_{12}\\
x_{21} & x_{22}\\
\end{bmatrix} &= 1 \times
\begin{bmatrix}
x_{11} & x_{12}\\
\end{bmatrix} + 0 \times
\begin{bmatrix}
x_{21} & x_{22}\\
\end{bmatrix}\\
\notag
\ \\
\notag
\begin{bmatrix}
-2 & 1\\
\end{bmatrix}
\begin{bmatrix}
x_{11} & x_{12}\\
x_{21} & x_{22}\\
\end{bmatrix} &= -2 \times
\begin{bmatrix}
x_{11} & x_{12}\\
\end{bmatrix} + 1 \times
\begin{bmatrix}
x_{21} & x_{22}\\
\end{bmatrix}
\end{align}
$$
对应地，单次初等列变换等价于 $\pmb{X}$ 右乘一个操作矩阵 $\pmb{C}$，例如矩阵**第一列除以 2**、**第二列乘 3 以后加上第一列变换后的 5 倍**：
$$
\begin{bmatrix}
\dfrac{x_{11}}{2} & 3x_{12}+\dfrac{5 x_{11}}{2}\\
\\
\dfrac{x_{21}}{2} & 3x_{22}+\dfrac{5 x_{21}}{2}\\
\end{bmatrix} \ \Longleftrightarrow \
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