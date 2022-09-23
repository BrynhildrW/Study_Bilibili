## 5.3 Cramer's Rule, Inverse and Volumes
### 5.3.1 逆矩阵公式
在接触行列式的概念之后，我们终于可以获得一个通用的矩阵求逆公式（尽管这个公式操作起来未必就比 LU 分解或者行变换简单），首先以二阶方阵为例：
$$
    \pmb{A}^{-1} = 
    \begin{bmatrix}
        a & b\\ c & d\\
    \end{bmatrix}^{-1} = \dfrac{1}{ad-bc}
    \begin{bmatrix}
        d & -b\\ -c & a\\
    \end{bmatrix} = \dfrac{1}{\det(\pmb{A})}
    \begin{bmatrix}
        C_{1,1} & C_{1,2}\\ C_{2,1} & C_{2,2}
    \end{bmatrix}^T
$$
推广到 $\pmb{A} \in \mathbb{R}^{n \times n}$：
$$
    \pmb{A}^{-1} = \dfrac{1}{\det(\pmb{A})}\pmb{C}^T, \ \pmb{C} = 
    \begin{bmatrix}
        C_{1,1} & C_{1,2} & \cdots & C_{1,n}\\
        C_{2,1} & C_{2,2} & \cdots & C_{2,n}\\
        \vdots & \vdots & \ddots & \vdots\\
        C_{n,1} & C_{n,2} & \cdots & C_{n,n}\\
    \end{bmatrix}
    \tag{5-3-1}
$$
接下来证明（5-3-1），即证明：
$$
    \pmb{A}\pmb{C}^T = \det(\pmb{A}) \pmb{I}
$$
$$
    \underbrace{
        \begin{bmatrix}
            a_{1,1} & a_{1,2} & \cdots & a_{1,n}\\
            a_{2,1} & a_{2,2} & \cdots & a_{2,n}\\
            \vdots & \vdots & \ddots & \vdots\\
            a_{n,1} & a_{n,2} & \cdots & a_{n,n}\\
        \end{bmatrix}}_{\pmb{A}}
    \underbrace{
        \begin{bmatrix}
            C_{1,1} & C_{2,1} & \cdots & C_{n,1}\\
            C_{1,2} & C_{2,2} & \cdots & C_{n,2}\\
            \vdots & \vdots & \ddots & \vdots\\
            C_{1,n} & C_{2,n} & \cdots & C_{n,n}\\
        \end{bmatrix}}_{\pmb{C}^T} = 
    \begin{bmatrix}
        \det{\pmb{A}} & & & \\
        & \det{\pmb{A}} & & \\
        & & \ddots & \\
        & & & \det{\pmb{A}}\\
    \end{bmatrix}
$$
我们先观察等式右边矩阵的第一个元素来源 $a_{1,1}C_{1,1} + a_{1,2}C_{1,2} + \cdots + a_{1,n}C_{1,n}$，不难发现这就是 $\det(\pmb{A})$ 的计算公式，同理右矩阵主对角线上的其它元素分别对应于依照 $\pmb{A}$ 的**不同行**展开的行列式计算公式。现在仅剩的问题是，矩阵其它部分都是 0 吗？换句话说，当展开行与余子式对应行不匹配时，矩阵元素与代数余子式的乘积是否为 0？即是否有：
$$
    a_{i,1}C_{j,1} + a_{i,2}C_{j,2} + \cdots + a_{i,n}C_{j,n} = \sum_{k=1,i \ne j}^{n} a_{i,k}C_{j,k} = 0
    \tag{5-3-2}
$$
当然我都给编号了，所以它是成立的。想直接证明这个等式是非常困难的，但是（5-3-2）的形式与行列式计算很相似，因此我们通过构造特殊矩阵的方式加以说明。

我们具体一点，取 $i=1$，$j=n$ 的情况进行分析，即第一行展开，但是用的是最后一行各元素对应的代数余子式。我们设想这样一种矩阵 $\pmb{X}$：第一行与最后一行相等。显然 $\pmb{X}$ 是奇异矩阵，即 $\det(\pmb{X})=0$。我们分别按第一行和最后一行展开行列式：
$$
    \begin{align}
    \notag \det(\pmb{A}) =
    \begin{vmatrix}
        a_{1,1} & a_{1,2} & \cdots & a_{1,n}\\
        a_{2,1} & a_{2,2} & \cdots & a_{2,n}\\
        \vdots & \vdots & \ddots & \vdots\\
        a_{n-1,1} & a_{n-1,2} & \cdots & a_{n-1,n}\\
        a_{n,1} & a_{n,2} & \cdots & a_{n,n}\\
    \end{vmatrix} &= a_{1,1}C_{1,1} - a_{1,2}C_{1,2} + \cdots + (-1)^{n+1}a_{1,n}C_{1,n} = 0\\
    \notag \ \\
    \notag &= a_{n,1}C_{n,1} - a_{n,2}C_{n,2} + \cdots + (-1)^{n+1}a_{n,n}C_{n,n} = 0
    \end{align}
$$
由于 $\pmb{A}$ 的第一行与最后一行相等，因此有：
$$
    C_{1,1} = (-1)^m C_{n,1}, C_{1,2} = (-1)^m C_{n,2},\ \cdots, \ C_{1,n} = (-1)^m C_{n,n}, \ (-1)^m \sum_{k=n}^n a_{n,k} C_{1,k}=0
$$
当我们构建 $\pmb{A}$ 的第 $i$、$j$ 行相等时，对应于（5-3-2）中 $i$、$j$ 取相应值的情况。至此，我们已经得到并证明了面向一般方阵的逆矩阵求解公式。

### 5.3.2 Cramer 法则
首先需要声明，*Cramer* 法则在线性方程组求解过程中并不实用，计算量大、数值不稳定等等都是它的缺点。据说在微分几何中有它的用武之地，不过这也不是我研究的方向。

在普通线代领域，*Cramer* 法则的理论价值在于方程组有解性的判断：对于方阵系数的方程组（方程个数与未知量个数相等），当系数行列式不等于 0 时，方程组具有唯一解；若系数行列式为 0，则方程组无解或有无数解。

接下来给出基本说明。对于有唯一解的 $n$ 阶线性方程组 $\pmb{Ax}=\pmb{b}$，其解为 $\pmb{x}=\pmb{A}^{-1} \pmb{b} = \dfrac{1}{\det{\pmb{A}}}\pmb{C}^T \pmb{b}$。*Cramer* 法则认为方程解可以由 $n$ 个分量构成：
$$
    \pmb{x}_1 = \dfrac{\det(\pmb{B}_1)}{\det(\pmb{A})}, \ \pmb{x}_2 = \dfrac{\det(\pmb{B}_2)}{\det(\pmb{A})}, \ \cdots \ , \ \pmb{x}_n = \dfrac{\det(\pmb{B}_n)}{\det(\pmb{A})}
    \tag{5-3-3}
$$
其中方阵 $\pmb{B}_1$、$\pmb{B}_2$、$\cdots$、$\pmb{B}_n$ 分别表示 $\pmb{A}$ 中第 $1$、$2$、$\cdots$、$n$ 列被 $\pmb{b}$ 替换后构成的方阵：
$$
    \pmb{B}_1 = 
    \begin{bmatrix}
        b_1 & a_{1,2} & \cdots & a_{1,n}\\
        b_2 & a_{2,2} & \cdots & a_{2,n}\\
        \vdots & \vdots & \ddots & \vdots\\
        b_n & a_{n,2} & \cdots & a_{n,n}\\
    \end{bmatrix}, \ \pmb{B}_2 = 
    \begin{bmatrix}
        a_{1,1} & b_1 & \cdots & a_{1,n}\\
        a_{2,1} & b_2 & \cdots & a_{2,n}\\
        \vdots & \vdots & \ddots & \vdots\\
        a_{n,1} & b_n & \cdots & a_{n,n}\\
    \end{bmatrix}, \ \cdots \ \pmb{B}_n = 
    \begin{bmatrix}
        a_{1,1} & a_{1,2} & \cdots & b_1\\
        a_{2,1} & a_{2,2} & \cdots & b_2\\
        \vdots & \vdots & \ddots & \vdots\\
        a_{n,1} & a_{n,2} & \cdots & b_n\\
    \end{bmatrix}
$$

### 5.3.3 行列式与几何
在阶数较低的时候，方阵行列式与欧式几何之间存在紧密联系（高维的时候相关规律也许依然存在，只是没那么好理解和运用了）。以二阶（平面）为例：

![三角形面积示意图](/figures/triangle.png)

如果不给定坐标系，我们求三角形面积的常用办法是“底乘高除以二”。如今我们在坐标系中，手拥线代这一把利器，一切都变得不一样了。如图所示的三角形，其中一个顶点在原点上，剩余两点的坐标分别为 $(a,b)$、$(c,d)$。我们先按照纯几何的角度分析一下问题，在完善辅助线后，不难发现目标区域面积 $S$ 为矩形减去周边三个三角形的面积：
$$
    S_{\triangle OAB} = bc - \dfrac{1}{2}ab - \dfrac{1}{2}cd - \dfrac{1}{2}(b-d)(c-a) = \dfrac{bd-ac}{2}
$$ 
这个形式与二阶矩阵的行列式非常相似：
$$
    \pmb{A} = 
    \begin{vmatrix}
        a & b\\ c & d\\
    \end{vmatrix}, \ S_{\triangle OAB} = \dfrac{1}{2} \left|\det{(\pmb{A})}\right|
$$
之所以要取绝对值，是因为行列式本身并不像面积一样满足非负性。数值的正负与图形的方向（或立体的手型）有关。

由这个案例可以看出，线性代数和向量对于几何分析带来的巨大帮助。当三角形顶点不在原点时，以向量的角度，无非是 $\vec{OA}$、$\vec{OB}$ 的起点发生了平移，三边之间的位置关系是不变的，因此只需将对应坐标修正至以原点为顶点的情况即可继续套用公式。例如三点坐标分别为 $A(x_1,y_1)$、$B(x_2,y_2)$ 以及 $C(x_3,y_3)$ 时，将 $A$ 平移至原点：
$$
    \begin{align}
        \notag
        \begin{cases}
            A(x_1,y_1) \rightarrow A^{'}(0,0)\\
            B(x_2,y_2) \rightarrow B^{'}(x_2-x_1,y_2-y_1)\\
            C(x_3,y_3) \rightarrow C^{'}(x_3-x_1,y_3-y_1)\\
        \end{cases}, \ S_{\triangle ABC} &= \dfrac{1}{2}
        abs\left(
            \begin{vmatrix}
                x_2-x_1 & y_2-y_1\\
                x_3-x_1 & y_3-y_1\\
            \end{vmatrix}
        \right)\\
        \notag &= \left|x_1(y_1-y_3) + x_2(y_3-y_1) + x_3(y_1-y_2)\right|
    \end{align}
    \tag{5-3-4}
$$
或者我们也可以用另外一个公式：
$$
    \pmb{S} = 
    \begin{bmatrix}
        x_1 & y_1 & 1\\
        x_2 & y_2 & 1\\
        x_3 & y_3 & 1\\
    \end{bmatrix}, \ S_{\triangle ABC}=\dfrac{1}{2}\left|\det(\pmb{S})\right|
    \tag{5-3-5}
$$
把（5-3-5）的 $\det(\pmb{S})$ 按第一列展开，结果与（5-3-4）是一样的。类似的结论可以沿用至平行四边形。因为在二维坐标系中决定一个平行四边形其实仅需要三个点即可，相应区域的面积即为三角形面积的两倍，故不再赘述。