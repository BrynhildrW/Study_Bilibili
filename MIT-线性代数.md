---
html:
    toc: true
print_background: true
---

# MIT-18.06：线性代数
[课程视频][total_lesson] 和 [配套教材][document]

如果我大二的时候学的是这套线代教程，我的人生轨迹就会发生变化：可能绩点会有大的改观，可能拿到保研资格，可能不会因为考研不考数学而跟医工院深度绑定，可能离开天津前往上海，可能硕士毕业就参加工作，可能不会在一个不喜欢的行业继续读博苟延残喘。

这份文档并非是面面俱到的笔记，而是我在学习相关课程中发现的重要结论与新的收获。

若无特殊说明，本文中大写字母表示二维矩阵，小写字母表示行向量。

[total_lesson]: https://www.bilibili.com/video/BV16Z4y1U7oU?p=1&vd_source=16d5baed24e8ef63bea14bf1532b4e0c
[document]: https://pan.baidu.com/s/139zZkqWUxa-sHpKJzU5vdg

***
## 1. 方程组的几何解释
### 1.1 矩阵乘法的“列视图”
对于多元一次线性方程组：
$$
    \begin{cases}
        a_{11}x_1 + a_{12}x_2 + a_{13}x_3 = b_{1}\\
        a_{21}x_1 + a_{22}x_2 + a_{23}x_3 = b_{2}\\
        a_{31}x_1 + a_{32}x_2 + a_{33}x_3 = b_{3}\\
    \end{cases}
    \tag {1-1}
$$
式 (1-1) 可以改写为矩阵乘法形式：
$$
    \begin{pmatrix}
        a_{11} & a_{12} & a_{13}\\
        a_{21} & a_{22} & a_{23}\\
        a_{31} & a_{32} & a_{33}\\
    \end{pmatrix}
    \begin{pmatrix}
        x_1\\
        x_2\\
        x_3\\
    \end{pmatrix} = 
    \begin{pmatrix}
        b_1\\
        b_2\\
        b_3\\
    \end{pmatrix}
    \tag {1-2}
$$
$$
    \pmb{A} \pmb{x}^T = \pmb{b}^T
    \tag{1-3}
$$
在面对式 (1-2) 这种类型的矩阵乘法时，我们通常习惯以“**元素的加权和**”这一视角来描述一种“数群”至“数字”的运算过程，即存在这样的思维习惯：$b_1 = a_{11}x_1 + a_{12}x_2 + a_{13}x_3$。此时我们接受系数矩阵 $\pmb{A}$ 的方式是逐行进行的，是一种略显复杂的动态过程。

“**列视图**”是一种我以前未曾发现的全新视角，其本质为矩阵列的线性组合，基于列视图，我们可以把式 (1-2) 改写为：
$$
    x_1
    \begin{pmatrix}
        a_{11}\\
        a_{21}\\
        a_{31}\\
    \end{pmatrix} + 
    x_2
    \begin{pmatrix}
        a_{12}\\
        a_{22}\\
        a_{32}\\
    \end{pmatrix} + 
    x_3
    \begin{pmatrix}
        a_{13}\\
        a_{23}\\
        a_{33}\\
    \end{pmatrix} = 
    \begin{pmatrix}
        b_1\\
        b_2\\
        b_3\\
    \end{pmatrix}
    \tag {1-4}
$$
显然，在面对形如“矩阵右乘向量”之类的运算时，使用“**列视图**”能够更好地简化思路，在实际应用中帮助我们明确信息流传递的真实意义。

***
## 2. 矩阵消元
### 2.1 矩阵乘法的“行视图”
对于矩阵乘法 $\pmb{aX}=\pmb{b}$ ：
$$
    \begin{pmatrix}
        a_1 & a_2 & a_3
    \end{pmatrix}
    \begin{pmatrix}
        x_{11} & x_{12} & x_{13}\\
        x_{21} & x_{22} & x_{23}\\
        x_{31} & x_{32} & x_{33}\\
    \end{pmatrix} = 
    \begin{pmatrix}
        b_1 & b_2 & b_3
    \end{pmatrix}
    \tag {2-1}
$$
与 (1-4) 不同，这是“矩阵左乘向量”的运算形式，因此可以使用“**行视图**”对其进行观察与分析：
$$
    a_1
    \begin{pmatrix}
        x_{11} & x_{12} & x_{13}
    \end{pmatrix} + 
    a_2
    \begin{pmatrix}
        x_{21} & x_{22} & x_{23}
    \end{pmatrix} + 
    a_3
    \begin{pmatrix}
        x_{31} & x_{32} & x_{33}
    \end{pmatrix} = 
    \begin{pmatrix}
        b_1 & b_2 & b_3
    \end{pmatrix}
    \tag{2-2}
$$
掌握矩阵的“**行视图**”与“**列视图**”有助于我们进一步理解矩阵初等变换以及逆矩阵的本质。

### 2.2 


***