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

# 1 Introduction to Vectors
## 1.2 Length and Dot Products
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

### 1.2.1 向量单位化
在此基础上，我们可以定义单位向量 $\pmb{u}$ 以及将一个任意非零向量 $\pmb{v}$ 单位化的过程：
$$
    \|\pmb{u}\|_2 = 1, \ \pmb{u}_{\pmb{v}} = \dfrac{\pmb{v}}{\|\pmb{v}\|_2}
    \tag{1-2-3}
$$

### 1.2.2 正交向量
对于点乘结果为 0 的特殊向量组合，将其称为正交向量组，即二者相互垂直：
$$
    \left<\pmb{x}, \pmb{y} \right> = 0 \ \Longleftrightarrow \ \pmb{x} \perp \pmb{y}
    \tag{1-2-4}
$$

### 1.2.3 向量夹角
点乘并非只能判断是否正交，还能进一步量化向量间的夹角。对于单位向量 $\pmb{u}$、$\pmb{v}$：
$$
    \pmb{u} \cdot \pmb{v} = cos \theta
    \tag{1-2-5}
$$

### 1.2.4 余弦公式 & 柯西不等式
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