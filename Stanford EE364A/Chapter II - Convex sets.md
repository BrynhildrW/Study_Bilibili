# Chapter II: Convex Sets
## 2.1 Affine and convex sets
### 2.1.1 Lines and line segments
假设 $\pmb{x}_1 \ne \pmb{x}_2$ 是 $\mathbb{R}^n$ 空间中的两个广义点（简单起见，以下段落中的线段、点等几何描述均为广义的），满足下式的点 $\pmb{y}$ 可构成一条同时经过 $\pmb{x}_1$ 与 $\pmb{x}_2$ 的直线（line）：
$$
    \pmb{y} = \theta \pmb{x}_1 + (1 - \theta) \pmb{x}_2, \ \ \theta \in \mathbb{R}
    \tag{1}
$$
关于式 (1)，我们可以从平面几何的角度加以理解：

![第二章图1](/Stanford%20EE364A/C2-f1.png)

向量 $\vec{BA}=\pmb{x}_1$，$\vec{BC}=\pmb{x}_2$。$\vec{BD}=\theta \pmb{x}_1$，$\vec{BE}=\vec{BD}+\vec{BF}$。已知点 $E$ 在 $l_{AC}$ 上，即证 $\vec{BF}=(1-\theta) \pmb{x}_2$。通过相似三角形性质易得：
$$
    \dfrac{BF}{BC}=\dfrac{AE}{AC}=\dfrac{AD}{AB}=1-\dfrac{DB}{AB}=1-\theta
$$
原式得证，即满足式 (1) 的 $\pmb{y}$ 的轨迹是一条经过 $\pmb{x}_1$ 与 $\pmb{x}_2$ 的直线。当参数 $\theta \in [0,1]$ 时，$\pmb{y}$ 构成 $\pmb{x}_1$ 与 $\pmb{x}_2$ 之间的闭合线段（line segment）。

从另一个角度看：
$$
    \pmb{y} = \pmb{x}_2 + \theta(\pmb{x}_1-\pmb{x}_2)
    \tag{2}
$$
$\pmb{y}$ 可理解为基点（base point）与放缩后的方向（direction）之和：基点为 $\pmb{x}_2$（$\theta=0$），放缩系数为 $\theta$, 方向为 $\pmb{x}_1-\pmb{x}_2$。因此 $\theta$ 决定了 $\pmb{y}$ 在 $l_{\pmb{x}_1 \pmb{x}_2}$ 上的移动方式：

当 $\theta$ 从 0 增加至 1 时，$\pmb{y}$ 从 $\pmb{x}_2$ 移动至 $\pmb{x}_1$；
当 $\theta>1$ 时，$\pmb{y}$ 继续移动直至落在 $\pmb{x}_1$ 点外的直线上。

![第二章图2](/Stanford%20EE364A/C2-f2.png)
***

### 2.1.2 Affine sets