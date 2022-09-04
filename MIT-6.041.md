---
html:
    toc: true
print_background: true
---

# MIT-6.041：概率论
[课程视频][total_lesson] 在这里！

[total_lesson]: https://www.bilibili.com/video/BV1Ea411S7ug?vd_source=1d3139f5ceaa14318494042a5c956e44
***

## 1. Probability Models and Axioms
### 1.1 样本空间
样本空间 $\Omega$ 表示一个包含各种可能产生的结果的集合。

> Set must be **mutually exclusive** and **collectively exhaustive**.

样本空间应当包含所有可能出现的结果，同时各个结果在单次实验过程中是相互排斥的。

类似线代中的向量空间、泛函中的线性空间，样本空间自然也存在“子空间”。$\Omega$ 的子空间 $A$ 满足以下性质（公理）：

（1）非负性：$P(A) \geqslant 0$；

（2）标准化：$P(\Omega)=1$；

（3）可加性：$A \cap B= \varnothing \ \Longrightarrow \ P\left(\{A,B\}\right) = P(A \cup B)=P(A)+P(B)$。该性质可以进一步拓展，若 $A_1,A_2,\cdots,A_n$ 是互不相交的，则 $P\left(\{A_1,A_2,\cdots,A_n\}\right) = P\left( \bigcup_{i=1}^n A_i \right) = P(A_1)+P(A_2)+\cdots+P(A_n)$。

### 1.2 离散均匀定律
假设样本空间 $\Omega$ 中的所有结果是等可能发生的，则有：
$$
    P(A) = \dfrac{number \ of \ elements \ of \ A}{total \ number \ of \ sample \ points}
    \tag{1-2-1}
$$