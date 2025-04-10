## 3.3 有噪信道编码定理
### 3.3.1 有噪信道编码定理
根据之前对离散信道的分类介绍可知，有噪信道，意味着发送端输入的符号有一定概率被噪声干扰，导致接收端输出错误的符号。例如 BSC 信道，教材评价说 “对于一般的传输系统，当 $p=0.01$ 时错误概率太高，不可接受”。好家伙，原来这就不可接受了。想想通讯型脑-机接口，能做到 1% 的误码率那不得乐翻天。言归正传，本节通过一个简单案例，介绍了平衡传输可靠性和有效性的理论实现方法。

**简单重复编码**，正如其名，是一种简单的提高通信可靠性的编码方法。以 BSC 信道为例，在发送端重复输入多次相同的消息，在接收端使用相应的判别方法，能够减小错误概率。比如重复三次，把输入的 “0”、“1” 替换成 “000”、“111”，可能收到共计 8 种码字。接收端通常采用的译码准则是**最大似然译码准则**（Maximum Likelihood Decoding, MLD）：对于接受到的码字 $R$，MLD 将所有输入码字 $S$ 中转移到 $R$ 的信道转移概率最大的一个判为 $S^*$，即 $P(R|S^*) \geqslant P(R|S^{'})$，（$S^* \ne S^{'}$）。

在当前案例中，输出码字若包括两个及以上的 “0” 或 “1”，就会被识别为 “0” 或 “1”。换句话说，单个字符的编码长度增加了，同时错误概率 $P_E$ 从 $0.01$ 降低为：$p^3 + C_3^1 \bar{p} p^2 = 3 \times 10^{-4}$，大幅提高了系统的可靠性。不难发现，如果继续增加编码长度 $n$，错误概率 $P_E$ 能够进一步减小：
$$
    \begin{align}
        \notag n&=5, \ \ P_E = p^5 + C_5^1 \bar{p} p^4 + C_5^2 \bar{p}^2 p^3 \approx 0.985 \times 10^{-5}\\
        \notag n&=7, \ \ P_E = p^7 +\sum_{i=1}^{3} C_7^i \bar{p}^i p^{7-i} \approx 4 \times 10^{-7}\\
        \notag n&=9, \ \ P_E = p^9 + \sum_{i=1}^{4} C_9^i \bar{p}^i p^{9-i} \approx 1 \times 10^{-8}\\
        \notag \vdots \\
        \notag n&=2k+1, \ \ P_E = p^{2k+1} + \sum_{i=1}^k C_{2k+1}^i \bar{p}^i p^{2k+1-i}
    \end{align}
$$
与之伴随而来的代价，是信息传输率的大幅下降。假设信源输出 $M$ 个等概率的、编码符号长度为 $n$ 的消息，则信息传输率为 $\dfrac{1}{n} \log_2 M$（单位 $\rm bit/code$），即信息传输率与编码长度成反比关系，这就体现了信息传输的可靠性（低错误概率）与有效性（高传输率）的矛盾。这种矛盾可以通过合适的编码、译码方法得到有效缓解。

接下来，我们不加证明地给出一种示例解决方案—— (5,2) 线性码：假设编码消息数 $M=4$，码长 $n=5$，此时信息传输率为 $\dfrac{1}{5} \log_2 4 = 0.4 \ {\rm bit/code}$。具体编码方法如下：记码元为 $c_i = (c_{i_1},c_{i_2},c_{i_3},c_{i_4},c_{i_5})$，$i=1,2,3,4$。$c_i$ 的前两位编码有效信息（“0-1” 二元编码），后三位作为校验码（纠错码），由以下方程组给定：
$$
    \begin{cases}
        c_{i_3} = c_{i_1} \oplus c_{i_2}\\
        c_{i_4} = c_{i_1}\\
        c_{i_5} = c_{i_1} \oplus c_{i_2}
    \end{cases}
$$
其中 $\oplus$ 表示模 2 加运算（二进制 “异或” 计算），译码依然采用 MLD 准则，具体规则如下所示：

![(5,2)线性码](/信息论与编码基础/figures/3-8.png)

此时，译码错误概率 $P_E$ 为 $1- \left( \bar{p}^5 + C_3^1 p \bar{p}^4 + C_2^1 \bar{p}^4 p + 2 \times \bar{p}^3 p^2 \right) \approx 7.86 \times 10^{-4}$。(5,2) 线性码与重复码（$n=3$）的错误概率基本在同一个量级，但前者的信息传输率（0.4）与可编码指令数目（4）却明显高于后者（0.33，2）。由此可见，适当增加消息数 $M$ 和编码长度 $n$，采用合适的编解码方法，既能降低 $P_E$，又不会减少信息传输率。

**香农第二定理（有噪信道编码定理）**：设某信道有 $s$ 个发送符号，$r$ 个接受符号，信道容量为 $C$。只要码长 $n$ 足够长，总能在输入的 $s^n$ 个码集中找到 $M$ 个码词组成的等概率编码与对应的解码规则，使得信道输出的错误概率 $P_E$ 达到任意小（$\epsilon$）的水平，其中 $M \leqslant 2^{n(C-\epsilon)}$。

### 3.3.2 编码定理的指导意义
考虑按香农第二定理编码的信道，码长为 $n$，码词数目为 $M$，单个符号的信息传输率 $R$ 为：
$$
    R = \dfrac{1}{n} \log_2 M \leqslant C - \epsilon
$$
根据先前所述，有 $R = H(X) - H(X|Y)$。