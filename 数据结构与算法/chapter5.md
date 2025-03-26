# 递归
递归（Recursion）是一种解决问题的方法，其要义在于将问题分解为规模更小的**相同问题**，持续分解直到问题规模小到可以用非常简单直接的方式解决，然后再层层上升并解决总体复杂问题。递归的问题分解方式在算法方面的独特之处在于，算法流程中会**调用本体**（自身），在系统资源允许的前提下，应用递归方法既可以极大提高解决问题的效率，又能简化代码。

以数组求和为例，首先声明该方法不一定是最快最好的，只是展示一下递归的基本流程。
```python
def list_sum(nums: List[Union[float, int]]) -> Union[float, int]:
    """Sum an list using recursion."""
    if len(nums) == 1:  # edge condition
        return nums[0]
    else:
        return nums[0] + list_sum(nums=nums[1:])
```
递归算法需要满足三个条件：
（1）必须有**基本结束条件**，即最小规模问题的直接解决方法，此时不能再调用函数本身。在代码中往往表现为一个前置的 `if` 语句，在基本结束条件中将直接 `return` 相应的结果；
（2）必须能改变状态并向基本结束条件演进，即问题规模需要逐步减小（收敛）；
（3）必须调用自身，不能调用外部函数，即每一步都是在解决减小规模的相同问题。

---
# 递归的应用
## 整数转换为任意进制
该问题在之前的 `Stack` 章节有讨论过，这里用递归方法再试一次。以十进制为例，十以内的数我们可以直接得出结果，对于十以上的数，我们需要将其分解为十位、百位、千位等等不同位数，再把每一个位数上的值保留下来，最后组成一个完整的十进制数。那么我们把原始数逐步拆解直至获得一个十以内的数的过程，就是一个典型的递归过程：逐步减小相同问题的规模直至可以直接得到结果。

上述递归过程需要用到的直接方法是整数除（`//`）以及求余数（`%`）：
- 对原始数按进制基（`base`）进行整数除法，余数必然小于 `base`，即达到了“**基本结束条件**”，可以直接查表获取数字（或符号）；
- 整数除法获得的整数商是“**更小规模问题**”（在它小于 `base` 之前），通过递归调用自身可以进一步缩小规模。
```python
def dec_to_str(num: int, base: int, mapping: str) -> str:
    """Implementing integer conversion for arbitrary bases.
    
    Parameters
    -------
    num : int.
        Input number.
    base : int.
        base = len(map).
    mapping : str.
        The conversion rules for digits and symbols in the current base.

    Returns
    -------
    out : str.
    """
    if num < base:
        return mapping[num]
    else:
        # quotient, remainder = divmode(nums, base)
        # return dec_to_str(quotient, base, mapping) + mapping[remainder]
        return dec_to_str(
            num=num // base,
            base=base,
            mapping=mapping
        ) + mapping[num % base]
```

## 递归调用的实现
在上一节中，我们用递归解决了之前用 `Stack` 结构解决的问题，之所以这么方便，是因为递归本身就是在利用栈解决问题。当一个函数被调用时，系统会把调用时的现场数据（**栈帧**）压入到**系统调用栈**，当函数返回时，要从调用栈的栈顶取得返回地址，恢复现场，弹出栈帧，按地址返回。这个顺序就是递归过程中，先分解问题、再逐层解决的顺序。

在调试递归算法程序时，很容易遇到的问题是 `RecursionError`，俗话说“堆栈溢出”，这是因为递归层数过多（或者程序有误导致无限递归），超过了系统容纳的最大调用栈数目。此时算法向基本结束条件演进的速度太慢，我们要考虑优化演进规则，或者减小递归处理的总问题规模。在 Python 中，递归深度限制是可以调整的（当然一般建议往低了调以限制资源占用，超出硬件限制后容易产生严重的系统错误）
```python
import sys
sys.getrecursionlimit()
sys.setrecursionlimit(3000)
```

---
# 递归可视化
## 分形树
Python 内置了一个海龟作图系统（`turtle`），其意象为模拟海龟在沙滩上爬行而留下的足迹。主要函数包括：
- `forward(n)`、`backward(n)`：向前（后）爬行距离 `n`；
- `left(a)`、`right(a)`：向左（右）转向角度 `a`；
- `penup()`、`pendown()`：抬笔或放笔；
- `pensize(s)`、`pencolor(c)`：设置笔刷粗细和颜色属性。 

利用 `turtle` 能够比较简单直观地绘制分形（Fractal）树。分形是 Mandelbrot 于 1975 年开创的新学科，指的是自相似递归图形，即一个粗糙或零碎的几何形状，可以分成数个部分，且每个部分都（至少近似）是整体缩小后的形状。螺旋线是一种简单的分形树，接下来是一个绘制螺旋线的 demo：
```python
import turtle

t = turtle.Turtle()

def draw_spiral(t: Turtle, line_len: int):
    if line_len > 0:  # stop condition: line_len <=0
        t.forward(line_len)
        t.right(90)
        draw_spiral(t=t, line_len=line_len - 5)

draw_spiral(t, 100)
turtle.done()
```
对于一个广义的分形树，我们可以把树分解为三个部分：树干以及左、右小树，这三个部分就对应了递归程序的“**基本结束条件**”以及“更小规模的相同问题”（小树同样包含树干和更小的小小树）。接下来的 demo 展示了如何绘制一个二叉树：
```python
def tree(branch_len: int):
    if branch_len > 5:  # stop condition: branch_len <=5
        t.forward(branch_len)  # draw the branch
        t.right(20)  # toward right sub-branch
        tree(branch_len - 15)  # config the length of right sub-branch
        t.left(40)  # toward left sub-branch
        tree(branch_len - 15)  # config the length of left sub-branch 
        t.right(20)  # back to the middle direction
        t.backward(branch_len)  # back to the start point

t = turtle.Turtle()

# move to the bottom
t.left(90)
t.penup()
t.backward(100)
t.pendown()

# config pen parameters
t.pencolor('green')
t.pensize(2)

# main process
tree(75)
t.hideturtle()
turtle.done()
```
这段递归代码最抽象的地方在于两点：（1）`tree()` 内部的 `tree` 函数到底做了什么事；（2）如何保证每次画完之后返回原点。换句话说，程序是以什么方式（顺序）画出分形树的每一笔？

（1）首先，每一个 `tree` 函数都涵盖了**绘制树干**、**转向至右侧并绘制右子树**、**转向至左侧并绘制左子树**、**转回中间方向**，**退回树干起点**这五个基本步骤，在绘制子树的部分进行了递归嵌套。每一次嵌套又将在相同的位置进入下一层嵌套（输入参数 `branch_len` 逐步缩小），直至达到终止条件（`branch_len <= 5`）。从图像上看，绘制的路径是一条从主干一直延伸到最右侧的最小枝干的折线段，没有任何分支；

（2）其次，当达到终止条件后（例如此时的 `branch_len = 20`），程序没有执行 `tree(branch_len - 15)`，而是往下执行了 `t.left(40)`（转向左侧），同时下一行 `tree(branch_len - 15)` 也不予执行，执行了 `t.right(20)`（转回中间方向）、`t.backward(branch_len)` 返回上一个节点。从图像上看，海龟从最小分支的终点回到了该分支的起点，什么也没画；

（3）上一步结束的本质是程序完成了 `tree(branch_len - 15)`（此时 `branch_len = 35`），接下来要执行的命令是 `t.left(40)` 以及 `tree(branch_len - 15)`，后者很快又将达到递归终止条件，即在子递归步骤中只画了 `t.forward(branch_len)` 这一笔，就不再绘制轨迹并返回分支节点。图像上表现为海龟完成了最小分支的左子干后回到该分支起点；

（4）综合（2）、（3）可知，当递归处于终止前的一层（`branch_len = 35`）时，程序执行两次 `tree(branch_len - 15)` 画完左右两个子干后回到它们的起点，接下来执行 `t.right(20)`（转回起点所属上层子干的正向方向），`t.backward(branch_len)`（以 `branch_len = 35` 条件返回上层子干的起点）；

（5）从一般角度来看，程序完成了一个 `tree(branch_len - 15)` 的递归栈后，需要进行下一个 `tree(branch_len - 15)` 绘制另一个子分支，或者 `t.backward(branch_len)` 回到上一层分支的起点。这样我们就明确了绘制的总体规律，以一个 4 层二叉树为例，最小分支从左到右依次编号为 1-16，则递归的绘制顺序就是 1-16。

## Sierpinski 三角形
Sierpinski 三角形是由三个尺寸减半的 Sierpinski 三角形按照品字形拼叠而成（自相似性）。与一般用来演示的图像不同，真正的 Sierpinski 三角形是完全不可见的，其面积为零，但周长无穷，是介于一维和二维之间的分数维构造。从绘图角度来看，我们只能绘制出有限维度（`degree`）的 Sierpinski 三角形：在 `degree` 有限的情况下，`degree = n` 的三角形由 3 个 `degree = n - 1` 的三角形按品字形拼叠而成，同时这 3 个小三角形的边长均为大三角形的一半（规模减小）；当 `degree = 0`，结果为等边三角形，即递归基本结束条件。
```python
def drawTriangle(
        points: Dict[str, Tuple[Union[float, int],
                                Union[float, int]]],
        color: str):
    # draw an equilateral triangle by using turtle module.
    t.fillcolor(color)
    t.penup()
    t.goto(points['top'])
    t.pendown()
    t.begin_fill()
    t.goto(points['left'])
    t.goto(points['right'])
    t.goto(points['top'])
    t.end_fill()


def getMid(
        p1: Tuple[Union[float, int],
                  Union[float, int]],
        p2: Tuple[Union[float, int],
                  Union[float, int]]) -> Tuple[Union[float, int],
                                               Union[float, int]]:
    # get the midpoint between p1 and p2
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def sierpinski(
        degree: int,
        points: Dict[str, Tuple[Union[float, int],
                                Union[float, int]]]):
    colormap = ['blue', 'red', 'green', 'white', 'yellow', 'orange']
    drawTriangle(points, colormap[degree])
    if degree > 0:
        sierpinski(
            degree=degree - 1,
            points={
                'left': points['left'],
                'top': getMid(points['left'], points['top']),
                'right': getMid(points['left'], points['right'])
            }
        )  # draw the left sub-triangle
        sierpinski(
            degree=degree - 1,
            points={
                'left': getMid(points['left'], points['top']),
                'top': points['top'],
                'right': getMid(points['right'], points['top'])
            }
        )  # draw the top sub-triangle
        sierpinski(
            degree=degree - 1,
            points={
                'left': getMid(points['left'], points['right']),
                'top': getMid(points['top'], points['right']),
                'right': points['right']
            }
        )  # draw the right sub-triangle


# main process
t = turtle.Turtle()
points = {
    'left': (-200, -100),
    'top': (0, 200),
    'right': (200, -100)
}
sierpinski(degree=5, points=points)
turtle.done()
```

## 汉诺塔
汉诺塔问题是法国数学家 Edouard Lucas 于 1883 年根据印度传说设计的、人尽皆知的数学问题，此处就不介绍了。我们来分析一下如何把该问题分解为递归形式。已知柱子共有 3 个（编号从左到右依次为 a、b、c 柱），假设有 5 个盘子（从小到大编号为 1-5 盘）穿在 a 柱，目标是将其全部移动到 c 柱。

（1）完成这一目标的“上一阶段”是：1-4 盘已经被移动到 b 柱上，5 盘还在 a 柱上，接下来只要把 5 盘穿在 c 柱上，之后 b 柱上的 1-4 盘怎么来的，就怎么去 c 柱。这样我们就把原本 5 个盘子的问题分解成了“把 4 个盘子从一个柱子移动到另一个柱子上”，问题规模减小了，问题性质却依然不变；

（2）类似地，把 1-4 盘从 a 柱移动到 b 柱的“上一阶段”是：1-3 盘在 c 柱上，4-5 盘在 a 柱。接下来把 4 盘移到 b 柱，1-3 盘怎么来的 c 柱就怎么去 b 柱；

（3）再往前的“上一阶段”是：1-2 盘在 b 柱上，3-5 盘在 a 柱。接下来把 3 盘移到 c 柱，1-2 盘怎么来的 b 柱就怎么去 c 柱；

（4）再往前，就只剩一个盘子的状态需要改变了：1 盘在 c 柱，2-5 盘在 a 柱。接下来把 2 盘移到 b 柱，1 盘从 c 柱移到 b 柱，就完成了（3）中所需的“上一阶段”。此时已经到了递归问题的基本结束条件，即把 1 盘从 a 柱移动到 c 柱，是单步可完成的基本操作。

总结一下汉诺塔问题的递归思路：将上层 N-1 个盘从**开始柱**，经由**目标柱**，移动到**中间柱**；然后将第 N 号盘从**开始柱**移动到**目标柱**；最后将**中间柱**的 N-1 个盘经由**开始柱**移动到**目标柱**。每一个子问题的形式都是如此。基本结束条件是 1 个盘的移动问题。
```python
def move_disk(
        disk: int,
        from_pole: str,
        to_pole: str):
    print('Moving disk {} from {} to {}'.format(disk, from_pole, to_pole))


def move_tower(
        height: int,
        from_pole: str,
        with_pole: str,
        to_pole: str):
    if height >= 1:
        move_tower(
            height=height - 1,
            from_pole=from_pole,
            with_pole=to_pole,
            to_pole=with_pole
        )
        move_disk(
            disk=height,
            from_pole=from_pole,
            to_pole=to_pole
        )
        move_tower(
            height=height - 1,
            from_pole=with_pole,
            with_pole=from_pole,
            to_pole=to_pole
        )


move_tower(height=5, from_pole='a', with_pole='b', to_pole='c')
```

---
# 分治策略 & 贪心策略
## 分治策略（Divide and Conquer）
分治策略是一种递归解决问题的方法，它将一个复杂的问题分解为若干个规模较小的子问题，独立地解决这些子问题，然后将子问题的解合并得到原问题的解。

分治策略的基本步骤包括**分解**、**解决**以及**合并**。
- “分解”指将原问题分解为若干个规模更小且相互独立的子问题；
- “解决”指递归地解决这些子问题。如果子问题的规模足够小，可以直接求解；
- “合并”指将子问题的解合并，得到原问题的解。

分治策略适用于可以分解为多个独立子问题的问题，且子问题的解可以通过某种方式合并得到原问题的解。例如排序问题、矩阵乘法、快速傅里叶变换等等。由于分治策略与递归的联系非常紧密，这里不给出代码地举两个实际应用中的案例：

（1）归并排序（Merge Sort）：
- 分解：将数组分为两半，分别对左半部分和右半部分进行排序；
- 解决：递归地对左半部分和右半部分进行排序；
- 合并：将两个已排序的子数组合并为一个有序数组。

（2）快速排序（Quick Sort）：
- 分解：选择一个基准元素，将数组分为小于基准和大于基准的两部分；
- 解决：递归地对这两部分进行排序；
- 合并：由于排序是原地进行的，不需要额外的合并步骤。

## 贪心策略（Greedy Algorithm）
贪心策略是一种在每一步选择中都采取当前状态下最优的选择，从而希望导致全局最优的算法设计思想。其基本步骤包括：
- 建立数学模型：将问题转化为数学模型；
- 选择贪心标准：确定每一步的贪心选择标准，即在当前状态下选择最优的局部解；
- 逐步求解：按照贪心标准逐步求解，直到得到最优解。

接下来以“找零问题”为例说明贪心策略的应用以及局限。对于目前世界上绝大多数国家的货币体系而言，当我们想以最少货币数目实现任意数额的金钱时，使用贪心策略总是有效的。例如我国现有的 1、5、10、20、50、100 元纸币（仅考虑不小于一元的面额），对于 63 元这个数额，可以先考虑不大于该数值的最大面额纸币，即 50 元（只需 1 张）；再考虑不大于剩余金额（13）的最大面额纸币，即 10 元（仅需 1 张）；接下来是 1 元（3 张）。这样只需要 5 张纸币即可凑齐，使用任何其它组合都将超出 5 张纸币。但假如我们还有面额为 21 元的纸币，在按上述步骤操作时就会出现问题：63 元可用 3 张 21 元纸币凑齐，比 5 张纸币更少。

不难发现，在“找零问题”中使用的贪心策略本身具有一定程度的递归属性，但递归分解问题的方法有漏洞：从上往下地进行单次搜索容易陷入局部最优点。我们换个思路，自下而上地考虑问题：假如在兑换的某个阶段，只剩下 1 元，那么仅需 1 张 1 元纸币即可完成；假如只剩 5 元，那么用 1 元或者用 5 元去兑换都行，但肯定是用 5 元所需的纸币数目最小。所以我们把问题分解成了：当总金额减去某个面值的纸币之后，是否达到了基本结束条件（可正好用某面额纸币兑换完成）。同时对于所有可能面额的纸币都进行一遍递归尝试，取纸币数目最小的选择结束当前递归，最终得到答案。
```python
def solution_v1(coin_value_list: List[int], change: int) -> int:
    min_coins = change
    if change in coin_value_list:
        return 1
    else:
        for i in [coin for coin in coin_value_list if coin <= change]:
            num_coins = 1 + rec_MC(
                coin_value_list=coin_value_list,
                change=change - i
            )
            if num_coins < min_coins:
                min_coins = num_coins
    return min_coins
```
这个方法能够保证找到全局最优解，但是代价就是运行效率，毕竟自下而上的方法类似于排列全遍历，即使全局最优解已经找到，`for` 循环不把所有可能性算完是不会结束的。更关键的是该过程还存在很多重复计算：例如先去除 1 元再去除 5 元，和先去 5 元再去 1 元是一样的，但分别属于两个不同的 `for` 循环进程。消除重复计算可以使用查表法，即将计算过的中间结果保存，在计算之前检查是否已经处理过这种组合方式；而检查最优解的操作稍微复杂一些，我们根据程序来解释：
```python
def solution_v2(
        coin_value_list: List[int],
        change: int,
        known_results: List[int]) -> int:
    min_coins = change
    if change in coin_value_list:
        known_results[change] = 1  # optimized result
        return 1
    elif known_results[change] > 0:  # has been record
        return known_results[change]
    else:
        for i in [coin for coin in coin_value_list if coin <= change]:
            num_coins = 1 + rec_DC(
                coin_value_list=coin_value_list,
                change=change - i,
                known_results=known_results
            )
            if num_coins < min_coins:
                min_coins = num_coins
                known_results[change] = min_coins  # update record
    return min_coins
```
- 语句 `if change in coin_value_list:` 用来判断是否达到基本结束条件，如果剩余金额可用单张纸币兑齐，则在 `known_results` 中记录这种情况（键名表示金额，值表示所需纸币数）。由于这是基本结束条件，所以必然是最优解（当前金额条件下）；
- 语句 `elif known_results[change] > 0:` 表示剩余金额虽然不能用一张纸币解决，但在 `known_results` 中找到了可以兑齐的选项，此时就没必要再重复一遍寻找答案的递归过程了，同时由于我们假定 `known_results` 中存储的都是最优解（事实上程序也总是在尝试达到这种情况），直接调用表中的结果即可；
- 如果前两种情况都不满足，说明当前剩余金额是未处理过的情况，那么按照正常的递归流程去判定即可。需要注意的是，当 `num_coins` 计算完成并以此更新 `min_coins` 时，需要同步更新 `known_results` 的结果以保证表中存储的都是最优解。
- 最后，`known_results` 在初始化阶段会设为全 0 的字典或列表（字典的 `in` 查询更快），如果为 `Dict` 结构，在判断 `known_results[change]` 时要注意防止出现 `KeyError` 的异常。

在排除冗余计算项之后，程序递归次数大幅减少。以 `[1,5,10,25]` 面额、目标金额 `63` 的任务而言，递归次数从 67716925 次减少为 221 次，是改进前的约**三十万分之一**，其效果可见一斑。

---
# 动态规划（Dynamic Programming）
先前提到的中间结果记录可以很好地解决递归解法冗余计算太多的问题，这种方法的学名叫**记忆化/函数值缓存**（Memorization）。一种更有条理的解决方法是动态规划。动态规划是一种算法思想，用于解决具有**重叠子问题**和**最优子结构**特性的问题：
- **重叠子问题**：传统递归求解过程中，很多子问题可能会被重复计算。动态规划通过 `List` 或 `Dict` 等结构存储子问题的解，当再次需要该子问题的解时，可以直接查表获得；
- **最优子结构**：一个问题的最优解包含其子问题的最优解，换句话说我们可以通过求解子问题的最优解来构建原问题的最优解。（例如最短路径问题中，从起点到终点的最短路径中，任意一段子路径也必然是该子路径的最短路径）。

我们给出动态规划的一般性求解步骤：
- **定义状态**：状态时动态规划中用于描述问题的变量，其选择需要满足两个条件，一是能够唯一地描述问题；二是能够通过状态之间的关系推导出问题的解；
- **状态转移方程**：状态转移方程是动态规划的核心，描述了状态之间的关系；
- **初始化和边界条件**：二者分别是动态规划的起点和终止条件；
- **计算顺序**：动态规划有自底向上（从小到大）和自顶向下（递归 + 记忆化）两种计算顺序，其中前者更常见。

## 找零问题
还是以找零问题为例，假设所需金额是 11，有 `[1,5,10,25]` 五种面额的纸币。动态规划从金额为 1 开始逐步递增，每一步都计算最优解并存储，具体计算方法与上一节自下而上的思路相似：每次减去一种可用面额后（基本结束条件）判断最优解，在计算过程中如果遇到已经算过的结果就直接查表，具体流程如下所示：
- 1 元只需一张 1 元纸币，记录 1 元时最优解；
- 2 元只能减去 1 元，之后查表得知仍需一张纸币，记录 2 元时的最优解；
- 3 元只能减去 1 元，查表得知 2 元情况下需两张纸币，记录 3 元时的最优解；
- ……
- 6 元可以减去 5 元，查表得知仍需一张纸币；也可以减去 1 元，查表得知也需一张纸币。两者比较后记录 6 元时的最优解；
- ……
- 11 元可以减去 1 元、5 元或者 10 元，剩余金额为 10 元、6 元或 1 元，分别查表得知最优解为 1 张、2 张或 1 张，比较后记录最优解（2 张）。
```python
def solution_dp(
        coin_value_list: List[int],
        change: int) -> int:
    known_results = {0: 0}
    for value in range(1, change + 1):
        n_coin = value  # initialization for maximum number of coins
        for i in [c for c in coin_value_list if c <= value]:
            try:
                if known_results[value - i] < n_coin:  # find & check existed record
                    n_coin = known_results[value - i] + 1  # update optimized result
            except KeyError:
                pass
        known_results[value] = n_coin  # update result sheet
    return known_results[change]
```
从思路上看，动态规划与递归似乎很像，但动态规划一般不会调用自身，每一步子问题虽然是更小规模的同类问题，但解决方法可能是继续分解，也可能是直接查表。换句话说，动态规划的基本结束条件不是固定不变的，而是动态变化的。在此基础上，如果还要求返回具体的币种组合，需要对 `solution_dp()` 做一些扩展。已知 `known_results` 记录的是每一种金额所需的最少总纸币数，该数值的得来需要当前金额减去某个面额的纸币之后，对剩余金额进行查表。类似地可以设计另一个数据表（假设为 `used_coins`），记录当前金额下需减去的第一个纸币的面额，剩余金额对应的数值是下一次需要减去的纸币面额，直至剩余金额为零。还是以 11 元为例：
- 第一个减去的应是 10 元，则有 `used_coins[11] = 10`；
- 减去 10 元后还剩 1 元，下一次减去的就是 1 元，即 `used_coins[1] = 1`；
- 此时剩余金额为零，回顾输出结果是 `10`、`1`，即需要一张 10 元和一张 1 元。
```python
def solution_dp_v2(
        coin_value_list: List[int],
        change: int) -> Tuple[int, Dict[int, int]]:
    known_results, used_coins = {0: 0}, {}
    new_coin = 1
    for value in range(change + 1):
        n_coin = value
        for i in [c for c in coin_value_list if c <= value]:
            if known_results[value - i] < n_coin:  # find optimized result
                n_coin = known_results[value - i] + 1
                new_coin = i
        known_results[value] = n_coin  # update result sheet
        used_coins[value] = new_coin
    return known_results[value], used_coins


def print_coins(used_coins: Dict[int, int], change: int):
    remained_coin = change
    while remained_coin > 0:
        current_coin = used_coins[remained_coin]
        remained_coin -= current_coin
        print(current_coin)
```

## 背包问题
严格来说，本节讨论的是“01背包问题”。给定一组物品，每个物品有重量（`weight[i]`）与价值（`value[i]`）。现有容量为 `W` 的背包，要求选择若干物品放入背包，使得背包中物品的总重量不超过背包容量，且总价值最大。按照动态规划的一般思路分析该问题：
- **状态定义**：`dp[i][j]` 表示前 `i` 个物品在背包容量为 `j` 时的最大价值，需要注意的是，这并不代表前 `i` 个物品全部都在背包中，`i` 仅表示对待拿取物品的索引；
- **状态转移方程**：对于待取的第 `i` 个物品，从“背包能否装下”这个角度出发，有两种可能：
    - 若背包容量允许（`j >= weight[i]`），那就可以考虑。拿了该物品之后，背包所剩空间为 `j - weight[i]`，背包内其余物品的最大总价值为 `dp[i - 1][j - weight[i]]`，所以**现状态**为 `dp[i - 1][j - weight[i]] + value[i]`，至于拿了这个物品是否为最优解（是否赋值给 `dp[i][j]`），还需要进一步判断：
      - `dp[i][j] = max(dp[i - 1][j - weight[i]] + value[i], dp[i - 1][j])`。此处 `dp[i - 1][j]` 不能简单地认为是“拿第 `i` 个物品”前的背包，否则装了新东西自然是比没装价值更高的，这个判断就没有意义了，需要明确 `dp[i - 1][j]` 表示考虑前 `i - 1` 个物品的条件下可达到的最大价值。假如第 `i` 个物品重量很大，拿了可能要取出背包中其它物品（背包容量上限下降为 `j - weight[i]`），这样反倒使得总价值下降。那么 `dp[i - 1][j - weight[i]] + value[i]` 是有可能小于 `dp[i - 1][j]` 的，就不应该拿；
    - 若背包容量超了（`j < weight[i]`），那就算把背包清空也装不下，只能放弃了，即 `dp[i][j] = dp[i - 1][j]`；
- **初始化**：令 `dp[0][j] = 0`，即没有物品时背包的价值应当为 0；同时当 `j = 0` 时，背包（没空间）当然也没有价值；
- **计算顺序**：`i = 1` 至 `i = n`，从 `j = 0` 至 `j = W`，依次计算 `dp[i][j]`。由于在考虑拿取物品时需要对比 `i` 较大而 `j` 较小的情况，所以应先循环 `j`（内层），再循环 `i`（外层）。
```python
treasure = []
for (w, v) in zip(
    [2, 3, 4, 5, 9],
    [3, 4, 8, 8, 10]):
    treasure.append({'w': w, 'v': v})


def solution(
        max_w: int,
        treasure: List[Dict[str, Union[int, float]]]) -> Union[int, float]:
    if max_w == 0 or treasure == []:
        return 0

    # initialization
    dp = {(i, w): 0 for i in range(len(treasure)) for w in range(max_w + 1)}

    # main process
    for i in range(1, len(treasure) + 1):
        for j in range(1, max_w + 1):
            if treasure[i - 1]['w'] > j:  # over-weight
                dp[(i, j)] = dp[(i - 1, j)]
            else:
                dp[(i, j)] = max(
                    dp[(i - 1, j)],
                    dp[(i - 1, j - treasure[i - 1]['w'])] + treasure[i - 1]['v']
                )
    return dp[(len(treasure), max_w)]


print(solution(max_w=20, treasure=treasure))
```
不难发现，程序中的语句基本上就是之前分析过程中的语句，只要前期分析得当，动态规划的编程实施是非常简单直观的。