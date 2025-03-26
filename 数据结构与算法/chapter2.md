# 算法分析基础概念
## 算法时间度量指标
一个算法所实施的操作数量或步骤数可作为独立于具体程序/机器的度量指标。控制流语句（如循环、判断等）仅仅起到了组织语句的作用，并不实施处理，所以控制流语句通常不作为度量指标。而**赋值语句**同时包含了计算（表达式）与存储（变量）两个基本资源，适合作为度量指标。

赋值语句数量通常记为 $T(n)$，以求和函数为例：
```python
def sum_of_n(n) -> int:
    result = 0
    for i in range(1, n + 1):
        result = result + i
    return result
```
函数内第一行为赋值语句（$1$），第二行为控制流（不计入规模），第三行为赋值语句，运行 $n$ 次，所以该程序的 $T(n) = 1 + n$。其中 $n$ 通常表示“**问题规模**”，指的是影响算法执行时间的主要因素。算法分析的目标即找出问题规模如何影响算法的执行时间。

## 数量级函数 Order of Magnitude
$T(n)$ 的精确值往往不重要，重要的是 $T(n)$ 中起决定性因素的主导部分。数量级函数（又称“大 O” 表示法）$O \left(f(n) \right)$ 描述了 $T(n)$ 中随着 $n$ 增加而增加速度最快的主导部分。以函数 ```sum_of_n()``` 为例，其运行时间的数量级为 $O \left(n \right)$；对于另一个案例，如 $T(n) = 5n^2 + 3n + 1005$，其运行时间数量级为 $O \left(n^2 \right)$。

需要注意的是，影响算法真实运行时间的因素不仅仅是问题规模，数据本身的特性（如排序问题中，输入序列的规整程度）、硬件属性等等其它因素也会对结果产生影响。客观公正的算法分析流程应当关注实现过程中的问题规模，不能被少部分特殊情况迷惑。

常见的大 O 数量级函数主要有七种，当 $n$ 较大时，它们**从小到大**的排序分别为：（1）**常数** $O \left(1 \right)$；（2）**对数** $O \left(\log (n) \right)$；（3）**线性** $O \left(n \right)$；（4）**对数线性** $O \left( n \log (n) \right)$；（5）**平方** $O \left( n^2 \right)$；（6）**立方** $O \left( n^3 \right)$；（7）**指数** $O \left( 2^n \right)$。当 n 较小时，仅靠大 O 函数难以确定数量级差异，需根据 $T(n)$ 判断。

---
# 变位词判断问题
## 问题描述与目标
“变位词”指的是两个词之间存在组成字母的重新排列关系，如 earth 与 heart。为简单起见，假设参与判断的词语仅由小写英文字母构成，且输入词语长度相等。解题目标是设计一个 bool 函数，以两个词作为参数，返回是否为变位词的判断结果。

## 方法 1：逐字检查
对于输入的词语 A 中的每一个字符，分别在词语 B 中检查是否存在相同的字符并匹配消除：当发现 A 中存在 B 中没有的字符时，判断为 ```False```；顺利消除完毕则判断为 ```True```。由于限定了输入词语长度相等，因此无需考虑 B 包含了 A 中所有的字符但存在其它额外字符的情况。在 Python 中，由于字符串为不可变变量，因此需要将词语 B 转换为字符串序列。示例代码如下：
```python
def anagram_solution_1(s1: str, s2: str) -> bool:
    """Solve anagram problem.

    Args:
        s1 (str). Input string 1.
        s2 (str). Input string 2.

    Returns:
        still_ok (bool). Bool result.
    """
    s2_list = list(s2)  # transform s2 into List[str]
    pos_s1 = 0
    still_ok = True
    while pos_s1 < len(s1) and still_ok:  # match every character of s1
        pos_s2 = 0
        found_exist = False  # existed symbol in s2
        while pos_s2 < len(s2_list) and not found_exist:  # check characters of s2
            if s1[pos_s1] == s2_list[pos_s2]:
                found_exist = True  # break
            else:  # check the next character of s2
                pos_s2 = pos_s2 + 1
        if found_exist:
            s2_list[pos_s2] = None  # erase the existed character of s2
        else:  # s1 has character that is not in s2
            return False  # break, not anagram
        pos_s1 = pos_s1 + 1  # match the next character of s1
    return still_ok
```
接下来开始分析该算法：

- 首先我们确定问题规模，应当为词语包含的字符总个数 $n$；

- 其次我们观察循环结构，共有两层循环：第一层需要遍历字符串 ```s1``` 中的每个字符（即 $n$ 次）；第二层检查字符串 ```s2``` 的每个字符。

    - 这里尤其需要注意第二层的运行次数，新手很容易在这里犯错。所以我们暂停一下，先来考虑一种似乎“最费时间”的场景，即 ```s1``` 与 ```s2``` 是完全相反排列的词语，且字符无重复。对于 ```s1``` 中的第一个字符，需要检查 $n$ 次才能找到匹配项；第二个字符则需要检查 $n - 1$ 次；第三个字符需要 $n - 2$ 次……不难发现计算两层循环的执行总数是一个等差数列求和问题，即 $T(n) = \dfrac{n(n+1)}{2}$。

    - 但是这真的是“最”费时间的吗？我们换个情况，假设 ```s1``` 与 ```s2``` 完全相同，第一个字符只用检查 1 次，第二个字符要检查 2 次（```s2``` 中的第一个字符虽然被消除了，但序列长度是静态的，没有发生变化，所以将第一个字符赋值为 ```NoneType``` 只能确保匹配不出问题，但还是需要消耗一次匹配次数），第三个字符要检查 3 次……最终 $T(n)$ 其实不变。

综上所述，该方法的数量级为 $O \left( n^2 \right)$。

## 方法 2：排序比较
将 ```s1``` 与 ```s2``` 按某个统一规则分别排序后，再逐个对比字符是否相同。这里我们暂时先不讨论不同排序算法的性能优劣，仅考虑内置的基本方法 ```sort()```。示例代码如下：
```python
def anagram_solution_2(s1: str, s2: str) -> bool:
    """Solve anagram problem. Check details in anagram_solution_1()."""
    # transform str into List[str]
    s1_list = list(s1)
    s2_list = list(s2)

    # sorting
    s1_list.sort()
    s2_list.sort()

    # compare each character individually
    pos = 0
    matches = True
    while pos < len(s1_list) and matches:
        if s1_list[pos] == s2_list[pos]:
            pos = pos + 1  # switch to the next character
        else:
            return False
    return matches
```
这样“匹配”阶段的问题规模就减少为 $n$，但是排序过程并不是无代价的。根据排序方法的不同，其时间数量级通常在 $O \left( n \log (n) \right)$ 到 $O \left( n^2 \right)$ 之间，因此上述方法的时间主导步骤是排序步骤，时间数量级取决于排序过程。

## 方法 3：暴力法
接下来展示一种听上去就很差的方法，穷尽 ```s1``` 所有可能的字符组合，检查 ```s2``` 是否出现在全排列列表中。前者的总可能数为 $n!$，阶乘的增长速率是超过指数的，所以无需多言，暴力法不是好方法。

## 方法 4：计数比较
计数比较是一种相对更合理的解决方法，相比于前述三种方法，计数法跳出了“逐个字符对比是否相同”的逻辑，抽象出了变位词的本质特征，即组成变位词的每类字符的数目都相等。以英文字母为例，为每个词设置一个 26 位的计数器，分别检查每个词语，并在计数器中分别记录每个字母出现的次数。在计数完成后，比较两个词语的计数器是否相同，如果相同则二者即为变位词。
```python
def anagram_solution_4(s1: str, s2: str) -> bool:
    """Solve anagram problem. Check details in anagram_solution_1()."""
    # initialize the character counter
    c1 = [0] * 26
    c2 = [0] * 26

    # start counting
    for i in range(len(s1)):
        pos = ord(s1[i]) - ord('a')
        c1[pos] = c1[pos] + 1
    for i in range(len(s2)):
        pos = ord(s2[i]) - ord('a')
        c2[pos] = c2[pos] + 1

    # compare the counter
    j = 0
    still_ok = True
    while j < 26 and still_ok:
        if c1[j] == c2[j]:
            j = j + 1
        else:
            return False
    return still_ok
```
不难发现，该方法的时间主导步骤在于生成词语的计数器，且为线性级 $O \left(n \right)$，其余步骤时间数量级都是常数级，因此这是目前四种方法中最快的一种。当然这种算法的主要缺点在于存储空间。对于大字符集，或者超长字符串，生成相应计数器需要的存储空间不可小觑。这种时间与空间的取舍与权衡，是经常需要考量的互相制约因素，根据使用场景、目标需求等其它现实因素，往往需要灵活选择不同的算法来完成目标。

---
# Python 数据类型的性能
列表 ```List``` 与 ```Dict``` 是两种重要的 Python 内置数据类型，其它数据结构（如堆栈、数等）基本上都能够通过这两种容器（可变）类型得以实现。因此评估这两种类型在进行各种操作时的大 O 数量级，对于基于 Python 的数据结构与算法分析而言是非常有必要的。按照操作类型，主要分为 7 种：

- **索引**： ```List``` 的索引依靠自然数；```Dict``` 的索引依赖与不可变类型值 ```key```；
- **添加**： ```List``` 添加元素的方法主要有 ```append()```、```extend()``` 以及 ```insert()``` 三种；```Dict``` 则靠键值对指定，即 ```dict['new_key'] = new_value```；
- **删除**： ```List``` 删除元素有 ```pop()``` 以及 ```remove()``` 两种；```Dict``` 通过 ```pop()``` 方法删除已有键值对；
- **更新**： ```List``` 通过数字索引并重新赋值来更新已有元素，即 ```List[index] = new_value```；```Dict``` 通过键索引并重新赋值更新元素，即 ```Dict['key'] = new_value```；
- **正查**：正查指的是通过外部索引 ```index``` 或 ```'key'``` 查找相应元素的内容。 ```List``` 通过数字索引或切片，即 ```List[index]``` 或者 ```List[index1:index2]```；```Dict``` 通过键名查找，即 ```Dict['key']``` 或 ```copy()```；
- **反查**：反查指的是已知元素内容 ```value```，搜索其位置或出现次数。 ```List``` 的相应方法为 ```index(value)``` 与 ```count(value)```；```Dict``` 不支持通过值来反向检索键；
- **其它**：```List``` 的特殊方法还包括反向 ```reverse()```、排序 ```sort()```；```Dict``` 的特殊方法还有判断键是否存在 ```has_key()```、更新字典 ```update()``` 等。

Python 语言设计的总体思路是，让最常用的操作性能最好，牺牲不太常用的操作，即 **80/20 准则**：80% 的功能其使用率只占 20%。在选择具体数据类型时，也应当尽量遵照 80/20 准则的基本思路。

## 列表数据类型
对于 ```List``` 结构而言，最常用的功能是**按索引取值和赋值**。由于列表的随机访问特性，这两种操作的执行时间与列表大小无关，均为 $O \left( 1 \right)$；另一个常用功能是**列表增长**，有两种时间不一样的实现方法。假设初始列表 ```list_ori``` 长度为 $n$，待添加列表 ```list_add``` 长度为 $k$：第一种方法是 ```list_ori.append(list_add)```，执行时间为 $O \left( 1 \right)$；另一种方法是内置的 ```__add__()```，即 ```list_ori = list_ori + list_add```，执行时间为 $O \left( n + k \right)$。后者在本质上是生成了一个新长度的序列，再把原始序列与待加序列的元素值分别复制进去，所以运行时间更慢一些。

这里我们给出四种实现，其目的都是构造一个从 1 到 1000、步长为 1 的数值序列。
```python
def func1() -> List[int]:
    """__add__()."""
    l = []
    for i in range(1000):
        l = l + [i]
    return l


def func2() -> List[int]:
    """append()."""
    l = []
    for i in range(1000):
        l.append(i)
    return l


def func3() -> List[int]:
    """List comprehension."""
    return [i for i in range(1000)]


def func4() -> List[int]:
    """Transform range() into List."""
    return list(range(1000))
```
经测试可以发现，列表连接（concat）类方法最慢，append 类方法速度大约快 20 倍。但列表推导式更快，最快的是第四种。后两种之所以效率更高，是因为它们没有直接调用 ```List``` 类的基础方法，而是先借由迭代器（```Iterator```）或生成器（```Generator```）生成了列表中元素的内容，再将其转换成完整的列表。```List``` 类虽然是基础类，但仍然是由一系列语法定义的数据类型。与计算机能直接识别或执行的二进制码不同，Python 数据类基本操作的实现同样属于需要编译的高级命令。那么编译后命令的执行速度不一（例如 ```range()``` 比 ```List``` 内置方法快）当然是很正常的现象。

对于 ```List``` 类的其它操作，我们没必要记住所有操作的大 O 时间，只需要了解一些通用规律以及部分特例：

- 由前述可知，列表的索引取（赋）值、```append()``` 添加都是 $O \left( 1 \right)$ 的，相对应的剔除末尾 ```pop()``` 同样也是 $O \left( 1 \right)$。这些常数级的操作是最快的；
- 大多数需要使用索引下标的操作是线性级 $O \left( n \right)$ 的，其中 $n$ 一般表示列表长度，比如删除特定位置的元素 ```pop(index)```、在特定位置插入元素 ```insert(index, item)```、反向 ```reverse()```等；
- 有些操作虽然也是线性级，但与（2）略有差异，比如从原有列表中产生长度为 $k$ 的切片 ```List[index1:index2]``` 是 $O \left( k \right)$ 的；获取切片后取集合是 $O \left( n + k \right)$ 的；批量复制（乘法）列表 ```k * List``` 是 $O \left( nk \right)$ 的；
- ```sort()``` 排序是 $O \left( n \log (n) \right)$ 的。

关于 ```pop()``` 操作，当不指定参数时，该方法默认移除列表末尾的元素，是 $O \left(1 \right)$ 的；指定索引下标后，会从列表对应位置移除元素，时间增加到 $O \left(n \right)$。为什么会有这样的差异呢？其原因在于 ```pop()``` 功能的具体实现方法。当 ```pop()``` 从列表中部移除元素时，Python 会把之后的元素全部向前挪位并复制一遍。这样做的目的是为了保证列表的索引不出错，从而使得列表最常用功能（按索引取、赋值）的操作始终最快，这其实就是 80/20 准则的体现。

## 字典数据类型
与 ```List``` 略有不同，```Dict``` 根据键 ```key``` 找到数据项，而前者通过索引位置 ```index``` 实现相应功能。字典最常用的操作同样也是取（赋）值，性能为 $O \left( 1 \right)$。此外，删除键值对、判断是否存在键也是 $O \left( 1 \right)$ 的。迭代 ```iteration()``` 以及复制 ```copy()``` 的性能与字典大小有关，是 $O \left( n \right)$ 的。

需要注意的是，在 ```List``` 中判断元素是否存在（```in``` 操作）是 $O \left(n \right)$ 的，因为 ```List``` 需要逐个比对其中元素是否匹配，同时列表本身具有随机访问性质，```Dict``` 在这一项功能上是具备明显优势的。

除了上述两种基本类型，Python 其它数据类型的大 O 可以在[官方网址][ref1]查到，这里就不一一叙述了。

---
[ref1]: https://wiki.python.org/moin/TimeComplexity