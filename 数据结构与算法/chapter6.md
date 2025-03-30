# 排序与查找
## 顺序查找（Sequential Search）
如果数据项保存在如 `List` 这样的集合中，则称这些数据项具有线性或顺序关系。以 Python 的 `List` 为例，这些数据项的存储位置为有序整数，即下标（Index）。通过下标，可以按照顺序访问和查找数据项，即“**顺序查找**”。对于 `List` 数据集，设计顺序查找函数：首先从列表的第一个数据项开始，按照下标增长的顺序逐个比对数据项；如果到最后一个都未发现，则查找失败。
```python
def sequantial_search(alist: List, item: Any) -> bool:
    pos = 0
    found = False
    while pos < len(alist) and not found:
        if alist[pos] == item:
            found = True
        else:
            pos += 1
    return found
```
查找算法性能的基本计算步骤是进行数据项的比对，比对的次数决定了算法复杂度。在顺序查找算法中，通常假定列表中的数据项没有按值的大小排列顺序，而是随机放置在列表中的各个位置。同时比对次数与数据项是否在列表中也有很大关联：若数据项不在列表中，则需要比对所有数据项才能知道，比对次数必然是 `n`；如果数据项在列表中，则最好的情况是一次就找到，最差的情况则需要 `n` 次比对，平均下来比对次数应为 `n/2`，所以顺序查找的算法复杂度是线性级。

假如数据表已经排序了，且还要用顺序查找的话，当数据项不存在于表内时，相比无序表可能会节省一些步骤（可以根据前后项的大小判定有没有可能在表内）。但实际上，就算法复杂度而言（$O \left( n \right)$）并没有变化。

## 二分查找（Binary Search）
显而易见，在有序数据表中进行顺序查找的效率太低了，此时二分查找是一个更合适的选择。具体来说，从数据表的中间开始比对：
- 假如正好相等，则一次性找到；
- 若当前数据项小于待查项，则说明前半部分都小于待查项，一次性排除了一半长度的数据项；反之则排除后半部分的数据项。
- 更进一步地，从剩余部分的中间再进行一次比对，重复上述流程，不断地缩小排查范围，直至最终找到相同项（或确认没有相同项）。
```python
def binary_search(alist: List, item: Any) -> bool:
    fisrt_idx, last_idx = 0, len(alist - 1)
    found = False
    while not found:
        mid_idx = (first_idx + last_idx) // 2
        if alist[mid_idx] == item:
            found = True
        else:
            if item < alist[mid_idx]:
                last_idx = mid_idx - 1
            else:
                first_idx = mid_idx + 1
    return found
```
二分查找体现了“分治法”解决问题的策略：将总体问题分解为若干更小规模的部分，通过解决每一个小规模部分问题，将结果汇总得到原问题的解。既然能用二分法做，当然程序可以用递归实现：
```python
def binary_search_recursion(alist: List, item: Any) -> bool:
    if len(alist) == 0:
        return False
    else:
        mid_idx = len(alist) // 2
        if alist[mid_idx] == item:
            return True
        else:
            if item < alist[mid_idx]:
                return binary_search_recursion(alist=alist[:mid_idx], item=item)
            else:
                return binary_search_recursion(alist=alist[mid_idx + 1:], item=item)
```
假设列表长度为 $n$，二分法最多需要比对 $x$ 次，则有：$2^x = n$，即 $x = \log_2(n)$，所以二分法的算法复杂度是 $O \left(\log(n) \right)$。

不过需要注意的是，递归程序中有列表切片的调用，而切片操作的复杂度是 $O \left(k \right)$，这样会使得整个算法的时间复杂度稍有增加。当然也可以把程序输入设计为索引，这样可以节约部分开销。此外，尽管二分法非常高效，但它的前提是数据表需要排序。如果缺少这种前提，或者数据集经常变动而查找次数相对较少（即排序需要多次进行），就要谨慎考虑使用二分法。

## 冒泡排序（Bubble Sort）
如无特殊说明，接下来关于排序算法的描述都限定于升序排列。冒泡排序的算法思路在于对无序表进行多轮比较交换，每轮包括多次两两比较，将逆序的数据项互换位置，最终将本轮的最大项就位（仅最大项就位，其它项不一定按序分布）。最终经过 $n - 1$ 轮比较后，所有数据项都排序完毕。随着轮次进行，每轮需要比较的次数也在逐渐减少，总计比较次数是一个等差数列的和，即冒泡排序的时间复杂度是 $O \left( n^2 \right)$。如果考虑交换次数，平均下来也是 $O \left( n^2 \right)$。
```python
def bubble_sort(alist: List[Union[float, int]]):
    for passnum in range(len(alist) - 1, 0, -1):
        for i in range(passnum):
            if alist[i] > alist[i + 1]:  # switch
                alist[i], alist[i + 1] = alist[i + 1], alist[i]
```
总结来看，冒泡排序通常作为时间效率较差的基准排序算法。由于每个数据项在找到最终位置之前，都必须要经过多次比对和依次交换，其中大部分操作其实是无效的。例如列表 `[7,5,4,3,2,1,6]`，在对 `6` 进行排序时，明明我们知道 `6` 只比 `7` 小，但是必须进行一系列与 `1`、`2` 等数据项的比较、交换之后才能就位。当然冒泡排序也有好处，就是无需任何额外的存储空间，所有的数据项交换步骤都是原地进行、原地修改列表的。此外，如果我们发现在某一轮比对中没有发生数据项的交换，说明此时排序已经完成，可以提前输出结果了，这一点也是大部分排序算法无法做到的。
```python
def bubble_sort_v2(alist: List[Union[float, int]]):
    exchange = True
    passnum = len(alist) - 1
    while passnum > 0 and exchange:
        exchange = False
        for i in range(passnum):
            if alist[i] > alist[i + 1]:
                exchange = True
                alist[i], alist[i + 1] = alist[i + 1], alist[i]
        passnum -= 1
```

## 选择排序（Selection Sort）
选择排序对冒泡排序进行了一定的改进，保留了基本的多轮对比思路：每轮都使得当前最大项就位。在交换部分进行了一定的删减：每轮过程中仅记录最大项的所在位置，最后进行 1 次交换。选择排序的比对次数与冒泡排序一致，即时间复杂度还是 $O \left(n^2 \right)$，但交换次数减少为 $O \left(n \right)$。
```python
def selection_sort(alist: List[Union[float, int]]):
    for fillslot in range(len(alist) - 1, 0, -1):
        pos_of_max = 0
        for loc in range(1, fillslot + 1):
            if alist[loc] > alist[pos_of_max]:
                pos_of_max = loc
        alist[fillslot], alist[pos_of_max] = alist[pos_of_max], alist[fillslot]
```

## 插入排序（Insertion Sort）
插入排序的时间复杂度还是 $O \left(n^2 \right)$，但算法思路与冒泡、选择排序不同；同时复杂度数量级虽然一样，但总体用时会稍少一些。插入排序维持一个已排序的子列表，其位置始终在列表的**前部**，之后逐步扩大这个子列表直至全体列表都被涵盖。通俗地说，插入排序的过程类似于扑克牌游戏中整理手牌的过程：
- 第一轮：子列表仅包含第 1 个数据项（不考虑其大小），将第 2 个数据项作为“新项”插入到子列表的合适位置（比第 1 个大就在后，小就在前）；
- 第二轮：子列表此时包含 2 个已排好序的数据项，将第 3 个数据项跟前两者比对，移动比其大的数据项，空出位置来把第 3 个数据项插入到合适的位置；
- 循环上述过程，直至 $n-1$ 轮比对和插入结束后，子列表扩展到全表，排序完成。 

插入排序的比对操作主要用来寻找“新项”合适的插入位置。最差的情况是：手牌完全是逆序的，即每轮中新项都需要与子列表的全体项目进行比对，数量级为 $O \left(n^2 \right)$；最好的情况是：手牌已经排好序，即每轮仅需 1 次比对，总次数为 $O \left(n \right)$。具体到插入步骤：首先要在子列表的基础上新增一个空位，每次比对时移动比“新项”大的数据项，直至空位索引前的数据项比“新项”小、索引后的数据项比“新项”大，则该索引即为正确插入位置。
```python
def insertion_sort(alist: List[Union[float, int]]):
    for idx in range(1, len(alist) + 1):
        value = alist[idx]
        pos = idx
        while (pos > 0) and (alist[pos - 1] > value):  # find insert position
            alist[pos] = alist[pos - 1]  # alist[pos] has been saved as value, not lost
            pos -= 1
        alist[pos] = value
```

## 谢尔排序（Shell Sort）
在插入排序中，比对次数的最好情况是 $O \left(n \right)$，即列表已经有序。事实上列表越接近有序，插入排序的比对次数应该越少。从这个情况入手，Shell 排序以插入排序为基础，对无序表进行间隔划分子列表，每个子列表都进行插入排序，此为一轮；随着轮次进行，间隔数目越来越少，直至为 1，即执行一次标准的插入排序。但由于总列表整体的排序情况在不断好转，最终插入排序所需执行的操作步骤是很少的。

以一个长度为 16 的无序列表为例，一般来说，划分的间隔数为 $\frac{n}{2}$、$\frac{n}{4}$、$\frac{n}{8}$ 等：
- 首先进行一次间隔为 8 的子列表划分，即将索引为 `[0,8]`、`[1,9]`、`[2,10]` …… `[7,15]` 的元素划分为 8 个子列表，之后分别进行插入排序。排序之后的子列表在原列表中的区域不变，即 `[0,8]` 子列表的 2 个元素，依然位于原列表中相应位置范围内，只不过具体顺序可能发生了改变；
- 接下来依次进行间隔为 4、2 的子列表划分与排序，操作与上述相同；
- 最后进行间隔为 1 的划分与排序，即插入排序。

```python
def gap_insertion_sort(
        alist: List[Union[float, int]],
        start: int,
        gap: int):
    for i in range(start + gap, len(alist), gap):
        value = alist[i]
        pos = i
        while (pos >= gap) and (alist[pos - gap] > value):
            alist[pos] = alist[pos - gap]
            pos -= gap
        alist[pos] = value


def shell_sort(alist: List[Union[float, int]]):
    gap = len(alist) // 2
    while gap > 0:
        for start in range(gap):
            gap_insertion_sort(alist=alist, start=start, gap=gap)
        gap = gap // 2
```
Shell 排序的时间复杂度大致介于 $O \left( n \right)$ 和 $O \left( n^2 \right)$ 之间，具体性能当然与划分间隔有关。如果将间隔保持在 $2^k - 1$（1、3、7、15、31 等等），Shell 排序的时间复杂度约为 $O \left( n^{\frac{3}{2}} \right)$。

## 归并排序（Merge Sort）
归并排序是一种递归算法，其思路是将数据表持续分裂为两半并分别进行归并排序。
- 基本结束条件：数据表仅有 1 个数据项，当然是有序的；
- 缩小规模：将数据表分裂为相等长度的两部分，规模缩减为原来的二分之一；
- 调用自身：将两部分分别调用自身排序，之后再归并，得到排好序的数据表。

```python
def merge_sort(alist: List[Union[float, int]]):
    if len(alist) > 1:
        mid = len(alist) // 2
        left_half, right_half = alist[:mid], alist[mid:]

        # recursion
        merge_sort(left_half)
        merge_sort(right_half)

        # merge process
        i, j, k = 0, 0, 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                alist[k] = left_half[i]
                i += 1
            else:
                alist[k] = right_half[j]
                j += 1
            k = k + 1
            

```

## 快速排序

---
# 散列

## 完美散列函数

## 区块链技术

## 散列函数设计

## 冲突解决方案

## 映射抽象数据类型

---
