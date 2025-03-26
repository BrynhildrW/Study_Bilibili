# 1 - 10
## 88. 合并两个有序数组
有两个按**非递减顺序**排列的整数数组 `nums1` 和 `nums2`，另有两个整数 `m` 和 `n`，分别表示 `nums1` 和 `nums2` 中的有效元素数目。现在需要合并 `nums1` 和 `nums2`，使得合并后的数组同样按非递减顺序排列。
- 需要注意的是，合并后的数组不由函数返回，而是直接修改在数组 `nums1` 中。即 `nums1` 的初始长度为 `m + n`，其中前 `m` 个为有效元素，后 `n` 个均为 0，需要替换；`nums2` 的长度为 `n`。

示例：
> 输入：`nums = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3`
> 输出：`[1,2,2,3,5,6]`

> 输入：`nums = [1], m = 1, nums2 = [], n = 0`
> 输出：`[1]`

> 输入：`nums = [0], m = 0, nums2 = [1], n = 1`
> 输出：`[1]`

### 初始方法
这一题麻烦的地方在于，仅修改输入变量，不输出新的数组，也许是为了限制内存资源的占用。不通过索引而直接对 `nums1` 赋值会生成新的临时变量，并不会直接修改原变量 `nums1`，所以赋值过程要包含索引：
```python
class Solution(object):
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """Do not return anything, modify nums1 in-place instead."""
        nums1[m:] = nums2
        nums1.sort()
```
这种方法快当然是快，不过占用了不少内存。同时复杂度比线性略高一些。主要原因在于排序 `sort()` 浪费了已有条件，即两个输入数组都是非递减排序好的。

### 优化方法
根据已知条件，我们并不需要重新进行排序。可以依次读取两个数组的元素，比较大小之后按次序拼接就行了。很自然地，我们会想到设计一个中间变量存储排序结果，再赋值到 `nums1` 中。这是因为，如果我们在扫描过程中直接把 `nums2` 的元素赋值到 `nums1` 中，可能会覆盖掉原本位置的有效元素。

这种方法还有优化空间，注意到 `nums1` 的后半部分是空的，如果从后往前赋值，就不会产生干扰（在后方空余位置用完之前，`nums1` 已有的有效元素不可能被覆盖掉），此时赋值的标准就从“更小的元素”变成了“更大的元素”，在扫描 `nums1`（有效元素）和 `nums2` 时，也是从后往前的：
```python
class Solution(object):
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """Do not return anything, modify nums1 in-place instead."""
        idx_1, idx_2 = m - 1, n - 1
        idx_final = m + n - 1
        while (idx_1 >= 0) or (idx_2 >= 0):
            if idx_1 == -1:  # nums1 finished, add nums2 only
                nums1[idx_final] = nums2[idx_2]
                idx_2 -= 1
            elif idx_2 == -1:  # nums2 finished, add nums1 only
                nums1[idx_final] = nums1[idx_1]
                idx_1 -= 1
            elif nums1[idx_1] > nums2[idx_2]:
                nums1[idx_final] = nums1[idx_1]
                idx_1 -= 1
            else:
                nums1[idx_final] = nums2[idx_2]
                idx_2 -= 1
            idx_final -= 1
```
这种方法的复杂度是 $O \left( m + n \right)$，同时原地修改数组，不占用额外内存空间。

---
## 27. 移除元素
给定一个数组 `nums` 和一个值 `val`，需要原地移除所有数值等于 `val` 的元素，返回 `nums` 中与 `val` 不同的元素的数量。
- 要求变量 `nums` 被修改。具体来说，假如 `nums` 中共有 `m` 个不等于 `val` 的值，在经过函数处理后，这些值要位于该数组的前 `m` 个，至于后面有多少值、分别是多少都无所谓。

### 初始方法
尽管函数最终返回的是 `int`，但由于要求修改变量，所以简单的遍历并计数是不行的。一个合理的方法是在遍历过程中通过 `pop()` 方法动态修改 `nums` 的大小，同时通过 `IndexError` 控制 `while` 循环的终止条件，最后提交一遍过。要是所有算法题都这么简单就好了。
```python
class Solution(object):
    def removeElement(self, nums: List[int], val: int) -> int:
        stop = False
        idx = 0
        while not stop:
            try:
                if nums[idx] == val:
                    _ = nums.pop(idx)
                else:
                    idx += 1
            except IndexError:
                stop = True
        return len(nums)
```

---
## 26. 删除有序数组中的重复项
已知一个非严格递增排列的数组 `nums`，需要原地删除重复出现的元素，使数组中每个元素只出现一次，返回唯一元素的个数，元素的相对位置要保持不变。
- 假设 `nums` 的唯一元素数量为 `k`，要求 `nums` 被更改，使得 `nums` 的前 `k` 个元素包含唯一元素，并按最初的相对顺序排列。`nums` 其余元素与 `nums` 的大小无所谓。

### 初始方法
与 **Q.27** 类似，按顺序遍历一次 `nums`，检测到重复元素时 `pop()` 当前元素，检测到新元素时索引增加。最终 `nums` 将仅包含唯一元素。
```python
class Solution(object):
    def removeDuplicates(self, nums: List[int]) -> int:
        stop = False
        idx = 1
        while not stop:
            try:
                if nums[idx] == nums[idx - 1]:
                    _ = nums.pop(idx)
                else:
                    idx += 1
            except IndexError:
                stop = True
        return len(nums)
```

### 优化方法
初始方法的运行速度有点慢，主要原因在于每次 `pop()` 中间位置的元素都会导致数组后段重新创建并赋值（数组长度发生变化）。优化方法仅改变对应索引位置的值，不改变数组长度。从理论来说，用这个思路做 **Q.27** 应该也会更快一些。
```python
class Solution(object):
    def removeDuplicates(self, nums: List[int]) -> int:
        k = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[k - 1]:  # new number
                nums[k] = nums[i]
                k += 1
            # else: switch to next index i, keep k
        return k
```
同时需要注意判断语句 `if nums[i] != nums[k - 1]:`，这里把后者索引换成 `i - 1` 也能通过。之所以要使用 `k - 1` 索引，是因为随着程序运行，数组只修改了前 `k` 个元素，与这些元素的最大值比较才是真正要做的。此外，从 C 语言的角度来看，这一题实际上是用到了快慢指针的思想。其中慢指针是 `k`，快指针是 `i`。慢指针用来保证输出结果，快指针用来遍历输入元素。

---
## 80. 删除有序数组中的重复项 II
已知一个有序数组 `nums`，需要原地删除重复出现的元素，使得出现次数超过两次的元素只出现两次，返回新数组的长度。

### 初始方法
该题与 **Q.26**、**Q.27** 思路一致，可以考虑快慢指针（索引），也可以使用 `pop()` & `except IndexError` 动态改变数组长度。这里仅给出快慢索引方法：
```python
class Solution(object):
    def removeDuplicates(self, nums: List[int]) -> int:
        slow = 2
        for fast in range(2, len(nums)):
            if nums[fast] != nums[slow - 2]:
                nums[slow] = nums[fast]
                slow += 1
        return slow
```

---
## 169. 多数元素
给定一个大小为 `n` 的数组 `nums`，返回其中的多数元素。多数元素指出现次数至少为长度一半的元素。假定 `num` 中一定有一个多数元素。

### 初始方法
最简单暴力的方法就是计算清楚 `nums` 中每个元素分别有多少个，然后再找出出现次数最多的元素并返回，这同时也是“众数”的直观定义。在具体操作过程中，可以引入 `Dict` 结构减少对 `List` 的查找操作，尽可能提高一点速度：
```python
class Solution(object):
    def majorityElement(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]

        count = {}
        major_element = 0
        major_num = 0
        for num in nums:
            try:
                count[num] += 1
                if count[num] > major_num:
                    major_num = count[num]
                    major_element = num
            except KeyError:
                count[num] = 1
        return major_element
```

### 优化方法
题干给出的条件其实比初始方法理解的“众数”要更严格一些，即多数元素不仅是众数，还是占比过半的众数。此时设想一个场景，假如一个多数元素匹配到另一个其它元素时会发生“湮灭”，那么最终数组中会有且仅有至少一个的多数元素。再进一步落实到操作：假如我们从头开始遍历 `num`，只要当前元素与上一个元素不同，就消灭掉上一个元素（“湮灭”），最终 `num` 中将只剩下多数元素（可能只有一个，也可能剩下多个相同的元素）。

---
## 189. 轮转数组
给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。要求原地修改数组。

示例：
> 输入：`nums = [1,2,3,4,5,6,7], k = 3`
> 输出：`[5,6,7,1,2,3,4]`

### 初始方法
为了尽量减少不必要的内存占用（变量赋值），利用索引切片直接拼接两个片段，并以索引方法赋值给 `nums` 以实现原地修改。
```python
class Solution(object):
    def rotate(self, nums: List[int], k: int) -> None:
        """Do not return anything, modify nums in-place instead."""
        n = len(nums)
        if k > n:
            k = k % n
        nums[:] = nums[-k:] + nums[:-k]
```
这题只有一个注意点，就是当 `k` 大于或等于数组长度时的处理方法。当 `k` 等于数组长度时，相当于没有轮转；当 `k` 大于数组长度时，需要先通过取余判断真正轮转的数目，接下来的步骤就与 `k` 小于数组长度时一样了。

---
## 121. 买卖股票的最佳时机
给定一个数组 `prices`，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。用户只能选择某一天买入这只股票，并选择在之后的某一天卖出。设计一个算法来计算能获取的最大利润。
- 函数要返回从这笔交易中可获得的最大理论，如果无法获利则返回 0。

示例：
> 输入：[7,1,5,3,6,4]
> 输出：5（在第二天买入，第五天卖出，可获得最大利润 5）

### 初始方法
从总体思路来看，最大利润与最低买入价有关，因此在遍历过程中找最低价格是循环的一个重要目的，但最终输出并不绝对与最低价绑定。与之对应的一种特殊情况是：第一天买进，第二天疯涨，然后一直下跌。这样虽然我们找到了最低价（例如最后一天），但已经错过了交易时间。因此要把“找最低价格”与“计算可能收益”设置为两条并行的路线。具体来说，把第一天的价格初始化为最低可买入价格，最大收益为 0，从第二天开始遍历：假如当天价格低于最低价（初始值为第一天），则更新最低价，最大收益不变；若当天价格高于最低价，则判断是否更新最大收益。
```python
class Solution(object):
    def maxProfit(self, prices: List[int]) -> int:
        min_price = prices[0]
        max_profit = 0
        for i in prices[1:]:
            if i < min_price:
                min_price = i
            else:
                max_profit = max(max_profit, i - min_price)
        return max_profit
```

---
## 122. 买卖股票的最佳时机 II
给定一个数组 `prices`，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。在每一天都可以决定是否购买或卖出股票，同时间内最多只能持有一股股票。当天内允许先买后卖或者先卖后买。要求函数返回可获得的最大利润。

### 初始方法
由于题目特殊性，有一种逃课方法：既然当天可以既买又卖，同时题目允许我们开天眼预知后续的价格走向，那么只要：
- 如果明天会涨，那今天就买；若要跌，今天就不买；
- 每天对比前一天，涨了立马卖掉；如果下一天继续涨，就当天再买回来。

这样只要第 `i + 1` 天涨价，相比第 `i` 天的收益就都可以计入最大收益中。
```python
class Solution(object):
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(1, len(prices)):
            if prices[i] - prices[i - 1] > 0:
                profit += prices[i] - prices[i - 1]
        return profit
```

### 优化方法
鉴于逃课方法过于简单，很难说这个优化是不是真优化了。我们从动态规划的角度来看看正经角度该如何分析这种问题：
- **定义状态**：`dp[i][0]` 表示第 `i` 天交易完后手里没有股票的最大利润，`dp[i][1]` 表示手里持有一支股票的最大利润（`i` 从 0 开始）；
- **状态转移方程**：
  - 考虑 `dp[i][0]` 的转移方程（当天交易完以后手里没有股票）：假如前一天就没有股票（`dp[i - 1][0]`），那收益自然不变；假如前一天有股票（`dp[i - 1][1]`），那今日卖掉后获得的收益应当为 `prices[i]`。为了收益最大化，对比二者的收益获取最佳选择：`dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])`；
  - 考虑 `dp[i][1]` 的转移方程（当天交易完以后手里持有股票）：假如前一天也持有股票（`dp[i - 1][1]`），那么相当于没动；假如前一天未持有股票（`dp[i - 1][0]`），今天就买进了股票，收益要减去 `prices[i]`，最佳选择为：`dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])`；
- **初始化**：`dp[0][0] = 0`、`dp[0][1] = -prices[0]`；同时最后一天要是买进，必然是要亏损的，即 `dp[n - 1][1] < dp[n - 1][0]`，所以返回的答案应为 `dp[n - 1][0]`。
- **计算顺序**：仅需迭代天数即可。
```python
class Solution(object):
    def maxProfit(self, prices: List[int]) -> int:
        # initialization
        n = len(prices)
        dp = {(i, j): 0 for i in range(n) for j in range(2)}
        dp[(0, 1)] = -1 * prices[0]

        # main process
        for i in range(1, n):
            dp[(i, 0)] = max(dp[(i - 1, 0)], dp[(i - 1, 1)] + prices[i])
            dp[(i, 1)] = max(dp[(i - 1, 1)], dp[(i - 1, 0)] - prices[i])
        return dp[(n - 1, 0)]
```
事实上，动态规划并没有逃课方法快。但是从原则上来看，动态规划没有考虑当天还能卖出再买进的操作，可能更符合真实交易环境，所以这种方法还是需要掌握的。

---
## 55. 跳跃游戏
给定一个非负整数数组 `nums`，初始位置为数组的第一个下标，数组中的每个元素代表在该位置可跳跃的最大长度，判断能否到达数组终点，返回 `bool` 变量。
- 可跳跃的最大长度不一定要跳满

示例：
> 输入：`nums = [2,3,1,1,4]`
> 输出：`True`（先跳一步到 `3`，然后跳三步到达终点）

> 输入：`nums = [3,2,1,0,4]`
> 输出：`False`（无论怎样都会停在 `0` 的位置）

### 初始方法
该问题的考虑方法不是寻找能抵达终点的组合，而是考虑每一个点位可向后移动的最大距离。以 `nums = [2,3,1,1,4]` 为例，初始位置可移动到的索引有 `1`、`2`，则最远位置为 `2`，且在最远位置范围内的所有点均可达；所以接下来可以考虑从 `1` 或 `2` 出发，分别寻找可达的最远位置，直到最远位置达到（或超过）终点索引。需要注意的有两点：
- 可以参与循环的点只能在最远位置范围内，最远位置是随着遍历进程更新的；
- 若最远位置达不到终点索引，则说明不论如何组合都不可能跳到终点。

```python
class Solution(object):
    def canJump(self, nums: List[int]) -> bool:
        max_length = 0
        n = len(nums)
        for i in range(n):
            if i <= max_length:
                max_length = max(max_length, i + nums[i])
                if max_length >= (n - 1):
                    return True
        return False
```

---
## 45. 跳跃游戏 II
