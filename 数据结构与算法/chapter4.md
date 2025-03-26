# 队列抽象数据类型
队列（Queue）是一种有次序的数据集合，其特征是：新数据项添加总发生在一端（尾端，rear），已有数据项的移除总发生在另一端（首端，front），不允许插队。队列的出入规则简称为 FIFO：First-in first out（先进先出、后进后出）。

队列常用于计算机的进程调度。在计算机系统实际运行过程中，进程数远远多于 CPU 核心数，有些进程还需要等待不同类型的 I/O 事件，所以操作系统核心往往会采用多个队列来对系统中同时运行的进程进行调度，其主要原则综合了“先到先服务”、“资源充分利用”两个出发点。此外，键盘缓冲区也是日常使用过程中非常重要的队列应用。对于手速较快的用户，其敲击键盘输入的字符会先保存至队列性质的缓冲区，然后根据 FIFO 性质依次输出和显示相应的字符，确保顺序不出错误。这种“预输入”性质在部分动作类型游戏中同样至关重要。

具体来说，构建一个抽象数据类型“队列”需要满足或支持以下操作：

- ```Queue()```：创建一个空队列，不包含任何数据项；
- ```enqueue(item)```：将数据项 ```item``` 添加到队尾，无返回值；
- ```dequeue()```：从队首移除数据项并返回该数据项，同时修改队列；
- ```isEmpty()```：检查并返回 bool 变量，判断队列是否为空队列；
- ```size()```：返回队列中数据项的总数目。

在 Python 语言中，可以利用基础类 ```List``` 来实现抽象类 ```Queue```，将 ```List``` 的首端作为 ```Queue``` 的尾端，```List``` 的尾端作为 ```Queue``` 的首端，这样 ```enqueue()``` 的复杂度是 $O \left( n \right)$，而 ```dequeue()``` 的复杂度是 $O \left( 1 \right)$。与 ```Stack()``` 类似地，ADT Queue 的实现方式也不是唯一的，我们当然也可以把 ```List``` 的首端作为 ```Queue``` 的首端，这样添加/移除数据项的复杂度就会互换。不难发现，没有一种方式能够同时使得出列与入列的复杂度都最小，所以我们需要根据实际应用场景决定具体实现的方法。
```python
class Queue(object):
    def __init__(self):
        self.items = []

    def enqueue(self, item: Any):
        self.items.insert(0, item)

    def dequeue(self) -> Any:
        return self.items.pop()

    def isEmpty(self) -> bool:
        return self.items == []

    def size(self) -> int:
        return len(self.items)
```

---
# 队列的应用
## Josephus 问题
该问题描述如下：传说犹太人反叛罗马人，被抓后落入困境，约瑟夫（Josephus）与其它 39 名战俘坐成一圈，按数字 1-7 从小到大依次报数，报到 7 的战俘将被处刑，之后继续上述流程，直至剩余最后一人。约瑟夫给自己安排了一个特殊的位置，最终得以幸存。这个问题有一些特点需要注意：

- 每次处刑之后，报数过程重置并从下一人开始进行。即假如连续的三个人分别为 $a$、$b$、$c$，其中 $b$ 报到了 7 被处刑，则下一轮报数从 $c$ 开始，$c$ 应当报 1；
- 当剩余人数不足 7 人时，如仅剩 $a$、$b$、$c$ 三人，$a$ 开始报数直至 $c$ 报完 3 之后，重新轮到 $a$ 继续报数（即 $a$ 报 4、$b$ 报 5……），直至轮到 $a$ 报 7，$a$ 被处刑；

Josephus 问题的广义数学描述是：有 $n$ 个人围成一圈，从第一个人开始报数，报到第 $m$ 个人时将其淘汰，然后从下一个人重新开始报数，继续报到第 $m$ 个人时再次淘汰，如此循环，直到剩下最后一个人为止。问题是：最后剩下的那个人的初始位置是多少？正向分析解决这个问题是有些困难的，我们可以通过工程手段测试予以确定：即生成一个包含 $n$ 个元素的列表，模拟上述过程，直至列表中只剩下最后一个元素。

我们很难直接基于索引模拟 Josephus 问题，因为元素索引下标会经常发生变化，所以我们需要一种数据结构，既能在动态变化（增加/删除）中保持元素间相对位置不变，又能轻松实现对全体元素的循环遍历，队列就很适合处理这类问题。把全体人名按顺序依次入列，之后开始模拟报数，表现为队首的元素出队并计数，然后再到队尾入队，即完成一次报数。当计数结果为 $m$ 时，该元素直接删除不再入队，即“处刑”。
```python
def solve_josephus(name_list: List[str], num: int) -> int:
    """Solve Josephus problem."""
    # initialization
    name_queue = Queue()
    for name in name_list:
        name_queue.enqueue(name)

    # main process
    while name_queue.size() > 1:
        for i in range(num - 1):
            name_queue.enqueue(name_queue.dequeue())
        _ = name_queue.pop()
    return name_queue.dequeue()
```
这种基于 ```Queue``` 结构的方法，时间复杂度是 $O \left( mn \right)$，不算太高效，但非常节约内存空间。此处不加证明地给出基于递归的另一种方法，仅供性能方面的参考。假设 $f(n,m)$ 表示在 $n$ 个人中，每次报数为 $m$ 时，最终剩下的人的初始位置，该问题可通过递归公式解决：
$$
f(n, m) = 
\begin{cases}
0 \phantom{(f(n - 1, m) + m) \mod n} & \ n = 1 \\
(f(n - 1, m) + m) \mod n & \ n > 1
\end{cases}
$$
```python
def solve_josephus_v2(total_num: int, target_num: int) -> int:
    """Solve Josephus problem by recursion."""
    if total_num > 1:
        return (solve_josephus_v2(total_num - 1, target_num) + target_num) % total_num
    elif total_num == 1:
        return 0
```
当 $m$ 为 100、$n$ 为 1000 时，使用 ```Queue``` 求解耗时约 0.047 秒，而递归耗时约 0.00025 秒。递归的运算速度显然是更快的，但倘若问题复杂度提高一个数量级，比如 $n$ 增加到 10000 时，我的机器可能就会报错了（递归溢出）。递归嵌套层数即为人数 $n$，$m$ 仅作为计算步骤中的操作数，因此递归基本不受 $m$ 的限制：当 $m$ 为 1000、$n$ 为 1000 时，```Queue``` 用时约 4.65 秒（与时间复杂度 $O \left( mn \right)$ 是吻合的），而递归用时约 0.00030 秒，与 $m$ 为 100、$n$ 为 1000 时没什么差别。综上所述，```Queue``` 求解是一种空间友好型方法，即便问题规模很大，依然可以顺利运行，只是运行效率会受到影响；递归法虽然快，但确是一种危险的算法，仅适合处理小规模（特指 $n$）Josephus 问题，否则递归嵌套层数过多会无法正常运行。

## 打印任务
假设如下场景：教室中有 10 名学生，大家共享一台打印机。打印机采取“先到先服务”的队列策略执行任务，且有两种工作模式：（1）草稿模式打印质量低，但是速度快，每分钟 10 页；高质量模式速度慢，每分钟 5 页。每人平均每小时打印 2 次，每次打印页数为 1 至 20 页。我们应当选择哪种打印模式，使得全体用户等待时间不太久的情况下，尽可能地提高打印质量？这种决策支持问题无法通过某种规则直接计算，但可以通过程序模拟得出不同策略下的应用结果，并提供分析依据。

对上述具体问题进行抽象建模，涉及的对象有三个：**打印任务**、**打印队列**以及**打印机**。

- 打印任务的属性是任务提交时间以及打印页数；
- 打印队列的属性是具有“先进先出”（FIFO）性质的任务队列；
- 打印机的属性是打印速度以及当前是否被占用。

问题的过程包含**生成**、**提交打印任务**以及**实施打印**。

- 以上述场景为例，任务生成具有概率属性，指的是单位时间内（如秒）有多少个打印任务可能被提交，即 $\frac{10 \times 2}{3600} = \frac{1}{180}$；
- 打印任务指的是每次打印的页数，在当前任务下是 1 至 20 页等概率分布；
- 打印过程包含两个要素，其一是当前正在打印的作业，表示工作状态占用；其二是倒计时，当倒计时为 0 时才解除占用状态，可以处理下一个任务。

仿真模拟需要创建打印队列，同时设计一个统一的时间框架，在时间流逝（计数）过程中，按照概率生成打印作业任务，加入打印队列，如果打印机空闲且队列不空，则取出队首作业按照既定速度开始打印，并进入占用状态。如果占用状态过程中产生了新的打印作业任务，则记录等待时间。在时间框架计数结束（时间用尽）后，统计平均等待时间。每一项作业有两个时间参数，一是等待时间，即开始打印时的时间戳减去生成作业时的时间戳；二是打印时间，即开始打印时的页数（生成作业时初始化）除以打印速度。
```python
import random

class Task(object):
    def __init__(self, time: int):
        """Basic configuration.

        Args:
            time (int): The timestamp when the task starts.
        """
        # initialization
        self.time_stamp = time
        self.pages = random.randrange(1, 21)

    def get_stamp(self) -> int:
        return self.time_stamp

    def get_pages(self) -> int:
        return self.pages

    def wait_time(self, current_time: int) -> int:
        return current_time - self.time_stamp


class Printer(object):
    def __init__(self, ppm: int):
        """Basic configuration.

        Args:
            ppm (int): Pages per-minute, i.e. the working mode of the printer.
        """
        # initialization
        self.page_rate = ppm  # page per-minute
        self.current_task = None
        self.time_remaining = 0  # time for executing the task (seconds)
        self.busy = False

    def tick(self):
        """Simulate printing operation within a unit of time."""
        if self.current_task != None:
            # self.busy = True
            self.time_remaining -= 1
            if self.time_remaining <= 0:  # finish the current task
                self.current_task = None
                # self.busy = False

    def busy(self) -> bool:
        """Determine whether the printer is currently working."""
        return self.current_task != None

    def start(self, new_task: Task):
        """Start a new task.

        Args:
            new_task (Task object). See details in Task().
        """
        self.current_task = new_task
        self.time_remaining = 60 * new_task.get_pages() / self.page_rate


def new_print_task(prob: int = 180) -> bool:
    """Generate tasks according to the given probability. i.e. 1/prob."""
    return random.randrange(1, prob + 1) == prob


def simulation(total_time: int, ppm: int) -> Union[float, int]:
    """Stimulation of printer problem.

    Args:
        total_time (int): The total duration (seconds) of the simulation process.
        ppm (int): Pages per-minute, i.e. the working mode of the printer.

    Returns:
        avg_wait_time (Union[float, int]): Averaged waiting time.
    """
    # initialization
    printer = Printer(ppm=ppm)
    print_queue = Queue()
    waiting_time = []

    # main process
    for current_second in range(total_time):
        if new_print_task():  # a new printing task has been generated
            print_queue.enqueue(Task(time=current_second))

        # check printer's work state & mission queue
        if (not printer.busy()) and (not print_queue.isEmpty()): 
            new_task = print_queue.dequeue()
            waiting_time.append(new_task.wait_time(current_second))
            printer.start(new_task)
        printer.tick()  # printer is working
    # return np.mean(waiting_time)
    return sum(waiting_time) / len(waiting_time)
```
在主程序 ```simulation()``` 中，仿真过程是这样的：每经过一个单位时间（```current_second```），模拟是否有学生要去打印（```new_print_task()```），如果有，则在任务队列 ```print_queue``` 中入列一个任务对象 ```Task```，并记录时间戳（```Task.time_stamp```）；与此同时检查打印机的状态，如果打印机正忙（```printer.busy()```），那就让它先工作（```printer.tick()```），等它完成任务了再出列位于 ```print_queue``` 队首的新任务，记录该任务的等待时间（```new_task.wait_time()```）并继续打印。

代码中还有一些其它细节需要指出：

- ```Printer.start()``` 中，类属性 ```time_remaining``` 不需要考虑取整，因为在类方法 ```tick()``` 中对于打印任务结束的判断是 ```time_remaining``` 为非负数，即已进行了向上取整；
- ```Task``` 类的 ```get_stamp()``` 以及 ```get_pages()``` 方法本质上只是查看了类属性，在实际应用中可以用 ```Task.time_stamp```、```Task.pages``` 这种返回类属性的语句直接替代，但在示例代码中还是特地给出了接口函数。这种习惯对于面向对象编程是比较重要的，不仅可以提高代码的可读性、避免类属性变化导致的潜在错误，还能在函数层面支持对类属性的安全修改与返回；
- 仿真过程设计了 ```Printer``` 与 ```Task``` 两个对象、```new_print_task()``` 与 ```simulation()``` 两个事件函数，成功化繁为简，抽象出关键过程。利用代码模拟系统对现实问题的仿真，能够在不耗费真实资源的情况下帮助我们合理决策，这种框架设计以及问题抽象化方法需要认真学习。

---
# 双端队列抽象数据类型
双端队列（Deque）与队列相似，两端可以分别视为“首”、“尾”端，但 ```Deque``` 中数据项既可以从队首加入，也可以从队尾加入，移除同样也可以在双端进行，相当于集成了 ```Stack``` 和 ```Queue``` 的能力。需要注意的是，```Deque``` 虽然可以用来模拟 ```Stack``` 和 ```Queue```，但不具有内在的 LIFO 或 FIFO 特性，与 Python 内置的 ```List``` 一样，需要自行维护抽象类的性质。

一个 ```Deque``` 类应满足的定义操作包括：

- ```Deque()```：创建一个空的双端队列；
- ```addFront(item)```：将 ```item``` 加入队首；
- ```addRear(item)```：将 ```item``` 加入队尾；
- ```removeFront()```：从队首移除数据项，并返回该数据项；
- ```removeRear()```：从队尾移除数据项，并返回该数据项；
- ```isEmpty()```：判断双端队列是否为空；
- ```size()```：返回双端队列中数据项的总数目。

不难发现，```Deque``` 与 Python 内置类 ```List``` 有很多相似之处，在用队列实现 ```Stack```、```Queue``` 时封锁的类方法在此处可以更多地用上。不失一般性地，我们默认 ```List``` 的右端为队首，左端为队尾（与 ```Queue``` 一致）：
```python
class Deque(object):
    def __init__(self):
        self.items = []

    def addFront(self, item: Any):
        self.items.append(item)

    def addRear(self, item: Any):
        self.items.insert(0, item)

    def removeFront(self) -> Any:
        return self.items.pop()

    def removeRear(self) -> Any:
        return self.items.pop(0)

    def isEmpty(self) -> bool:
        return self.items == []

    def size(self) -> int:
        return len(self.items)
```
在上述设计下，```addFront(item)```、```removeFront()``` 的操作复杂度都是常数级；```addRear(item)```、```removeRear()``` 的复杂度是线性级（与总长度有关）。若将 ```List``` 的右端设为队尾，则以上操作的复杂度对调。

---
# 双端队列的应用
## “回文词”判定
“回文词”指正读和反读都一样的词，如 radar、toot 等；在中文语境中，还可以衍生为回文句，例如“上海自来水来自海上，山东落花生花落东山”。回文词用 ```Deque``` 很容易解决，只需要把字符串加入队列中，再从两端分别出列字符并判断是否相同，只要有一次不同则原文不是回文词：
```python
def palindrome_detect(x: str) -> bool:
    """Determine whether the input string is a palindrome.

    Args:
        x (str): Input string.

    Returns:
        palindrome (bool).
    """
    # initialization
    if len(x) % 2 == 1:
        return False
    words = Deque()
    for word in x:
        words.addRear(word)

    # main process
    for i in range(int(words.size() / 2)):
        front = words.removeFront()
        end = words.removeRear()
        if front != end:
            return False
    return True
```
这里我写的版本比视频课程给出的案例稍稍有些不同，加入了一些提高运行效率的小 tricks。判断是否为“回文词”的标准在前述部分已有说明，这里我做出的改进在于两点：（1）如果文本长度是奇数，则不可能为回文词。对于大型字符串而言，可以省去入列的时间；（2）使用固定轮次的 for 循环替代 while 循环，减少不必要的判断控制流。

对于小型文本，其实怎么样都很快，内存占用也几乎没有变化。但是当文本长度达到一定水平时，或者进行 I/O 密集型任务时，这种优化积累起来还是比较可观的。

---
# 无序表 & 有序表
目前，我们已经用 ```List``` 结构实现了很多其它的抽象数据类。但并非所有的编程语言都有类似 Python 中 ```List``` 这样简单强大的数据集类型，有时候需要程序员自行定义。

## 无序表
无序表（```UnorderedList```）是一种数据项按照相对位置存放的数据集，数据项只按照存放位置索引，与数据项本身属性（如数值大小）无关，这一特性与 ```List``` 其实是很像的。无序表需包含的操作如下所示：

- ```UnorderedList()```：创建一个空列表；
- ```add(item)```：添加一个数据项到无序表中（假设 ```item``` 原先不存在于表中）；
- ```remove(item)```：从表中移除项目 ```item```，表被修改（假设 ```item``` 原先存在于表中）；
- ```search(item)```：在表中查找项目 ```item```，返回 bool 类型值；
- ```isEmpty()```：判断列表是否为空；
- ```size()```：返回列表包含了多少数据项；
- ```append(item)```：添加一个数据项到表的末尾（假设 ```item``` 原先不存在于列表中）；
- ```index(item)```：返回数据项在表中的位置；
- ```insert(pos, item)```：将数据项 ```item``` 插入到 ```pos``` 位置处（假设 ```item``` 原先不存在于列表中，且原列表长度足够覆盖位置 ```pos```）；
- ```pop()```：从列表末尾移除并返回该数据项（假设 ```item``` 原先存在于列表中）；
- ```pop(pos)```：从位置 ```pos``` 移除数据项（假设原列表存在位置 ```pos```）。

当我们用惯了 Python 的内置类，面对这种需要另起炉灶的问题感到毫无头绪是很正常的。我们需要以更加底层的方式思考列表的特点：尽管它要求保持数据项的相对位置，但从硬件角度来看，数据项不一定要依次存放在连续的存储空间，换句话说，我们实际上要实现的关键功能是“**链接指向**”，在访问到当前数据项时，应当同时明确下一个数据项的位置，即链表。链表实现的最基本元素是节点（Node），每个节点至少包含两个信息：数据项的值以及指向下一个节点的引用信息。对于链表头和链表尾，为了区分它们与其它节点，需要进行额外的标定（如链表尾不存在下一个节点，相关信息可以定义为 ```None```）。

那么首先，我们先实现节点抽象类 ```Node```。一个 ```Node``` 应当具备以下功能：

- ```Node(init_data)```：创建一个初始数据为 ```init_data``` 的节点，在初始化过程中暂时不需要定义指向下一个节点的信息；
- ```getData()```：返回节点包含的数据值；
- ```getNext()```：返回指向下一个节点所需的信息，即下一个节点实例 ```Node```；
- ```setData(new_data)```：修改节点包含的数据项具体值为 ```new_data```；
- ```setNext(new_next)```：修改指向下一个节点的索引（即下一个节点实例 ```Node```）。

```python
class Node(object):
    def __init__(self, init_data: Any):
        self.data = init_data
        self.next = None

    def getData(self) -> Any:
        return self.data

    def getNext(self) -> Any:
        return self.next

    def setData(self, new_data: Any):
        self.data = new_data  # any type

    def setNext(self, new_next):
        self.next = new_next  # Node object
```
接下来我们要利用链表结构实现无序表 ```UnorderedList```。具体来说，数据项的数值与下一项的索引存储在 ```Node``` 类中，我们还需要实现的功能是链接节点。
```python
class UnorderedList(object):
    def __init__(self):
        self.head = None

    def isEmpty(self) -> bool:
        return (self.head is None)

    def add(self, item: Any):
        # create new node for item
        temp = Node(item)  # new value
        temp.setNext(self.head)  # next node is the last node (the old head)
        self.head = temp  # set new node as the new head

    def size(self) -> int:
        current_node = self.head
        n_count = 0
        while current_node is not None:
            n_count += 1
            current_node = current_node.getNext()
        return n_count

    def search(self, item: Any) -> bool:
        current_node = self.head
        found = False
        while (not found) and (current_node is not None):
            if current_node.getData() != item:
                current_node = current_node.getNext()
            else:
                found = True
        return found

    def index(self, item: Any) -> int:
        # combine search() & size()
        current_node = self.head
        found = False
        n_count = 0
        while (not found) and (current_node is not None):
            if current_node.getData() != item:
                current_node = current_node.getNext()
                n_count += 1
            else:
                found = True
        return n_count

    def remove(self, item: Any):
        # initialization
        current_node = self.head
        found = False
        previous_node = None

        # search for the input item
        while not found:
            if current_node.getData() == item:
                found = True
            else:
                previous_node = current_node
                current_node = current_node.getNext()

        # update index information in nodes
        if current_node == self.head:  # head (first node)
            self.head = current_node.getNext()
        else:
            previous_node.setNext(current_node.getNext())

    def append(self, item: Any):
        # from head node to the last node
        current_node = self.head
        while current_node.getNext() is not None:
            current_node = current_node.getNext()

        # append a new node
        current_node.setNext(Node(item))

    def insert(self, pos: int, item: Any):
        # from head node to the input position
        current_node = self.head
        if pos == 0:
            self.add(item)
        else:
            for i in range(pos):
                previous_node = current_node
                current_node = current_node.getNext()
            temp = Node(item)
            temp.setNext(current_node)
            previous_node.setNext(temp)

    def pop(self, pos: Optional[int] = None) -> Any:
        # from head node to the last node
        current_node = self.head
        if pos is None:  # pop the last node
            while current_node.getNext() is not None:
                previous_node = current_node
                current_node = current_node.getNext()
            previous_node.setNext(None)
        elif pos == 0:  # pop the original head
            self.head = current_node.getNext()  # change the head to the next node
        elif pos >= 1:
            for i in range(pos):
                previous_node = current_node  # head
                current_node = current_node.getNext()  # the 2nd node
            previous_node.setNext(current_node.getNext())  # update the link relationship
        return current_node.getData()  # return the data of the unlinked node
```
我们先给出上述代码，而后再根据功能逐一展开详述：
- 首先是**初始化对象** ```__init__()```：```UnorderedList``` 需要定义链表表头 ```head```，该属性在列表初始化时设为 ```None```，随着列表增长，```head``` 始终指向对首个 ```Node``` 的引用。因此判断列表是否为空，实质上是判断 ```head``` 指向的是否为空节点（即 ```None```），这样我们顺便也实现了 ```isEmpty()``` 功能；

- 其次实现**新增数据项** ```add(item)``` 的功能：```UnorderedList``` 中的数据是无序存储的，有序的仅是导向下一个 ```Node``` 的索引，所以要访问整条链上的任意数据项，都需要从表头 ```head``` 开始一直 ```next``` 下去，直至达到相应位置。基于这种特点不难判断，添加新数据项最快的方式是放在表头而不是放在表尾；
  - 此处与 Python 内置 ```List``` 类不一样，后者的 ```insert(0, item)``` 复杂度是线性的，而 ```UnorderedList``` 在表头添加数据是常数级的；
  - 在链表中，加入的新数据项应当占据 ```head``` 的位置，其后向索引指向的是原表头对应的 ```Node```，就像 ```Queue``` 结构中的 FIFO 一样。所以应当先建立指向 ```Node``` 的索引，再更新链表的 ```head```。

- 之后实现**判断列表数据项总数** ```size()``` 的功能：由于 ```UnorderedList``` 只存储了 ```head``` 这一个节点，每个 ```Node``` 又仅存储当前数据项与下一个 ```Node``` 的引用，所以不管是 ```UnorderedList``` 还是 ```Node```，都不具备像 ```len(List)``` 那样可以直接计数的功能。我们需要从 ```head``` 遍历至链表尾，并计数统计经过的节点个数，以此表示数据项总数；
  - 之所以要用这种遍历计数的方式（线性级），是因为实际数据存储不是连续的，只能挨个计数才能确认总数；当数据连续存储时，可以用末端数据地址减去首端后除以单位长度计算总数（常数级），但这种存储方式耗费的系统调度资源是远远多于离散存储的。

- **查找** ```search(item)``` 功能是 Python 内置 ```List``` 类不具备的功能，具体实现原理也很简单，挨个判断是否为搜索项即可。与之类似地，索引 ```index(item)``` 功能则更像是 ```search(item)``` 与 ```size()``` 的结合，一边 search，一边计数。找到了就输出当前数字作为索引；

- 在能够找到相应数据项的基础上，我们可以实现移除 ```remove(item)``` 功能了。删除的本质是链表不再连接该节点，具体原理是将当前节点的前、后两个节点直接相连，从而去除当前节点的索引。从硬件角度看，被删除的数据（节点）其实还储存在某个区域，但指向该区域的地址索引已经没有了，该区域的数据在下一次存入新数据时随时可以被覆盖。
  - 当数据项位于链表头时，没有前一个节点，直接把 ```head``` 设置为后一个节点即可；
  - 当数据项位于链表中央（有后节点）或末尾（后节点为 ```None```）时，把前节点的 ```next``` 设置为后节点即可；
  - 由于每个 ```Node``` 只有向后的索引，即链表是单向的，所以在查找过程中，我们需要随时记录并更新前节点（```previous_node```）；对于未来介绍的双向链表而言，就无需这步操作了。但是双向链表的维护本身又需要占用一定的资源，所以一个反复强调的点是，没有哪种结构能够适合所有的应用场景。

- 接下来实现末尾增加 ```append(item)``` 功能。首先为 ```item``` 新增一个 ```Node```，找到原链表的最后一个 ```Node```，将其 ```next``` 属性设为新建的 ```Node``` 即可。不难发现，此时 ```append(item)``` 的复杂度就变成了线性的（与 ```add(item)``` 相对应）。进一步可知，对于线性结构，要么像 ```List``` 一样 ```append(item)``` 最快，要么像 ```UnorderedList``` 一样 ```add(item)``` 最快，不可能实现在头、尾添加数据都一样快；

- 按索引插入 ```insert(pos, item)``` 在 ```UnorderedList``` 结构中略有些麻烦，主要原因在于 ```UnorderedList``` 总是按照数据项内容进行索引 ```index(item)```，按下标索引是需要额外实现的功能：
  - 当 ```pos``` 为 0 时，```insert(0, item)``` 等价于 ```add(item)```；
  - 当 ```pos``` 为其它值时，```insert(pos, item)``` 相当于在第 pos - 1 与 pos + 1 号节点之间插入一个以 ```item``` 为值的新节点（第 0 号节点指的是第 1 个节点）。插入的具体流程是：先将新节点的 ```next``` 属性设置为第 pos + 1 号节点，再修改第 pos - 1 号节点的 ```next``` 属性为新节点。

- ```pop()``` 与 ```pop(pos)``` 功能没有本质区别，理由在上一项功能的说明中已有解释：```UnorderedList``` 想要找到第 pos 号节点需要从 ```head``` 一直遍历直至该节点；假如 ```pos``` 为 ```None```，则需要遍历至最后一个节点。
  - 完成 ```pop``` 需要进行的第一步操作与 ```remove(item)``` 类似，将当前节点的前节点直接连接到后节点（若 ```pop()``` 则前节点的 ```next``` 属性为 ```None```），即完成删除链表中节点的操作；
  - 第二步是返回当前节点的 ```data``` 属性。需要注意的是，在完成第一步后，我们已经无法从链表中访问到被删除的节点，但是程序中有为该节点保存一个临时变量 ```current_node```，因此返回数据值还是可以通过 ```current_node.getData()``` 实现的。

至此，我们基本上完成了 ```UnorderedList``` 所必需的全部基本功能。当然，上述程序还是有一些潜在问题的：

（1）一些前置假设条件没有设计单独的检测语句，在有些情况下可能导致溢出或错误；

（2）虽然链表不再访问那些被删除的节点，但我们并没有告诉操作系统这些空间是“可用的”了。在 Python 环境中，这些变量占据的实体空间可能还处于无法写入新数据的状态，这需要我们采用更进一步的回收步骤才能真正完成空间的再利用。

由于对这些异常的封装以及内存回收不属于抽象类的关键功能，因此我们在刚开始入门时不需要被这些需求过度分散了注意力。

## 有序表
有序表（```OrderedList```）是一种数据集，其中数据项依照某可比性质（如整数大小、字母表先后等）来决定其在列表中的位置，在这种比较规则下，越“小”的数据项越靠近列表头（靠前）。```OrderedList``` 定义的操作与无序表基本一致，具体包括：

- ```OrderedList()```：创建一个空表；
- ```add(item)```：添加一个数据项到表中，并保持整体顺序（假设 ```item``` 原先并不存在）；
- ```remove(item)```：从表中移除已有数据项 ```item```，有序表被修改；
- ```search(item)```：查找数据项并返回 ```bool``` 变量；
- ```isEmpty()```：判断是否空表；
- ```size()```：返回表中数据项的个数；
- ```index(item)```：返回数据项在表中的位置（默认存在）；
- ```pop()```：移除并返回表中最后一项（表中至少有一项数据）；
- ```pop(pos)```：移除并返回表中指定位置的数据项（默认存在该位置）

既然 ```List``` 不让用了，我们要实现列表还得用链表结构，唯一的区别在于，每次加入数据项的时候，不能直接在队尾节点中添加新节点，而是要选择合适的位置加入。因此重点修改的方法在于 ```search()``` & ```add()```，其它方法与数据项的次序无关，无需修改。

- 首先说 ```search(item)```：无序表需要遍历所有数据项直至找到对应项（或返回 ```False```）；而有序表可以通过比较当前数据项与 ```item``` 的大小，判断是否还有必要继续查询；

- 其次是 ```add(item)```：有序表的添加包含了一定程度的遍历过程，具体来说，需要沿着链表找到第一个比 ```item``` 大的数据项，并将 ```item``` 插入到该项前面，以此维护有序性。此时还有一个潜在的困难是，链表并不具备“前驱”节点的引用信息，因此在每轮查询过程中，需要同时记录前节点和后节点的信息，才能完整实现节点的插入；

```python
class OrderedList(object):
    def __init__(self):
        self.head = None

    def isEmpty(self) -> bool:
        return (self.head is None)

    def add(self, item: Any):
        # find the insert place: previous & current node
        current_node = self.head
        previous_node = None
        stop = False
        while (current_node is not None) and (not stop):
            if current_node.getData() > item:
                stop = True
            else:
                previous_node = current_node
                current_node = current_node.getNext()

        # insert data
        temp_node = Node(item)
        if previous_node is None:  # insert at the head
            temp_node.setNext(self.head)
            self.head = temp_node
        else:  # in the middle
            temp_node.setNext(current_node)
            previous_node.setNext(temp_node)

    def size(self) -> int:
        current_node = self.head
        n_count = 0
        while current_node is not None:
            n_count += 1
            current_node = current_node.getNext()
        return n_count

    def search(self, item: Any) -> bool:
        current_node = self.head
        found = False
        while (not found) and (current_node is not None):
            if current_node.getData() < item:
                current_node = current_node.getNext()
            elif current_node.getData() > item:
                return False  # found = False
            else:
                found = True
        return found

    def index(self, item: Any) -> int:
        # combine search() & size()
        current_node = self.head
        found = False
        n_count = 0
        while (not found) and (current_node is not None):
            if current_node.getData() != item:
                current_node = current_node.getNext()
                n_count += 1
            else:
                found = True
        return n_count

    def remove(self, item: Any):
        # initialization
        current_node = self.head
        found = False
        previous_node = None

        # search for the input item
        while not found:
            if current_node.getData() == item:
                found = True
            else:
                previous_node = current_node
                current_node = current_node.getNext()

        # update index information in nodes
        if current_node == self.head:  # head (first node)
            self.head = current_node.getNext()
        else:
            previous_node.setNext(current_node.getNext())

    def append(self, item: Any):
        # from head node to the last node
        current_node = self.head
        while current_node.getNext() is not None:
            current_node = current_node.getNext()

        # append a new node
        current_node.setNext(Node(item))

    def pop(self, pos: Optional[int] = None) -> Any:
        # from head node to the last node
        current_node = self.head
        if pos is None:  # pop the last node
            while current_node.getNext() is not None:
                previous_node = current_node
                current_node = current_node.getNext()
            previous_node.setNext(None)
        elif pos == 0:  # pop the original head
            self.head = current_node.getNext()  # change the head to the next node
        elif pos >= 1:
            for i in range(pos):
                previous_node = current_node  # head
                current_node = current_node.getNext()  # the 2nd node
            previous_node.setNext(current_node.getNext())  # update the link relationship
        return current_node.getData()  # return the data of the unlinked node
```

对比 Python 内置类 ```List```，基于链表结构实现的 ```UnorderedList``` 以及 ```OrderedList``` 在部分相同方法的实现上具有不同的时间复杂度，这主要是因为 ```List``` 基于顺序存储实现，同时在底层语言上进行了一定的优化。最后我们来总结一下链表结构的复杂度，其关键要点在于待分析方法是否涉及到链表的遍历：

- ```isEmpty()``` 以及无序表的 ```add(item)``` 都是常数级的，前者只需要检查表头，后者只需要将新节点链接到表头；
- ```size()```、```search(item)```、```remove(item)``` 以及有序表的 ```add(item)``` 都是线性级的。其中只有 ```size()``` 必须遍历全部数据项，其余三种方法都是最高可能遍历全体，按概率来看平均操作次数其实只有总长的一半。
---