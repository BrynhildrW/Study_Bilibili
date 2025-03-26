# 线性结构 Linear Structure
线性结构是一种有序数据项的集合，其中每个数据项都有唯一的前驱（首项没有）和后继（末项没有）。新的数据项加入到数据集中时，只会加入到原有的某个数据项之前或之后。根据数据结构的不同，两端的称呼可能存在差异（左右、前后、顶底等），当然这不重要，关键性区别在于数据项增减的方式。有些结构只允许数据项从一端添加（如堆栈），有些则没有这种限制。

我们首先从 4 个简单结构入手，分别是堆栈（Stack）、队列（Queue）、双端队列（Deque）以及列表（List）。这些结构的共同点在于，数据项之间只存在先后次序关系，都是线性结构。它们的区别则在于数据项的增减方式。

---
# 栈抽象数据类型
堆栈（Stack）是一种有次序的数据项集合，在栈中，数据项的加入和移除都仅发生在同一端，该端为栈顶（top），另一端为栈底（base）。托盘堆是一种物理栈的表现形式，我们拿走、放置空盘都只能在盘子堆的顶部，不能从底部或中间抽出空盘。

栈的出入规则简称为 LIFO：Last In, Fisrt Out（后进先出）。LIFO 是一种基于数据项保存时间的次序，距离栈底越近的数据项，留在栈中的时间越长；最新加入栈的数据项会被最先移除。LIFO 特性在某些计算机操作上是很重要的，比如浏览器的“后退”，Word 中的“Undo”，```Ctrl + Z```等等，这些撤销类的操作都是优先处理最近执行的步骤。

具体来说，构建一个抽象数据类型“栈”需要满足或支持以下操作：

- ```Stack()```：创建一个空栈，不包含任何数据项；
- ```push(item)```：将新数据项 ```item``` 加入栈顶，无返回值；
- ```pop()```：将栈顶数据项移除并返回该数据项，同时修改栈；
- ```peek()```：检索栈顶数据项并返回该数据项，但不修改栈，即不移除该数据项；
- ```isEmpty()```：检查并返回 bool 变量，判断栈是否为空栈；
- ```size()```：返回栈中数据项的总数目。

用 Python 实现抽象数据类（Abstract data type, ADT） Stack，主要利用的是面向对象编程特性。由于 Stack 本身通常作为数据集使用，因此可以用 Python 内置的 ```List``` 类来实现，这样 ```push()``` 和 ```pop()``` 功能可以很方便地用 ```List``` 的内置功能实现。由于 ```List``` 结构自带 ```pop()``` 功能，因此可以把索引为 0 的值作为栈底，索引为 -1 的值作为栈顶。在 ```Stack``` 层面仅允许访问栈顶，不允许访问栈底。
```python
class Stack(object):
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self) -> Any:
        return self.items.pop()

    def peek(self) -> Any:
        return self.items[-1]

    def isEmpty(self) -> bool:
        return self.items == []

    def size(self) -> int:
        return len(self.items)
```
这里需要注意的是，```Stack``` 类的 Python 实现方式不是唯一的。以栈底定义方式为例，我们当然可以在 ```List``` 中把索引为 -1 的值定义为栈底，这样我们在实现 ```Stack.push()```、```Stack.pop()``` 功能时，操作的对象都是 ```List``` 中的首位元素。不难发现这样设计的话，相应功能的运行效率会大打折扣。根据上一章内容可知，```List.pop()```、```List.append(value)``` 都是 $O \left(1 \right)$ 的，而 ```List.pop(index)```、```List.insert(index, value)``` 都是 $O \left( n \right)$ 的，所以 ```Stack.push()```、```Stack.pop()``` 在示例代码中的时间复杂度是 $O \left(1 \right)$，而修改过后就增加为 $O \left( n \right)$。这就提醒我们，在实现 ADT 时，一定要从具体实现方法的实际性能出发，尽管 ADT 本身的接口是稳定的，但不同实现方法在某些功能上可能存在较大的性能差异。同时这种差异不一定是单方面的，可能互有优劣，此时我们就需要进一步结合使用场景的需求来决定到底需要使用哪种实现方法。

---
# 栈的应用
## 简单括号匹配
对左右括号是否正确匹配的识别，是很多语言编译器的基础算法。正确匹配一般包括两个基本规则：（1）每个开括号（左）要对应一个闭括号（右）；（2）每对括号要正确地嵌套。例如 ```((), (), (()))``` 是匹配的，```((), ))(``` 是不匹配的。换句话说，假如从左到右扫描括号，最后记录到的开括号应当最先遇到闭括号，这种“先进后出”的特性很符合 ```Stack``` 类的操作逻辑。

下图展示了算法流程图。当然这个算法很基础，只能支持同类括号的匹配，不具备对不同类括号（如圆括号和中括号）的区分和匹配功能。

![括号匹配流程图](/Study_Bilibili/数据结构与算法/figures/20250131.png)

```python
def bracket_pair_detect(input_list: List[str]) -> bool:
    """Parenthesis matching detection."""
    # initialization
    s = Stack()
    assert len(input_list) != 0, 'Empty input list!'

    # matching
    for char in input_list:
        if char == '(':
            s.push(char)
        else:
            if s.isEmpty():
                return False
            else:
                _ = s.pop()
    return s.isEmpty()
```
想要进一步拓展括号种类，需要在闭括号判断为真后，再单独进行判断是否与栈顶记录的括号种类一致，在流程图中表现起来可能有些复杂，但是在代码里其实也就几行的事：
```python
open_brackets = '([{'
close_brackets = ')]}'

def bracket_match(bracket1: str, bracket2: str) -> bool:
    """Check whether the types of parentheses are consistent."""
    return open_brackets.index(bracket1) == close_brackets.index(bracket2)


def bracket_pair_detect_v2(input_list: List[str]) -> bool:
    """Upgraded parenthesis matching detection.

    Args:
        
    """
    # initialization
    s = Stack()
    assert len(input_list) != 0, 'Empty input list!'

    # matching
    for char in input_list:
        if char in open_brackets:
            s.push(char)
        else:
            if s.isEmpty() or not bracket_match(s.peek(), char):
                return False
            else:
                _ = s.pop()
    return s.isEmpty()
```
类似括号匹配，HTML 语言中还有一些其它格式上的匹配规则，同样也可以视为不同类型的括号匹配问题。

## 进制转换
十进制是人类日常生活中最常使用的数学进制，而计算机原理上通用的指令是二进制的。因此十进制与二进制的转换是计算机程序一项不可或缺的重要基础功能。从运算角度来看，十进制数（为简单起见以整数为例）转化为二进制使用的具体方法是不断地除以 2 并取余数，直至无法再整除，最后反向输出余数序列。例如数字 233：

- 233 除以 2 等于 116 余 1；
- 116 除以 2 等于 58 余 0；
- 58 除以 2 等于 29 余 0；
- 29 除以 2 等于 14 余 1；
- 14 除以 2 等于 7 余 0；
- 7 除以 2 等于 3 余 1；
- 3 除以 2 等于 1 余 1；
- 1 除以 2 等于 0 余 1；

最后反向输出 “10010111” 即为结果 “11101001”。不难发现，余数序列 “10010111” 的保存顺序是与计算流程一致，即正向的，但是实际结果需要倒向输出。这种“先进后出”的特点很适合用栈结构来实现。
```python
def dec_to_bin(decimal_num: int) -> str:
    """Convert a decimal number to a binary number(string).

    Args:
        decimal_num (int): Decimal number.

    Returns:
        bin_string (str): Binary string of decimal_num.
    """
    # initialization
    remainder_stack = Stack()

    # main process
    quotient = decimal_num
    while quotient > 0:
        quotient, remainder = divmod(quotient, 2)
        remainder_stack.push(remainder)

    # backward-output
    bin_string = ''
    while not remainder_stack.isEmpty():
        bin_string = bin_string + str(remainder_stack.pop())
    return bin_string
```
十进制与其它进制的转换是换汤不换药。例如八进制，只需把除以 2 改为除以 8 即可；对于十六进制，除了把除数改为 16，还需要用字符表映射处理一下大于 9 的余数，将其变更为十六进制中相应的字母（即 A-10、B-11、C-12、D-13、E-14、F-15），同时在输出序列前加上 “0x” 或 “#” 以与其它数制表示进行区分。
```python
def dec_to_oct(decimal_num: int) -> str:
    """Convert a decimal number to a octal number(string).

    Args:
        decimal_num (int): Decimal number.

    Returns:
        oct_string (str): Octal string of decimal_num.
    """
    # initialization
    remainder_stack = Stack()

    # main process
    quotient = decimal_num
    while quotient > 0:
        quotient, remainder = divmod(quotient, 8)
        remainder_stack.push(remainder)

    # backward-output
    oct_string = ''
    while not remainder_stack.isEmpty():
        oct_string = oct_string + str(remainder_stack.pop())
    return oct_string
```
```python
def dec_to_hex(decimal_num: int) -> str:
    """Convert a decimal number to a hexadecimal number(string).

    Args:
        decimal_num (int): Decimal number.

    Returns:
        hex_string (str): Hexadecimal string of decimal_num.
    """
    # initialization
    remainder_stack = Stack()
    mapping = '0123456789ABCDEF'

    # main process
    quotient = decimal_num
    while quotient > 0:
        quotient, remainder = divmod(quotient, 16)
        remainder_stack.push(mapping[remainder])

    # backward-output
    hex_string = '0x'
    # hex_string = '#'
    while not remainder_stack.isEmpty():
        hex_string = hex_string + remainder_stack.pop()
    return hex_string
```

## 表达式转换
人们日常使用的表达式属于“中缀表达式”，即操作符（operator）介于操作数（operand）中间的表示法（如 $b + c$）。但是这种表示法对计算机系统是不太友好的，比如 $a + b \times c$，计算机可能会对先做加法还是先做乘法会产生混淆。人（除了九漏鱼）之所以不会出问题，是因为我们对操作符有“优先级”的概念设定，同时我们还会引入括号表示强制优先级，当括号嵌套时，内层优先级最高。

对于人而言，我们总是追求尽可能地简便书写、凝练精华；而对于计算机，信息表面上的简练不一定总是有益的，有时可能需要额外增加很多判定步骤，规则上的简单才是真正的简单。所以我们引入全括号中缀表达式，即在所有表达式项两端都加上括号，如 $a + b \times c - d$ 就表示为 $((a + (b \times c)) - d)$。虽然看上去有很多冗余，但是此时计算机不需要再判断是否有括号（强制优先级），不需要判断运算符之间的优先级（先乘除后加减等等），实际需要处理的信息量其实变少了。

在中缀表达式中，计算机需要根据括号内嵌情况来决定真实计算的优先级，这个环节还有优化空间。我们希望达到的结果是，在获取输入序列的同时，计算机就知道什么要先算，什么要后算，而不是获取完整序列之后再扫描、判断。所以由“中缀”引申出了“前缀”和“后缀”两种表示，又称波兰（Polish notation, PN）、逆波兰表示法（Reverse Polish notation, RPN）。这两种方法是基于操作符相对于操作数的位置来定义的，运算符的位置决定了运算顺序。对于 PN 而言，遇到运算符时取其后紧邻的两个变量进行运算；而 RPN 则取的是其前紧邻的两个变量。例如：

|中缀表达式|前缀表达式|后缀表达式|
|--|--|--|
|$a + b \times c - d$|$- + a \times b c d$|$a b c \times + d -$|
|$(a + b) \times (c - d)$|$\times + a b - c d$|$a b + c d - \times$|
|$a \times b + c \times d$|$+ \times a b \times c d$|$a b \times c d \times +$|
|$a + b - c + d$|$+ - + a b c d$|$a b + c - d +$|

为表达式设计转换方法是我们接下来的任务，首先着眼于全括号的中缀表达式。以 $a + b \times c - d$ 为例，其全括号形式为 $((a + (b * c)) - d)$，其前、后缀形式在上述案例中已有展示。将其转化为前缀表达式的步骤如下：

- 从左到右扫描表达式，遇到的第一个运算符是 $+$，将其移动至最邻近左括号的左边，并删除左括号，即 $( + a (b \times c)) - d)$；
- 继续扫描，遇到运算符 $\times$，移动至最邻近左括号的左边并删除左括号，即 $( + a \times b c)) - d)$；
- 继续，遇到运算符 $-$，移动并删除，即 $- + a \times b c d)))$；
- 最后删掉所有的右括号，得到 $- + a \times b c d$。

相应地，转化为后缀表达式时，将运算符移动至最邻居右括号的右边，并删除右括号即可。

对于通用型的中缀表达式，括号仅用于标定强制优先级，上述转换方法自然就失效了。那么能不能从普通中缀表达式转换为前缀或后缀表达式呢，答案显然是肯定的。接下来我们以 $a + b \times c$ 的 “中转后”为例，以人的视角来分析这种转换过程是如何进行的：

- 从左到右扫描表达式，先在表达式序列中记录发现的第一个操作数 $a$，然后在操作符堆栈中记录操作符 $+$；
- 之后记录第二个操作数 $b$，此时先不急着出栈操作符 $+$，再看看后面的操作符是不是优先级更高，果然是（$\times$ 更高），按理应该先算乘法，再算加法，从后缀的书写顺序来看就是 $\times$ 在前，$+$ 在后，所以先入栈更高级的操作符 $\times$；
- 继续扫描记录操作数 $c$。再往后就没有操作符了，开始出栈并存进表达式序列，最终结果即为 $a b c \times +$。

这是没有括号的情况，我们加点难度，看看 $a + (b - c) \times d - e$。处理括号时需要把左括号当成特殊的操作符，并向后搜寻匹配的右括号，一旦匹配成功，立马出栈输出两个括号包含范围内的全部操作符，直至出栈左括号。其余操作与上述步骤基本一致：

- 在表达式序列中记录 $a$，操作符堆栈入栈 $+$、$($，启动对 $)$ 的搜寻匹配；
- 表达式序列记录 $b$，操作符栈入栈 $-$，继续检测到操作符 $)$，匹配成功，开始依次出栈 $-$、$($。$($ 直接删除不记入表达式序列，到这一步的表达式序列应为 $abc-$；
- 检测到操作符 $\times$，比栈中 $+$ 优先级高，因此无需出栈 $+$，直接入栈 $\times$。继续记录操作数 $d$；
- 检测到操作符 $-$，优先级低于栈顶的 $\times$，因此出栈直至遇到优先级更低的操作符（此处没有更低的，所以要全部出栈），再入栈 $-$。此时表达式序列为 $abc-d \times +$；
- 记录操作数 $e$，没有其它操作符或操作数了。出栈 $-$，最终表达式结果为 $abc-d \times + e -$。

总结一下，中缀表达式转后缀表达式的基本流程如下所示：

>- 创建空栈存储运算符，创建空列表存储后缀表达式；
>- 从左到右读取中缀表达式中的每个元素：
>>- 如果元素是操作数，添加到列表末尾；
>>- 如果元素是左括号，入栈并注意搜索右括号；
>>- 如果元素是右括号，依次出栈运算符并添加到列表末尾，直至遇到左括号，弹出并删除左括号；
>>- 如果元素是运算符，则从栈中弹出所有优先级大于或等于当前运算符的运算符，并添加至列表末尾，然后把当前运算符入栈；
>- 读取完所有元素后，出栈所有剩余的运算符并加到列表末尾。输出列表元素即为后缀表达式。
```python
def infix_to_postfix(infix_expr: str) -> str:
    """Convert an infix expression to a postfix expression.

    Args:
        infix_expr (str): Infix expression.

    Returns:
        postfix_expr (str): Postfix expression.
    """
    # initialization
    token_list = list(infix_expr.upper())
    postfix_list = []
    op_stack = Stack()  # Stack for operators
    priority = {'*': 3, '/': 3, '+': 2, '-': 2, '(': 1}
    symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    numbers = '0123456789'

    # main process
    for token in token_list:
        if token in symbols or token in numbers:  # operands
            postfix_list.append(token)
        elif token == '(':  # forced priority, search backward for ')'
            op_stack.push(token)
        elif token == ')':  # search forward for '('
            top_token = op_stack.pop()
            while top_token != '(':
                postfix_list.append(top_token)
                top_token = op_stack.pop()
        else:  # operators: +-/*
            while (not op_stack.isEmpty()) and (priority[op_stack.peek()] >= priority[token]):
                postfix_list.append(op_stack.pop())
            op_stack.push(token)
    while not op_stack.isEmpty():  # finish
        postfix_list.append(op_stack.pop())
    return ''.join(postfix_list)
```
关于前缀表达式，其可读性会略逊于后缀表达式，但在某些特定的递归计算场景下具有更好的适应性。与后缀表达式的扫描顺序不同，前缀表达式是从右向左扫描中缀表达式序列的。具体流程如下所示：

>- 创建两个空栈，分别存储运算符和前缀表达式元素；
>- 从右到左读取中缀表达式中的每个元素：
>>- 如果元素是操作数，推入表达式栈；
>>- 如果元素是右括号，推入运算符栈，注意搜索左括号；
>>- 如果元素是左括号，依次出栈运算符并推入表达式栈，直至遇到右括号，弹出并删除右括号；
>>- 如果元素是运算符，则从栈中弹出所有优先级大于当前运算符的运算符，并依次推入表达式栈，然后将当前运算符入栈；
>- 读取完所有元素后，将运算符栈中剩余的元素全部出栈并依次推入表达式栈；
>- 最后弹出表达式栈的所有元素并组成字符串序列，即为前缀表达式结果。
```python
def infix_to_prefix(infix_expr: str) -> str:
    """Convert an infix expression to a prefix expression.

    Args:
        infix_expr (str): Infix expression.

    Returns:
        prefix_expr (str): Prefix expression.
    """
    # initialization
    token_list = list(infix_expr.upper())
    token_list.reverse()
    prefix_stack = Stack()
    prefix_list = []
    op_stack = Stack()  # Stack for operators
    priority = {'*': 3, '/': 3, '+': 2, '-': 2, ')': 1}
    symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    numbers = '0123456789'

    # main process
    for token in token_list:
        if token in symbols or token in numbers:  # operands
            prefix_list.insert(0, token)
        elif token == ')':  # forced priority, search for '('
            op_stack.push(token)
        elif token == '(':  # search backward for ')'
            top_token = op_stack.pop()
            while top_token != ')':
                prefix_list.insert(0, top_token)
                top_token = op_stack.pop()
        else:  # operators: +-/*
            while (not op_stack.isEmpty()) and (priority[op_stack.peek()] > priority[token]):
                prefix_list.insert(0, op_stack.pop())
            op_stack.push(token)
    while not op_stack.isEmpty():
        prefix_list.insert(0, op_stack.pop())
    return ''.join(prefix_list)
```

## 后缀表达式求值
相比于中缀和前缀表达式，计算机在处理后缀表达式的时候效率更高，所以在本章节的最后，我们讨论一下后缀表达式求值的问题。由于操作符在操作数的后边，所以操作数是暂存的，当遇到操作符时再处理最近的两个操作数。与转换成后缀表达式时的步骤不同（操作符存在栈中），此时操作数是存储在栈中的。具体代码如下（规定表达式中只有数字和操作符，没有字母变量）：
```python
def compute_postfix(
        postfix_expr: str,
        variables: Dict[str, Union[int, float]]) -> Union[int, float]:
    """Compute the result of a postfix expression.

    Args:
        postfix_expr (str)： Postfix expression.
        variables: (Dict): {'name': value}.

    Returns:
        value (int or float): Result.
    """
    # initialization
    result_stack = Stack()
    postfix_list = list(postfix_expr.upper())
    operators = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y
    }

    # main process
    for token in postfix_list:
        if token in operators:  # +-*/
            right_num = result_stack.pop()
            left_num = result_stack.pop()
            result_stack.push(operators[token](left_num, right_num))
        else:  # variables
            result_stack.push(variables[token])
    return result_stack.pop()
```
代码中有一些细节需要单独强调一下：

- 在操作数出栈时，先弹出的是右操作数，后弹出的是左操作数。这一点对于减法、除法尤为重要；
- 在完成一个子表达式的计算后，需要把计算结果压入栈顶，以支持后续计算；
- 处理完所有的操作符后，栈中应当只留下一个操作数，即最终结果。
---