class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)


def parenthesis_checker(string):
    def matches(open, close):
        opens = "([{"
        closers = ")]}"
        return opens.index(open) == closers.index(close)

    stack = Stack()
    is_balanced = True
    string = [c for c in string if c in "([{}])"]
    i = 0
    while i < len(string) and is_balanced:

        char = string[i]

        if char == '(':
            stack.push(char)

        else:
            if stack.isEmpty():
                is_balanced = False
            else:
                top = stack.pop()
                if not matches(top, char):
                    is_balanced = False

        i += 1

    if is_balanced and stack.isEmpty():
        return True
    else:
        return False


def convert_base(integer, base):
    """Divide by base algorithm"""
    stack = Stack()
    chars = "0123456789ABCDEF"

    while integer > 0:
        remainder = integer % base
        stack.push(remainder)
        integer = integer // base

    return ''.join([chars[stack.pop()] for x in range(stack.size())])


def infix_to_postfix(expr):
    precedence = {}
    precedence["*"] = 3
    precedence["/"] = 3
    precedence["+"] = 2
    precedence["-"] = 2
    precedence["("] = 1

    opstack = Stack()
    output = []
    tokens = expr.split()
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    for token in tokens:
        if token in chars:
            output.append(token)
        elif token == '(':
            opstack.push(token)
        elif token == ')':
            top = opstack.pop()
            while top != '(':
                output.append(top)
                top = opstack.pop()

        else:
            while (not opstack.isEmpty()) and prec[opstack.peek()] >= prec[token]:
                output.append(opstack.pop())

            opstack.push(token)

        while not opstack.isEmpty():
            output.append(opstack.pop())

        return ''.join(output)


def eval_postfix(expr):
    opstack = Stack()
    tokens = expr.split()
    chars = "0123456789"

    for token in tokens:
        if token in chars:
            opstack.push(token)

        else:
            x1 = opstack.pop()
            x2 = opstack.pop()
            result = eval('%s %s %s' % (x2, token, x1))
            opstack.push(result)

    return opstack.pop()
