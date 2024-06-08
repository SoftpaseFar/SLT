class TestA:
    def __init__(self, arg):
        print("testA()")
        print(arg)


class TestB:
    def __init__(self, arg):
        print("testB()")
        print(arg)


class TestC:
    def __init__(self, arg):
        print("testC()")
        print(arg)


if __name__ == '__main__':
    eval("TestA")(666)
