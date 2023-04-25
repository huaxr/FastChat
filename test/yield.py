
def generator():
    i = 0
    while True:
        value = yield i
        if value is not None:
            i = value
        else:
            i += 1

if __name__ == "__main__":
    gen = generator()

    print(next(gen))  # 输出 0
    print(next(gen))  # 输出 1
    print(gen.send(None))  # 输出 10，替换了 yield i 中的 i 的值
    print(next(gen))  # 输出 11
