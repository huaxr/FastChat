import torch

# torch.as_tensor() 是一个用于创建张量（tensor）的函数，它接受一个数组或序列，并将其转换为张量。如果输入的对象已经是张量，则该函数将返回原始张量，
# 否则它将复制输入的对象数据并返回新的张量。这个函数可以在创建张量时提高性能，因为它使用的是与输入数据相同的内存而不是复制数据。该函数也可以用于在不需要复制的情况下共享数据。
my_list = [1, 2, 3, 4, 5]
my_tensor = torch.as_tensor(my_list)
print(my_tensor)

import torch

# 创建一个 2x3 的张量，每个元素都为 1
a = torch.ones(2, 3)
print(a)