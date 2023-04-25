
import torch

# torch.softmax(input, dim=None, dtype=None) -> Tensor 是一个计算 softmax 函数的函数，它接受一个张量 input 和一个可选的 dim 参数。
#
# input 张量可以有任意形状，它的最后一个维度会被视为 softmax 函数的作用域，也就是在这个维度上，函数会对张量的所有元素进行归一化处理，使得每个元素都变成一个概率值。在归一化的过程中，函数会对张量的最后一个维度的每个元素做指数运算，并将运算结果除以这个维度上的所有元素的指数和，从而得到每个元素的概率值。
#
# dim 参数用来指定在哪个维度上进行 softmax 运算，默认值为 None，表示对最后一个维度进行运算。
#
# dtype 参数用来指定输出的张量的数据类型，默认值为 None，表示输出张量的数据类型与输入张量的数据类型相同。
# 构造输入数据
x = torch.ones(2, 3)

print(x)
# 对每个行向量进行 softmax 计算
softmax_output = torch.softmax(x, dim=1)

# 输出结果
print(softmax_output)
