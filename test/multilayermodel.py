import torch.nn as nn

# 在 PyTorch 中，我们可以通过 nn.Sequential() 组合多个层，构建一个模型。
# nn.Sequential() 接受一个层的有序序列，按照序列顺序依次将输入传递到每一层，
# 最终返回最后一层的输出结果。下面是一个简单的例子，演示了如何使用 nn.Sequential() 组合两个全连接层构建一个模型：


# 在这个例子中，我们首先定义了一个类 MyModel，继承自 nn.Module，这意味着我们的模型具有 PyTorch 模型的基本功能，
# 例如参数管理和序列化。在 init() 方法中，我们创建了一个 nn.Sequential() 对象，其中包含两个 nn.Linear() 层、
# 一个 nn.ReLU() 激活层和一个 nn.Sigmoid() 激活层。最后，我们将 nn.Sequential() 对象赋值给 self.model。
# 在 forward() 方法中，我们将输入数据 x 传递给 self.model，并将输出结果返回。通过这种方式，
# 我们可以使用 nn.Sequential() 快速组合多个层构建一个模型。
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

