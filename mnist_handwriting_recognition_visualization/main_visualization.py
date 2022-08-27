import torch
from torch.utils.data import DataLoader  # 我们要加载数据集的
import torchvision
from torchvision import datasets  # pytorch十分贴心的为我们直接准备了这个数据集
import torch.nn.functional as F  # 激活函数
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from math import sqrt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  	# 添加这条可以让图形显示中文


def plt_need_detach(x, position, title):
    """画图 这个提示需要detach，否则报错
    这个是把1个通道的各层画到1个图上
    """
    if is_plt:
        if is_only_plt_test:
            if is_entey_test:
                img = x.cpu().detach().numpy()
                print(f"len(img):{len(img[0])}")
                img_data = img[0][0]
                plt.subplot(position)
                plt.title(title)
                plt.imshow(img_data, cmap='gray')
                # # plt.imshow(img_data, cmap='gray_r')
                if title == "relu2 output":  # 最后一个也plt完的时候就show出来
                    plt.show()  # 显示
        else:
            img = x.cpu().detach().numpy()
            print(f"len(img):{len(img[0])}")
            img_data = img[0][0]
            plt.subplot(position)
            plt.title(title)
            plt.imshow(img_data, cmap='gray')
            # # plt.imshow(img_data, cmap='gray_r')
            if title == "relu2 output":  # 最后一个也plt完的时候就show出来
                plt.show()  # 显示


def plt_need_detach2(x, position, title):
    """画图 这个提示需要detach，否则报错
    这里是分开画，每一层单独一个图
    """
    if is_plt:
        if is_only_plt_test:
            if is_entey_test:
                img = x.cpu().detach().numpy()
                # print(f"len(img):{len(img[0])}")
                plt_number = len(img[0])
                for i in range(plt_number):
                    img_data = img[0][i]
                    if sqrt(plt_number) - int(sqrt(plt_number)) < 0.01:
                        plt.subplot(int(sqrt(plt_number)), int(sqrt(plt_number)), i + 1)
                        # 只在最后一张图片显示size，不然看起来太乱
                        if (i+1) == plt_number:
                            plt.xlabel(f"{x.size()}")
                    else:
                        plt.subplot(int(sqrt(plt_number)) + 1, int(sqrt(plt_number)) + 1, i + 1)
                        # 只在最后一张图片显示size，不然看起来太乱
                        if (i + 1) == plt_number:
                            plt.xlabel(f"{x.size()}")
                    plt.title(f"{title:}{i+1}")
                    # plt.ylabel("纵坐标")
                    plt.imshow(img_data, cmap='gray')
                    # # plt.imshow(img_data, cmap='gray_r')
                plt.show()  # 显示
        else:
            img = x.cpu().detach().numpy()
            # print(f"len(img):{len(img[0])}")
            plt_number = len(img[0])
            for i in range(plt_number):
                img_data = img[0][i]
                if sqrt(plt_number) - int(sqrt(plt_number)) < 0.01:
                    plt.subplot(int(sqrt(plt_number)), int(sqrt(plt_number)), i + 1)
                else:
                    plt.subplot(int(sqrt(plt_number)) + 1, int(sqrt(plt_number)) + 1, i)
                plt.title(f"{title:}{i + 1}")
                plt.imshow(img_data, cmap='gray')
                # # plt.imshow(img_data, cmap='gray_r')
            plt.show()  # 显示

def plt_conv_kernel(model):
    """画图 这个提示需要detach，否则报错
    这里是分开画，每一层单独一个图
    """
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.detach().cpu().numpy()
    print(params['conv1.weight'])

    plt_number = len(params['conv1.weight'])
    for i in range(plt_number):
        img_data = params['conv1.weight'][i][0]
        if sqrt(plt_number) - int(sqrt(plt_number)) < 0.01:
            plt.subplot(int(sqrt(plt_number)), int(sqrt(plt_number)), i + 1)
            # 只在最后一张图片显示size，不然看起来太乱
            if (i+1) == plt_number:
                plt.xlabel(f"conv1.weight")
        else:
            plt.subplot(int(sqrt(plt_number)) + 1, int(sqrt(plt_number)) + 1, i + 1)
            # 只在最后一张图片显示size，不然看起来太乱
            if (i+1) == plt_number:
                plt.xlabel(f"conv1.weight")
        plt.title(f"conv_kernel{i}")
        # plt.ylabel("纵坐标")
        plt.imshow(img_data, cmap='gray')
        # # plt.imshow(img_data, cmap='gray_r')
    plt.show()  # 显示

    x = np.linspace(1, 10, 10)
    y = params['conv1.bias']
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('bias取值')
    plt.title(f"conv1.bias")
    plt.show()

    plt_number = len(params['conv2.weight'])
    for i in range(plt_number):
        img_data = params['conv2.weight'][i][0]
        if sqrt(plt_number) - int(sqrt(plt_number)) < 0.01:
            plt.subplot(int(sqrt(plt_number)), int(sqrt(plt_number)), i + 1)
            # 只在最后一张图片显示size，不然看起来太乱
            if (i + 1) == plt_number:
                plt.xlabel(f"conv2.weight")
        else:
            plt.subplot(int(sqrt(plt_number)) + 1, int(sqrt(plt_number)) + 1, i + 1)
            # 只在最后一张图片显示size，不然看起来太乱
            if (i + 1) == plt_number:
                plt.xlabel(f"conv2.weight")
        plt.title(f"conv_kernel{i}")
        # plt.ylabel("纵坐标")
        plt.imshow(img_data, cmap='gray')
        # # plt.imshow(img_data, cmap='gray_r')
    plt.show()  # 显示


    x = np.linspace(1, 20, 20)
    y = params['conv2.bias']
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('bias取值')
    plt.title(f"conv2.bias")
    plt.show()

    x = np.linspace(1, 3200, 3200)
    y = params['fc.weight']
    y = y.flatten()
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('fc.weight')
    plt.title(f"fc.weight")
    plt.show()

    x = np.linspace(1, 10, 10)
    y = params['fc.bias']
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('fc.bias')
    plt.title(f"fc.bias")
    plt.show()



# 接下来我们看一下模型是怎么做的
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义了我们第一个要用到的卷积层，因为图片输入通道为1，第一个参数就是1
        # 输出的通道为10，kernel_size是卷积核的大小，这里定义的是5x5的
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
        # 再定义一个池化层
        self.pooling = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 最后是我们做分类用的线性层
        self.fc = torch.nn.Linear(in_features=320, out_features=10)

    # 下面就是计算的过程
    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        # print(f"x.shape:{x.shape}")
        # print(f"x:{x}")
        batch_size = x.size(0)  # 这里面的0是x大小第1个参数，自动获取batch大小

        plt_need_detach2(x=x, position=331, title="original input")
        # print(x.size())

        # 输入x经过一个卷积层，之后经历一个池化层，最后用relu做激活
        x = self.conv1(x)  # 输入： batch_size*1*28*28 输出：batch_size*10*24*24 其中：24 = [(28-5)/1] + 1
        plt_need_detach2(x=x, position=332, title="conv1 output")
        # print(x.size())

        x = self.pooling(x)  # 输入： batch_size*10*24*24 输出：batch_size*10*12*12 其中:12 = [(24-2)/s=2] + 1
        plt_need_detach2(x=x, position=333, title="Maxpool1 output")
        # print(x.size())

        x = F.relu(input=x)  # 输入：batch_size*10*12*12  输出：batch_size*10*12*12
        plt_need_detach2(x=x, position=334, title="relu1 output")
        # print(x.size())

        x = self.conv2(x)  # 输入：batch_size*10*12*12 输出：batch_size*20*8*8 其中：8= [(12-5+0)/s=1] + 1 = 8
        plt_need_detach2(x=x, position=335, title="conv2 output")
        # print(x.size())

        x = self.pooling(x)  # 输入：batch_size*20*8*8 输出：batch_size*20*4*4 其中：4 = [(8-2+0)/s=2] + 1
        plt_need_detach2(x=x, position=336, title="Maxpool2 output")
        # print(x.size())

        x = F.relu(input=x)  # 不改变维度
        plt_need_detach2(x=x, position=337, title="relu2 output")
        # print(x.size())

        # 为了给我们最后一个全连接的线性层用
        # 我们要把一个二维的图片（实际上这里已经是处理过的）20x4x4张量变成一维的
        x = x.view(batch_size, -1)  # flatten  # 输入：batch_size*20*4*4  输出：batch_size*320
        # print(x.size())
        # 经过线性层，确定他是0~9每一个数的概率
        x = self.fc(x)  # # 输入：batch_size*320  输出：batch*10
        # print(x.size())

        return x


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):  # 每次取一个样本
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        # 优化器清零
        optimizer.zero_grad()
        # 正向计算一下
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, target)
        # 反向求梯度
        loss.backward()
        # 更新权重
        optimizer.step()
        # 把损失加起来
        running_loss += loss.item()
        # 每300次输出一下数据
        if batch_idx % 300 == 0:
            print('[%d, %5d] loss: %.6f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0


def test():
    global is_entey_test
    is_entey_test = True
    correct = 0
    total = 0
    with torch.no_grad():  # 不用算梯度
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            # 我们取概率最大的那个数作为输出
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            # 计算正确率
            correct += (predicted == target).sum().item()
            print(f"目标target为：{target}，实际预测predicted为：{predicted}")
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    """定义超参数"""
    hyperparameter = {
        "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        "batch_size": 32,
        "epochs": 1,
        "transform": torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()  # 将图片转化为tensor
            # torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # 标准化  参数分别是均值和标准差 这些系数都是数据集提供方计算好的数据,
        ]),
    }
    print(f"hyperparameter:{hyperparameter}")
    # 我们拿到的图片是pillow,我们要把他转换成模型里能训练的tensor也就是张量的格式

    # 在函数中输入download=True，他在运行到这里的时候发现你给的路径没有，就自动下载
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=hyperparameter["transform"])
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=hyperparameter["batch_size"])
    print(f"train_loader shape:{train_loader}")
    # 同样的方式加载一下测试集
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=hyperparameter["transform"])
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=hyperparameter["batch_size"])
    print(f"test_loader shape:{test_loader}")

    model = Net()  # 实例化模型

    # 把计算迁移到GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # device = torch.device("cpu")

    # 定义一个损失函数，来计算我们模型输出的值和标准值的差距
    criterion = torch.nn.CrossEntropyLoss()
    # 定义一个优化器，训练模型咋训练的，就靠这个，他会反向的更改相应层的权重
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)  # lr为学习率

    """控制画图的开关"""
    global is_plt, is_entey_test, is_only_plt_test
    is_entey_test = False
    is_only_plt_test = True
    is_plt = True
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    for epoch in range(hyperparameter["epochs"]):
        train(epoch)

        # if epoch % 10 == 9:
        #     test()

        """可视化卷积核"""
        plt_conv_kernel(model=model)

        if epoch + 1:
            test()

    """打印参数情况"""
    # 打印某一层的参数名
    for name in model.state_dict():
        print(name)
    # Then I konw that the name of target layer is '1.weight'

    # schemem1(recommended)
    print(model.state_dict()['conv1.weight'])


    # scheme3
    params = {}  # change the tpye of 'generator' into dict
    for name, param in model.named_parameters():
        params[name] = param.detach().cpu().numpy()
    print(params['conv1.weight'])


    # 打印每一层的参数名和参数值
    # schemem1(recommended)
    for name, param in model.named_parameters():
        print(name, param)

        # scheme2
        for name in model.state_dict():
            print(name)
            print(model.state_dict()[name])


    """保存参数"""
    torch.save(model, '.\\save_model\\model.pt')
    # 保存网络中的参数, 速度快，占空间少
    torch.save(model.state_dict(), '.\\save_model\\parameter.pkl')  # 只保存权重
