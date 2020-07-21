import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

DATADIR = 'H:/eyeWork/PALM-Training400/PALM-Training400'   # 这里有改动

file1 = 'N0012.jpg'
file2 = 'P0095.jpg'

# 读取图片
img1 = Image.open(os.path.join(DATADIR,file1))
img1 = np.array(img1)
img2 = Image.open(os.path.join(DATADIR,file2))
img2 = np.array(img2)

# 画出读取的数据
plt.figure(figsize=(16,8))
f = plt.subplot(121)
f.set_title('Normal',fontsize=20)
plt.imshow(img1)
f = plt.subplot(122)
f.set_title('PM',fontsize=20)
plt.imshow(img2)
plt.show()




# 定义数据读取器
# 使用opencv从磁盘读取图片，将每张图片放缩到224*224大小，并且将像素值调整到[-1,1] 之间
import cv2
import random
import numpy as np

# 对读入的图像数据进行处理
def transform_img(img):
    # 图片尺寸放缩到 224*224
    img = cv2.resize(img,(224,224))
    # 读入图片的格式是[H,W,c]
    # 使用转置操作将其变成[C,H,W]
    img = np.transpose(img,(2,0,1))
    img = img.astype('float32')
    # 将数据范围调整到-1，1之间
    img = img / 255
    img = img * 2.0 - 1.0
    return img

# 定义训练集数据读取器
def data_loader(datadir, batch_size=10, mode='train'):
    # 将datadir目录下的文件列出来，每条文件都要读入   run时打印看结果
    filenames = os.listdir(datadir)   # 整体的文件名
    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir,name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            if name[0] == 'H' or name[0] == 'N':
                # H开头的文件表示高度近视，N开头的文件表示为正常
                # 高度近视和正常视力的样本都不是病理性的，属于负样本，标签为0
                label = 0
            elif name[0] == 'P':
                label = 1
            else: raise('Not file name')  # 若执行此句，后面不执行
            # 每读取一个样本数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 把这些数据当作一个mini-batch,并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
                yield imgs_array,labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
            yield imgs_array, labels_array  # 生成器 到yield就返回，下一次执行从yield下面的一条语句执行

    return reader

# 定义验证集数据读取器
def valid_data_loader(datadir, csvfile, batch_size = 10, mode = 'valid'):
    # 训练集读取时通过文件名来确定样本标签，验证集通过csvfile来读取每个图片对应的标签
    # 请查看解压后的验证集标签数据，观察csvfile文件里面所包含的内容
    # csvfile文件所包含的内容格式如下，每一行代表一个样本，
    # 其中第一列是图片id，第二列是文件名，第三列是图片标签，
    # 第四列和第五列是Fovea的坐标，与分类任务无关
    # ID,imgName,Label,Fovea_X,Fovea_Y
    # 1,V0001.jpg,0,1157.74,1019.87
    # 2,V0002.jpg,1,1285.82,1080.47
    # 打开包含验证集标签的csvfile，并读入其中的内容
    filelists = open(csvfile).readlines()
    def reader():
        batch_imgs = []
        batch_labels = []
        for line in filelists[1:]:
            line = line.strip().split(',')  # 这里根据数据格式而定
            name = line[1]
            label = int(line[2])
            # 根据图片名加载图片，并对图片做预处理
            filepath = os.path.join(datadir,name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            # 每读取一个样本的数据集，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 把这些数据当作一个mini-batch,并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array
    return reader


# 查看数据形状
DATADIR = 'H:/eyeWork/PALM-Training400/PALM-Training400/'
train_loader = data_loader(DATADIR,batch_size=10,mode='train')
data_reader = train_loader()
data = next(data_reader)
print(data)
print(data[0].shape,data[1].shape)


# 识别眼疾图片
import os
import random
import paddle
import paddle.fluid as fluid

DATADIR = 'H:/eyeWork/PALM-Training400/PALM-Training400/'
DATADIR2 = 'H:/eyeWork/PALM-Validation400/'
CSVFILE = 'H:/eyeWork/valid_gt/PALM-Validation-GT/Labelss.csv'

# 定义训练过程
def train(model):
    print("Start training……")
    with fluid.dygraph.guard():
        model.train()
        epoch_num = 5
        # 定义优化器  学习率   动量因子  指定优化参数
        opt = fluid.optimizer.Momentum(learning_rate=0.001,momentum=0.9,parameter_list=model.parameters())
        # 定义数据读取器，训练数据读取器和测试数据读取器
        train_loader = data_loader(DATADIR,batch_size=10,mode='train')
        valid_loader = valid_data_loader(DATADIR2,CSVFILE)
        for epoch in range(epoch_num):
            for batch_id,data in enumerate(train_loader()):
                x_data,y_data=data
                img = fluid.dygraph.to_variable(x_data)  # 数据格式的转换
                label = fluid.dygraph.to_variable(y_data)
                # 进行模型前向计算，得到预测值
                logits = model(img)
                # 进行loss计算
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits,label)
                avg_loss = fluid.layers.mean(loss)

                if batch_id % 10 == 0:
                    print("epoch:{},batch_id:{},loss is:{}".format(epoch, batch_id, avg_loss.numpy()))
                # 反向传播，更新权重，清楚梯度
                avg_loss.backward()  # 从该节点开始执行反向
                # 为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameter_list中的Parameters，最小化网络损失值loss
                opt.minimize(avg_loss)
                model.clear_gradients()  # 清除需要优化的参数的梯度

            model.eval() # 精确率的计算结果。标量输出，float类型
            accuracies = []
            losses = []
            for batch_id,data in enumerate(valid_loader()):
                x_data,y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                # 运行模型前向计算，得到预测值
                logits = model(img)
                # 二分类，sigmoid计算后的结果为0.5为阈值分两个类别
                # 计算sigmoid后的预测概率，进行loss计算
                pred = fluid.layers.sigmoid(logits)
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits,label)
                # 计算预测概率小于0.5的类别
                pred2 = pred * (-0.1) + 1.0
                # 得到两个类别的预测概率，并沿第一个维度级联
                pred = fluid.layers.concat([pred2,pred],axis=1)
                acc = fluid.layers.accuracy(pred,fluid.layers.cast(label,dtype='int64'))
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
            print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies),np.mean(losses)))
            model.train()

        # save params of model
        fluid.save_dygraph(model.state_dict(),'mnist')
        # save optimizer state
        fluid.save_dygraph(opt.state_dict(),'mnist')

# ResNet模型代码
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable


# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一化块
class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        """

        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        act, 激活函数类型，默认act=None不使用激活函数
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # 创建第三个卷积 1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)


# 定义ResNet模型
# y = f(x) + x
# f(x) = y - x
# 残差块：瓶颈结构  bottleNeck
# 34和 50 101 152 有差别
# 34：输入64d 3*3*64  3*3*64  +
# 50以上：输入256d  1*1*64  3*3*63  1*1*256 +  还有跳转
# 特征图相加
class ResNet(fluid.dygraph.Layer):
    def __init__(self, layers=50, class_dim=1):
        """

        layers, 网络层数，可以是50, 101或者152
        class_dim，分类标签的类别数
        """
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        # 断言
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            # ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            # ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            # ResNet152包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
            depth = [3, 8, 36, 3]

        # 残差块中使用到的卷积的输出通道数
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        # ResNet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,  # c3、c4、c5将会在第一个残差块使用stride=2；其余所有残差块stride=1
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = Pool2D(pool_size=7, pool_type='avg', global_pooling=True)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        # 创建全连接层，输出大小为类别数目
        self.out = Linear(input_dim=2048, output_dim=class_dim,
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv, stdv))) # 随机均匀分布初始化器

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, [y.shape[0], -1])
        y = self.out(y)
        return y
import datetime
if __name__ == '__main__':
    starttime = datetime.datetime.now()
    with fluid.dygraph.guard():
        model = ResNet()

    train(model)
    endtime = datetime.datetime.now()
    print("运行时间：", endtime - starttime)