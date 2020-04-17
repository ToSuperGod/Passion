import numpy as np
import paddle as paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt

# 定义一个多层感知器
def multilayer_perceptron(input):
    # 定义一个全连接层，激活函数为ReLu
    hidden1 = fluid.layers.fc(input=input,size=100,act='relu')
    # 第二个全连接层，激活函数relu
    hidden2 = fluid.layers.fc(input=hidden1,size=100,act='relu')
    # 以softmax为激活层的全连接输出层，大小为label大小
    fc = fluid.layers.fc(input=hidden2,size=100,act='softmax')
    return fc

# 卷积神经网络
def convolutional_neural_network(input):
    # 第一个卷积层，卷积核大小为3*3，一共有32个卷积核
    conv1 = fluid.layers.conv2d(input=input,
                                num_filters=32,
                                filter_size=3,
                                stride=1)
    # 第一个池化层，池化大小为2*2，步长为1，最大池化
    pool1 = fluid.layers.pool2d(input=conv1,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')
    # 第二个卷积核，卷积核大小为3*3，一共有64个卷积核
    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=64,
                                filter_size=3,
                                stride=1)
    # 第二个池化层，池化大小为2*2，步长为1，最大池化
    pool2 = fluid.layers.pool2d(input=conv2,
                                pool_size=2,
                                pool_stride=1,
                                pool_type='max')
    # 以softmax为激活函数的全链接输出层，大小为label大小
    fc = fluid.layers.fc(input=pool2,size=10,act='softmax')
    return fc
# 定义输入层
image = fluid.layers.data(name='image',shape=[1,28,28],dtype='float32')
label = fluid.layers.data(name='label',shape=[1],dtype='int64')

# 获取分类器
model = convolutional_neural_network(image)

# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=model,label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model,label=label)

# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamaxOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

# 获取MNIST数据
train_reader = paddle.batch(mnist.train(),batch_size=128)
test_reader = paddle.batch(mnist.test(),batch_size=128)

# 定义一个CPU执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据的维度
feeder = fluid.DataFeeder(place=place,feed_list=[image,label])

# 开始训练和测试
for pass_id in range(5):
    # 进行训练
    for batch_id,data in enumerate(train_reader()):
        train_cost,train_acc = exe.run(program=fluid.default_main_program(),
                                       feed=feeder.feed(data),
                                       fetch_list=[avg_cost,acc])
        # 每100个batch 打印一次信息
        if batch_id%100 == 0:
            print('Pass:%d,Batch:%d,Cost:%0.5f,Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))
    # 进行测试
    test_accs = []
    test_costs = []
    for batch_id,data in enumerate(test_reader()):
        test_cost,test_acc = exe.run(program=test_program,
                                     feed=feeder.feed(data),
                                     fetch_list=[avg_cost,acc])
        test_accs.append(test_acc[0])
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, Accuray:%0.5f'%(pass_id,test_cost,test_acc))

# 预测图形 对图形进行预处理
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28,28),Image.ANTIALIAS)
    im = np.array(im).reshape(1,1,28,28).astype(np.float32)
    im = im/ 255.0*2.0 - 1.0
    return  im

# 下载图片问题
# !wget https://github.com/yeyupiaoling/LearnPaddle2/blob/master/note4/infer_3.png?raw=true -O 'infer_3.png'
img = Image.open('infer_3.png')
plt.imshow(img)
plt.show()

# 加载数据并开始预测
img = load_image('./')
result = exe.run(program=test_program,
                 feed={'image':img,"label":np.array([[1]]).astype('int64')},
                 fetch_list=[model])
lab = np.argsort(result)










