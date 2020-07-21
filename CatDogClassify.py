import paddle as paddle
import paddle.fluid as fluid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

'''
BATCH_SIZE = 128  # 批尺寸
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.cifar.train10(),buf_size=128*100),
    batch_size=BATCH_SIZE
    )
test_reader = paddle.batch(paddle.dataset.cifar.test10(),batch_size=BATCH_SIZE)


# 模型配置
def convolutional_neural_network(img):
    # 第一个卷积层池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(  # 没有卷积步长？
        input=img, # 输入
        filter_size=5, # 滤波器大小
        num_filters=20, # 滤波器数量
        pool_size=2, # 池化核大小   默认最大池化
        pool_stride=2, # 池化步长
        act="relu" # 激活函数
    )
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)  #可用作卷积和全连接操作的批正则化函数，根据当前批次数据按通道计算的均值和方差进行正则化
    # 2
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu"
    )
    conv_pool_2 = fluid.layers.batch_norm(conv_pool_2)
    # 3
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_2,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu"
    )
    prediction = fluid.layers.fc(input=conv_pool_3,size=10,act='softmax')
    return prediction

# 定义数据
data_shape = [3,32,32]
images = fluid.layers.data(name="images",shape=data_shape,dtype='float32')
label = fluid.layers.data(name='label',shape=[1],dtype='int64')

# 获取分类器，用cnn分类
predict = convolutional_neural_network(images)

# 定义损失函数和准确率
cost = fluid.layers.cross_entropy(input=predict,label=label) # 交叉熵，分类任务常用
avg_cost = fluid.layers.mean(cost) # 计算cost中所有元素平均值
acc = fluid.layers.accuracy(input=predict,label=label) # 使用输入和标签计算准确率

# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.Adam(learning_rate=0.001)  # Adam优化方法，学习率0.001
optimizer.minimize(avg_cost)
print("ok")
# 到此模型配置完毕

# 定义使用CPU还是GPU进行训练，use_cuda = True使用GPU
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# 创建执行器，初始化参数
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 定义数据映射器
feeder = fluid.DataFeeder([images,label],place=place)  # 负责将reader返回的数据转换成一种特殊的结构，是他们呢可以输入到Executor

# 定义绘制训练过程的损失值和准确率变化趋势的方法
all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]

def draw_train_process(title,iters,costs,accs,label_cost,label_acc):
    plt.title(title,fontsize=24)
    plt.xlabel("iter",fontsize=20)
    plt.ylabel("cost/acc",fontsize=20)
    plt.plot(iters,costs,color="red",label=label_cost)
    plt.plot(iters,accs,color="green",label=label_acc)
    plt.legend()
    plt.grid()
    plt.show()

# 训练并保存模型,每个pass结束进行验证
EPOCH_NUM = 20
model_save_dir = "/home/aistudio/work/catdog.inference.model"

for pass_id in range(EPOCH_NUM):
    # 开始训练
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader的迭代器，并为其数据加上索引batch_id
        train_cost,train_acc = exe.run(program=fluid.default_main_program(), # 运行主程序
                                       feed=feeder.feed(data), # 喂入一个batch的数据
                                       fetch_list=[avg_cost,acc]) # fetch均方差和准确率
        all_train_iter = all_train_iter + BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

        # 每100次batch打印一次训练，并进行测试
        if batch_id % 100 == 0:
            print('Pass:%d,Batch:%d,Cost:%0.5f,Accracy:%0.5f'%(pass_id,batch_id,train_cost[0],train_acc[0]))

    # 开始测试
    test_costs = [] # 测试的损失值
    test_accs = [] # 测试的准确率
    for batch_id,data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program, # 测试运行程序
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost,acc] # fetch误差准确率
                                      )
        test_costs.append(test_cost[0])
        test_accs.append(test_acc[0])

    # 求误差准确率的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f, ACC:%0.5f'%(pass_id,test_cost,test_acc))

# 模型保存
# 如果路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save model to %s'%(model_save_dir))
fluid.io.save_inference_model(model_save_dir,['images'],[predict],exe)
print("模型训练完毕！")
draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"training cost","trainning acc")
'''

# 模型预测  此后可本地运行

model_save_dir = "D:/PROGRAMS/python/Paddle/trainModel/catDogModel"
# 定义使用CPU还是GPU进行训练，use_cuda = True使用GPU
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# 创建预测用的Executor
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()

# 图片预处理
def load_image(file):
    # 打开图片
    im = Image.open(file)
    # 将图片尺寸调整为跟训练数据一样大小  32*32
    im = im.resize((32,32),Image.ANTIALIAS)  # resize 放缩  抗锯齿

    plt.imshow(im)
    plt.show()
    # 建立图片矩阵 类型为float32
    im = np.array(im).astype(np.float32)
    # 矩阵转置
    im = im.transpose((2,0,1))
    # 归一化
    im = im/255.0
    im = np.expand_dims(im,axis=0)
    print('im_shape的维度',im.shape)
    return im

# 开始预测
with fluid.scope_guard(inference_scope):
    # 从指定目录中加载 推理model
    [inference_program, # 预测用的program
     feed_target_names, # 是一个str列表，它包含需要在PRogram 中提供数据的变量的名称
     fetch_targets] = fluid.io.load_inference_model(model_save_dir, # fetch_targets是一个Variable列表，从中我们可以推断结果
                                                    infer_exe)
    infer_path="D:/study/picture/PaddleDog2.jpg"
    img = Image.open(infer_path)
    plt.imshow(img)
    plt.show()

    img = load_image(infer_path)
    result = infer_exe.run(inference_program, # 运行预测程序
                           feed={feed_target_names[0]:img}, # 喂入预测img
                           fetch_list=fetch_targets) # 得到推测结果
    # print('result',result)
    label_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
        "ship", "truck"]
    print("infer results:%s" % label_list[np.argmax(result[0])])  # np.argmax 求最大值索引




