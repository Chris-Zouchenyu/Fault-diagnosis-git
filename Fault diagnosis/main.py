from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from scipy.io import loadmat
import torch.optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torchkeras
# 函数
from CNN import CNNnet
from MLP import MLPnet
from LSTMnet import LSTM_model
from ResNet import ResNet18
import matplotlib.pyplot as plt

#故障数据标签
FAULT_LABEL_DICT = {'97': 0,
                    '105': 1,
                    '118': 2,
                    '130': 3,
                    '169': 4,
                    '185': 5,
                    '197': 6,
                    '209': 7,
                    '222': 8,
                    '234': 9}
#选取驱动端的数据进行建模
AXIS = '_DE_time'
#随机数种子
seed = 102
np.random.seed(seed)
class CWRUDataset():
    """
    继承paddle.io.Dataset类
    """
    def __init__(self, data_dir, time_steps=1024, window=128, mode='train', val_rate=0.3, test_rate=0.5, \
                 noise=False, snr=None, network='MLP'):
        """
        实现构造函数,定义数据读取方式,划分训练和测试数据集
        time_steps: 样本的长度
        window：相邻样本之间重合的点数
        mode：数据集合
        val_rate:
        test_rate:
        noise：是否添加噪声
        snr：添加噪声的分贝数
        network：网络类型(决定生成的数据格式)
        
        """
        super(CWRUDataset, self).__init__()
        self.time_steps = time_steps
        self.mode = mode
        self.noise = noise
        self.snr = snr
        self.network = network
        self.feature_all, self.label_all = self.transform(data_dir)
        self.window = window
        #训练集和验证集的划分
        train_feature, val_feature, train_label, val_label = \
        train_test_split(self.feature_all, self.label_all, test_size=val_rate, random_state=seed)
        #标准化
        train_feature, val_feature = self.standardization(train_feature, val_feature)
        #验证集和测试集的划分
        val_feature, test_feature, val_label, test_label = \
        train_test_split(val_feature, val_label, test_size=test_rate, random_state=seed)
        if self.mode == 'train':
            self.feature = train_feature
            self.label = train_label
        elif self.mode == 'val':
            self.feature = val_feature
            self.label = val_label
        elif self.mode == 'test':
            self.feature = test_feature
            self.label = test_label
        else:
            raise Exception("mode can only be one of ['train', 'val', 'test']")
        
    def transform(self, data_dir) :
        """
        转换函数,获取数据
        """
        feature, label = [], []
        for fault_type in FAULT_LABEL_DICT:
            lab = FAULT_LABEL_DICT[fault_type]
            totalaxis = 'X' + fault_type + AXIS
            if fault_type == '97':
                totalaxis = 'X0' + fault_type + AXIS
            #加载并解析mat文件
            mat_data = loadmat(data_dir + '\\' + fault_type + '.mat')[totalaxis]
            #start, end = 0, self.time_steps
            #每隔self.time_steps窗口构建一个样本，指定样本之间重叠的数目
            for i in range(0, len(mat_data) - self.time_steps, window):# 这里time_steps: 1024  window: 128
                sub_mat_data = mat_data[i: (i+self.time_steps)].reshape(-1,)
                #是否往数据中添加噪声
                if self.noise:
                    sub_mat_data = self.awgn(sub_mat_data, self.snr)
                feature.append(sub_mat_data)
                label.append(lab)
        return np.array(feature, dtype='float32'), np.array(label, dtype="int64")
    
    def __getitem__(self, index):
        """
        实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据
        """
        feature = self.feature[index]
        if self.network == 'CNNNet':
            #增加一列满足cnn的输入格式要求
            feature = feature[np.newaxis,:]
        elif self.network == 'ResNet':
            #增加一列并将通道复制三份满足resnet的输入要求
            n = int(np.sqrt(len(feature)))
            feature = np.reshape(feature, (n, n))
            feature = feature[np.newaxis,:]
            feature = np.concatenate((feature, feature, feature), axis=0)
        label = self.label[index]
        feature = feature.astype('float32')
        label = np.array([label], dtype="int64")
        return feature, label

    def __len__(self):
        """
        实现__len__方法，返回数据集总数目
        """
        return len(self.feature)

    def awgn(self, data, snr, seed=seed):
        """
        添加高斯白噪声
        """
        np.random.seed(seed)
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(data ** 2) / len(data)
        npower = xpower / snr
        noise = np.random.randn(len(data)) * np.sqrt(npower)
        return np.array(data + noise)
    
    def standardization(self, train_data, val_data):
        """
        标准化
        """
        scalar = preprocessing.StandardScaler().fit(train_data)
        train_data = scalar.transform(train_data)
        val_data = scalar.transform(val_data)
        return train_data, val_data

def train_model(lr,batch_size,epoch,num_classes,network,x,y,path):
    '''
    用于代码训练
    '''
    #torchkeras训练方式
    x = torch.tensor(x)
    y = torch.tensor(y)
    if network == 'MLPNet':
        mymodel = torchkeras.Model(MLPnet())
    elif network == 'CNNNet':
        mymodel = torchkeras.Model(CNNnet(num_classes))
    elif network == 'LSTMNet':
        mymodel = torchkeras.Model(LSTM_model(1,num_classes))
        x = torch.reshape(x,(x.shape[0],x.shape[1],1))
        y = torch.reshape(y,(y.shape[0],1))
    elif network == 'ResNet':
        mymodel = torchkeras.Model(ResNet18(num_classes))
    ds_train = TensorDataset(x,y)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0)
    #优化器
    optimizer = torch.optim.SGD(mymodel.parameters(),lr=lr)
    #损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    mymodel.compile(loss_func=loss_fn, optimizer=optimizer)
    history = mymodel.fit(epochs=epoch,dl_train=dl_train)
    torch.save(mymodel,path)
    return history

def evaluate(path,x,y,network):
    '''
    模型评价用
    '''
    model = torch.load(path,weights_only=False)
    # model.eval()
    all_preds = []
    all_labels = []
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    if network == 'CNNNet':
        x = x
    elif network == 'LSTMNet':
        x = torch.reshape(x,(x.shape[0],x.shape[1],1))
        y = torch.reshape(y,(y.shape[0],1))
    with torch.no_grad():
        outputs = model(x)
        _, preds = torch.max(outputs, 1)  # 获取预测类别
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Recall: {recall:.4f}')
    return accuracy

def plotgraph(history):
    '''
    该函数绘制loss epoch图
    '''
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12, 8))
    # 绘制每条曲线
    plt.plot(range(50), np.array(history['CNNNet']), label='CNNNet', linestyle='-', linewidth=2)
    plt.plot(range(50), np.array(history['ResNet']), label='ResNet', linestyle='--', linewidth=2)
    plt.plot(range(50), np.array(history['MLPNet']), label='MLPNet', linestyle='-.', linewidth=2)
    # 设置标题和标签
    plt.title('Loss - Epoch', size=30)
    plt.xlabel('Epoch', size=14)
    plt.ylabel('Loss', size=14)
    # 设置图例
    plt.legend(fontsize=12, loc='upper right')  # 调整图例字体大小和位置
    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.6)
    # 显示图像
    plt.show()

if __name__ == '__main__':
    #时序步长
    time_steps = 1024
    #相邻样本之间重叠的数目
    window = 128
    #是否添加噪声
    noise = True
    #添加的噪声分贝数
    snr = -10
    #验证集比例（从全体数据中取多少比例作为验证集）
    val_rate = 0.3
    #测试集比例(从验证集中取多少比例作为测试集)
    test_rate = 0.5
    #网络类型
    networks = ['CNNNet','MLPNet','ResNet']
    #超参数
    lr = 1e-3
    epoch = 50
    batch_size = 16
    num_classes = 10
    result = {}
    history = {}
    # 对每个模型进行训练
    for network in networks:
        train_data = CWRUDataset('D:\python\Deep learning\Fault diagnosis\data',time_steps=1024,window=128,mode='train',
                                val_rate=0.3,test_rate=0.5,noise=noise,snr=snr,network=network)
        val_dataset = CWRUDataset('D:\python\Deep learning\Fault diagnosis\data', time_steps=time_steps, window=window, mode='val',
                                    val_rate=val_rate, test_rate=test_rate, noise=noise, snr=snr, network=network)
        test_dataset = CWRUDataset('D:\python\Deep learning\Fault diagnosis\data', time_steps=time_steps, window=window, mode='test',
                                    val_rate=val_rate, test_rate=test_rate, noise=noise, snr=snr, network=network)

        # print (train_data.__len__())# 7285
        # print (val_dataset.__len__())# 1561
        # print (test_dataset.__len__())# 1562

        # print (train_data.feature.shape)# (7285, 1024)
        # print (train_data.label.shape)# (7285,)
        # print (val_dataset.feature.shape)# (1561, 1024)
        # print (val_dataset.label.shape)# (1561,)
        # print (test_dataset.feature.shape)# (1562, 1024)
        # print (test_dataset.label.shape)# (1562,)
        
        #训练部分
        path = r'Deep learning\Fault diagnosis\n50_lr0.001_' + network# 保存每一个模型的权重
        hist= train_model(lr=lr,
                        batch_size=batch_size,
                        num_classes=num_classes,
                        network=network,
                        x=train_data.feature,
                        y=train_data.label,
                        epoch=epoch,
                        path=path)
        #验证部分
        result[network] = evaluate(path,val_dataset.feature,val_dataset.label,network)#保存每一个模型的accuracy
        history[network] = hist #保存每一个模型的loss图
    torch.save(result,r'D:\python\Deep learning\Fault diagnosis\result.pth')# 保存结果
    torch.save(history,r'D:\python\Deep learning\Fault diagnosis\history.pth')
    plotgraph(history)#画图
    print('CNN: {},MLP: {},Res: {}'.format(result['CNNNet'],result['MLPNet'],result['ResNet']))#展示各模型的acc
