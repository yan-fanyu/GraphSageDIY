import sys
import os

from collections import defaultdict
import numpy as np


class DataCenter(object):
    """数据中心类，用于加载和处理数据集"""

    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def load_dataSet(self, dataSet='cora'):
        """加载数据集的方法

        参数:
            dataSet (str): 要加载的数据集名称，默认为'cora'
        """
        if dataSet == 'cora':
            # 获取文件路径
            cora_content_file = self.config['file_path.cora_content']
            cora_cite_file = self.config['file_path.cora_cite']

            feat_data = []  # 存储特征数据的列表
            labels = []  # 存储节点标签序列的列表
            node_map = {}  # 将节点映射到节点ID的字典
            label_map = {}  # 将标签映射到标签ID的字典

            # 读取特征数据文件
            with open(cora_content_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    feat_data.append([float(x) for x in info[1:-1]])  # 提取特征数据并转换为浮点数列表
                    node_map[info[0]] = i  # 将节点映射到节点ID

                    if info[-1] not in label_map:
                        #意思就是，label_map[info[-1]]不是一个新的标签的id吗，现在 len(label_map)这么长，但序号是从0数，所以序号到 len(label_map)-1，那新来的就是 len(label_map)
                        label_map[info[-1]] = len(label_map)  # 将标签映射到标签ID
                    labels.append(label_map[info[-1]])  # 将节点标签ID添加到labels列表中
            feat_data = np.asarray(feat_data)  # 将特征数据转换为NumPy数组
            labels = np.asarray(labels, dtype=np.int64)  # 将标签列表转换为NumPy数组

            adj_lists = defaultdict(set)  # 存储邻接列表的字典
            print("adj_lists",type(adj_lists))

            # 读取引用关系文件
            with open(cora_cite_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    assert len(info) == 2
                    paper1 = node_map[info[0]]  # 获取第一个节点的ID,是一个一个的int
                    paper2 = node_map[info[1]]  # 获取第二个节点的ID

                    adj_lists[paper1].add(paper2)  # 将paper2添加到paper1的邻接列表中
                    adj_lists[paper2].add(paper1)  # 将paper1添加到paper2的邻接列表中

            assert len(feat_data) == len(labels) == len(adj_lists)  # 断言特征数据、标签和邻接列表的长度相同

            # 将数据集分割为测试集、验证集和训练集，包含了节点信息
            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
            # setattr函数，它用于设置对象的属性值，这行代码的作用是将test_indexs这个变量的值设置为self对象的一个名为dataSet+'_test'的属性的值
            setattr(self, dataSet + '_test', test_indexs)  # 设置测试集属性
            setattr(self, dataSet + '_val', val_indexs)  # 设置验证集属性
            setattr(self, dataSet + '_train', train_indexs)  # 设置训练集属性

            setattr(self, dataSet + '_feats', feat_data)  # 设置特征数据属性
            setattr(self, dataSet + '_labels', labels)  # 设置标签属性
            setattr(self, dataSet + '_adj_lists', adj_lists)  # 设置邻接列表属性

        elif dataSet == 'pubmed':
            # 获取文件路径
            pubmed_content_file = self.config['file_path.pubmed_paper']
            pubmed_cite_file = self.config['file_path.pubmed_cites']

            feat_data = []  # 存储特征数据的列表
            labels = []  # 存储节点标签序列的列表
            node_map = {}  # 将节点映射到节点ID的字典
            with open(pubmed_content_file) as fp:
                fp.readline()  # 跳过文件的第一行
                feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
                for i, line in enumerate(fp):
                    info = line.split("\t")
                    node_map[info[0]] = i  # 将节点映射到节点ID
                    labels.append(int(info[1].split("=")[1]) - 1)  # 提取标签并转换为整数，并减去1
                    tmp_list = np.zeros(len(feat_map) - 2)  # 创建全零数组
                    for word_info in info[2:-1]:
                        word_info = word_info.split("=")
                        tmp_list[feat_map[word_info[0]]] = float(word_info[1])  # 将单词特征添加到数组中
                    feat_data.append(tmp_list)  # 将节点特征数据添加到feat_data列表中

            feat_data = np.asarray(feat_data)  # 将特征数据转换为NumPy数组
            labels = np.asarray(labels, dtype=np.int64)  # 将标签列表转换为NumPy数组

            adj_lists = defaultdict(set)  # 存储邻接列表的字典
            with open(pubmed_cite_file) as fp:
                fp.readline()  # 跳过文件的第一行
                fp.readline()  # 跳过文件的第二行
                for line in fp:
                    info = line.strip().split("\t")
                    paper1 = node_map[info[1].split(":")[1]]  # 获取第一个节点的ID
                    paper2 = node_map[info[-1].split(":")[1]]  # 获取第二个节点的ID
                    adj_lists[paper1].add(paper2)  # 将paper2添加到paper1的邻接列表中
                    adj_lists[paper2].add(paper1)  # 将paper1添加到paper2的邻接列表中

            assert len(feat_data) == len(labels) == len(adj_lists)  # 断言特征数据、标签和邻接列表的长度相同

            # 将数据集分割为测试集、验证集和训练集
            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

            setattr(self, dataSet + '_test', test_indexs)  # 设置测试集属性
            setattr(self, dataSet + '_val', val_indexs)  # 设置验证集属性
            setattr(self, dataSet + '_train', train_indexs)  # 设置训练集属性

            setattr(self, dataSet + '_feats', feat_data)  # 设置特征数据属性
            setattr(self, dataSet + '_labels', labels)  # 设置标签属性
            setattr(self, dataSet + '_adj_lists', adj_lists)  # 设置邻接列表属性

    def _split_data(self, num_nodes, test_split=0.2, val_split=0.4):
        """将数据集分割为测试集、验证集和训练集的方法
        2:4:4

        参数:
            num_nodes (int): 数据集中节点的数量
            test_split (int): 测试集的分割比例，默认为  1/5
            val_split (int): 验证集的分割比例，默认为   1/5

        返回:
            test_indexs (list): 测试集的索引列表
            val_indexs (list): 验证集的索引列表
            train_indexs (list): 训练集的索引列表
        """



        rand_indices = np.random.permutation(num_nodes)  # 对节点进行随机排列

        test_size = int(num_nodes * test_split)  # 计算测试集大小
        val_size = int(num_nodes * val_split)  # 计算验证集大小
        train_size = num_nodes - (test_size + val_size)  # 计算训练集大小


        print("node num = ", num_nodes)
        print("test size = ", test_size)
        print("val size = ", val_size)
        print("train size = ", train_size)

        # while True:
        #     pass

        test_indexs = rand_indices[:test_size]  # 提取测试集索引
        val_indexs = rand_indices[test_size:(test_size + val_size)]  # 提取验证集索引
        train_indexs = rand_indices[(test_size + val_size):]  # 提取训练集索引

        return test_indexs, val_indexs, train_indexs  # 返回测试集、验证集和训练集的索引列表
