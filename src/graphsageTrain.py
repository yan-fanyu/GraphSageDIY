import math
import sys
import os
import torch
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import torch.nn as nn
import numpy as np

def classcificationACC(true_labels,predict_labels,dataloader):
	correctACC = 0
	correctACC += true_labels.eq(predict_labels.data.view_as(true_labels)).cpu().sum()
	acc = 100. * correctACC / len(dataloader)
	return acc

#可以试着将其新建一个valtest
#定义函数evaluate，用于评估模型性能
def evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, cur_epoch):
	#这几段代码的意思是通过getattr函数从datacenter中获取ds（dataset，有两种cora和pub）的各个属性值，
	#test_nodes可能是一个列表、数组或其他数据结构，用于存储测试集的节点信息
	test_nodes = getattr(dataCenter, ds+'_test')
	print("testnodes.tpye:",type(test_nodes))
	val_nodes = getattr(dataCenter, ds+'_val')
	labels = getattr(dataCenter, ds+'_labels')

	models = [graphSage, classification]
	#以下代码用于冻结模型中的参数，防止它们在训练中被更新，思考可否使用model.eval()
	#创建一个空列表params，用于存储需要设置“require_grad”属性的参数
	params = []
	#遍历模型列表“models”中的每一个参数，也就是两个模型
	for model in models:
		#对于每个模型，遍历其参数列表model.parameters()中的每个参数
		for param in model.parameters():
			#如果参数的requires_grad属性为True，则将其设置为False，表示不需要计算其梯度
			if param.requires_grad:
				param.requires_grad = False
				#将需要设置属性的参数添加到params列表中
				params.append(param)
	'''
	#使用 GraphSage 模型获取验证集节点的嵌入
	'''
	print("val_nodes.shape", val_nodes.shape)
	print("val_nodes.type", type(val_nodes))
	embs = graphSage(val_nodes)
	print("embs经过graphsage。shape：", embs.shape)
	print("embs经过graphsage.type：", type(embs))
	#使用分类器模型对这些嵌入进行分类，得到分类得分
	logists = classification(embs)
	#形状为torch.Size([451, 7])
	print("logists.type", type(logists))
	print(logists.shape)
	print("logists.shape")
	#对分类得分取最大值，得到每个节点的预测类别，torch.max(logists, 1)返回logists张量沿着指定维度的最大值以及对应的索引。
	#torch.max返回一个元组 (values, indices)，其中 values 是沿着指定维度的最大值组成的张量，indices 是对应的索引组成的张量
	#_, predicts = 将返回的元组进行解包，但是只保留索引部分，最大值部分被 _ 所忽略.predicts是一个张量，包含了预测类别的索引
	#沿着每行，横向走，再纵向走
	_, predicts = torch.max(logists, 1)
	#获取验证集节点的真实标签
	labels_val = labels[val_nodes]
	#断言验证集节点的真实标签数量与预测结果数量相同
	assert len(labels_val) == len(predicts)
	comps = zip(labels_val, predicts.data)
	labels_valT = torch.tensor(labels_val).cuda()
	ValACC = classcificationACC(labels_valT,predicts,val_nodes)

	print("Validation ACC:", ValACC)
	print(type(ValACC))
	#调用f1函数，，val真实标签，predicts.cpu().data预测结果（不包括梯度），使用微平均计算F1参数
	vali_f1 = f1_score(labels_val, predicts.cpu().data, average="micro")
	print("Validation F1:", vali_f1)
#用于检测验证集F1是否超过了之前记录的最大验证集F1，如果超过了，就更新，并执行操作
	#如果不好，就不存
	if vali_f1 > max_vali_f1:
		max_vali_f1 = vali_f1
		#获取测试节点图表征向量
		embs = graphSage(test_nodes)


		#输入分类器获得结果
		logists = classification(embs)
		_, predicts = torch.max(logists, 1)
		labels_test = labels[test_nodes]
		assert len(labels_test) == len(predicts)
		comps = zip(labels_test, predicts.data)

		test_f1 = f1_score(labels_test, predicts.cpu().data, average="micro")
		print("Test F1:", test_f1)

		for param in params:
			param.requires_grad = True

		torch.save(models, '../models/model_best_{}_ep{}_{:.4f}.torch'.format(name, cur_epoch, test_f1))
	#因为是一个epoch一个个训练，需要冲洗把模型属性设为true
	for param in params:
		param.requires_grad = True

	return max_vali_f1

#embeddings
def get_gnn_embeddings(gnn_model, dataCenter, ds):
	print('正在从训练好的 GraphSAGE 模型加载图表征向量.')
	features = np.zeros((len(getattr(dataCenter, ds+'_labels')), gnn_model.out_size))
	#创建一个包含所有节点索引的列表，.tolist()用于将数组和张量转换为python列表的形式
	nodes = np.arange(len(getattr(dataCenter, ds+'_labels'))).tolist()
	#batch——size设置为500
	b_sz = 500
	#计算总共有多少个batch，ceil向上取整
	batches = math.ceil(len(nodes) / b_sz)
	#创建一个空列表，用于存储每个节点的图表征向量
	embs = []
	for index in range(batches):
		#从节点列表中提取当前批次节点，500一取
		nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
		#调用 GNN 模型，获取当前批次节点的图表征向量
		embs_batch = gnn_model(nodes_batch)
		#断言当前批次节点的图表征向量数量与节点数量相同
		assert len(embs_batch) == len(nodes_batch)
		#将当前批次节点的图表征向量添加到 embs 列表中
		embs.append(embs_batch)
		# if ((index+1)*b_sz) % 10000 == 0:
		#     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')
	assert len(embs) == batches
	# 将 embs 列表中的图表征向量拼接成一个张量，沿着行的方向进行拼接
	embs = torch.cat(embs, 0)
	#断言拼接后的张量的长度与节点列表的长度相同
	assert len(embs) == len(nodes)
	print('Embeddings loaded.')
	#返回加载的图表征向量，并调用 detach() 方法，使其与计算图分离，以避免梯度传播到原始的 GNN 模型参数
	return embs.detach()

#只在无监督学习过程中使用
def train_classification(dataCenter, graphSage, classification, ds, device, max_vali_f1, name, epochs=800):
	print('Training Classification ...')
	#创建优化器优化模型参数
	c_optimizer = torch.optim.SGD(classification.parameters(), lr=0.5)
	# train classification, detached from the current graph
	#classification.init_params()
	#batch——size为50
	b_sz = 50
	#训练集节点和标签，numpy.ndarray格式
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')
	#tensor形式的features
	features = get_gnn_embeddings(graphSage, dataCenter, ds)
	#featurs经过graphsage。shape： torch.Size([2708, 128])
	#featuregraphsage.type： <class 'torch.Tensor'>
	print("featurs经过graphsage。shape：", features.shape)
	print("featuregraphsage.type：", type(features))
	for epoch in range(epochs):
		#对训练节点进行随机洗牌，以增加训练的随机性
		train_nodes = shuffle(train_nodes)
		#向上取整计算batch数量
		batches = math.ceil(len(train_nodes) / b_sz)
		#集合是一种无序且不重复的数据结构，在这里用于存储已经访问过的节点。通过这个集合，可以确保在遍历数据时不重复处理相同的节点
		visited_nodes = set()
		#一批一批抽取数据
		for index in range(batches):
			nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]

			visited_nodes |= set(nodes_batch)
			#获得当前批次对应的标签
			labels_batch = labels[nodes_batch]
			#获得当前批次的图表征向量
			embs_batch = features[nodes_batch]
			print("embs_batch。shape：", embs_batch.shape)
			print("embs_batch.type：", type(embs_batch))
			#使用分类器模型对图表征向量进行分类得分预测，logists是模型中的输出的张量
			logists = classification(embs_batch)
			print("Trainlogists经过classification。shape：", logists.shape)
			print("Trainlogists经过classification.type：", type(logists))
			#计算交叉熵损失
			loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			#将总的负预测得分除以批次中的节点数量，以得到每个节点的平均损失
			loss /= len(nodes_batch)
			# print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(epoch+1, epochs, index, batches, loss.item(), len(visited_nodes), len(train_nodes)))

			loss.backward()
			#对梯度进行裁剪，以防止梯度爆炸
			nn.utils.clip_grad_norm_(classification.parameters(), 5)
			#更新优化器
			c_optimizer.step()
			#清除梯度
			c_optimizer.zero_grad()
		#在每个训练周期结束时，评估分类器模型的性能
		max_vali_f1 = evaluate(dataCenter, ds, graphSage, classification, device, max_vali_f1, name, epoch)
	#函数返回训练好的分类器模型以及最大的验证集 F1 分数
	return classification, max_vali_f1