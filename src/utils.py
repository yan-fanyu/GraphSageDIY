import sys
import os
import torch
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


# 该类的作用是根据节点的嵌入向量和节点之间的关系计算损失，用于无监督学习任务中
class UnsupervisedLoss(object):
	"""docstring for UnsupervisedLoss"""

	def __init__(self, adj_lists, train_nodes, device):
		super(UnsupervisedLoss, self).__init__()
		self.Q = 10
		self.N_WALKS = 6
		self.WALK_LEN = 1
		self.N_WALK_LEN = 5
		self.MARGIN = 3
		self.adj_lists = adj_lists
		self.train_nodes = train_nodes
		self.device = device

		self.target_nodes = None
		self.positive_pairs = []
		self.negtive_pairs = []
		self.node_positive_pairs = {}
		self.node_negtive_pairs = {}
		self.unique_nodes_batch = []

	def get_loss_sages(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i] == self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n: i for i, n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			# Q * Exception(negative score)
			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
			# print(neg_score)

			# multiple positive score
			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score = torch.log(torch.sigmoid(pos_score))
			# print(pos_score)

			nodes_score.append(torch.mean(- pos_score - neg_score).view(1, -1))

		loss = torch.mean(torch.cat(nodes_score, 0))

		return loss

	def get_loss_margins(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i] == self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n: i for i, n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

			nodes_score.append(
				torch.max(torch.tensor(0.0).to(self.device), neg_score - pos_score + self.MARGIN).view(1, -1))
		# nodes_score.append((-pos_score - neg_score).view(1,-1))

		loss = torch.mean(torch.cat(nodes_score, 0), 0)

		# loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))

		return loss

	def extend_nodess(self, nodes, num_neg=6):
		self.positive_pairs = []
		self.node_positive_pairs = {}
		self.negtive_pairs = []
		self.node_negtive_pairs = {}

		self.target_nodes = nodes
		self.get_positive_nodes(nodes)
		# print(self.positive_pairs)
		self.get_negtive_nodes(nodes, num_neg)
		# print(self.negtive_pairs)
		self.unique_nodes_batch = list(
			set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
		assert set(self.target_nodes) < set(self.unique_nodes_batch)
		return self.unique_nodes_batch

	# 获得正样本节点
	def get_positive_nodes(self, nodes):
		return self._run_random_walks(nodes)

	# 获得负样本节点
	def get_negtive_nodes(self, nodes, num_neg):
		for node in nodes:
			neighbors = set([node])
			frontier = set([node])
			for i in range(self.N_WALK_LEN):
				current = set()
				for outer in frontier:
					current |= self.adj_lists[int(outer)]
				frontier = current - neighbors
				neighbors |= current
			far_nodes = set(self.train_nodes) - neighbors
			neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
			self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
			self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
		return self.negtive_pairs

	# 运行随机游走
	def _run_random_walks(self, nodes):
		for node in nodes:
			if len(self.adj_lists[int(node)]) == 0:
				continue
			cur_pairs = []
			for i in range(self.N_WALKS):
				curr_node = node
				for j in range(self.WALK_LEN):
					neighs = self.adj_lists[int(curr_node)]
					next_node = random.choice(list(neighs))
					# self co-occurrences are useless
					if next_node != node and next_node in self.train_nodes:
						self.positive_pairs.append((node, next_node))
						cur_pairs.append((node, next_node))
					curr_node = next_node

			self.node_positive_pairs[node] = cur_pairs
		return self.positive_pairs


"""
代码核心运行过程
"""
def apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
	test_nodes = getattr(dataCenter, ds+'_test')
	val_nodes = getattr(dataCenter, ds+'_val')
	train_nodes = getattr(dataCenter, ds+'_train')
	labels = getattr(dataCenter, ds+'_labels')
	#num_neg 是用于无监督学习的负采样数量，其值不同可能会影响模型的学习效果
	if unsup_loss == 'margin':
		num_neg = 6
	elif unsup_loss == 'normal':
		num_neg = 100
	else:
		print("unsup_loss can be only 'margin' or 'normal'.")
		sys.exit(1)
	#随机打乱
	train_nodes = shuffle(train_nodes)
	#models列表包含了要优化的模型
	models = [graphSage, classification]
	params = []
	for model in models:
		#遍历每个模型的参数，检查是否需要计算梯度，把需要优化的参数添加到 params 列表中
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)
	#使用 params 列表初始化一个 SGD（随机梯度下降）优化器，设置学习率为 0.7
	optimizer = torch.optim.SGD(params, lr=0.7)
	#梯度归零
	optimizer.zero_grad()
	for model in models:
		model.zero_grad()
	#向上取整获得batches数
	batches = math.ceil(len(train_nodes) / b_sz)
	#设置一个无序不重复空集合对象
	visited_nodes = set()
	for index in range(batches):
		#循环导入一批批节点
		nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]

		# extend nodes batch for unspervised learning
		# no conflicts with supervised learning
		#调用了 unsupervised_loss 对象的 extend_nodes 方法，
		# 该方法可能会扩展给定的节点批次 nodes_batch，并返回扩展后的节点列表。参数 num_neg 用于指定扩展节点的数量。
		nodes_batch = np.asarray(list(unsupervised_loss.extend_nodess(nodes_batch, num_neg=num_neg)))
		#将扩展后的节点列表添加到 visited_nodes 集合中，以标记这些节点已被访问
		visited_nodes |= set(nodes_batch)

		# get ground-truth for the nodes batch，根据节点批次获取相应的标签，这些标签将用于计算损失
		labels_batch = labels[nodes_batch]

		# feed nodes batch to the graphSAGE
		# returning the nodes embeddings
		'''
		将节点批次输入到 graphSage 模型中，以获取节点的嵌入表示（特征）
		'''
		embs_batch = graphSage(nodes_batch)

		if learn_method == 'sup':
			# superivsed learning
			#通过分类器模型 classification 计算节点的预测得分 logists
			logists = classification(embs_batch)
			#损失的计算采用了交叉熵损失函数的形式，通过对每个节点的预测得分和真实标签进行相应位置的计算来得到损失值，
			#logists[range(logists.size(0)), labels_batch] 部分用于获取每个节点的预测得分，
			# -torch.sum(..., 0) 部分用于计算所有节点的损失总和，最后除以节点批次的大小来得到平均损失
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			loss = loss_sup
		elif learn_method == 'plus_unsup':
			# superivsed learning
			logists = classification(embs_batch)
			loss_sup = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
			loss_sup /= len(nodes_batch)
			# unsuperivsed learning
			#接下来，根据无监督损失类型 unsup_loss 的不同进行计算。
			#
			if unsup_loss == 'margin':
				loss_net = unsupervised_loss.get_loss_margins(embs_batch, nodes_batch)
			elif unsup_loss == 'normal':
				loss_net = unsupervised_loss.get_loss_sages(embs_batch, nodes_batch)

			loss = loss_sup + loss_net
		else:
			if unsup_loss == 'margin':
				loss_net = unsupervised_loss.get_loss_margins(embs_batch, nodes_batch)
			elif unsup_loss == 'normal':
				loss_net = unsupervised_loss.get_loss_sages(embs_batch, nodes_batch)
			loss = loss_net

		print('Step [{}/{}], Loss: {:.4f}, Dealed Nodes [{}/{}] '.format(index+1, batches, loss.item(), len(visited_nodes), len(train_nodes)))
		loss.backward()
		for model in models:
			nn.utils.clip_grad_norm_(model.parameters(), 5)
		optimizer.step()

		optimizer.zero_grad()
		for model in models:
			model.zero_grad()

	return graphSage, classification