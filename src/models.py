import sys, os
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
from Aggregate import aggregate

class Classification(nn.Module):
	#接受嵌入向量大小和类别数量
	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		#创建一个线性层神经网络模型，输入为emb_size，输出为num_classes。
		#这个线性层将节点的嵌入向量映射到类别得分
		self.layer = nn.Sequential(
								nn.Linear(emb_size, num_classes)	  
								#nn.ReLU()
							)
		#用于初始化模型参数。
		self.init_params()

	#对于模型中的每个参数，如果其大小是二维的（例如，线性层的权重），则使用 Xavier 初始化方法进行初始化
	def init_params(self):
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		#它接受节点的嵌入向量 embeds 作为输入，并将其传递给模型的线性层。然后，对线性层的输出进行 Log Softmax 操作，
		# 得到预测的类别概率分布，并返回这些概率值
		logists = torch.log_softmax(self.layer(embeds), 1)
		return logists
#这个模型主要用于将节点的嵌入向量转换为类别得分，以进行节点分类任务

'''
GCN是一种用于图数据的深度学习模型，它通过聚合每个节点的邻居信息来更新节点的表示。
在这个SageLayer模型中，如果self.gcn参数为True，则表示只使用邻居节点的信息进行更新，
类似于GCN的操作
'''
class SageLayer(nn.Module):
	"""
	用于实现图的节点嵌入编码，采用了"convolutional" GraphSage方法
	"""
	def __init__(self, input_size, out_size, gcn=False): 
		super(SageLayer, self).__init__()

		self.input_size = input_size
		self.out_size = out_size


		self.gcn = gcn
		#如果gcn为True，则权重矩阵的形状为(out_size, input_size)，否则形状为(out_size, 2 * input_size)
		#nn.Parameter(): 将这个张量包装成一个可学习的参数，这样在模型训练过程中，PyTorch会自动计算和更新这个参数的梯度
		self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))

		self.init_params()

	#初始化参数方法
	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	#self_feats: 当前节点的特征向量，aggregate_feats: 聚合的邻居节点特征向量（已经聚合为一个向量），neighs: 可选参数，表示邻居节点的列表
	def forward(self, self_feats, aggregate_feats, neighs=None):
		"""
		Generates embeddings for a batch of nodes.

		nodes	 -- list of nodes
		"""
		if not self.gcn:
			#将当前节点特征向量和聚合的邻居特征向量拼接在一起，然后经过ReLU激活函数
			combined = torch.cat([self_feats, aggregate_feats], dim=1)
		else:
			#则直接使用聚合的邻居特征向量，不进行拼接操作
			combined = aggregate_feats
		#将矩阵转置
		'''
		在第一次计算时，combined转置后的结果是：[1433*2,(原batch+batch*邻居)],此时第一层的weight矩阵形状是[128,1433*2]，
		weight.mm(combined.t())的矩阵形状为[128,原batch+batch*邻居],之后转置得到[原batch+batch*邻居，128]，
		也就是第一层输出的pre_hidden_embs = cur_hidden_embs
		
		
		第二次计算时，combined转置后的结果是：[128*2,batch]；此时第2层的weight矩阵形状是[128,128*2]
		weight.mm(combined.t())的矩阵形状为[128,原batch],之后转置得到[原batch，128]
		'''
		combined = F.relu(self.weight.mm(combined.t())).t()
		#combined torch.Size([2161, 128])
		#combined torch.Size([1025, 128])
		#print('combined', combined.shape)
		return combined


#GraphSage的核心思想是利用节点的邻居信息来更新节点的表示，通过多层的聚合操作逐步提取更高阶的特征。
class GraphSage(nn.Module):
	"""docstring for GraphSage"""
	def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN'):
		super(GraphSage, self).__init__()

		#input_size表示输入特征的维度
		self.input_size = input_size
		#out_size表示输出特征的维度
		self.out_size = out_size
		#表示GraphSage模型的层数
		self.num_layers = num_layers
		#表示是否用gcn方式聚合，不是真的用gcn
		self.gcn = gcn
		self.device = device
		self.agg_func = agg_func
		#原始特征
		self.raw_features = raw_features
		#邻接表
		self.adj_lists = adj_lists
		#torch.Size([2708, 1433])
		#print("self.raw_features.shape",self.raw_features.shape)

		#每一层都是一个sageLayer
		for index in range(1, num_layers+1):
			#首先确定当前层的输入特征维度layer_size，如果当前层是第一层，则输入特征维度为input_size，否则为out_size
			layer_size = out_size if index != 1 else input_size
			#利用setattr方法动态地将创建的SageLayer实例设置为GraphSage模型的属性，属性名为'sage_layer'加上当前层的序号，例如'sage_layer1'、'sage_layer2'等
			'''Sagelayer1:它的函数是SageLayer(input_size=1433, out_size=128, gcn=self.gcn)
			Sagelayer2:它的函数是SageLayer(out_size=128, out_size=128, gcn=self.gcn)
			'''
			setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

	def forward(self, nodes_batch):
		"""
		Generates embeddings for a batch of nodes.
		nodes_batch	-- batch of nodes to learn the embeddings
		"""
		#初始化了lower_layer_nodes变量，其值为输入节点批次的列表
		lower_layer_nodes = list(nodes_batch)
		#lower_layer_nodes [1, 3, 4, 7, 8, 9, 10, 12, 16, 18
		print("lower_layer_nodeslong", len(lower_layer_nodes))
		#初始化了nodes_batch_layers变量，它是一个包含元组的列表，每个元组代表一层节点的信息
		#第一层的信息通过(lower_layer_nodes,)元组存储，表示该层节点的列表；
		#接下来，通过循环从第二层到第num_layers层，依次计算每一层节点的信息，并插入到nodes_batch_layers列表中
		#具体计算过程在_get_unique_neighs_list方法中实现
		#如果你想创建一个只有一个元素的元组，你需要在元素后面加上一个逗号，这是为了与普通的括号区分开。因此，(lower_layer_nodes,)表示一个包含一个元素 lower_layer_nodes 的元组
		nodes_batch_layers = [(lower_layer_nodes,)]
		#[([0, 1, 3, 4, 6, 8, 9, 10, 12, 18, 22, 24
		#print("nodes_batch_layers", nodes_batch_layers)

		# self.dc.logger.info('get_unique_neighs.')
		#print("self.num_layers",self.num_layers),为2
		'''#循环遍历每一层，对每一层节点进行特征聚合和更新'''
		for i in range(self.num_layers):
			#在每次循环中，首先调用_get_unique_neighs_list方法，该方法接收当前层的节点列表lower_layer_nodes作为输入，
			# 返回当前层节点的邻居节点列表、节点字典以及唯一节点列表。然后，将这些信息组成元组
			lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes)
			#将元组插入到nodes_batch_layers 列表的开头。这个元组包含了当前层节点的信息，这三者被一起视为一个元素了
			nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))
			#组合起来了，由于只有两层，所以长度为2和3
			#print("nodes_batch_layers-insert-long",len(nodes_batch_layers))
		#确保nodes_batch_layers列表的长度为num_layers + 1，即每一层节点都已经被正确添加到列表中
		#循环体中每次循环都将一个元组插入到列表的开头，所以列表的长度会增加，
		'''注意此时nodes_batch_layers已经是3个元组了'''
		assert len(nodes_batch_layers) == self.num_layers + 1

		'''
		这段代码是 GraphSage 模型的前向传播过程
		主要的思想是从后往前推理，先用邻居节点的邻居节点聚合邻居节点的特征
		再用邻居节点的特征反映目标节点的特征
		'''
		#将原始特征self.raw_features初始化给pre_hidden_embs
		pre_hidden_embs = self.raw_features
		#print("pre_hidden_embs初始", pre_hidden_embs.shape)
		#nodes_batch_layers长度一致。
		'''在 Python 中，range 函数的结束值是不包含在范围内的'''
		for index in range(1, self.num_layers+1):
			#获取当前层节点的索引，注意只有一层，所以是读第2个元组的第一个值，
			'''获取本层索引,nd第一次与nodes_batch_layers[1][0]相同，nodes_batch_layers[1]是312元组，第二次与nodes_batch_layers[2]相同，是最原始的值只有1个'''
			'''现在打印出的lower_layer_nodes是nodes_batch_layers[0][0]'''
			nb = nodes_batch_layers[index][0]
			#[0, 1, 3, 4, 6, 8, 9, 10, 12, 18, 22
			#print("nodes_batch_layers[1]", nodes_batch_layers[1])
			#print("nodes_batch_layers[2]", nodes_batch_layers[2])
			#print("nodes_batch_layers[0][0]", nodes_batch_layers[0][0])
			#print("lower_layer_nodes", lower_layer_nodes)
			#print("nb", nb)

			'''获取前一层的邻居节点列表，nodes_batch_layers[0]也就是312组成的元组'''
			pre_neighs = nodes_batch_layers[index-1]
			#print("pre_neighs,len",len(pre_neighs))#3
			# self.dc.logger.info('aggregate_feats.')
			'''
			对选好的节点进行特征聚合
			'''
			# 调用 aggregate 方法，计算当前层节点的聚合特征 aggregate_feats
			"""
			第一次生成的aggregate_feats维度为[batch+batch*邻居，1433]
			第一次生成的aggregate_feats维度为[batch,128]
			"""
			aggregate_feats = aggregate(self, nb, pre_hidden_embs, pre_neighs)
			#3
			#print("pre_neighs-len",len(pre_neighs))

			#print("pre_hidden_embs", len(pre_hidden_embs))#第一次为2708
			'''
			以下这里进行神经网络的计算
			'''
			#使用 getattr 方法获取当前层的 SageLayer 对象，并调用它的前向传播方法，
			# 计算当前层节点的隐藏表示 cur_hidden_embs
			sage_layer = getattr(self, 'sage_layer'+str(index))
			#在处理非首层节点时，将当前层的节点索引 nb 转换为前一层的节点索引，以便在下一层中使用
			if index > 1:
				#通过查找前一层节点的字典映射，将当前层节点索引映射到前一层节点索引。
				# 这样做的目的是确保在下一层的聚合过程中，使用正确的前一层节点表示
				nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
			# self.dc.logger.info('sage_layer.')
			#将当前层的节点特征和聚合特征输入到 SageLayer 中进行处理，得到当前层的隐藏表示
			'''
			Sagelayer1:它的函数是SageLayer(input_size=1433, out_size=128, gcn=self.gcn)
			Sagelayer2:它的函数是SageLayer(out_size=128, out_size=128, gcn=self.gcn)
			
			对于第二层，pre_hidden_embs[nb]与aggreate_feats形状都成了[batch,128]
			'''
			cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
										aggregate_feats=aggregate_feats)
			'''
			第一次的节点是原batch节点+sample出的邻居节点，第二次是原batch节点，但两次的pre_hidden_embs不同了
			
			经过一层sagelayer运算后，pre_hidden_embs变为[batch+batch*邻居，128]
			
			pre_hidden_embs初始 torch.Size([2708, 1433])
			pre_hidden_embs运算 torch.Size([2708, 1433])，注意pre_hidden_embs[nb]会根据nb变形
			pre_hidden_embs运算 torch.Size([2101, 128])
			初始lower_layer_nodeslong 1004
			pre_hidden_embs结束了 torch.Size([990, 128])
			
			所有计算结束后，输出的pre_hidden_embs形状为[batch,128]
			'''
			#print("pre_hidden_embs运算", pre_hidden_embs.shape)
			#将 cur_hidden_embs 更新为下一次迭代中的前一层隐藏表示 pre_hidden_embs
			pre_hidden_embs = cur_hidden_embs
			'''
			在 GraphSage 模型中，每一层的节点表示是由其上一层的节点表示通过聚合操作得到的。
			在每一层的 SageLayer 中，需要将当前层的节点表示和聚合特征输入到 SageLayer 中进行处理。
			为了确保当前层节点的聚合特征和其对应的节点特征匹配，
			需要通过 _nodes_map 方法将当前层节点的索引映射到上一层节点的索引，
			从而获取对应的节点特征。这样可以保证在 SageLayer 中处理的节点特征与其对应的聚合特征是匹配的。
			'''
		print("pre_hidden_embs结束了", pre_hidden_embs.shape)
		return pre_hidden_embs

	'''
	#将当前层的节点映射到唯一节点列表中的索引，nodes：当前层节点列表；
	# neighs：包含当前层节点的邻居节点信息的元组，
	'''
	def _nodes_map(self, nodes, hidden_embs, neighs):
		#其中包括三个部分：当前层节点列表layer_nodes、当前层节点的邻居节点集合samp_neighs和节点到索引的字典layer_nodes_dict
		layer_nodes, samp_neighs, layer_nodes_dict = neighs
		"""第二次循环的nb是原始数据batch，samp_neighs是pre_neighs = nodes_batch_layers[1]中的，也就是batch数"""
		#确保当前层节点的邻居节点集合samp_neighs和节点列表nodes的长度一致，因为samp_neighs是唯一邻居节点列表，所以有几个节点就有几个列表
		assert len(samp_neighs) == len(nodes)
		#将节点映射为其在唯一节点列表中的索引,12的索引是11，13的索引是12等等
		index = [layer_nodes_dict[x] for x in nodes]

		"""
		layer_nodes_dict {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 12: 11, 13: 12
		nodes [1, 7, 8, 9, 10, 12, 18, 24, 27,
		index [1, 7, 8, 9, 10, 11, 16, 20, 23, 26, 27,
		"""
		#print("layer_nodes_dict",layer_nodes_dict)
		#print("nodes", nodes)
		#print("index", index)
		#返回节点在唯一节点列表中的索引列表
		#nodes是[0, 1, 3, 4, 6, 8, 9, 10, 12, 14]，ayer_nodes_dict是{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 12: 11, 14: 12}
		#index 列表的结果将是 [0, 1, 3, 4, 6, 8, 9, 10, 11, 12]
		return index

	'''
	#获取唯一邻居节点列表，nodes：节点列表、num_sample:要抽样的邻居节点数
	'''
	def _get_unique_neighs_list(self, nodes, num_sample=10):
		#创建了一个指向内置的set函数的引用，并将其赋值给变量_set。
		# 这样做的目的可能是为了在后续代码中使用_set作为set函数的别名，以提高代码的可读性或简洁性
		_set = set

		'''
		#推导式遍历nodes中的每个节点，并将每个节点的邻居节点列表存储在to_neighs列表中。
		'''
		#self.adj_lists是一个字典，键是节点的标识符，值是与每个节点相邻的节点列表（邻接表），
		#就是遍历节点表中每个节点，然后把这个节点的邻接表存入toneighs
		to_neighs = [self.adj_lists[int(node)] for node in nodes]
		#[{544, 258, 8, 14, 435}, {344}, {601, 197, 463}, {170}, {490, 251}, {258}]第一个括号就是索引1的邻居节点序号
		#print("to_neighs", to_neighs)
		#1191: {1290, 692, 661, 2423, 1053}, 429: {1290, 1514, 29},
		#print("邻接表",self.adj_lists)
		if not num_sample is None:
			#将random.sample函数赋值给了变量_sample，以便后续在代码中使用_sample来调用random.sample函数
			_sample = random.sample

			'''
			#这行代码是一个列表推导式，用于将每个节点的邻居集合进行采样。
			'''
			# 具体来说：对于每个节点的邻居集合 to_neighs 中的每一个邻居集合 to_neigh，
			# 如果邻居集合的长度大于等于 num_sample，则使用 random.sample 函数对邻居集合进行随机采样，
			# 采样结果保留在 _sample(to_neigh, num_sample) 中，并用 _set 函数将其转换为集合。
			#否则保持邻居集合不变
			#{544, 258, 8, 14, 435}, {344}, {601, 197, 463}, {170}, {490, 251},还是这样，因为10个邻居节点有点多
			samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
			#print("samp_neighs", samp_neighs)
		else:
			#邻居节点集合保持不变
			samp_neighs = to_neighs

		'''
		#将每个节点的邻居集合 samp_neigh 与节点自身形成的单元素集合合并，确保每个节点自身也包含在其邻居中
		'''

		#samp_neigh是节点nodes[i]的邻居集合，set([nodes[i]])创建了一个只包含节点nodes[i]的集合。通过|将两个集合合并
		#for i, samp_neigh in enumerate(samp_neighs)，遍历列表中每一个元素，使用enumerate获取索引和值
		samp_neighs = [samp_neigh | {nodes[i]} for i, samp_neigh in enumerate(samp_neighs)]
		#{544, 0, 258, 435, 8, 14}, {344, 1}, {2, 565, 471, 552, 410}, {601, 3, 197, 463},
		#print("samp_neighs", samp_neighs)
		'''
		#将所有节点的邻居集合合并为一个唯一的节点列表
		'''
		#samp_neighs是一个包含多个集合的列表，每个集合代表一个节点的邻居节点集合
		#set.union(*samp_neighs)将所有邻居节点的集合合并为一个大集合，其中的重复元素会被去除
		#*用于解包，解包操作会将这些集合作为单独的参数传递给 set.union() 函数，用于合并所有邻居节点的集合
		"""把samp_neighs里的元素unique出来"""
		_unique_nodes_list = list(set.union(*samp_neighs))
		#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12
		#print("_unique_nodes_list", _unique_nodes_list)
		'''
		这行代码创建了一个列表 i，其中包含了从 0 到 _unique_nodes_list 长度减 1 的连续整数序列。
		这个列表的目的是为了为每个唯一的节点创建一个索引，以便稍后用于构建节点到索引的映射
		'''
		i = list(range(len(_unique_nodes_list)))
		#将唯一节点列表 _unique_nodes_list 和其对应的索引列表 i 组合成一个字典 unique_nodes，其中键是唯一节点，值是其对应的索引。
		unique_nodes = dict(list(zip(_unique_nodes_list, i)))
		#{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7，12: 11, 14: 12
		#print("unique_nodes", unique_nodes)
		#将采样的邻居列表 samp_neighs、节点到索引的映射字典 unique_nodes 和唯一节点列表 _unique_nodes_list 返回给调用者
		return samp_neighs, unique_nodes, _unique_nodes_list


