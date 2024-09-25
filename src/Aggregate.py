import torch
import random

import torch.nn as nn
import torch.nn.functional as F

'''
将邻居节点的特征进行聚合，生成每个节点的聚合特征
unique_nodes_list 存储了所有邻居节点的列表，
samp_neighs 是每个节点的邻居节点的集合列表，
unique_nodes 是一个字典，将每个节点映射到其在 unique_nodes_list 中的索引位置
'''


def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
    # 分别为唯一节点列表、唯一邻居节点列表、节点字典
    unique_nodes_list, samp_neighs, unique_nodes = pre_neighs
    # 通过断言来确保节点列表和邻居列表的长度相等

    # true      正常执行
    # false     抛出异常
    assert len(nodes) == len(samp_neighs)

    # 确保每个节点的邻居列表中都包含该节点本身，indicator 是一个布尔列表，
    # 其中的每个元素表示对应节点的邻居列表中是否包含该节点本身。
    # nodes[i] in samp_neighs[i] 遍历每个节点及其邻居列表，检查该节点是否在其自己的邻居列表中
    # 如果邻居列表不包括节点本身，那么对应indicator 列表中就会出现 False
    indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
    # 做一个检查，确保indicator中没有false
    assert (False not in indicator)


    if not self.gcn:
        # 去除每个节点的邻居列表中的节点本身，通过集合减法来实现，{544，0，8} -> {544，8}
        samp_neighs = [(samp_neighs[i] - {nodes[i]}) for i in range(len(samp_neighs))]
    # 如果 pre_hidden_embs（特征） 中的嵌入数量与 unique_nodes 中的唯一节点数量相同，那么说明每个节点的嵌入已经包含在 pre_hidden_embs 中
    #print("len(unique_nodes):",len(unique_nodes)),25xx,21xx
    '''
    在第一个sagelayer中，也就是用邻居的邻居反映邻居中，由于pre_hidden_embs的长度是2708；
    unique_nodes字典的长度是（原batch+邻居*batch（不是很严谨）+邻居*邻居），来自node-batch-layers[0]，所以二者不同；
    在第二个sagelayer中，也就是用邻居节点反映目标节点中，pre_hidden_embs的长度是原batch+邻居*batch（不是很严谨）；
    unique_nodes字典的长度是原batch+邻居*batch（不是很严谨），此时两者相等
    
    '''
    if len(pre_hidden_embs) == len(unique_nodes):
        '''
        在第二个sagelayer中，embed_matrix长度是原batch+邻居*batch（不是很严谨）
        '''
        embed_matrix = pre_hidden_embs
    else:
        # 需要根据 unique_nodes_list 中的节点索引，
        # 从 pre_hidden_embs 中选择相应的嵌入，形成一个新的嵌入矩阵 embed_matrix
        # 通过使用 torch.LongTensor(unique_nodes_list) 作为索引来实现，就是把unique_nodes_list中的出现的节点的特征拿出来
        '''
        在第一个sagelayers中，embed_matrix长度是原batch+邻居*batch（不是很严谨）+邻居*邻居
            
        '''
        embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        #[25xx,1433]
        #print("embed_matrix", embed_matrix.shape)
    '''
    这段代码用于生成一个掩码矩阵，以便在聚合过程中只考虑与每个节点相邻的节点
    '''
    # mask是一个全0的矩阵，len(samp_neighs) 表示节点批次中节点的数量，len(unique_nodes) 表示唯一节点的数量
    #mask[node-batch，unique_nodes]
    '''在第一个sagelayers中，mask的shape是行：原batch+邻居*batch（不是很严谨）个集合；列：原batch+邻居*batch（不是很严谨）+邻居*邻居
        在第2个sagelayers中，mask的shape是行：batch;列：原batch+邻居*batch（不是很严谨）
    '''

    mask = torch.zeros(len(samp_neighs), len(unique_nodes))

    '''
    column_indices 是一个列表，其中每个元素对应着 samp_neighs 中的每个节点的索引
    '''

    # 对于 samp_neighs 中的每个节点，它的邻居节点在 unique_nodes 中的索引被添加到 column_indices 中
    # for samp_neigh in samp_neighs 遍历 samp_neighs 列表中的每个元素，即遍历每个节点的邻居节点集合{544，8}，{344}
    # for n in samp_neigh 遍历当前节点的邻居节点集合中的每个节点544，8，344.。。
    # 整个列表推导式会将每个邻居节点在 unique_nodes 字典中的索引收集到 column_indices 列表中
    column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
    '''
	这段代码生成了 row_indices 列表，该列表包含了将要设置为1的位置在 mask 张量中的行索引
	'''
    # i遍历了 samp_neighs 列表的索引，即遍历了每个节点的邻居集合,即节点的数量，即{}这种结构总共有多少个；
    # j遍历了每个节点的邻居集合中的每个邻居节点的长度
    # 在嵌套的循环中，每次迭代生成的 i 表示节点的索引，而 j 表示当前节点的邻居节点的索引
    # 每次迭代时，都会将 i 添加到 row_indices 中，因为每个邻居节点对应一个行索引
    row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
    # row——indices为mask提供了行索引，就是想找哪个节点的邻居节点标注最终把每一行的邻居节点位置设置成了1
    mask[row_indices, column_indices] = 1
    #[2157,2558]
    #print("mask.shape", mask.shape)
    if self.agg_func == 'MEAN':
        # 计算邻居节点的数量,沿着列的方向，即向右
        num_neigh = mask.sum(1, keepdim=True)
        # 将 mask 张量中的每行除以相应的邻居数量，以确保对每个节点的特征进行平均聚合
        # 将张量 mask 移动到与 embed_matrix 张量相同的设备上
        mask = mask.div(num_neigh).to(embed_matrix.device)
        # 将 mask 与特征矩阵 embed_matrix 矩阵相乘，以获取每个节点的聚合特征向量
        # mask 的大小为 (num_nodes, num_unique_nodes)，
        # 而 embed_matrix 的大小为 (num_unique_nodes, embedding_dim（嵌入向量）)。
        # 所以结果将会是一个 (num_nodes, embedding_dim) 大小的张量，
        # 其中每行包含了对应节点的聚合特征
        aggregate_feats = mask.mm(embed_matrix)
        '''第二次的aggregate_feats为[batch,128]'''



    elif self.agg_func == 'MAX':
        # 找到所有值为1的元素的索引，x.nonzero() 是一个 PyTorch 函数，用于找到张量 x 中非零元素的索引。它返回一个包含非零元素索引的张量
        indexs = [x.nonzero() for x in mask == 1]
        #tensor([[ 463],[1541]]), tensor([[ 11],[671]]), tensor([[ 14]])
        #print("indexs", indexs)
        aggregate_feats = []
        # 对于每个索引，它从embed_matrix中获取对应的特征向量，并执行以下操作
        #x.squeeze() 是为了去除 x 中的所有维度大小为1的维度,所以tensor([[ 463],[1541]])就变成了[463,1541]
        for feat in [embed_matrix[x.squeeze()] for x in indexs]:
            #print("feat", feat)
            #feat.shape torch.Size([y, 1433])
            #print("feat.shape", feat.shape)
            if len(feat.size()) == 1:
                # 如果特征向量是一维的，则直接将其添加到aggregate_feats列表中
                aggregate_feats.append(feat.view(1, -1))
            else:
                # 如果特征向量是二维的，则使用torch.max函数沿着指定的维度0找到最大值
                aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
        # 最后，它将所有的特征向量连接起来，形成最终的聚合特征矩阵aggregate_feats
        aggregate_feats = torch.cat(aggregate_feats, 0)
        #21xx*1433
        ##print("aggregate_feats.shape", aggregate_feats.shape)



    elif self.agg_func == 'MEANPOOL':
        indexs = [x.nonzero() for x in mask == 1]
        aggregate_feats = []
        for feat in [embed_matrix[x.squeeze()] for x in indexs]:
            if len(feat.size()) == 1:
                aggregate_feats.append(feat.view(1, -1))
            else:
                # mean
                aggregate_feats.append(torch.mean(feat, 0).view(1, -1))
        aggregate_feats = torch.cat(aggregate_feats, 0)

    elif self.agg_func == 'LSTM':
        indexs = [x.nonzero() for x in mask == 1]
        aggregate_feats = []
        for feat in [embed_matrix[x.squeeze()] for x in indexs]:
            if len(feat.size()) == 1:
                # 如果特征向量的维度为1，则需要在其上添加一个维度，使其成为一个序列
                feat = feat.unsqueeze(0)
            # 对于每个邻居节点的特征向量，它们被添加到aggregate_feats列表中
            aggregate_feats.append(feat)
        # 通过遍历aggregate_feats列表，获取每个特征向量序列的长度，并将这些长度存储在seq_lengths列表中
        seq_lengths = [x.size(0) for x in aggregate_feats]
        # pad_sequence函数是PyTorch中的一个函数，用于将序列填充到相同长度。
        # 在这里，aggregate_feats是一个列表，其中包含每个邻居节点的特征向量序列
        # batch_first=True参数指示函数将批次维度放在序列维度之前，因此函数将填充序列以使它们具有相同的长度，并返回填充后的张量作为LSTM的输入
        lstm_input = nn.utils.rnn.pad_sequence(aggregate_feats, batch_first=True)
        lstm_input = lstm_input.to(self.device)  # 将输入移动到与 LSTM 参数相同的设备上
        # LSTM层，其输入和输出维度都设置为embed_matrix.size(1)
        lstm = nn.LSTM(embed_matrix.size(1), embed_matrix.size(1)).to(self.device)  # 将 LSTM 层移动到与输入数据相同的设备上
        # 这个函数将填充后的序列输入打包成PackedSequence格式，以便LSTM层能够忽略填充的部分并只处理有效的序列数据。
        # seq_lengths是填充前每个序列的长度列表，
        # batch_first=True指定输入数据的批次维度位于第一个维度，enforce_sorted=False表示不要求输入序列是按长度降序排列的
        packed_input = nn.utils.rnn.pack_padded_sequence(lstm_input, seq_lengths, batch_first=True,
                                                         enforce_sorted=False)
        # 将打包后的输入packed_input传递给LSTM层进行处理。
        # lstm的返回值包含两部分：第一个部分是输出序列，第二部分是最后一个时间步的隐藏状态和细胞状态（h_n, c_n）
        '''
		在这里，我们只关心最后一个时间步的隐藏状态，因此只取h_n
		'''
        _, (h_n, _) = lstm(packed_input)
        # 最后，使用squeeze(0)来去除第一维，因为h_n的形状是(num_layers * num_directions, batch_size, hidden_size)
        # 最后的aggregate_feats的维度是(batch_size, hidden_size)，其中batch_size是输入数据的批次大小，hidden_size是LSTM层的隐藏状态的大小
        aggregate_feats = h_n.squeeze(0)

    return aggregate_feats
