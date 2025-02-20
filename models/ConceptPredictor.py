import torch
from torch import nn
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ConceptPredictor_GCN(nn.Module):
    def __init__(self,
                 input_dim,
                 n_classes=5,
                 node_feature_dim=128,
                 in_channels=None,
                 dropout=0.1):
        """

        :param input_dim: d_model
        :param n_classes: 知识点掌握程度类别数
        :param node_feature_dim: gcn hidden_dim
        :param node_feature_dim: 知识点图节点特征维度
        :param dropout:
        """
        super(ConceptPredictor_GCN, self).__init__()
        self.input_dim = input_dim  # (bs, n_c, d_model)
        self.out_channels = n_classes  # (bs, n_c, n_classes)
        self.node_feature_dim = node_feature_dim
        self.in_channels = in_channels
        self.dropout = dropout

        self.masked_attn_head = MultiHeadAttention(input_dim, n_heads=1, kq_same=False)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout_mha = nn.Dropout(dropout)
        self.dropout_gcn = nn.Dropout(dropout)

        # 合并注意力输出和节点特征
        self.fc = nn.Linear(self.input_dim + self.node_feature_dim, self.in_channels)
        self.gcn1 = GCNConv(self.in_channels, self.in_channels)
        self.gcn2 = GCNConv(self.in_channels, self.out_channels)

        self.out = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            # nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, self.out_channels),
        )

    def forward(self, q, k, v, node_features, edge_index, edge_weight=None):
        """
        # 知识点映射c_embed为 query，know_params 为 key, 最后时刻 z为value, 通过注意力计算得到 z到知识点的映射
        :param q:  c_embed      (bs, n_concepts, d_model)
        :param k:  know_params  (bs, n_know, d_model)
        :param v:  z_last       (bs, n_know, d_model)
        :return:                (bs, n_concepts, d_model)
        """
        # mha
        query_, _ = self.masked_attn_head(q, k, v, mask=None, maxout=False)
        query = q + self.dropout_mha(query_)
        x = self.layer_norm(query)

        # 合并注意力输出和节点特征
        bs, n_c, d_model = x.shape  # (bs, n_concepts, d_model)
        node_features_expanded = node_features.unsqueeze(0).expand(bs, -1, -1)  # (32, 94, 128)

        x = torch.cat((x, node_features_expanded), dim=2)  # (bs, n_concepts, d_model + d_node)  (32, 94, 128+128)

        # 全连接层
        x = self.fc(x)
        x = x.view(-1, x.size(-1))  # (bs * n_concepts, hidden_dim)  (3008, 128)

        # GCN 层
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        x = self.dropout_gcn(x)
        x = self.gcn2(x, edge_index, edge_weight)

        x = x.view(bs, n_c, -1)  # (bs, n_concepts, n_classes)

        return x

    def get_loss(self, x, labels):
        """
        :param x: 能力z到知识点的映射 (bs, n_c, d_model)
        :param labels: 学生对每个知识点的掌握层级 (bs, n_c)
        :return loss: 损失值
        """
        outputs = self(x)
        class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0]).to(outputs.device)  # 根据类别不均衡情况设置权重
        # weight: 一个手动指定每个类别权重的张量，用于处理类别不平衡问题。
        # ignore_index: 指定某个类别不参与损失计算。
        # reduction: 指定如何对批次中的损失进行归约（'none', 'mean', 'sum'）
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # 将 outputs 展平成二维张量 (bs * n_know, num_classes)
        outputs_flattened = outputs.view(-1, outputs.size(-1))
        # 将 labels 展平成一维张量 (bs * n_know)
        labels_flattened = labels.view(-1).long()
        loss = criterion(outputs_flattened, labels_flattened)
        return loss

    def get_focal_loss(self, x, labels):
        outputs = self(x)
        # 使用 FocalLoss 函数计算损失
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        outputs_flattened = outputs.view(-1, outputs.size(-1))
        labels_flattened = labels.view(-1).long()
        loss = criterion(outputs_flattened, labels_flattened)
        return loss

    def get_semi_sparse_loss(self, q, k, v, node_features, edge_index, labels, edge_weight=None):
        outputs = self(q, k, v, node_features, edge_index, edge_weight)
        batch_size, num_nodes, n_classes = outputs.shape
        num_labeled_nodes = labels.shape[1]
        # 只选择有标签的节点的输出
        valid_outputs = outputs[:, :num_labeled_nodes, :]

        criterion = SemiSparseLoss()
        loss = criterion(outputs, labels)
        return loss

    def predict(self, q, k, v, node_features, edge_index, edge_weight=None):
        self.eval()
        with torch.no_grad():
            # 调用forward，需要传入gcn相关参数
            outputs = self(q, k, v, node_features, edge_index, edge_weight=None)  # (bs, n_concepts, n_classes)
            _, predictions = torch.max(outputs, dim=-1)  # 获取最大值所在的索引作为预测类别
        return predictions  # (bs, n_concepts)


class ConceptPredictor(nn.Module):
    def __init__(self,
                 input_dim,
                 n_classes=5,
                 hidden_dim=None,
                 num_layers=None,
                 dropout=0.1):
        """

        :param input_dim: d_model
        :param n_classes: 1
        :param hidden_dim:
        :param num_layers:
        :param dropout:
        """
        super(ConceptPredictor, self).__init__()
        self.input_dim = input_dim  # (bs, n_c, d_model)
        self.output_dim = n_classes  # (bs, n_c, n_classes)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.dropout = dropout

        self.masked_attn_head = MultiHeadAttention(input_dim, n_heads=1, kq_same=False)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout_mha = nn.Dropout(dropout)

        # self.fc1 = nn.Linear(input_dim, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, n_classes)
        # self.dropout = nn.Dropout(dropout)

        self.out = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            # nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, self.output_dim),
        )

    def forward(self, q, k, v):
        """
        # 知识点映射c_embed为 query，know_params 为 key, 最后时刻 z为value, 通过注意力计算得到 z到知识点的映射
        :param q:  c_embed      (bs, n_concepts, d_model)
        :param k:  know_params  (bs, n_know, d_model)
        :param v:  z_last       (bs, n_know, d_model)
        :return:                (bs, n_concepts, d_model)
        """
        # mha
        query_, _ = self.masked_attn_head(q, k, v, mask=None, maxout=False)
        query = q + self.dropout_mha(query_)
        x = self.layer_norm(query)

        # mlp
        bs, n_c, d_model = x.shape  # (bs, n_concepts, d_model)
        x = x.view(-1, d_model)  # (bs* n_concepts, d_model)
        x = self.out(x)  # (bs* n_concepts, n_classes)
        x = x.view(bs, n_c, -1)  # (bs, n_concepts, n_classes)

        return x

    def get_loss(self, x, labels):
        """
        :param x: 能力z到知识点的映射 (bs, n_c, d_model)
        :param labels: 学生对每个知识点的掌握层级 (bs, n_c)
        :return loss: 损失值
        """
        outputs = self(x)
        class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0]).to(outputs.device)  # 根据类别不均衡情况设置权重
        # weight: 一个手动指定每个类别权重的张量，用于处理类别不平衡问题。
        # ignore_index: 指定某个类别不参与损失计算。
        # reduction: 指定如何对批次中的损失进行归约（'none', 'mean', 'sum'）
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # 将 outputs 展平成二维张量 (bs * n_know, num_classes)
        outputs_flattened = outputs.view(-1, outputs.size(-1))
        # 将 labels 展平成一维张量 (bs * n_know)
        labels_flattened = labels.view(-1).long()
        loss = criterion(outputs_flattened, labels_flattened)
        return loss

    def get_focal_loss(self, x, labels):
        outputs = self(x)
        # 使用 FocalLoss 函数计算损失
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        outputs_flattened = outputs.view(-1, outputs.size(-1))
        labels_flattened = labels.view(-1).long()
        loss = criterion(outputs_flattened, labels_flattened)
        return loss

    def get_semi_sparse_loss(self, q, k, v, labels):
        outputs = self(q, k, v)
        criterion = SemiSparseLoss()
        loss = criterion(outputs, labels)
        return loss

    def predict(self, q, k, v):
        self.eval()
        with torch.no_grad():
            outputs = self(q, k, v)  # (bs, n_concepts, n_classes)
            _, predictions = torch.max(outputs, dim=-1)  # 获取最大值所在的索引作为预测类别
        return predictions  # (bs, n_concepts)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, ignore_index=-100):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        elif self.reduction == 'none':
            return focal_loss


class SemiSparseLoss(nn.Module):
    def __init__(self):
        super(SemiSparseLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets):
        # 实现半稀疏损失函数的计算逻辑
        # predictions: (batch_size, num_knowledge_points, num_classes)
        # targets: (batch_size, num_knowledge_points)
        batch_size, num_knowledge_points, num_classes = predictions.size()
        # 将预测结果和目标标签展平
        predictions = predictions.view(-1, num_classes)
        targets = targets.view(-1).long()
        # Calculate the loss for all points assuming target is 0 (class index 0)
        zero_targets = torch.zeros_like(targets, dtype=torch.long)
        # f(0, y_hat)
        loss_all = self.ce_loss(predictions, zero_targets)
        # 打标的mask
        labeled_mask = (targets != -1)  # 未打标的知识点标签为-1
        if labeled_mask.sum() > 0:
            # y_t<>0: f(y_t, y_hat)
            loss_non_zero = self.ce_loss(predictions[labeled_mask], targets[labeled_mask])
            # y_t<>0: f(0, y_hat)
            loss_all_non_zero = self.ce_loss(predictions[labeled_mask], zero_targets[labeled_mask])
            # Combine the losses
            semi_sparse_loss = loss_all.sum() + (loss_non_zero - loss_all_non_zero).sum()
            semi_sparse_loss /= len(targets)
        else:
            semi_sparse_loss = loss_all.mean()
        return semi_sparse_loss


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.out_linear = nn.Linear(d_model, d_model, bias=bias)
        # self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        # torch.nn.init.xavier_uniform_(self.gammas)
        self.gammas = None

    def forward(self, q, k, v, mask=None, maxout=False):
        bs = q.size(0)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_head).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_head).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_head).transpose(1, 2)

        output, scores = self.Attention(q, k, v, mask, gamma=self.gammas, maxout=maxout)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_linear(output)
        return output, scores

    def Attention(self, q, k, v, mask=None, gamma=None, maxout=False):
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        bs, head, seqlen, _ = scores.size()
        # e^(-gamma * d(▲t))
        if gamma is not None:
            x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
            x2 = x1.transpose(0, 1).contiguous()

            with torch.no_grad():
                if mask is not None:
                    scores_ = scores.masked_fill(mask == 0, -1e32)
                else:
                    scores_ = scores
                scores_ = F.softmax(scores_, dim=-1)
                # 累计和
                distcum_scores = torch.cumsum(scores_, dim=-1)
                # 总和
                disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)

                position_effect = torch.abs(x1 - x2)[None, None, :, :]
                # mask的数据dist_scores为 0，距离越远dist_scores越高
                dist_scores = torch.clamp(
                    (disttotal_scores - distcum_scores) * position_effect, min=0.0
                )
                dist_scores = dist_scores.sqrt().detach()
            gamma = -1.0 * gamma.abs().unsqueeze(0)
            # 距离越远total_effect越小
            total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)
            scores *= total_effect

        if mask is not None:
            scores.masked_fill(mask == 0, -1e32)

        scores = F.softmax(scores, dim=-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, 0)

        # max-out scores (bs, n_heads, seqlen, seqlen)
        if maxout:
            scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
            scores *= scale
        output = torch.matmul(scores, v)
        return output, scores
