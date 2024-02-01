# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from scripts import Constants
from model.layers import PositionalEmbedding, MLP, KLDivergence
from model.modules import SA, SGA, GA

class ShallowModule(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(ShallowModule, self).__init__()
        self.cross_attention = GA(hidden_size, head_num, ff_size, dropout, hidden_size_head)

    def forward(self, inputs, vis_feat, vis_mask_tmp, program_masks):
        enc_output = self.cross_attention(inputs, vis_feat, vis_mask_tmp)
        enc_output = enc_output * program_masks.unsqueeze(-1)
        return enc_output

"""
核心模块，用于实现一层的推理过程:
=> SA用于汇聚前面（上一阶段）的依赖信息；
=> GA用于从图像中获取当前的推理结果；
"""
class Module(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(Module, self).__init__()
        self.self_attention = SA(hidden_size, head_num, ff_size, dropout, hidden_size_head)
        self.cross_attention = GA(hidden_size, head_num, ff_size, dropout, hidden_size_head)

    def forward(self, inputs, mask, vis_feat, vis_mask_tmp, program_masks, alpha):
        alpha = alpha.unsqueeze(-1)
        trans_mask = (1 - mask).unsqueeze(1).to(torch.bool)
        enc_output = self.self_attention(inputs, trans_mask)  # Gather dependency semantics
        enc_output = self.cross_attention(enc_output, vis_feat, vis_mask_tmp)  # Gather image information
        enc_output = enc_output * program_masks.unsqueeze(-1)

        # 每次推理只更新激活的几个program，其它的保持不变
        return alpha * enc_output + (1 - alpha) * inputs

def init(module, weight_init, bias_init, gain=1):
    """

    :param module: module to initialize
    :param weight_init: initialization scheme
    :param bias_init: bias initialization scheme
    :param gain: gain for weight initialization
    :return: initialized module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

"""
\pi策略，用于选择action，最简单的实现方式就是使用状态向量和所有的视觉特征
通过dot product得到概率分布；
=> query定义：上一阶段结果 + 状态向量
=> key定义：所有的objects特征
"""
class ActionStrategy(nn.Module):
    def __init__(self, hidden_size):
        super(ActionStrategy, self).__init__()

        """Agent：用于选择action"""
        self.query = nn.Linear(hidden_size * 2, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)

        """Critic：用于预测累计奖励"""
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.critic = init_(nn.Linear(hidden_size, 1))  # 用于估计中间步骤的累计奖励
        self.critic_ans = init_(nn.Linear(hidden_size, 1))  # 用于估计最终步骤的奖励

    def forward(self, last_results, state, objects, objects_mask, beta=None):
        """
        :param last_results: [B, num_programs, d_h]
        :param state: [B, num_programs, d_h]
        :param objects: [B, num_objects, d_h]
        :param objects_mask: [B, num_objects], False indicates padding
        :param beta: use softmax if beta is not None,
            use deterministic action choosing if beta is None
        :return:
        """

        """Action：通过state与各个object特征的点积来当做action选择"""
        query = self.query(torch.cat([last_results, state], dim=-1))  # [B, num_programs, d_h]
        key = self.key(objects)  # [B, num_objects, d_h]

        d_h = query.shape[-1]
        logits_raw = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(d_h)  # [B, num_programs, num_objects]
        logits = torch.masked_fill(
            logits_raw,
            mask=~objects_mask.unsqueeze(1),
            value=float("-inf")
        )  # [B, num_programs, N]

        """beta为None的时候，采用强化学习策略
        beta为float的时候，采用softmax策略"""
        if beta is None:
            """Critic：直接通过state预测累计奖励"""
            alpha = torch.softmax(logits, dim=-1)  # [B, num_programs, N]
            cat = Categorical(logits=logits)
            action = cat.sample()  # [B, num_programs]

            """根据action得到当前推理步的中间结果"""
            current_results = torch.gather(
                input=objects,
                dim=1,
                index=action.unsqueeze(-1).expand((-1, -1, d_h))
            )

            output_dict = {
                "action": action,
                "log_prob": cat.log_prob(action),  # [B, num_programs]
                "entropy": cat.entropy(),  # [B, num_programs]
                "value": self.critic(state).squeeze(-1),  # [B, num_programs]
                "current_results": current_results,
                "logits_raw": logits_raw
            }
        else:
            assert isinstance(beta, float)
            alpha = logits * beta
            alpha = torch.softmax(alpha, dim=-1)  # [B, num_programs, N]
            current_results = torch.matmul(alpha, objects)  # [B, num_programs, d_h]

            output_dict = {
                "current_results": current_results,
                "logits_raw": logits_raw
            }

        return output_dict

"""
采用强化学习的action策略替换Module中的GA，使得这种选择过程具有可解释性和可优化性；
新的模块采用如下逻辑：
=> 融合上一层的推理结果；
=> SA汇聚前面的program依赖信息；
=> 根据\pi策略选择当前推理步的推理结果；
"""
class ModuleRL(nn.Module):
    def __init__(self, hidden_size, head_num, ff_size, dropout, hidden_size_head):
        super(ModuleRL, self).__init__()

        # 用于融合上一层信息
        self.mlp = MLP(
            in_size=hidden_size * 2,
            mid_size=hidden_size,
            out_size=hidden_size,
            dropout_r=dropout,
            use_relu=True
        )

        # 用于汇聚前面的program依赖信息
        self.self_attention = SA(
            hidden_size=hidden_size,
            head_num=head_num,
            ff_size=ff_size,
            dropout=dropout,
            hidden_size_head=hidden_size_head
        )

        # 用于实现\pi策略选择action
        self.pi = ActionStrategy(
            hidden_size=hidden_size
        )

    """programs中已经包含了batch_size条eposides，每一条eposides都指定了推理路径，
    现在主要是增加从环境获得反馈信息的模块，以及计算loss所需变量的模块"""
    def forward(self, last_results, last_logits, last_entropy, last_reward, last_value, last_logprob,
                programs, mask, vis_feat, vis_mask_tmp,
                program_masks, alpha, beta=None):
        """
        :param last_results: [B, num_programs, d_h]
        :param last_entropy: [B, num_programs]
        :param last_reward: [B, num_programs]
        :param last_value: [B, num_programs]
        :param last_logprob: [B, num_programs]
        :param programs: [B, num_programs, d_h]
        :param mask: [B, num_programs]
        :param alpha: [B, num_programs]
        :return:
        """

        # 将program融合上一层的结果得到当前的state
        state = self.mlp(
            torch.cat([last_results, programs], dim=-1)
        )

        # 汇聚前面的programs依赖信息
        trans_mask = (1 - mask).unsqueeze(1).to(torch.bool)
        state = self.self_attention(state, trans_mask)

        # 根据\pi策略选择当前的推理结果
        output_dict = self.pi(last_results, state, vis_feat, vis_mask_tmp, beta)
        current_results = output_dict["current_results"]
        logits = output_dict["logits_raw"]

        # 只更新当前推理层的结果
        state = state * program_masks.unsqueeze(-1)
        current_results = current_results * program_masks.unsqueeze(-1)
        logits = logits * program_masks.unsqueeze(-1)

        alpha = alpha.unsqueeze(-1)
        state = alpha * state + (1 - alpha) * programs
        current_results = alpha * current_results + (1 - alpha) * last_results
        logits = alpha * logits + (1 - alpha) * last_logits

        if beta is None:
            """当前为强化学习模式，需要计算和记录一些信息"""
            entropy = output_dict["entropy"]
            entropy = entropy * program_masks
            entropy = alpha.squeeze(-1) * entropy + (1 - alpha.squeeze(-1)) * last_entropy

            """从环境中获取当前推理步的reward，为了简化配置，直接中间步骤reward为0"""
            # TODO: 采用更好的奖励评估策略；
            batch_size, length = last_reward.shape
            reward = torch.zeros((batch_size, length),
                                 dtype=last_reward.dtype).to(last_reward.device)
            reward = reward * program_masks
            reward = alpha.squeeze(-1) * reward + (1 - alpha.squeeze(-1)) * last_reward

            """记录下每一步模型估计的累计奖励"""
            value = output_dict["value"]
            value = value * program_masks
            value = alpha.squeeze(-1) * value + (1 - alpha.squeeze(-1)) * last_value

            """记录下每一步对应action的概率对数"""
            logprob = output_dict["log_prob"]
            logprob = logprob * program_masks
            logprob = alpha.squeeze(-1) * logprob + (1 - alpha.squeeze(-1)) * last_logprob

            return state, current_results, logits, entropy, reward, value, logprob

        return state, current_results, logits

class ProgramTransformer(nn.Module):
    def __init__(self, vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
                 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers,
                 intermediate_layer):
        super(ProgramTransformer, self).__init__()

        self.num_regions = intermediate_dim

        # The question encoder
        self.embedding = nn.Embedding(vocab_size, 300,
                                      padding_idx=Constants.PAD)
        self.ques_proj = nn.Linear(300, hidden_dim)
        self.prog_proj = nn.Linear(300, hidden_dim // 8)

        self.ques_pos_emb = PositionalEmbedding(hidden_dim)
        self.intermediate_layer = intermediate_layer

        # The visual encoder
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
        self.pos_emb = nn.Embedding(self.num_regions, hidden_dim)

        self.ques_encoder = nn.ModuleList(
            [SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])
        self.vis_encoder = nn.ModuleList(
            [SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])

        # The program decoder
        self.idx_predictor = nn.Linear(hidden_dim, self.num_regions)

        # The self attention module beforehand
        self.post = nn.ModuleList([ShallowModule(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
                                    for _ in range(stacking)])

        # The self attention module and cross attention module
        self.module = Module(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)

        # Projection layer to retrieve final answer
        self.proj = nn.Linear(hidden_dim, answer_size)

    def forward(self, ques, ques_masks, program, program_masks, transition_masks,
                activate_masks, vis_feat, box_feat, vis_mask, index, depth):
        batch_size = ques.size(0)
        length = program.size(1)

        idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(ques.device)
        vis_feat = self.vis_proj(vis_feat) + self.coordinate_proj(box_feat) + self.pos_emb(idx)

        vis_mask_tmp = (1 - vis_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
        program_mask_tmp = (1 - program_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
        ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)

        # Encoding the question with self-attention
        ques_input = self.ques_proj(self.embedding(ques)) + self.ques_pos_emb(ques)
        for enc in self.ques_encoder:
            ques_input = enc(ques_input, ques_mask_tmp)
            ques_input *= ques_masks.unsqueeze(-1)  # 每一层都将padding的token置零

        # Encoding the visual feature
        for enc in self.vis_encoder:
            vis_feat = enc(vis_feat, ques_input, vis_mask_tmp, ques_mask_tmp)
            vis_feat *= vis_mask.unsqueeze(-1)  # 每一层都将padding的feat置零

        enc_output = self.prog_proj(self.embedding(program)).view(batch_size, program.size(1), -1) # [B, M, d_h]
        transition_masks = transition_masks.transpose(0, 1)
        activate_masks = activate_masks.transpose(0, 1)

        # Build the structure into the transformer
        for trans_mask, active_mask in zip(transition_masks, activate_masks):
            enc_output = self.module(enc_output, trans_mask, vis_feat, vis_mask_tmp, program_masks, active_mask)

        # Post-Processing the encoder output
        for layer in self.post:
            enc_output = layer(enc_output, vis_feat, vis_mask_tmp, program_masks)

        # Predict the intermediate results
        pre_logits = self.idx_predictor(enc_output)

        # The last program state
        lang_feat = torch.gather(enc_output, 1, index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))
        logits = self.proj(lang_feat.view(batch_size, -1))

        return pre_logits, logits

class ProgramTransformerE2E(nn.Module):
    def __init__(self, vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
                 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers,
                 intermediate_layer, intermediate_size, args):
        super(ProgramTransformerE2E, self).__init__()

        self.grid_model = None # self.load_grid_model(args)

        """编码问题输入"""
        self.embedding = nn.Embedding(vocab_size, 300,
                                      padding_idx=Constants.PAD)
        self.ques_proj = nn.Linear(300, hidden_dim)
        self.prog_proj = nn.Linear(300, hidden_dim // 8)

        self.ques_pos_emb = PositionalEmbedding(hidden_dim, max_len=64)
        self.intermediate_layer = intermediate_layer

        """编码图像输入"""
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
        self.pos_emb = nn.Embedding(args.feat_size ** 2 + 1, hidden_dim)

        """Transformer编码"""
        self.ques_encoder = nn.ModuleList(
            [SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])
        self.vis_encoder = nn.ModuleList(
            [SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])

        """Program推理"""
        self.module = Module(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
        self.post = nn.ModuleList([ShallowModule(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
                                   for _ in range(stacking)])
        self.idx_predictor = nn.Linear(hidden_dim, args.feat_size ** 2)

        """生成答案"""
        self.proj = nn.Linear(hidden_dim, answer_size)

        """用于loss计算"""
        self.dec_emb = nn.Embedding(intermediate_size, hidden_dim)
        initrange = 0.1
        self.dec_emb.weight.data.uniform_(-initrange, initrange)
        self.dec_pos = PositionalEmbedding(hidden_dim, max_len=64)
        decoder_layers = TransformerDecoderLayer(hidden_dim, n_head, 4 * hidden_dim, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 2)
        self.tgt_mask = nn.Parameter(
            data=torch.triu(torch.ones(args.intermediate_num * 4 + 1,
                                       args.intermediate_num * 4 + 1) * float('-inf'), diagonal=1),
            requires_grad=False
        )
        self.dec_out = nn.Linear(hidden_dim, intermediate_size)
        self.dec_out.bias.data.zero_()
        self.dec_out.weight.data.uniform_(-initrange, initrange)
        self.dec_ce = nn.CrossEntropyLoss(ignore_index=-1)

    def load_grid_model(self, args):
        from detectron2.config import get_cfg
        from detectron2.engine import default_setup
        from grid.grid_feats.config import add_attribute_config
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer

        """配置参数"""
        cfg = get_cfg()
        add_attribute_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.MODEL.RESNETS.RES5_DILATION = 1
        cfg.freeze()
        default_setup(cfg, args)
        self.cfg = cfg

        """载入checkpoint"""
        model = build_model(cfg)
        model = model.eval()
        DetectionCheckpointer(model, save_dir="tmp").resume_or_load(
            args.grid_ckpt, resume=True
        )
        # TODO: 这里先简化图像特征生成模型的优化
        model: nn.Module
        for param in model.parameters():
            param.requires_grad = False
        print("size_divisibility: {}".format(model.backbone.size_divisibility))

        return model

    def forward(self, feature, question, question_masks, program, program_masks,
                transition_masks, activate_mask, index, intermediate_idx, **kwargs):
        """
        :param image: [B, 3, H, W], H和W每次大小不一样
        :param intermediate_idx: [num_questions, 9, 1 + 25 + 1]
        :return:
        """
        vis_feat = feature
        batch_size = vis_feat.shape[0]

        """编码文本"""
        ques_mask_tmp = (1 - question_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
        ques_input = self.ques_proj(self.embedding(question)) + self.ques_pos_emb(question)
        for enc in self.ques_encoder:
            ques_input = enc(ques_input, ques_mask_tmp)
            ques_input *= question_masks.unsqueeze(-1)

        """编码图像"""
        idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(vis_feat.device)
        vis_feat = self.vis_proj(vis_feat) + self.pos_emb(idx)
        for enc in self.vis_encoder:
            vis_feat = enc(vis_feat, ques_input, None, ques_mask_tmp)

        """编码program"""
        batch_size = vis_feat.shape[0]
        enc_output = self.prog_proj(self.embedding(program)).view(batch_size, program.size(1), -1)  # [B, num_program, hidden_dim]
        transition_masks = transition_masks.transpose(0, 1)
        activate_masks = activate_mask.transpose(0, 1)
        for trans_mask, active_mask in zip(transition_masks, activate_masks):
            enc_output = self.module(enc_output, trans_mask, vis_feat, None, program_masks, active_mask)
        for layer in self.post:
            enc_output = layer(enc_output, vis_feat, None, program_masks)

        # The last program state
        lang_feat = torch.gather(enc_output, 1,
                                 index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))
        logits = self.proj(lang_feat.view(batch_size, -1))

        # Predict the intermediate results
        pre_logits = self.idx_predictor(enc_output)  # [num_questions, 9, feat_dim ** 2]
        if self.training:
            tgt = self.dec_emb(torch.clamp(intermediate_idx[:, :, 1:-1], min=0))  # [B, 9, 15, 512]
            b, np, nt, d = tgt.shape
            tgt = torch.cat([
                enc_output.reshape(b * np, 1, d),
                tgt.view(b * np, nt, d)
            ], dim=1)  # [B * 9, 1 + 25, 512]
            tgt = tgt + self.dec_pos(tgt)
            out = self.dec_out(self.transformer_decoder(tgt,
                                                        vis_feat.unsqueeze(1).expand((-1, np, -1, -1)).flatten(0, 1),
                                                        tgt_mask=self.tgt_mask))  # [B * 9, 1 + 25, num_cls]
            loss_pre = self.dec_ce(out.view(-1, out.shape[-1]),
                                   intermediate_idx[:, :, 1:].reshape(-1, ))
        else:
            loss_pre = 0.0

        if self.training:
            return pre_logits, logits, loss_pre
        else:
            return (enc_output, vis_feat), logits, loss_pre

    def cal_module_loss(self, pre_logits, intermediate_idx):
        """
        :param pre_logits: [N, 9, feat_dim ** 2]
        :param intermediate_idx: [N, 9, 4]
        :return:
        """
        b, n, f = pre_logits.shape

        grid_pt = self.mesh_grid.expand((b, n, -1, -1))  # [N, 9, feat_dim ** 2, 2]
        gt_pt = torch.stack([
            (intermediate_idx[..., 0] + intermediate_idx[..., 2]) / 2,
            (intermediate_idx[..., 1] + intermediate_idx[..., 3]) / 2
        ], dim=-1).unsqueeze(2)  # [N, 9, 1, 2]

        # [N, 9, feat_dim ** 2]
        gt_dist = ((grid_pt[..., 0] - gt_pt[..., 0]) ** 2 + (grid_pt[..., 1] - gt_pt[..., 1]) ** 2)
        gt_dist = 1 / torch.exp(gt_dist)

        # x1_inter = torch.max(torch.stack([grid_bbox[..., 0], gt_bbox[..., 0]], dim=-1), dim=-1)[0]
        # y1_inter = torch.max(torch.stack([grid_bbox[..., 1], gt_bbox[..., 1]], dim=-1), dim=-1)[0]
        # x2_inter = torch.min(torch.stack([grid_bbox[..., 2], gt_bbox[..., 2]], dim=-1), dim=-1)[0]
        # y2_inter = torch.min(torch.stack([grid_bbox[..., 3], gt_bbox[..., 3]], dim=-1), dim=-1)[0]
        #
        # # [N, 9, feat_dim ** 2]
        # intersect_area = torch.clamp(x2_inter - x1_inter, min=0.0) * \
        #                  torch.clamp(y2_inter - y1_inter, min=0.0)
        # grid_area = (grid_bbox[..., 2] - grid_bbox[..., 0]) * (grid_bbox[..., 3] - grid_bbox[..., 1])
        # gt_area = (gt_bbox[..., 2] - gt_bbox[..., 0]) * (gt_bbox[..., 3] - gt_bbox[..., 1])

        KL_loss = KLDivergence()
        # gt_dist = intersect_area / (grid_area + gt_area - intersect_area + 1e-4)  # [N, 9, feat_dim ** 2]
        gt_dist /= (gt_dist.sum(-1, keepdim=True) + 1e-4)  # Normalized to distribution

        loss_pre = KL_loss(gt_dist.view((-1, f)), pre_logits.view((-1, f)))

        return loss_pre

class ProgramTransformerRL(nn.Module):
    def __init__(self, vocab_size, answer_size, visual_dim, hidden_dim, coordinate_dim,
                 n_head, n_layers, stacking, dropout, intermediate_dim, pre_layers,
                 intermediate_layer, answer_matrix):
        super(ProgramTransformerRL, self).__init__()

        self.num_regions = intermediate_dim

        # The question encoder
        self.embedding = nn.Embedding(vocab_size, 300,
                                      padding_idx=Constants.PAD)
        self.ques_proj = nn.Linear(300, hidden_dim)
        self.prog_proj = nn.Linear(300, hidden_dim // 8)

        self.ques_pos_emb = PositionalEmbedding(hidden_dim)
        self.intermediate_layer = intermediate_layer

        # The visual encoder
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.coordinate_proj = nn.Linear(coordinate_dim, hidden_dim)
        self.pos_emb = nn.Embedding(self.num_regions, hidden_dim)

        self.ques_encoder = nn.ModuleList(
            [SA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])
        self.vis_encoder = nn.ModuleList(
            [SGA(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
             for _ in range(pre_layers)])

        # The program decoder
        self.idx_predictor = nn.Linear(hidden_dim, self.num_regions)
        self.empty_region = nn.Parameter(
            data=torch.randn((1, 1, hidden_dim)),
            requires_grad=True
        )

        # The self attention module beforehand
        self.post = nn.ModuleList([ShallowModule(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)
                                    for _ in range(stacking)])

        # The self attention module and cross attention module
        self.module = ModuleRL(hidden_dim, n_head, 4 * hidden_dim, dropout, hidden_dim // n_head)

        # Projection layer to retrieve final answer
        self.proj = nn.Linear(hidden_dim, answer_size)

        self.ans_idx = nn.Parameter(
            data=answer_matrix,
            requires_grad=False
        )
        self.ans_emb = nn.Linear(300 * answer_matrix.shape[-1], hidden_dim)

    def forward(self, ques, ques_masks, program, program_masks, transition_masks,
                activate_masks, vis_feat, box_feat, vis_mask, index, beta):
        batch_size = ques.size(0)
        length = program.size(1)

        idx = torch.arange(vis_feat.size(1)).unsqueeze(0).repeat(batch_size, 1).to(ques.device)
        vis_feat = self.vis_proj(vis_feat) + self.coordinate_proj(box_feat) + self.pos_emb(idx)

        # 增加一个空区域，用于表示无对应的区域
        vis_feat = torch.cat([vis_feat, self.empty_region.expand((batch_size, -1, -1))], dim=1)
        vis_mask = torch.cat([vis_mask, vis_mask[:, :1] * 0 + 1], dim=1)

        vis_mask_tmp = (1 - vis_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
        program_mask_tmp = (1 - program_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)
        ques_mask_tmp = (1 - ques_masks).unsqueeze(1).unsqueeze(2).to(torch.bool)

        # Encoding the question with self-attention
        """编码文本的语义"""
        ques_input = self.ques_proj(self.embedding(ques)) + self.ques_pos_emb(ques)
        for enc in self.ques_encoder:
            ques_input = enc(ques_input, ques_mask_tmp)
            ques_input *= ques_masks.unsqueeze(-1)  # 每一层都将padding的token置零

        # Encoding the visual feature
        """从文本中获取信息，并编码图像的特征"""
        for enc in self.vis_encoder:
            vis_feat = enc(vis_feat, ques_input, vis_mask_tmp, ques_mask_tmp)
            vis_feat *= vis_mask.unsqueeze(-1)  # 每一层都将padding的feat置零

        enc_output = self.prog_proj(self.embedding(program)).view(batch_size, program.size(1), -1) # [B, M, d_h]
        transition_masks = transition_masks.transpose(0, 1)
        activate_masks = activate_masks.transpose(0, 1)

        # Use RL for visual reasoning
        pre_logits = torch.zeros((batch_size, length, vis_feat.shape[1]),
                                 dtype=enc_output.dtype).to(enc_output.device)
        last_entropy = torch.zeros((batch_size, length), dtype=enc_output.dtype).to(enc_output.device)
        last_reward = torch.zeros((batch_size, length), dtype=enc_output.dtype).to(enc_output.device)
        last_value = torch.zeros((batch_size, length), dtype=enc_output.dtype).to(enc_output.device)
        last_logprob = torch.zeros((batch_size, length), dtype=enc_output.dtype).to(enc_output.device)
        last_results = torch.mean(vis_feat, dim=1, keepdim=True).expand((-1, length, -1))  # [B, num_programs, d_h]
        """基于program语义和结构，实现语义推理"""
        for trans_mask, active_mask in zip(transition_masks, activate_masks):
            if beta is not None:
                """softmax模式"""
                enc_output, last_results, pre_logits = self.module(
                    last_results, pre_logits, last_entropy, last_reward, last_value, last_logprob,
                    enc_output, trans_mask, vis_feat,
                    vis_mask.to(bool), program_masks, active_mask, beta)
            else:
                """RL模式"""
                enc_output, last_results, pre_logits, last_entropy, last_reward, last_value, last_logprob = self.module(
                    last_results, pre_logits, last_entropy, last_reward, last_value, last_logprob,
                    enc_output, trans_mask, vis_feat,
                    vis_mask.to(bool), program_masks, active_mask, beta)

        # Post-Processing the encoder output
        """最后从图像特征中获取所需的信息"""
        for layer in self.post:
            enc_output = layer(enc_output, vis_feat, vis_mask_tmp, program_masks)

        # Predict the intermediate results
        # pre_logits = self.idx_predictor(enc_output)

        # The last program state
        lang_feat = torch.gather(
            input=enc_output,
            dim=1,
            index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, enc_output.size(-1)))  # [B, 1, d_h]
        logits = self.proj(lang_feat.view(batch_size, -1))

        if beta is None:
            """如果是强化学习模式，那么估计出最终一步的reward"""
            pred = torch.argmax(logits, dim=-1)  # [B, ]
            answer_emb = self.ans_emb(
                self.embedding(self.ans_idx[pred]).view(batch_size, -1)
            )  # [B, hidden_dim]
            final_value = self.module.pi.critic_ans(lang_feat.squeeze(1) * answer_emb)  # [B, 1]
            last_value = torch.scatter(
                input=last_value,  # [B, num_programs]
                dim=1,
                index=index.unsqueeze(-1),  # [B, 1]
                src=final_value  # [B, 1]
            )

            return pre_logits, logits, last_entropy, last_reward, last_value, last_logprob

        return pre_logits, logits

    def calculate_loss(self, pre_logits, pre_label, logits, label, pre_weight,
                       last_entropy=None, last_reward=None, last_value=None, last_logprob=None,
                       program_masks=None, rl_weight=1.0, use_rl=False):
        cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        KL_loss = KLDivergence()

        length = pre_logits.size(-1)
        pre_loss = KL_loss(pre_label.view(-1, length),
                           pre_logits.view(-1, length))
        pred_loss = cross_entropy(logits, label)

        if pre_weight == 0.0:
            loss_vqa = pred_loss
        else:
            loss_vqa = pred_loss + pre_weight * pre_loss

        if not use_rl:
            return {
                "loss": loss_vqa,
                "pred_loss": pred_loss,
                "pre_loss": pre_loss
            }
        else:
            """根据last_reward计算环境给出的累计奖励"""
            discount = 0.9
            length = last_reward.shape[-1]
            i = -2
            while i >= -length:
                last_reward[:, i] = last_reward[:, i] + discount * last_reward[:, i + 1]
                i -= 1
            advantage = last_reward - last_value

            """优化agent loss"""
            policy_loss = ((- last_logprob * advantage.detach()).sum(-1) /
                           torch.clamp(program_masks.sum(-1), min=1.0)).mean()

            """优化critic loss"""
            value_loss = (advantage.pow(2).sum(-1) /
                          torch.clamp(program_masks.sum(-1), min=1.0)).mean()

            """优化action选择的熵"""
            entropy = last_entropy.sum(-1).mean()

            loss_rl = policy_loss + 0.5 * value_loss + 0.01 * entropy

            return {
                "loss": loss_vqa + rl_weight * loss_rl,
                "loss_vqa": loss_vqa,
                "pred_loss": pred_loss,
                "pre_loss": pre_loss,
                "loss_rl": loss_rl,
                "loss_policy": policy_loss,
                "loss_value": value_loss,
                "loss_entropy": entropy
            }
