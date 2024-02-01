# -*- coding: utf-8 -*-

"""
考虑采用文本生成来实现视觉问答，这样的话，在文本生成过程中可以嵌入explainable text；
"""

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import T5ForConditionalGeneration

from model.modules import SGA, AttFlat
from model.layers import PositionalEmbedding
from scripts import Constants

class T5ForConditionalGenerationWithImageFeature(T5ForConditionalGeneration):
    def __init__(self, config):
        super(T5ForConditionalGenerationWithImageFeature, self).__init__(config)

    """这里的inputs分成两个部分，一个部分是encoder的输入ids，
    另一个部分是编码得到的图像特征"""
    def _prepare_model_inputs(
        self,
        inputs = None,
        bos_token_id = None,
        model_kwargs = None,
    ):
        inputs, vis_feats = inputs
        if (
            self.config.is_encoder_decoder
            and hasattr(self, "encoder")
            and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed."
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 3. In the presence of `inputs_embeds` for text models:
        # - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
        # doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
        # input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
        # - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
        # pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(
                    inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
                )
                if not has_inputs_embeds_forwarding:
                    raise ValueError(
                        f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                        "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                        "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                    )
                # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
                # the attention mask) can rely on the actual model input.
                model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                    inputs, bos_token_id, batch_size=model_kwargs["inputs_embeds"].shape[0]
                )
            else:
                if inputs is not None:
                    raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(
            inputs, bos_token_id, model_kwargs.get("encoder_outputs")
        )

        """将图像特征放到model_kwargs"""
        model_kwargs["vis_feats"] = vis_feats
        return inputs, input_name, model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name = None
    ):
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        encoder_outputs = encoder(**encoder_kwargs)

        """将图像特征融合到encoder_outputs中"""
        encoder_outputs.last_hidden_state = torch.cat([
                model_kwargs["vis_feats"], encoder_outputs.last_hidden_state
            ], dim=1)
        model_kwargs["encoder_outputs"] = encoder_outputs

        return model_kwargs

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id,
        eos_token_id,
    ) -> torch.LongTensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            batch_size = inputs.shape[0]
            image_mask = torch.ones((batch_size, 10 ** 2), device=inputs.device, dtype=torch.long)

            return torch.cat([
                image_mask, inputs.ne(pad_token_id).long()
            ], dim=1)
        else:
            batch_size = inputs.shape[0]
            mask = torch.ones((batch_size, 10 ** 2 + inputs.shape[1]),
                              device=inputs.device, dtype=torch.long)

            return mask

class VQAGenT5(nn.Module):
    def __init__(self, t5_dir: str, args, answer_size, **kwargs):
        super(VQAGenT5, self).__init__()
        self.args = args
        self.answer_size = answer_size

        """T5模型作为文本编码模型，包含文本编码+文本解码模型"""
        self.text_module = T5ForConditionalGenerationWithImageFeature.from_pretrained(t5_dir)

        """堆叠N层Transformer作为视觉模型"""
        self.vis_proj = nn.Linear(args.visual_dim, args.hidden_dim)
        self.pos_emb = nn.Embedding(args.feat_size ** 2, args.hidden_dim)
        self.vis_enc = nn.ModuleList([
            SGA(args.hidden_dim, args.n_head, 4 * args.hidden_dim, args.dropout, args.hidden_dim // args.n_head)
            for _ in range(args.vis_layers)
        ])

        """答案分类层"""
        self.flat_img = AttFlat(args.hidden_dim, args.hidden_dim * 2, args.dropout)
        self.flat_que = AttFlat(args.hidden_dim, args.hidden_dim * 2, args.dropout)
        self.proj_norm = nn.LayerNorm(args.hidden_dim * 2)
        self.proj = nn.Linear(args.hidden_dim * 2, answer_size)
        self.bce = nn.BCELoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

        """答案生成层"""
        self.dec_emb = nn.Embedding(kwargs["vocab_size"], args.hidden_dim)
        initrange = 0.1
        self.dec_emb.weight.data.uniform_(-initrange, initrange)
        self.dec_pos = PositionalEmbedding(args.hidden_dim, max_len=128)
        decoder_layers = TransformerDecoderLayer(args.hidden_dim, args.n_head, 4 * args.hidden_dim,
                                                 args.dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 3)
        self.tgt_mask = nn.Parameter(
            data=torch.triu(torch.ones(47, 47) * float('-inf'), diagonal=1),
            requires_grad=False
        )
        self.dec_out = nn.Linear(args.hidden_dim, kwargs["vocab_size"])
        self.dec_out.bias.data.zero_()
        self.dec_out.weight.data.uniform_(-initrange, initrange)
        self.dec_ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.feat_weights = nn.Parameter(
            data=torch.randn((5, args.vis_layers)),
            requires_grad=True
        )
        self.starts = nn.Parameter(
            data=torch.randn((5, args.hidden_dim)),
            requires_grad=True
        )

    def forward(self, input_ids, input_mask, image_feature, answer_id, answer_tokens, **kwargs):
        """
        :param input_ids: [B, seq_length]
        :param input_mask: [B, seq_length], 1 indicates valid token, 0 indicates padding
        :param image_feature: [B, h * w, C]
        :param answer_id: [B, ]
        :param answer_tokens: [B, max_len]
        :return: loss
        """

        """编码文本"""
        text_outputs = self.text_module.encoder(
            input_ids=input_ids,
            attention_mask=input_mask,  # [batch_size, key_length], 1 indicates valid token
            inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
        )  # text_outputs.last_hidden_state: [batch_size, seq_length, hidden_size]

        """编码图像特征"""
        batch_size, num_grids = image_feature.shape[:2]
        idx = torch.arange(num_grids).unsqueeze(0).repeat(batch_size, 1).to(image_feature.device)
        vis_feat = self.vis_proj(image_feature) + self.pos_emb(idx)
        ques_mask = (1 - input_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
        all_vis_feats = []
        for enc in self.vis_enc:
            vis_feat = enc(vis_feat, text_outputs.last_hidden_state, None, ques_mask)
            all_vis_feats.append(vis_feat)
        all_vis_feats = torch.stack(all_vis_feats, dim=-1)  # [B, N, hidden_dim, 3]

        """答案多分类"""
        t_flat = self.flat_que(text_outputs.last_hidden_state, ques_mask)
        v_flat = self.flat_img(vis_feat, None)
        proj_feat = self.proj_norm(t_flat + v_flat)
        proj_feat = self.proj(proj_feat)
        proj_feat = torch.sigmoid(proj_feat)
        # loss_cls = self.ce(proj_feat, answer_id)
        # print(proj_feat.shape)
        # print(F.one_hot(answer_id, self.answer_size).to(proj_feat.dtype).shape)

        loss_cls = self.bce(proj_feat, F.one_hot(answer_id, self.answer_size).to(proj_feat.dtype))

        """答案生成"""
        if self.training:
            loss_gen = 0.0
            feat_weights = torch.softmax(self.feat_weights, dim=-1)  # [5, 3]
            for i in range(5):
                gen = answer_tokens[:, i]  # [B, 48]
                tgt = self.dec_emb(gen[:, 1:-1])  # [B, 46, 512]
                b, n, d = tgt.shape
                vis_feat = (feat_weights[i] * all_vis_feats).sum(-1)
                tgt = torch.cat([
                    self.starts[i].unsqueeze(0).unsqueeze(0).expand((b, -1, -1)),
                    tgt
                ], dim=1)  # [B, 47, 512]

                tgt = tgt + self.dec_pos(tgt)
                out = self.dec_out(self.transformer_decoder(tgt,
                                                            vis_feat,
                                                            tgt_mask=self.tgt_mask))
                label = torch.masked_fill(
                    gen,
                    mask=gen == Constants.PAD,
                    value=-1
                )
                loss_gen = loss_gen + self.dec_ce(out.view(-1, out.shape[-1]),
                                                  label[:, 1:].reshape(-1, ))
                # decoder_answer_tokens = self.text_module._shift_right(gen)
                # image_mask = torch.ones(vis_feat.shape[:2],
                #                         device=vis_feat.device)
                # encoder_hidden_states = torch.cat([
                #     vis_feat, text_outputs.last_hidden_state
                # ], dim=1)  # [batch_size, num_objects + seq_length, hidden_size]
                # encoder_attention_mask = torch.cat([
                #     image_mask, input_mask
                # ], dim=1)  # [batch_size, num_objects + seq_length]
                # decoder_outputs = self.text_module.decoder(
                #     input_ids=decoder_answer_tokens,
                #     attention_mask=None,
                #     inputs_embeds=None,
                #     past_key_values=None,
                #     encoder_hidden_states=encoder_hidden_states,
                #     encoder_attention_mask=encoder_attention_mask,
                #     head_mask=None,
                #     cross_attn_head_mask=None,
                #     use_cache=None,
                #     output_attentions=None,
                #     output_hidden_states=None,
                #     return_dict=None
                # )  # [B, answer_length, hidden_dim]
                # sequence_output = decoder_outputs[0]
                # if self.text_module.config.tie_word_embeddings:
                #     sequence_output = sequence_output * (self.text_module.model_dim ** -0.5)
                # lm_logits = self.text_module.lm_head(sequence_output)  # [B, answer_length, vocab_length]
                #
                # loss_gen = loss_gen + self.ce(lm_logits.view(-1, lm_logits.size(-1)),
                #                               gen.reshape((-1, )))  # [B * max_length, ]
                # loss_gen = (loss_gen * kwargs["label_weights"].view(-1)).sum() / (answer_tokens.view(-1) != 0).float().sum()
            loss_gen /= 5
        else:
            loss_gen = 0.0

        return {
            "logits": proj_feat,
            "loss_cls": loss_cls,
            "loss_gen": loss_gen
        }

    @torch.no_grad()
    def generate(self, input_ids,
                 input_mask,
                 image_feature,
                 **kwargs):
        self.eval()

        """编码文本"""
        text_outputs = self.text_module.encoder(
            input_ids=input_ids,
            attention_mask=input_mask,  # [batch_size, key_length], 1 indicates valid token
            inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
        )

        """编码图像特征"""
        batch_size, num_grids = image_feature.shape[:2]
        idx = torch.arange(num_grids).unsqueeze(0).repeat(batch_size, 1).to(image_feature.device)
        vis_feat = self.vis_proj(image_feature) + self.pos_emb(idx)
        ques_mask = (1 - input_mask).unsqueeze(1).unsqueeze(2).to(torch.bool)
        for enc in self.vis_enc:
            vis_feat = enc(vis_feat, text_outputs.last_hidden_state, None, ques_mask)

        """生成答案"""
        decoder_output = self.text_module.generate(
            inputs=(input_ids, vis_feat),
            attention_mask=input_mask
        )

        return decoder_output