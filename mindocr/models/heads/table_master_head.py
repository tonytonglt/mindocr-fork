# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/mmocr/models/textrecog/decoders/master_decoder.py
"""
from typing import Optional, Tuple
import numpy as np
import copy
import math
import mindspore as ms
from mindspore import nn, ops, Tensor


class TableMasterHead(nn.Cell):
    """
    Split to two transformer header at the last layer.
    Cls_layer is used to structure token classification.
    Bbox_layer is used to regress bbox coord.
    """

    def __init__(self,
                 in_channels,
                 out_channels=43,
                 headers=8,
                 d_ff=2048,
                 dropout=0.,
                 max_text_length=500,
                 loc_reg_num=4,
                 share_parameter=False,
                 stacks=3,
                 **kwargs):
        super(TableMasterHead, self).__init__()
        hidden_size = in_channels
        # self.layers = clones(
        #     DecoderLayer(headers, hidden_size, dropout, d_ff), 2)
        self.cls_layer = DecoderLayer(headers, hidden_size, dropout, d_ff)
        self.bbox_layer = DecoderLayer(headers, hidden_size, dropout, d_ff)
        self.cls_fc = nn.Dense(hidden_size, out_channels)
        self.bbox_fc = nn.SequentialCell(
            # nn.Linear(hidden_size, hidden_size),
            nn.Dense(hidden_size, loc_reg_num),
            nn.Sigmoid())

        # self.embedding = Embeddings(d_model=hidden_size, vocab=out_channels)
        # self.positional_encoding = PositionalEncoding(d_model=hidden_size)


        self.out_channels = out_channels
        self.loc_reg_num = loc_reg_num
        self.max_text_length = max_text_length

        self.share_parameter = share_parameter

        self.attention = nn.CellList(
            [
                MultiHeadAttention(headers, in_channels, dropout)
                for _ in range(1 if share_parameter else stacks)
            ]
        )

        self.source_attention = nn.CellList(
            [
                MultiHeadAttention(headers, in_channels, dropout)
                for _ in range(1 if share_parameter else stacks)
            ]
        )

        self.position_feed_forward = nn.CellList(
            [
                PositionwiseFeedForward(in_channels, d_ff, dropout)
                for _ in range(1 if share_parameter else stacks)
            ]
        )

        self.position = PositionalEncoding(in_channels, dropout)
        self.stacks = stacks
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm([hidden_size])
        self.layer_norm2 = nn.LayerNorm([hidden_size])
        self.layer_norm3 = nn.LayerNorm([hidden_size])
        self.layer_norm4 = nn.LayerNorm([hidden_size])
        self.embedding = nn.Embedding(out_channels, in_channels)
        self.sqrt_model_size = np.sqrt(in_channels)
        self.SOS = out_channels - 3
        self.PAD = out_channels - 1

        self.tril = ops.tril
        self.argmax = ops.Argmax(axis=2)

    def make_mask(self, targets):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        target_pad_mask = targets != self.PAD
        target_pad_mask = target_pad_mask[:, None, :, None]
        target_pad_mask = ops.cast(target_pad_mask, ms.int32)
        target_length = targets.shape[1]
        target_sub_mask = self.tril(ops.ones((target_length, target_length), ms.int32))
        target_mask = ops.bitwise_and(target_pad_mask, target_sub_mask)
        return target_mask

    def decode(self, feature, targets, src_mask=None, tgt_mask=None):
        # main process of transformer decoder.
        # x = self.embedding(input)  # x: 1*x*512, feature: 1*3600,512
        # x = self.positional_encoding(x)
        #
        # # origin transformer layers
        # for i, layer in enumerate(self.layers):
        #     x = layer(x, feature, src_mask, tgt_mask)
        #
        # # cls head
        # cls_x = self.cls_layer(x, feature, src_mask, tgt_mask)
        # cls_x = self.norm(cls_x)
        #
        # # bbox head
        # bbox_x = self.bbox_layer(x, feature, src_mask, tgt_mask)
        # bbox_x = self.norm(bbox_x)
        # return self.cls_fc(cls_x), self.bbox_fc(bbox_x)
        targets = self.embedding(targets) * self.sqrt_model_size
        targets = self.position(targets)
        output = targets
        for i in range(self.stacks):
            if self.share_parameter:
                actual_i = i
            else:
                actual_i = 0

            normed_output = self.layer_norm1(output)
            output = output + self.dropout(
                self.attention[actual_i](
                    normed_output, normed_output, normed_output, tgt_mask
                )
            )
            normed_output = self.layer_norm2(output)
            output = output + self.dropout(
                self.source_attention[actual_i](
                    normed_output, feature, feature, src_mask
                )
            )
            normed_output = self.layer_norm3(output)
            output = output + self.dropout(
                self.position_feed_forward[actual_i](normed_output)
            )

            # cls head
            cls_x = self.cls_layer(output, feature, src_mask, tgt_mask)
            cls_x = self.layer_norm4(cls_x)

            # bbox head
            bbox_x = self.bbox_layer(output, feature, src_mask, tgt_mask)
            bbox_x = self.layer_norm4(bbox_x)
            return self.cls_fc(cls_x), self.bbox_fc(bbox_x)

    # def greedy_forward(self, SOS, feature):
    #     input = SOS
    #     output = ops.zeros(
    #         [input.shape[0], self.max_text_length + 1, self.out_channels])
    #     bbox_output = ops.zeros(
    #         [input.shape[0], self.max_text_length + 1, self.loc_reg_num])
    #     # max_text_length = Tensor(self.max_text_length)
    #     for i in range(self.max_text_length + 1):
    #         target_mask = self.make_mask(input)
    #         out_step, bbox_output_step = self.decode(input, feature, None,
    #                                                  target_mask)
    #         prob = ops.softmax(out_step, axis=-1)
    #         next_word = prob.argmax(axis=2)
    #         next_word = ops.cast(next_word, ms.int32)
    #         input = ops.concat(
    #             [input, next_word[:, -1].unsqueeze(-1)], axis=1)
    #         if i == self.max_text_length:
    #             output = out_step
    #             bbox_output = bbox_output_step
    #     return output, bbox_output
    #
    # def forward_train(self, out_enc, targets):
    #     # x is token of label
    #     # feat is feature after backbone before pe.
    #     # out_enc is feature after pe.
    #     padded_targets = targets[0]
    #     src_mask = None
    #     tgt_mask = self.make_mask(padded_targets[:, :-1])
    #     output, bbox_output = self.decode(padded_targets[:, :-1], out_enc,
    #                                       src_mask, tgt_mask)
    #     return {'structure_probs': output, 'loc_preds': bbox_output}
    #
    # def forward_test(self, out_enc):
    #     batch_size = out_enc.shape[0]
    #     SOS = ops.zeros([batch_size, 1], dtype=ms.int32) + self.SOS
    #     output, bbox_output = self.greedy_forward(SOS, out_enc)
    #     output = ops.softmax(output)
    #     # return {'structure_probs': output, 'loc_preds': bbox_output}
    #     return (output, bbox_output)

    def construct(self, feat, targets=None):
        # feat = feat[-1]
        # b, c, h, w = feat.shape
        # feat = feat.reshape([b, c, h * w])  # flatten 2D feature map
        # feat = feat.transpose((0, 2, 1))
        # out_enc = self.positional_encoding(feat)
        # if self.training:
        #     return self.forward_train(out_enc, targets)
        #
        # return self.forward_test(out_enc)
        N = feat.shape[0]
        num_steps = self.max_text_length + 1
        b, c, h, w = feat.shape
        feat = feat.reshape([b, c, h * w])  # flatten 2D feature map
        feat = feat.transpose((0, 2, 1))
        out_enc = self.position(feat)
        if targets is not None:
            # training branch
            targets = targets[0]
            targets = targets[:, :-1]
            target_mask = self.make_mask(targets)
            output, bbox_output = self.decode(out_enc, targets, tgt_mask=target_mask)
            return output, bbox_output
        else:
            input = ops.zeros((N, 1), ms.int32) + self.SOS
            output = ops.zeros(
                [input.shape[0], self.max_text_length + 1, self.out_channels])
            bbox_output = ops.zeros(
                [input.shape[0], self.max_text_length + 1, self.loc_reg_num])
            # probs = list()
            for i in range(num_steps):
                target_mask = self.make_mask(input)
                out_step, bbox_output_step = self.decode(out_enc, input, tgt_mask=target_mask)
                prob = ops.softmax(out_step, axis=-1)
                next_word = self.argmax(prob)
                input = ops.concat(
                    [input, next_word[:, -1].unsqueeze(-1)], axis=1)
                # probs.append(probs_step[:, i])
                if i == self.max_text_length:
                    output = out_step
                    bbox_output = bbox_output_step
            # probs = ops.stack(probs, axis=1)
            output = ops.softmax(output, axis=-1)
        return output, bbox_output


class DecoderLayer(nn.Cell):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """

    def __init__(self, headers, d_model, dropout, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(headers, d_model, dropout)
        self.src_attn = MultiHeadAttention(headers, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # self.sublayer = clones(SubLayerConnection(d_model, dropout), 3)
        self.norm1 = nn.LayerNorm([d_model])
        self.dropout1 = nn.Dropout(p=dropout)
        self.norm2 = nn.LayerNorm([d_model])
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm3 = nn.LayerNorm([d_model])
        self.dropout3 = nn.Dropout(p=dropout)

    def construct(self, x, feature, src_mask, tgt_mask):
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # x = self.sublayer[1](
        #     x, lambda x: self.src_attn(x, feature, feature, src_mask))
        # return self.sublayer[2](x, self.feed_forward)
        normed_x = self.norm1(x)
        x = x + self.dropout1(self.self_attn(normed_x, normed_x, normed_x, tgt_mask))
        normed_x = self.norm2(x)
        x = x + self.dropout2(self.src_attn(normed_x, feature, feature, src_mask))
        normed_x = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(normed_x))
        return x


class MultiHeadAttention(nn.Cell):
    def __init__(
        self, multi_attention_heads: int, dimensions: int, dropout: float = 0.1
    ) -> None:
        """ """
        super(MultiHeadAttention, self).__init__()

        assert dimensions % multi_attention_heads == 0
        # requires d_v = d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(dimensions / multi_attention_heads)
        self.h = multi_attention_heads
        self.linears = nn.CellList([nn.Dense(dimensions, dimensions) for _ in range(4)])
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

        self.matmul = ops.BatchMatMul()

    def dot_product_attention(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:

        d_k = float(value.shape[-1])
        d_k_sqrt = ops.cast(ms.numpy.sqrt(d_k), query.dtype)
        score = self.matmul(query, key.transpose(0, 1, 3, 2)) / d_k_sqrt  # (N, h, seq_len, seq_len)

        if mask is not None:
            score = ops.masked_fill(
                score, mask == 0, -np.inf
            )  # score (N, h, seq_len, seq_len)

        p_attn = ops.softmax(score, axis=-1)
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        return self.matmul(p_attn, value), p_attn

    def construct(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        N = query.shape[0]

        # do all the linear projections in batch from d_model => h x d_k
        # (N, seq_len, d_m) -> (N, seq_len, h, d_k) -> (N, h, seq_len, d_k)
        query = (
            self.linears[0](query)
            .reshape(N, -1, self.h, self.d_k)
            .transpose(0, 2, 1, 3)
        )
        # print(self.linears[1])
        # print(key.shape)
        key = (
            self.linears[1](key).reshape(N, -1, self.h, self.d_k).transpose(0, 2, 1, 3)
        )
        value = (
            self.linears[2](value)
            .reshape(N, -1, self.h, self.d_k)
            .transpose(0, 2, 1, 3)
        )

        # apply attention on all the projected vectors in batch.
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        product_and_attention = self.dot_product_attention(query, key, value, mask=mask)
        x = product_and_attention[0]

        # "Concat" using a view and apply a final linear.
        # (N, seq_len, d_m)
        x = x.transpose(0, 2, 1, 3).reshape(N, -1, self.h * self.d_k)

        # (N, seq_len, d_m)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Cell):
    def __init__(
        self, dimensions: int, feed_forward_dimensions: int, dropout: float = 0.1
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Dense(dimensions, feed_forward_dimensions)
        self.w_2 = nn.Dense(feed_forward_dimensions, dimensions)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, input_tensor: Tensor) -> Tensor:
        return self.w_2(self.dropout(ops.relu(self.w_1(input_tensor))))


class SubLayerConnection(nn.Cell):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm([size])
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))




def clones(module, N):
    """ Produce N identical layers """
    return nn.CellList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Cell):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.sqrt_d_model = ops.sqrt(Tensor(d_model, ms.float32))

    def construct(self, *input):
        x = input[0]
        return self.lut(x) * self.sqrt_d_model


class PositionalEncoding(nn.Cell):
    def __init__(
        self, dimensions: int, dropout: float = 0.1, max_len: int = 5000
    ) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, dimensions), dtype=np.float32)
        position = np.arange(0, max_len)[..., None]
        div_term = np.exp(-np.arange(0, dimensions, 2) * np.log(10000) / dimensions)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None, ...]
        self.pe = Tensor(pe, dtype=ms.float32)

    def construct(self, input_tensor: Tensor) -> Tensor:
        input_tensor = (
            input_tensor + self.pe[:, : input_tensor.shape[1]]
        )  # pe 1 5000 512
        return self.dropout(input_tensor)
