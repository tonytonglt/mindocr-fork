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
https://github.com/JiaquanYe/TableMASTER-mmocr/tree/master/mmocr/models/textrecog/losses
"""

import mindspore as ms
from mindspore import nn, ops


class TableMasterLoss(nn.Cell):
    def __init__(self, ignore_index=-1):
        super(TableMasterLoss, self).__init__()
        self.structure_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction='mean')
        self.box_loss = nn.L1Loss(reduction='sum')
        self.eps = 1e-12

    def construct(self, predicts, structure, bboxes, bbox_masks):
        # structure_loss
        structure_probs = predicts['structure_probs']
        structure_targets = structure
        structure_targets = structure_targets[:, 1:]
        structure_probs = structure_probs.reshape(
            [-1, structure_probs.shape[-1]])
        structure_targets = structure_targets.reshape([-1])
        structure_targets = ops.cast(structure_targets, ms.int32)
        # import numpy as np
        # np.save('/home/tongli/structure_probs.npy', structure_probs.asnumpy())
        # np.save('/home/tongli/structure_targets.npy', structure_targets.asnumpy())
        # exit()
        # print('structure_probs.shape', structure_probs.shape)
        # print('structure_probs.view(-1)[:100]', structure_probs.view(-1)[:100])
        # print('structure_probs.dtype', structure_probs.dtype)
        # print('structure_targets.shape', structure_targets.shape)
        # print('structure_targets.view(-1)[:100]', structure_targets.view(-1)[:100])
        # print('structure_targets.dtype', structure_targets.dtype)
        structure_loss = self.structure_loss(structure_probs, structure_targets)


        print('structure_loss', structure_loss)
        structure_loss = structure_loss.mean()
        # losses = dict(structure_loss=structure_loss)
        # structure_loss = ms.Tensor(1.0, ms.float32)

        # box loss
        bboxes_preds = predicts['loc_preds']
        bboxes_targets = bboxes[:, 1:, :]
        bbox_masks = bbox_masks[:, 1:]
        # mask empty-bbox or non-bbox structure token's bbox.

        masked_bboxes_preds = bboxes_preds * bbox_masks
        masked_bboxes_targets = bboxes_targets * bbox_masks

        # horizon loss (x and width)
        horizon_sum_loss = self.box_loss(masked_bboxes_preds[:, :, 0::2],
                                         masked_bboxes_targets[:, :, 0::2])
        horizon_loss = horizon_sum_loss / (bbox_masks.sum() + self.eps)
        # vertical loss (y and height)
        vertical_sum_loss = self.box_loss(masked_bboxes_preds[:, :, 1::2],
                                          masked_bboxes_targets[:, :, 1::2])
        vertical_loss = vertical_sum_loss / (bbox_masks.sum() + self.eps)

        horizon_loss = horizon_loss.mean()
        vertical_loss = vertical_loss.mean()
        all_loss = structure_loss + horizon_loss + vertical_loss
        # print('loss', all_loss)
        # print('horizon_bbox_loss', horizon_loss)
        # print('vertical_bbox_loss', vertical_loss)
        # losses.update({
        #     'loss': all_loss,
        #     'horizon_bbox_loss': horizon_loss,
        #     'vertical_bbox_loss': vertical_loss
        # })
        return all_loss
