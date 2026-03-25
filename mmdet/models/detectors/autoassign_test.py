# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .single_stage_enhance import SingleStageTwoBranchDetector

# - 所在位置 : autoassign_test.py
# - 功能讲解 : 这是 GCC-Net 的整体封装类。它继承自经典的 AutoAssign 算法，但将其主干网络替换为 SwinFusionTransformer 。
# - 流程控制 : 它负责协调 img 和 img_retinex 同时输入到 Backbone 中，并处理最终的分类与回归头（Head）输出。
@DETECTORS.register_module()
class AutoAssign_fusion(SingleStageTwoBranchDetector):
    """Implementation of `AutoAssign: Differentiable Label Assignment for Dense
    Object Detection <https://arxiv.org/abs/2007.03496>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(AutoAssign_fusion, self).__init__(backbone, neck,
                                                bbox_head, train_cfg,
                                                test_cfg, pretrained)
