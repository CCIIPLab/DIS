# -*- coding: utf-8 -*-

import os
import sys
sys.path.append("grid-feats-vqa")
import argparse

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from grid_feats import (
    add_attribute_config,
    build_detection_test_loader_with_attributes,
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", default="configs/X-152-challenge.yaml", type=str)
    parser.add_argument("--ckpt_file", default="ckpts/X-152pp.pth", type=str)

    parser.add_argument("--image_dir", type=str, default="gqa")
    parser.add_argument("--save_dir", type=str, default="gqa_grid_152")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.config_file) and os.path.exists(args.image_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    """配置detectron参数"""
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.freeze()
    default_setup(cfg, args)

    """构建模型"""
    model = build_model(cfg)
    model = model.eval()
    DetectionCheckpointer(model, save_dir="output").resume_or_load(
        args.ckpt_file, resume=True
    )

    """遍历图像、提取特征"""
    tfm_gens = utils.build_transform_gen(cfg, False)
    with torch.no_grad():
        for fn in tqdm(os.listdir(args.image_dir)):
            image_id = fn.split(".")[0]

            """准备输入"""
            data_dict = {
                "file_name": os.path.join(args.image_dir, fn)
            }
            image = utils.read_image(data_dict["file_name"], format="BGR")
            utils.check_image_size(data_dict, image)
            image, transforms = T.apply_transform_gens(
                tfm_gens, image
            )
            data_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            """提取特征"""
            images = model.preprocess_image([data_dict])
            features = model.backbone(images.tensor)
            outputs = model.roi_heads.get_conv5_features(features)
            outputs : torch.Tensor
            outputs = F.adaptive_avg_pool2d(outputs, (10, 10)).view((2048, 100)).transpose(0, 1)  # [100, 2048]

            """保存特征"""
            save_path = os.path.join(args.save_dir, "{}.npz".format(image_id))
            outputs = outputs.cpu().numpy()
            np.savez_compressed(
                save_path,
                feature=outputs,  # [1, 2048, H, W]
                width=data_dict["width"],
                height=data_dict["height"]
            )






