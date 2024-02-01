# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import glob
import random
import resource
import itertools
from tqdm import tqdm

from nltk.tokenize import word_tokenize
import pyarrow as pa
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from model.vqa import ProgramTransformerE2E
from model.layers import KLDivergence
from scripts import Constants

device = torch.device('cuda')
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40000, rlimit[1]))

def parse_opt():
    parser = argparse.ArgumentParser()

    # Define flags
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_train_all', default=False, action="store_true", help="train the model using all-split")
    parser.add_argument('--do_finetune', default=False, action="store_true", help="finetune the model using balance-split")
    parser.add_argument('--do_submission', default=False, action="store_true", help="predict answers for submission split")
    parser.add_argument('--do_train_cons', default=False, action="store_true", help="train model to test consistency")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_val_aug', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--use_aug', default=False, action="store_true", help="use the augmented data")

    # Define dataset parameters
    parser.add_argument('--data_dir', type=str, default="gqa_bottom_up_features/",
                        help="whether to train or test the model")
    parser.add_argument('--image_folder', type=str, default="gqa_bottom_up_features/",
                        help="whether to train or test the model")
    parser.add_argument('--meta', default="meta_info/", type=str,
                        help="the path of meta files")
    parser.add_argument('--threshold', type=float, default=0.,
                        help="only keep the objects with confidence > threshold")
    parser.add_argument('--cutoff', type=float, default=0.5,
                        help="set the prob to zero for iou < cutoff")
    parser.add_argument('--length', type=int, default=9,
                        help="max_length of programs")
    parser.add_argument('--pack_num', type=int, default=8)

    # Define model parameters
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help="the hidden dim of Transformer")
    parser.add_argument('--n_head', type=int, default=8,
                        help="the number of heads in multi-head attention")
    parser.add_argument('--pre_layers', type=int, default=3,
                        help="num_layers of question encoder")
    parser.add_argument('--visual_dim', type=int, default=2048,
                        help="the dimension of visual features")
    parser.add_argument('--num_regions', type=int, default=48,
                        help="num region for images features")
    parser.add_argument('--num_tokens', type=int, default=32,
                        help="the max length of question tokens")
    parser.add_argument('--num_workers', type=int, default=16,
                        help="num_workers for dataloader")
    parser.add_argument('--coordinate_dim', type=int, default=4,
                        help="the dimension of bounding boxes")
    parser.add_argument('--weight', type=float, default=0.5,
                        help="the loss weight of intermediate supervision")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="the batch size for training and validation")
    parser.add_argument('--num_epochs', type=int, default=20,
                        help="total epochs for training")
    parser.add_argument('--max_layer', type=int, default=5,
                        help="max_layers for program-guided reasoning")
    parser.add_argument('--intermediate_layer', default=False, action="store_true",
                        help="not used")
    parser.add_argument('--stacking', type=int, default=2,
                        help="num_layers for gathering image features in the end")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="the dropout of model")
    parser.add_argument('--feat_size', type=int, default=8)
    parser.add_argument('--intermediate_num', type=int, default=5)  # 中间步骤预测多少个object
    parser.add_argument('--object_grids', type=int, default=256)

    # Define training parameters
    parser.add_argument('--model', type=str, default="Tree",
                        help="VQA model")
    parser.add_argument('--load_from', type=str, default="",
                        help="load the pretrained model")
    parser.add_argument('--resume', type=int, default=-1,
                        help="resume epoch")
    parser.add_argument('--word_glove', type=str, default="meta_info/en_emb.npy",
                        help="the glove embedding file for vocab")
    parser.add_argument('--distribution', default=False, action='store_true',
                        help="use distributed training mode")
    parser.add_argument('--output', type=str, default="models",
                        help="output path for saving models")
    parser.add_argument('--id', type=str, default="default",
                        help="the unique id of the model")
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--config_file', type=str, default="grid/configs/X-101-grid.yaml")
    parser.add_argument('--grid_ckpt', type=str, default="grid/ckpts/X-101.pth")

    # Optimizer parameters
    parser.add_argument('--lr_decrease_start', type=int, default=10,
                        help="the start epoch for lr decay")
    parser.add_argument('--lr_default', type=float, default=1e-4,
                        help="the base learning rate")
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help="decay ratio of learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--accumulate', type=int, default=1)

    args = parser.parse_args()
    return args

"""
Programs with multiple intermediate objects:
- image_id
- question
- bboxes/true/false
- names/true/false
- program
- connection
- question_id
- answer

Programs with one intermediate object:
- image_id
- question
- bbox/true/false
- program
- connection
- question_id
- answer
"""
class GQA(Dataset):
    def __init__(self, data_dir, mode, split, word_vocab, answer_vocab,
                 num_tokens, num_regions, cutoff, length, max_layer):
        self.mode = mode
        self.split = split
        self.vocab = word_vocab
        self.answer_vocab = answer_vocab
        self.num_tokens = num_tokens
        self.num_regions = num_regions
        self.cutoff = cutoff
        self.LENGTH = length
        self.MAX_LAYER = max_layer
        self.data = []

        if split == "trainval_bal":
            self.data_path = os.path.join(data_dir, "trainval_balance_inputs.json")
        elif split == "trainval_all":
            self.data_path = os.path.join(data_dir, "trainval_all_inputs.json")
            self.more_path = os.path.join(data_dir, "trainval_all_inputs_additional.json")
        elif split == "testdev_bal":
            self.data_path = os.path.join(data_dir, "testdev_balance_inputs.json")
        elif split == "submission":
            self.data_path = os.path.join(data_dir, "submission_inputs.json")
        elif split == "train":
            self.data_path = os.path.join(data_dir, "train_inputs.json")
        elif split == "val":
            self.data_path = os.path.join(data_dir, "val_inputs.json")
        elif split == "train_aug":
            self.data_path = os.path.join(data_dir, "train_aug_inputs.json")
        elif split == "val_aug":
            self.data_path = os.path.join(data_dir, "val_aug_inputs.json")
        else:
            raise ValueError("Invalid mode {}".format(mode))

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

class GQA_v1(GQA):
    def __init__(self, data_dir, folder, mode, split, word_vocab, answer_vocab,
                 num_tokens, num_regions, cutoff, length, max_layer, threshold,
                 **kwargs):
        super(GQA_v1, self).__init__(data_dir, mode, split, word_vocab, answer_vocab,
                 num_tokens, num_regions, cutoff, length, max_layer)
        self.folder = folder
        self.threshold = threshold
        self.feat_size = kwargs["feat_size"]
        self.object_vocab = kwargs["object_vocab"]
        self.bbox_grids = kwargs["bbox_grids"]
        self.intermediate_num = kwargs["intermediate_num"]

        self.data = []
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        if hasattr(self, "more_path") and os.path.exists(self.more_path):
            with open(self.more_path, "r") as f:
                self.data: list
                self.data.extend(json.load(f))
        self.data = pa.array(self.data)

        print("loading data from {}".format(self.data_path))
        print("there are in total {} instances".format(len(self.data)))

    """
    - image_id
    - question
    - bboxes/true/false
    - names/true/false
    - program
    - connection
    - question_id
    - answer
    """
    def __getitem__(self, index):
        entry = json.loads(self.data[index].as_py())

        data_dict = dict()
        image_id = entry[0]
        data_dict["image_id"] = image_id

        """处理question"""
        question = entry[1]
        idxs = word_tokenize(question)[:self.num_tokens]
        question = [self.vocab.get(_, Constants.UNK) for _ in idxs]
        question += [Constants.PAD] * (self.num_tokens - len(idxs))
        question = np.array(question, 'int64')
        question_masks = np.zeros((len(question),), 'float32')
        question_masks[:len(idxs)] = 1.
        data_dict["question"] = question
        data_dict["question_masks"]= question_masks

        """处理programs"""
        inputs = entry[4]
        length = min(len(inputs), self.LENGTH)
        program = np.zeros((self.LENGTH, 8), 'int64')
        depth = np.zeros((self.LENGTH,), 'int64') + self.MAX_LAYER  # value range from 0 ~ self.MAX_LAYER
        for i in range(length):
            for j, text in enumerate(inputs[i]):
                if text is not None:
                    program[i][j] = self.vocab.get(text, Constants.UNK)
        program_masks = np.zeros((self.LENGTH,), 'float32')
        program_masks[:length] = 1.
        data_dict["program"] = program
        data_dict["program_masks"]= program_masks

        """处理program的转换矩阵"""
        connection = entry[5]
        questionId = entry[-2]
        transition_masks = np.zeros(
            (self.MAX_LAYER, self.LENGTH, self.LENGTH), 'uint8')
        activate_mask = np.zeros((self.MAX_LAYER, self.LENGTH), 'float32')
        for i in range(self.MAX_LAYER):
            if i < len(connection):
                for idx, idy in connection[i]:
                    transition_masks[i][idx][idy] = 1
                    depth[idx] = i
                    activate_mask[i][idx] = 1

            for j in range(self.LENGTH):
                if activate_mask[i][j] == 0:
                    # As a placeholder
                    transition_masks[i][j][j] = 1
                else:
                    pass
        data_dict["transition_masks"] = transition_masks
        data_dict["activate_mask"] = activate_mask
        data_dict["questionId"] = questionId
        data_dict["depth"] = depth

        """准备图像特征"""
        try:
            bottom_up = np.load(os.path.join(
                self.folder, '{}.npz'.format(image_id)))
            feature = bottom_up["feature"]  # [100, 2048]  # , [1, 2048, h, w]
        except:
            print(image_id)
            bottom_up = {
                "feature": np.zeros((100, 2048), dtype=np.float32),
                "width": 1000,
                "height": 1000
            }
            feature = bottom_up["feature"]
        data_dict["feature"] = feature

        """生成Ground Truth的Bbox，用于后续监督中间结果"""
        if self.mode == 'train':
            returns_bbox = entry[2]
            returns_name = entry[3]
            intermediate_idx = np.full(
                (self.LENGTH, 1 + self.intermediate_num * 4 + 1), -1, 'int64')

            for idx in range(length - 1):
                if returns_bbox[idx] in ["true", "false"]:
                    intermediate_idx[idx][:3] = np.array([self.object_vocab.get("[START]"),
                                                          self.object_vocab.get(returns_bbox[idx]),
                                                          self.object_vocab.get("[END]")], dtype=np.int64)
                elif isinstance(returns_bbox[idx], list):
                    if len(returns_bbox[idx]) == 0:
                        intermediate_idx[idx][:3] = np.array([self.object_vocab.get("[START]"),
                                                              self.object_vocab.get("[NULL]"),
                                                              self.object_vocab.get("[END]")], dtype=np.int64)
                    else:
                        tmp = []
                        returns = list(zip(returns_bbox[idx], returns_name[idx]))
                        random.shuffle(returns)
                        # bboxes = returns_bbox[idx]
                        # names = returns_name[idx]
                        for bbox, name in returns[:self.intermediate_num]: # zip(bboxes[:self.intermediate_num], names[:self.intermediate_num]):
                            x1, y1, w, h = bbox
                            x2, y2 = x1 + w, y1 + h
                            tmp.extend([
                                int(min(x1 / bottom_up["width"], 1.0) * self.bbox_grids) + len(self.object_vocab),
                                int(min(y1 / bottom_up["height"], 1.0) * self.bbox_grids) + len(self.object_vocab),
                                int(min(x2 / bottom_up["width"], 1.0) * self.bbox_grids) + len(self.object_vocab),
                                int(min(y2 / bottom_up["height"], 1.0) * self.bbox_grids) + len(self.object_vocab)
                            ])
                            # tmp.append(self.object_vocab.get(name, self.object_vocab["[UNK]"]))
                        tmp = [self.object_vocab.get("[START]")] + tmp + [self.object_vocab.get("[END]")]
                        intermediate_idx[idx][:len(tmp)] = np.array(tmp, dtype=np.int64)
                else:
                    raise ValueError("No such bbox: {}".format(returns_bbox[idx]))

            """returns = entry[2]
            intermediate_idx = np.full(
                (self.LENGTH, 4), 0, 'float32')

            for idx in range(length - 1):
                if isinstance(returns[idx], list):
                    if returns[idx] == [-1, -1, -1, -1]:
                        pass
                    else:
                        x1, y1, w, h = returns[idx]
                        intermediate_idx[idx] = np.array([x1, y1, x1 + w, y1 + h],
                                                         dtype=np.float32) * scale_ratio  # [4, ]"""
        else:
            intermediate_idx = 0
        data_dict["intermediate_idx"] = intermediate_idx

        """准备输出"""
        index = length - 1
        answer_id = self.answer_vocab.get(entry[-1], Constants.UNK)
        data_dict["index"] = index
        data_dict["answer_id"] = answer_id

        return data_dict

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        batch_out = dict()

        for key in keys:
            if isinstance(batch[0][key], torch.Tensor):
                batch_out[key] = torch.stack([b[key] for b in batch], dim=0)
            elif isinstance(batch[0][key], np.ndarray):
                batch_out[key] = torch.from_numpy(np.stack([b[key] for b in batch], axis=0))
            elif isinstance(batch[0][key], int) or isinstance(batch[0][key], float):
                batch_out[key] = torch.tensor([b[key] for b in batch])
            else:
                batch_out[key] = [b[key] for b in batch]

        return batch_out

def to_device(batch: dict):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to("cuda")

    return batch

if __name__ == "__main__":
    args = parse_opt()
    if args.do_submission:
        args.length += 1

    # Fixed seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # LR scheduler
    lr_decay_step = 2
    lr_decay_epochs = range(args.lr_decrease_start, args.num_epochs, lr_decay_step)
    gradual_warmup_steps = [1.0 * args.lr_default, 1.0 *
                            args.lr_default, 1.5 * args.lr_default, 2.0 * args.lr_default]
    if args.do_train_all:
        args.num_epochs = 5
        gradual_warmup_steps = [1.0 * args.lr_default,
                                1.0 * args.lr_default,
                                1.5 * args.lr_default,
                                2.0 * args.lr_default,
                                1.0 * args.lr_default]
    if args.do_finetune:
        args.num_epochs = 5
        gradual_warmup_steps = [args.lr_default / 1.0,
                                args.lr_default / 1.0,
                                args.lr_default / 4.0,
                                args.lr_default / 4.0,
                                args.lr_default / 16.0]

    # Output dir
    print(args)
    repo = os.path.join(args.output, args.id)
    if not os.path.exists(repo):
        os.makedirs(repo)

    # Load meta data, including vocab and answer
    with open('{}/full_vocab.json'.format(args.meta), 'r') as f:
        vocab = json.load(f)
        ivocab = {v: k for k, v in vocab.items()}

    with open('{}/answer_vocab.json'.format(args.meta), 'r') as f:
        answer = json.load(f)
        remove = []
        for ans, ind in answer.items():
            if ind >= 1845:
                remove.append(ans)
        answer: dict
        for ans in remove:
            del answer[ans]

        inv_answer = {v: k for k, v in answer.items()}  # ind -> str

    with open('{}/objects_gen.json'.format(args.meta), 'r') as f:
        objects = json.load(f)
        objects += ['true', 'false', '[UNK]', '[START]', '[END]', '[NULL]']
        inv_object = {v: k for k, v in enumerate(objects)}

    MAX_LAYER = args.max_layer
    if args.model == "ProgramTransformer":
        model = ProgramTransformerE2E(vocab_size=len(vocab), stacking=args.stacking, answer_size=len(answer),
                                      visual_dim=args.visual_dim, coordinate_dim=args.coordinate_dim,
                                      hidden_dim=args.hidden_dim, n_head=args.n_head, n_layers=MAX_LAYER,
                                      dropout=args.dropout, intermediate_dim=args.num_regions + 1, pre_layers=args.pre_layers,
                                      intermediate_layer=args.intermediate_layer,
                                      intermediate_size=len(objects) + args.object_grids + 1,
                                      args=args)
        print("Running Modular Transformer model with {} layers with post layer".format(args.stacking))

        model.embedding.weight.data.copy_(torch.from_numpy(np.load(args.word_glove)))
        print("loading embedding from {}".format(args.word_glove))
    else:
        raise NotImplementedError

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           lr=args.lr_default * 2.0)

    if args.load_from != "":
        assert os.path.exists(args.load_from), "Checkpoint {} does not exist!".format(args.load_from)
        print("loading model from {}".format(args.load_from))
        ckpt_dict = torch.load(args.load_from, map_location="cpu")
        state_dict = ckpt_dict["state_dict"]
        model.load_state_dict(state_dict)

    # Resume from last checkpoint
    if args.resume != -1:
        filename = glob.glob('{}/{}/model_ep{}*'.format(args.output, args.id, args.resume))[0]
        print("resuming from {}".format(filename))
        ckpt = torch.load(filename)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    model.to(device)

    if args.do_train or args.do_train_all or args.do_finetune or args.do_train_cons:
        if args.do_train_all:
            train_split = "trainval_all"
            test_split = "testdev_bal"
        elif args.do_train_cons:
            if args.use_aug:
                train_split = "train_aug"
                test_split = "val_aug"
            else:
                train_split = "train"
                test_split = "val"
        else:
            train_split = "trainval_bal"
            test_split = "testdev_bal"
        train_dataset = GQA_v1(data_dir=args.data_dir,
                               folder=args.image_folder,
                               mode='train',
                               split=train_split,
                               word_vocab=vocab,
                               answer_vocab=answer,
                               num_tokens=args.num_tokens,
                               num_regions=args.num_regions,
                               cutoff=args.cutoff,
                               length=args.length,
                               max_layer=MAX_LAYER,
                               threshold=args.threshold,
                               feat_size=args.feat_size,
                               object_vocab=inv_object,
                               bbox_grids=args.object_grids,
                               intermediate_num=args.intermediate_num)
        test_dataset = GQA_v1(data_dir=args.data_dir,
                              folder=args.image_folder,
                              mode='test',
                              split=test_split,
                              word_vocab=vocab,
                              answer_vocab=answer,
                              num_tokens=args.num_tokens,
                              num_regions=args.num_regions,
                              cutoff=args.cutoff,
                              length=args.length,
                              max_layer=MAX_LAYER,
                              threshold=args.threshold,
                              feat_size=args.feat_size,
                              object_vocab=inv_object,
                              bbox_grids=args.object_grids,
                              intermediate_num=args.intermediate_num)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      prefetch_factor=2,
                                      collate_fn=train_dataset.collate_fn)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch_size // 2,
                                     shuffle=False,
                                     num_workers=8,
                                     pin_memory=True,
                                     prefetch_factor=2,
                                     collate_fn=test_dataset.collate_fn)

        cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        KL_loss = KLDivergence()

        scaler = GradScaler()
        best_acc = 0.0
        for epoch in range(args.resume + 1, args.num_epochs):
            # Adjust learning rate
            if epoch < len(gradual_warmup_steps):
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = gradual_warmup_steps[epoch]
                print('lr', optimizer.param_groups[-1]['lr'])
            if epoch in lr_decay_epochs:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] *= args.lr_decay_rate
                print('lr', optimizer.param_groups[-1]['lr'])
            else:
                print('lr', optimizer.param_groups[-1]['lr'])

            model.train()
            start_time = time.time()
            success_train = 0.
            total_train = 0.
            for i, batch in enumerate(train_dataloader):
                to_device(batch)

                # with autocast():
                results = model(**batch)
                if isinstance(results, tuple):
                    pre_logits, logits, loss_pre = model(**batch)
                else:
                    logits = model(**batch)

                loss = cross_entropy(logits, batch["answer_id"])
                loss = loss + loss_pre * args.weight
                loss = loss / args.accumulate

                # scaler.scale(loss).backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
                # scaler.step(optimizer)
                # scaler.update()

                loss.backward()
                if (i + 1) % args.accumulate == 0:
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                # Calculate accuracy on train split
                preds = torch.argmax(logits, -1)
                success_or_not = (preds == batch["answer_id"]).float()
                success_train += torch.sum(success_or_not).item()
                total_train += success_or_not.size(0)

                if i % 200 == 0 and i > 0:
                    acc = round(success_train / max(1, total_train), 4)
                    print("epoch: {}, iteration {}/{}: module loss = {:.4f}, pred_loss = {:.4f}, accuracy = {:.4f}, used time = {:.4f}".
                          format(epoch, i, len(train_dataloader), loss_pre.item(),
                                 loss.item(), acc, time.time() - start_time))
                    success_train = 0.
                    total_train = 0.

                start_time = time.time()

            model.eval()
            success, total = 0, 0
            for i, batch in enumerate(test_dataloader):
                to_device(batch)

                with torch.no_grad():
                    results = model(**batch)
                    if isinstance(results, tuple):
                        logits = results[1]
                    else:
                        logits = results
                preds = torch.argmax(logits, -1)
                success_or_not = (preds == batch["answer_id"]).float()

                success += torch.sum(success_or_not).item()
                total += success_or_not.size(0)
            
            acc = round(success / (total + 0.), 4)
            best_acc = max(acc, best_acc)
            print("epoch {}, accuracy = {}, best accuracy = {}".format(epoch, acc, best_acc))

            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "acc": acc
            }, os.path.join(repo, 'model_ep{}'.format(epoch)))
            model.train()
    elif args.do_val or args.do_val_aug:
        os.makedirs("results_consistency", exist_ok=True)

        test_dataset = GQA_v1(data_dir=args.data_dir,
                              folder=args.image_folder,
                              mode='test',
                              split="val_aug",
                              word_vocab=vocab,
                              answer_vocab=answer,
                              num_tokens=args.num_tokens,
                              num_regions=args.num_regions,
                              cutoff=args.cutoff,
                              length=args.length,
                              max_layer=MAX_LAYER,
                              threshold=args.threshold,
                              feat_size=args.feat_size,
                              object_vocab=inv_object,
                              bbox_grids=args.object_grids,
                              intermediate_num=args.intermediate_num)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch_size // 2,
                                     shuffle=False,
                                     num_workers=8,
                                     pin_memory=True,
                                     prefetch_factor=2,
                                     drop_last=False,
                                     collate_fn=test_dataset.collate_fn)

        model.eval()
        success, total = 0, 0
        success_ori, total_ori = 0, 0
        success_aug, total_aug = 0, 0
        result_ori = []
        result_aug = []
        for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            to_device(batch)

            with torch.no_grad():
                results = model(**batch)
                if isinstance(results, tuple):
                    logits = results[1]
                else:
                    logits = results
            preds = torch.argmax(logits, -1)
            success_or_not = (preds == batch["answer_id"]).long().detach().cpu()

            for qid, pred, succ in zip(batch["questionId"], preds, success_or_not):
                if "_" in qid:
                    result_aug.append({
                        "questionId": qid,
                        "prediction": inv_answer[pred.item()]
                    })
                    if succ.item() == 1:
                        success_aug += 1
                        success += 1
                    total_aug += 1
                else:
                    result_ori.append({
                        "questionId": qid,
                        "prediction": inv_answer[pred.item()]
                    })
                    if succ.item() == 1:
                        success_ori += 1
                        success += 1
                    total_ori += 1

                total += 1

        print("total {:.4f}, ori {:.4f}, aug {:.4f}".format(
            success / total,
            success_ori / total_ori,
            success_aug / total_aug
        ))

        with open("results_consistency/val_predictions_da.json", "w") as fp:
            json.dump(
                result_ori,
                fp,
                indent=2
            )
        with open("results_consistency/val_aug_predictions_da.json", "w") as fp:
            json.dump(
                result_aug,
                fp,
                indent=2
            )

    elif args.do_submission:
        dev_dataset = GQA_v1(data_dir=args.data_dir,
                             folder=args.image_folder,
                             mode='test',
                             split="testdev_bal",
                             word_vocab=vocab,
                             answer_vocab=answer,
                             num_tokens=args.num_tokens,
                             num_regions=args.num_regions,
                             cutoff=args.cutoff,
                             length=args.length,
                             max_layer=MAX_LAYER,
                             threshold=args.threshold,
                             feat_size=args.feat_size,
                             object_vocab=inv_object,
                             bbox_grids=args.object_grids,
                             intermediate_num=args.intermediate_num)
        dev_dataloader = DataLoader(dev_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=8,
                                    pin_memory=True,
                                    prefetch_factor=2,
                                    collate_fn=dev_dataset.collate_fn)

        test_dataset = GQA_v1(data_dir=args.data_dir,
                              folder=args.image_folder,
                              mode='test',
                              split="submission",
                              word_vocab=vocab,
                              answer_vocab=answer,
                              num_tokens=args.num_tokens,
                              num_regions=args.num_regions,
                              cutoff=args.cutoff,
                              length=args.length,
                              max_layer=MAX_LAYER,
                              threshold=args.threshold,
                              feat_size=args.feat_size,
                              object_vocab=inv_object,
                              bbox_grids=args.object_grids,
                              intermediate_num=args.intermediate_num)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=8,
                                     pin_memory=True,
                                     prefetch_factor=2,
                                     collate_fn=test_dataset.collate_fn)

        model.eval()
        success, total = 0, 0
        for i, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            to_device(batch)

            results = model(**batch)
            if isinstance(results, tuple):
                logits = results[1]
            else:
                logits = results
            preds = torch.argmax(logits, -1)
            success_or_not = (preds == batch["answer_id"]).float()

            success += torch.sum(success_or_not).item()
            total += success_or_not.size(0)

        acc = round(success / (total + 0.), 4)
        print("dev accuracy = {}".format(acc))

        submission = []
        for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            to_device(batch)

            results = model(**batch)
            if isinstance(results, tuple):
                logits = results[1]
            else:
                logits = results
            preds = torch.argmax(logits, -1).detach().cpu()  # [B, ]

            for qid, pred in zip(batch["questionId"], preds):
                submission.append({
                    "questionId": qid,
                    "prediction": inv_answer[pred.item()]
                })

        with open("submission.json", "w") as fp:
            json.dump(submission, fp)