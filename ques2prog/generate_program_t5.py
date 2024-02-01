# -*- coding: utf-8 -*-

import os
import sys
root_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
print("append project dir {} to environment".format(root_dir))
sys.path.append(root_dir)
import json
import argparse
import resource
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm

import pyarrow as pa
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40000, rlimit[1]))
device = torch.device('cuda')

abbrs = [("relate_inv_name", "relate inverse name"),
         ("relate_inv", "relate inverse"), ("relate_attr", "relate attribute"),
         ("filter_h", "filter horizon"), ("filter_v", "filter vertical"),
         ("verify_h", "verify horizon"), ("verify_v", "verify vertical"),
         ("query_h", "query horizon"), ("query_v", "query vertical"),
         ("query_n", "query name"), ("query_f", "query feature"),
         ("verify_f", "verify feature"), ("verify_rel_inv", "verify relation inverse"),
         ("verify_rel", "verify relation"), ("choose_n", "choose name"),
         ("choose_h", "choose horizon"), ("choose_v", "choose vertical"),
         ("choose_subj", "choose subject"), ("choose_attr", "choose attribute"),
         ("choose_f", "choose feature"), ("choose_rel_inv", "choose relation inverse"),
         ("same_attr", "same attribute"), ("different_attr", "different attribute")]

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_preprocess', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--data_dir', type=str, default="preprocess")
    parser.add_argument('--save_dir', type=str, default="preprocess")

    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_train_prog', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_val_prog', default=False, action="store_true", help="whether to train or test the model")

    parser.add_argument('--do_trainval_prog', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_testdev_prog', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_submission_prog', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--total_splits', default=1, type=int)
    parser.add_argument('--split_index', default=0, type=int)

    parser.add_argument('--do_test', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_testdev', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_submission', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_trainval_split', default=False, action="store_true")

    parser.add_argument('--batch_size', default=1024, type=int, help="The batch size during training")
    parser.add_argument('--hidden_dim', default=128, type=int, help="The hidden size of the state")
    parser.add_argument('--version', default='', type=str, help="The hidden size of the state")
    parser.add_argument('--beam_size', default=1, type=int, help="The hidden size of the state")
    parser.add_argument('--load_from', type=str, default="", help="whether to train or test the model")
    parser.add_argument('--max_len', default=100, type=int, help="The hidden size of the state")
    parser.add_argument('--debug', default=False, action="store_true", help="Whether to debug it")
    parser.add_argument('--max_epoch', default=10, type=int, help="The hidden size of the state")
    parser.add_argument('--output', default="nl2prog/", type=str, help="The hidden size of the state")
    parser.add_argument('--meta', default="meta_info/", type=str, help="The hidden size of the state")
    args = parser.parse_args()
    return args


class GQA(Dataset):
    def __init__(self, data_dir, split):
        assert os.path.exists(data_dir)

        if split == "train_all":
            data_path = os.path.join(data_dir, "train_all_pairs.json")
        elif split == "train_bal":
            data_path = os.path.join(data_dir, "train_balanced_pairs.json")
        elif split == "testdev_bal":
            data_path = os.path.join(data_dir, "testdev_balanced_pairs.json")
        else:
            raise ValueError("No such split: {}".format(split))

        self.max_src = 40
        self.max_trg = 100
        self.split = split
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

        with open(data_path, "r") as fp:
            self.data = pa.array(json.load(fp))

    def __getitem__(self, index):
        src, dst = self.data[index].as_py()

        src = "transform question into program: {}".format(src).lower()

        return src, dst

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        src = [b[0] for b in batch]
        dst = [b[1] for b in batch]

        src = self.tokenizer(
            src,
            padding="longest",
            max_length=self.max_src,
            truncation=True,
            return_tensors="pt"
        ).input_ids  # [B, LM]
        dst = self.tokenizer(
            dst,
            padding="longest",
            max_length=self.max_trg,
            truncation=True,
            return_tensors="pt"
        ).input_ids  # [B, LN]

        return src, dst

class GQAQuestion(Dataset):
    def __init__(self, questions):
        super(GQAQuestion, self).__init__()

        self.max_src = 40
        self.max_trg = 100
        self.split = split
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

        self.data = pa.array([json.dumps(q) for q in questions])
        print("there are {} questions".format(len(self.data)))

    def __getitem__(self, index):
        entry = json.loads(self.data[index].as_py())
        src = entry[1]

        src = "transform question into program: {}".format(src).lower().replace(" ?", "?").replace(" ,", ",")

        dst = "none"
        return src, dst, entry[0], entry[1], entry[-2], entry[-1]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        src = [b[0] for b in batch]

        src = self.tokenizer(
            src,
            padding="longest",
            max_length=self.max_src,
            truncation=True,
            return_tensors="pt"
        )  # [B, LM]

        return src.input_ids, src.attention_mask, \
               [b[2] for b in batch], [b[3] for b in batch], \
               [b[4] for b in batch], [b[5] for b in batch]

def split(string):
    output = []
    buf_str = ""
    for s in string:
        if s == "(":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append("(")
            buf_str = ""
        elif s == ")":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append(")")
            buf_str = ""
        elif s == ",":
            string = buf_str.strip()
            if string:
                output.append(string)
            output.append(",")
            buf_str = ""
        else:
            buf_str += s
    return output

# 将entry中的program转化为用于训练program生成模型的格式
def generate_pairs(entry):
    if entry[2]:
        output = []
        for r in entry[2]:
            _, p = r.split('=')
            sub_p = split(p)
            output.extend(sub_p)
            output.append(";")
        del output[-1]
    else:
        output = []

    output = "".join(output)
    output = output.replace(";", "; ").replace(",", ", ")
    # 将其中的program名称转化为更易懂
    abbrs = [("relate_inv", "relate inverse"), ("relate_attr", "relate attribute"),
             ("filter_h", "filter horizon"), ("filter_v", "filter vertical"),
             ("verify_h", "verify horizon"), ("verify_v", "verify vertical"),
             ("query_h", "query horizon"), ("query_v", "query vertical"),
             ("query_n", "query name"), ("query_f", "query feature"),
             ("verify_f", "verify feature"), ("verify_rel_inv", "verify relation inverse"),
             ("verify_rel", "verify relation"), ("choose_n", "choose name"),
             ("choose_h", "choose horizon"), ("choose_v", "choose vertical"),
             ("choose_subj", "choose subject"), ("choose_attr", "choose attribute"),
             ("choose_f", "choose feature"), ("choose_rel_inv", "choose relation inverse"),
             ("same_attr", "same attribute"), ("different_attr", "different attribute")]
    for i, j in abbrs:
        output = output.replace(i, j)
    output = output.replace("_", " ")

    return entry[1], "".join(output)

def create_pairs(filename, split):
    examples = []
    with open(filename, "r") as f:
        data = json.load(f)
    print("total {} programs".format(len(data)))
    if split == 'submission':
        data = map(lambda x: (x[1]['imageId'],
                              x[1]['question'],
                              [],
                              x[0],
                              'unknown'), data.items())

    cores = multiprocessing.cpu_count()
    print("using parallel computing with {} cores".format(cores))
    pool = Pool(cores)

    r = pool.map(generate_pairs, data)

    pool.close()
    pool.join()

    with open('{}/{}_pairs.json'.format(args.output, split), 'w') as f:
        json.dump(r, f)

def get_batch(data, batch_size):
    examples = []
    length = len(data)
    intervals = (length // batch_size) + 1
    for i in range(intervals):
        yield data[i * batch_size: min(length, (i + 1) * batch_size)]

"""
采用T5来训练program生成模型
"""
def train(option):
    # Initialize model
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    if option == 'train':
        train_dataset = GQA(
            data_dir=args.data_dir,
            split="train_all"
        )
        test_dataset = GQA(
            data_dir=args.data_dir,
            split="testdev_bal"
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            collate_fn=train_dataset.collate_fn
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size // 2,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            collate_fn=test_dataset.collate_fn
        )

        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        for epoch in range(args.max_epoch):
            model.train()
            for idx, batch in enumerate(train_dataloader):
                batch = tuple(Variable(t).to(device) for t in batch)
                input_ids, labels = batch

                model.zero_grad()
                optimizer.zero_grad()

                loss = model(input_ids=input_ids, labels=labels).loss

                loss.backward()
                optimizer.step()

                if idx % 1000 == 0:
                    print("step {}/{}, loss = {:.4f}".format(idx, len(train_dataloader), loss.item()))

            # 计算测试集上的loss
            model.eval()
            total_loss = 0.0
            for idx, batch in enumerate(test_dataloader):
                batch = tuple(Variable(t).to(device) for t in batch)
                input_ids, labels = batch
                loss = model(input_ids=input_ids, labels=labels).loss.item()
                total_loss += loss
            print("test loss: {:.4f}".format(total_loss / len(test_dataloader)))

            torch.save(model.state_dict(), 'models/seq2seq_ep{}.pt'.format(epoch))

    elif option in ["train_aug", "val_aug"]:
        model.load_state_dict(torch.load(args.load_from))
        model.to(device)
        print("loading the model from {}".format(args.load_from))
        model.eval()

        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        print("processing {}".format(option))
        programs = []
        if option == "train_aug":
            data_path = os.path.join(args.data_dir, "train_dialog_balanced_questions.json")

            with open(data_path, "r") as fp:
                questions = list(json.load(fp)["questions"].items())
        elif option == "val_aug":
            data_path = os.path.join(args.data_dir, "val_balanced_questions.json")
            aug_path = os.path.join(args.data_dir, "val_sub_balanced_questions.json")

            with open(data_path, "r") as fp:
                questions = list(json.load(fp).items())
            with open(aug_path, "r") as fp:
                questions = questions + list(json.load(fp).items())
        else:
            raise ValueError("No such option: {}".format(option))

        batch_size = 32
        steps = (len(questions) // batch_size) \
            if (len(questions) % batch_size == 0) \
            else (len(questions) // batch_size + 1)
        for step in tqdm(range(steps)):
            batch = []
            for i in range(step * batch_size, min((step + 1) * batch_size, len(questions))):
                batch.append(
                    "transform question into program: {}".format(questions[i][1]["question"]).lower()
                )
            batch = tokenizer(
                batch,
                padding="longest",
                max_length=40,
                truncation=True,
                return_tensors="pt"
            )  # [B, LM]
            output = model.generate(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                max_length=100,
                num_beams=1
            )
            output = tokenizer.batch_decode(output, skip_special_tokens=True)

            for qtup, out in zip(questions[step * batch_size: (step + 1) * batch_size],
                                 output):
                try:
                    # print(qtup[1]["question"], out)
                    for i, j in abbrs:
                        out = out.replace(j, i)
                    out = out.split("; ")
                    out_new = []
                    for o in out:
                        prefix, suffix = o.split("(")
                        prefix: str
                        prefix = prefix.replace(" ", "_")
                        out_new.append("(".join([prefix, suffix]))

                    programs.append([
                        qtup[1]["imageId"],
                        qtup[1]["question"],
                        out_new,
                        qtup[0],
                        qtup[1]["answer"]
                    ])
                except:
                    print(qtup[1]["question"], out)

        the_split = "train" if option == "train_aug" else "val"
        save_path = os.path.join(args.data_dir, "{}_questions_program.json".format(the_split))
        with open(save_path, "w") as fp:
            json.dump(
                programs,
                fp
            )
    elif option in ['trainval_prog', 'testdev_prog', 'submission_prog']:
        model.load_state_dict(torch.load(args.load_from))
        model.to(device)
        print("loading the model from {}".format(args.load_from))
        model.eval()

        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        print("processing {}".format(option))
        if option == "trainval_prog":
            data_path = os.path.join(args.data_dir, "trainval_balanced_programs.json")
            with open(data_path, "r") as fp:
                questions = json.load(fp)
        elif option == "testdev_prog":
            data_path = os.path.join(args.data_dir, "testdev_balanced_programs.json")
            with open(data_path, "r") as fp:
                questions = json.load(fp)
        elif option == "submission_prog":
            data_path = os.path.join(args.data_dir, "submission_all_questions.json")
            with open(data_path, "r") as fp:
                data = json.load(fp)
            questions = [[
                qdict["imageId"],
                qdict["question"],
                qid,
                "unknown"
            ] for qid, qdict in data.items()]

            # 按照total_splits分割
            length_per_split = len(questions) // args.total_splits + 1
            questions = questions[args.split_index * length_per_split: (args.split_index + 1) * length_per_split]
            print("process split {} / {}, {} samples".format(args.split_index, args.total_splits, len(questions)))
        else:
            raise ValueError("No such option: {}".format(option))

        dataset = GQAQuestion(questions)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            drop_last=False
        )

        errors = []
        programs = []
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            src_ids, src_mask, iids, qs, qids, answers = batch
            output = model.generate(
                input_ids=src_ids.to("cuda"),
                attention_mask=src_mask.to("cuda"),
                max_length=100,
                num_beams=1
            )
            output = tokenizer.batch_decode(output, skip_special_tokens=True)

            for iid, q, qid, ans, out in zip(iids, qs, qids, answers, output):
                try:
                    for i, j in abbrs:
                        out = out.replace(j, i)
                    out = out.split("; ")
                    out_new = []
                    for o in out:
                        prefix, suffix = o.split("(")
                        prefix: str
                        prefix = prefix.replace(" ", "_")
                        out_new.append("(".join([prefix, suffix]))

                    programs.append([
                        iid,
                        q,
                        out_new,
                        qid,
                        ans
                    ])
                except:
                    if option == 'submission_prog':
                        errors.append([iid, qid])
                    print(q, out)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if option == "trainval_prog":
            save_path = os.path.join(args.save_dir, "trainval_pred_programs.json")
        elif option == "testdev_prog":
            save_path = os.path.join(args.save_dir, "testdev_pred_programs.json")
        elif option == "submission_prog":
            save_path = os.path.join(args.save_dir, "submission_programs.json")
        else:
            raise ValueError("No such option: {}".format(option))

        with open(save_path, "w") as fp:
            json.dump(programs, fp, indent=2)

if __name__ == "__main__":
    args = parse_opt()

    if args.do_preprocess:
        """
        python generate_program_t5.py --do_preprocess --data_dir /home/yuhang/data/gqa/mmn/questions --output /home/yuhang/data/gqa/question2program
        """
        trainval_all_path = os.path.join(args.data_dir, "trainval_all_programs.json")
        trainval_bal_path = os.path.join(args.data_dir, "trainval_balanced_programs.json")
        testdev_bal_path = os.path.join(args.data_dir, "testdev_balanced_programs.json")

        create_pairs(trainval_all_path, 'train_all')
        create_pairs(trainval_bal_path, 'train_balanced')
        create_pairs(testdev_bal_path, 'testdev_balanced')
    elif args.do_train:
        """
        python generate_program_t5.py --do_train --data_dir "/home/yuhang/data/gqa/question2program" --batch_size 16 --max_epoch 10
        """
        train('train')
    elif args.do_train_prog:
        """
        python question2program/generate_program_t5.py --do_train_prog --load_from /home/yuhang/MYCODE/VQAVR/models/seq2seq_ep0.pt --data_dir "/home/yuhang/data/gqa/consistency/questions"
        """
        train('train_aug')
    elif args.do_trainval_prog:
        """
        python question2program/generate_program_t5.py --do_trainval_prog --load_from /home/yuhang/MYCODE/VQAVR/models/seq2seq_ep0.pt --data_dir "/home/yuhang/data/gqa/mmn/questions"
        """
        train('trainval_prog')
    elif args.do_testdev_prog:
        """
        python question2program/generate_program_t5.py --do_testdev_prog --load_from /home/yuhang/MYCODE/VQAVR/models/seq2seq_ep0.pt --data_dir "/home/yuhang/data/gqa/mmn/questions"
        """
        train('testdev_prog')
    elif args.do_submission_prog:
        """
        python question2program/generate_program_t5.py --do_submission_prog --load_from /home/yuhang/MYCODE/VQAVR/models/seq2seq_ep0.pt --data_dir "/home/yuhang/data/gqa/mmn/questions" --total_splits 5 --split_index 0
        """
        train('submission_prog')
    elif args.do_val_prog:
        train('val_aug')
    elif args.do_submission:
        train('submission')
    else:
        print("unsupported")
