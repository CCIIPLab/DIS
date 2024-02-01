# -*- coding: utf-8 -*-

import os
import re
import sys
root_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
print("append project dir {} to environment".format(root_dir))
sys.path.append(root_dir)
from tqdm import tqdm
import json
import sys
import random
import argparse

import h5py
import numpy as np

from scripts import Constants
from scripts.trace_sg import GQASceneGraphTrace

def add1(string, extra):
    #nums = string[1:-1].split(',')
    #nums = [str(int(_) + extra) for _ in nums]
    new_string = ""
    for c in string:
        if c.isdigit():
            new_string += str(int(c) + extra)
        else:
            new_string += c
    return new_string


def filter_field(string):
    output = re.search(r' ([^ ]+)\b', string).group()[2:]
    if 'not(' in output:
        return re.search(r'\(.+$', output).group()[1:], True
    else:
        return output, False


def filter_parenthesis(string):
    objects = re.search(r'\(.+\)', string).group()[1:-1]
    if objects == '-':
        return '[]'
    else:
        return '[{}]'.format(objects)


def filter_squre(string):
    indexes = re.search(r'\[.+\]', string).group()
    if ',' in indexes:
        return ','.join(['[{}]'.format(_.strip()) for _ in indexes[1:-1].split(',')])
    else:
        return indexes


def extract_rel(string):
    subject = re.search(r'^([^,]+),', string).group()[:-1]
    relation = re.search(r',(.+),', string).group()[1:-1]
    try:
        o_s = re.search(r',(o|s) ', string).group()[1:-1]
        if 's' in o_s:
            return subject, relation, True
        else:
            return subject, relation, False
    except:
        return subject, relation, None


def extract_query_key(string):
    if 'name' in string:
        return 'name'
    elif 'hposition' in string:
        return 'hposition'
    elif 'vposition' in string:
        return 'vposition'
    else:
        return 'attributes'


def split_rel(string):
    subject = re.search(r'([^,]+),', string).group()[:-1]
    relation1 = re.search(r',(.+)\|', string).group()[1:-1]
    relation2 = re.search(r'\|(.+),', string).group()[1:-1]
    o_s = re.search(r',(o|s)', string).group()[1:-1]
    if 's' in o_s:
        return subject, relation1, relation2, True
    else:
        return subject, relation1, relation2, False


def split_attr(string):
    attr1 = re.search(r'(.+)\|', string).group()[2:-1]
    attr2 = re.search(r'\|(.+) ', string).group()[1:-1]
    return attr1, attr2


def shuffle(string):
    attrs = string.split('|')
    random.shuffle(attrs)
    attr1, attr2 = attrs
    return attr1, attr2


def preprocess(raw_data, output_path, formal=False):
    symbolic_programs = []
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    keys = list(raw_data.keys())  # all question keys
    print("total {} programs".format(len(keys)))
    success, fail = 0, 0

    for idx in range(len(keys)):
        imageId = raw_data[keys[idx]]['imageId']
        question = raw_data[keys[idx]]['question']
        program = raw_data[keys[idx]]['semantic']
        answer = raw_data[keys[idx]]['answer']

        new_programs = []
        # try:
        for i, prog in enumerate(program):
            # Parse dependency
            if prog['dependencies']:
                subject = ",".join(["[{}]".format(_) for _ in prog['dependencies']])

            if '(' in prog['argument'] and ')' in prog['argument'] and 'not(' not in prog['argument']:
                result = filter_parenthesis(prog['argument'])
            else:
                result = '?'

            if prog['operation'] == 'select':
                if prog['argument'] == 'scene':
                    # new_programs.append('{}=scene()'.format(result))
                    flag = 'full'
                else:
                    new_programs.append('{}=select({})'.format(
                        result, lemmatizer.lemmatize(prog['argument'].split(' ')[0])))
                    flag = 'partial'

            elif prog['operation'] == 'relate':
                # print prog['argument']
                name, relation, reverse = extract_rel(prog['argument'])
                if reverse == None:
                    new_programs.append('{}=relate_attr({}, {}, {})'.format(result, subject, relation, name))
                else:
                    if reverse:
                        if name != '_':
                            name = lemmatizer.lemmatize(name)
                            new_programs.append('{}=relate_inv_name({}, {}, {})'.format(
                                result, subject, relation, name))
                        else:
                            new_programs.append('{}=relate_inv({}, {})'.format(result, subject, relation))
                    else:
                        if name != '_':
                            name = lemmatizer.lemmatize(name)
                            new_programs.append('{}=relate_name({}, {}, {})'.format(result, subject, relation, name))
                        else:
                            new_programs.append('{}=relate({}, {})'.format(result, subject, relation))

            elif prog['operation'].startswith('query'):
                if prog['argument'] == "hposition":
                    new_programs.append('{}=query_h({})'.format(result, subject))
                elif prog['argument'] == "vposition":
                    new_programs.append('{}=query_v({})'.format(result, subject))

                elif prog['argument'] == "name":
                    new_programs.append('{}=query_n({})'.format(result, subject))
                else:
                    if flag == 'full':
                        new_programs.append('{}=query_f({})'.format(result, prog['argument']))
                    else:
                        new_programs.append('{}=query({}, {})'.format(result, subject, prog['argument']))

            elif prog['operation'] == 'exist':
                new_programs.append('{}=exist({})'.format(result, subject))

            elif prog['operation'] == 'or':
                new_programs.append('{}=or({})'.format(result, subject))

            elif prog['operation'] == 'and':
                new_programs.append('{}=and({})'.format(result, subject))

            elif prog['operation'].startswith('filter'):
                if prog['operation'] == 'filter hposition':
                    new_programs.append('{}=filter_h({}, {})'.format(result, subject, prog['argument']))

                elif prog['operation'] == 'filter vposition':
                    new_programs.append('{}=filter_h({}, {})'.format(result, subject, prog['argument']))

                else:
                    negative = 'not(' in prog['argument']
                    if negative:
                        new_programs.append('{}=filter_not({}, {})'.format(result, subject, prog['argument'][4:-1]))
                    else:
                        new_programs.append('{}=filter({}, {})'.format(result, subject, prog['argument']))

            elif prog['operation'].startswith('verify'):
                if prog['operation'] == 'verify':
                    new_programs.append('{}=verify({}, {})'.format(result, subject, prog['argument']))
                elif prog['operation'] == 'verify hposition':
                    new_programs.append('{}=verify_h({}, {})'.format(result, subject, prog['argument']))
                elif prog['operation'] == 'verify vposition':
                    new_programs.append('{}=verify_v({}, {})'.format(result, subject, prog['argument']))
                elif prog['operation'] == 'verify rel':
                    name, relation, reverse = extract_rel(prog['argument'])
                    name = lemmatizer.lemmatize(name)
                    if reverse:
                        new_programs.append('{}=verify_rel_inv({}, {}, {})'.format(result, subject, relation, name))
                    else:
                        new_programs.append('{}=verify_rel({}, {}, {})'.format(result, subject, relation, name))
                    # if reverse:
                    #    new_programs.append('?=relate_inv_name({}, {}, {})'.format(subject, relation, name))
                    #    new_programs.append('{}=exist([{}])'.format(result, len(new_programs) - 1))
                    # else:
                    #    new_programs.append('?=relate_name({}, {}, {})'.format(subject, relation, name))
                    #    new_programs.append('{}=exist([{}])'.format(result, len(new_programs) - 1))
                else:
                    if flag == 'full':
                        new_programs.append('{}=verify_f({})'.format(result, prog['argument']))
                    else:
                        new_programs.append('{}=verify({}, {})'.format(result, subject, prog['argument']))

            elif prog['operation'].startswith('choose'):
                if prog['operation'] == 'choose':
                    attr1, attr2 = shuffle(prog['argument'])
                    if flag == "full":
                        new_programs.append('{}=choose_f({}, {})'.format(result, attr1, attr2))
                    else:
                        new_programs.append('{}=choose({}, {}, {})'.format(result, subject, attr1, attr2))

                elif prog['operation'] == 'choose rel':
                    name, relation1, relation2, reverse = split_rel(prog['argument'])
                    relation1, relation2 = shuffle('{}|{}'.format(relation1, relation2))
                    name = lemmatizer.lemmatize(name)
                    if reverse:
                        new_programs.append('{}=choose_rel({}, {}, {}, {})'.format(
                            result, subject, name, relation1, relation2))
                    else:
                        new_programs.append('{}=choose_rel_inv({}, {}, {}, {})'.format(
                            result, subject, name, relation1, relation2))

                elif prog['operation'] == 'choose hposition':
                    attr1, attr2 = shuffle(prog['argument'])
                    new_programs.append('{}=choose_h({}, {}, {})'.format(result, subject, attr1, attr2))

                elif prog['operation'] == 'choose vposition':
                    attr1, attr2 = shuffle(prog['argument'])
                    new_programs.append('{}=choose_v({}, {}, {})'.format(result, subject, attr1, attr2))

                elif prog['operation'] == 'choose name':
                    attr1, attr2 = shuffle(prog['argument'])
                    attr1 = lemmatizer.lemmatize(attr1)
                    attr2 = lemmatizer.lemmatize(attr2)
                    new_programs.append('{}=choose_n({}, {}, {})'.format(result, subject, attr1, attr2))

                elif ' ' in prog['operation']:
                    attr = prog['operation'].split(' ')[1]
                    if len(prog['argument']) == 0:
                        new_programs.append('{}=choose_subj({}, {})'.format(result, subject, attr))
                    else:
                        attr1, attr2 = shuffle(prog['argument'])
                        if flag == "full":
                            new_programs.append('{}=choose_f({}, {})'.format(result, attr1, attr2))
                        else:
                            new_programs.append('{}=choose_attr({}, {}, {}, {})'.format(
                                result, subject, attr, attr1, attr2))

            elif prog['operation'].startswith('different'):
                if ' ' in prog['operation']:
                    attr = prog['operation'].split(' ')[1]
                    new_programs.append('{}=different_attr({}, {})'.format(result, subject, attr))
                else:
                    new_programs.append('{}=different({})'.format(result, subject))

            elif prog['operation'].startswith('same'):
                if ' ' in prog['operation']:
                    attr = prog['operation'].split(' ')[1]
                    new_programs.append('{}=same_attr({}, {})'.format(result, subject, attr))
                else:
                    new_programs.append('{}=same({})'.format(result, subject))

            elif prog['operation'] == 'common':
                new_programs.append('{}=common({})'.format(result, subject))

            else:
                raise ValueError("Unseen Function {}".format(prog))
            # if answer == "yes":
            #    answer = True
            # elif answer == "no":
            #    answer = False
            # elif 'choose' in new_programs[-1]:
            #    _, _, arguments = parse_program(new_programs[-1])
            #    if answer not in arguments:
            #        import pdb
            #        pdb.set_trace()
            # elif answer == "right" and 'choose' in new_programs[-1]:
            #    answer = 'to the right of'
            # elif answer == "left" and 'choose' in new_programs[-1]:
            #    answer = 'to the left of'

        symbolic_programs.append((imageId, question, new_programs, keys[idx], answer))
        success += 1

        # except Exception:
        #    print(program)
        #    fail += 1

        if idx % 10000 == 0:
            sys.stdout.write("finished {}/{} \r".format(success, fail))

    print("finished {}/{}".format(success, fail))
    with open(output_path, 'w') as f:
        json.dump(symbolic_programs, f, indent=2)


def create_inputs(program_files, output):
    def find_all_nums(strings):
        nums = []
        for s in strings:
            if '[' in s and ']' in s:
                nums.append(int(s[1:-1]))
        return nums

    results = []
    for split in program_files:
        # if split == 'submission':
        #    with open('questions/{}_programs_pred.json'.format(split)) as f:
        #        data = json.load(f)
        # else:
        with open(split, "r") as f:
            data = json.load(f)
            print("loading {}".format(split))

        count = 0
        for idx, entry in enumerate(data):
            # for prog in entry[2]:
            programs = entry[2]
            rounds = []
            depth = {}
            cur_depth = 0
            tmp = []
            connection = []
            inputs = []
            returns = []
            tmp_connection = []
            for i, program in enumerate(programs):
                if isinstance(program, list):
                    _, func, args = Constants.parse_program(program[1])
                    returns.append(program[0])
                else:
                    _, func, args = Constants.parse_program(program)
                try:
                    if func == 'relate' or func == 'relate_inv':
                        inputs.append([func, None, None, args[1], None, None, None, None])
                    elif func == 'relate_attr':
                        inputs.append([func, None, None, args[1], args[2], None, None, None])
                    elif func == 'relate_name' or func == 'relate_inv_name':
                        inputs.append([func, None, None, args[1], args[2], None, None, None])
                    elif func == 'select':
                        inputs.append([func, None, None, None, args[0], None, None, None])
                    elif func == 'filter' or func == 'filter_not':
                        inputs.append([func, None, args[1], None, None, None, None, None])
                    elif func == 'filter_h' or func == 'filter_v':
                        inputs.append([func, None, None, None, None, args[1], None, None])
                    elif func == 'verify_h' or func == 'verify_v':
                        inputs.append([func, None, None, None, None, args[0], None, None])
                    elif func == 'query_n':
                        inputs.append([func, None, None, None, None, None, None, None])
                    elif func == 'query_h' or func == 'query_v':
                        inputs.append([func, None, None, None, None, None, None, None])
                    elif func == 'query':
                        inputs.append([func, args[1], None, None, None, None, None, None])
                    elif func == 'query_f':
                        inputs.append([func, args[0], None, None, None, None, None, None])
                    elif func == 'verify':
                        inputs.append([func, None, args[1], None, None, None, None, None])
                    elif func == 'verify_f':
                        inputs.append([func, None, args[0], None, None, None, None, None])
                    elif func == 'verify_rel' or func == 'verify_rel_inv':
                        inputs.append([func, None, None, args[1], args[2], None, None, None])
                    elif func in ['choose_n', 'choose_h', 'choose_v']:
                        inputs.append([func, None, None, None, None, None, args[1], args[2]])
                    elif func == 'choose':
                        inputs.append([func, None, None, None, None, None, args[1], args[2]])
                    elif func == 'choose_subj':
                        inputs.append([func, None, args[2], None, None, None, None, None])
                    elif func == 'choose_attr':
                        inputs.append([func, args[1], None, None, None, None, args[2], args[3]])
                    elif func == 'choose_f':
                        inputs.append([func, None, None, None, None, None, args[0], args[1]])
                    elif func == 'choose_rel_inv':
                        inputs.append([func, None, None, None, args[1], None, args[2], args[3]])
                    elif func in ['same_attr', 'different_attr']:
                        inputs.append([func, None, args[2], None, None, None, None, None])
                    elif func in ['exist', 'or', 'and', 'different', 'same', 'common']:
                        inputs.append([func, None, None, None, None, None, None, None])
                    else:
                        raise ValueError('unknown function {}'.format(func))
                except Exception:
                    print(program)
                    inputs.append([func, None, None, None, None, None, None, None])

                assert len(inputs[-1]) == 8
                if len(find_all_nums(args)) == 0:
                    tmp.append(program)
                    depth[i] = cur_depth
                    tmp_connection.append([i, i])

            connection.append(tmp_connection)
            cur_depth += 1
            rounds.append(tmp)

            while len(depth) < len(programs):
                tmp = []
                tmp_depth = {}
                tmp_connection = []
                for i, program in enumerate(programs):
                    if i in depth:
                        continue
                    else:
                        if isinstance(program, list):
                            _, func, args = Constants.parse_program(program[1])
                        else:
                            _, func, args = Constants.parse_program(program)
                        reliance = find_all_nums(args)
                        if all([_ in depth for _ in reliance]):
                            tmp.append(program)
                            tmp_depth[i] = cur_depth
                            for r in reliance:
                                if r > i:
                                    r = i - 1
                                tmp_connection.append([i, r])
                        else:
                            continue

                if len(tmp_depth) == 0 and len(tmp) == 0 and len(tmp_connection) == 0:
                    break
                else:
                    connection.append(tmp_connection)
                    rounds.append(tmp)
                    cur_depth += 1
                    depth.update(tmp_depth)

            results.append([entry[0], entry[1], returns, inputs, connection, entry[-2], entry[-1]])
            sys.stdout.write("finished {}/{} \r".format(idx, len(data)))

    with open(output, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=str)
    parser.add_argument("--question_dir", type=str, default="questions")
    parser.add_argument("--sg_dir", type=str, default="sceneGraphs")
    parser.add_argument("--meta_dir", type=str, default="meta")
    parser.add_argument("--image_feature_dir", type=str, default="objects")
    parser.add_argument("--glove_path", type=str, default="glove/glove.6B.300d.txt")
    parser.add_argument("--preprocess_dir", type=str, default="preprocess")
    parser.add_argument("--output_dir", type=str, default="preprocess")
    args = parser.parse_args()

    if args.process == "trainval_all":
        raw_data = {}
        for i in range(10):
            train_all_path = os.path.join(args.question_dir, "train_all_questions", "train_all_questions_{}.json".format(i))
            print("loading {}".format(train_all_path))
            with open(train_all_path, "r") as fp:
                raw_data.update(json.load(fp))
        val_all_path = os.path.join(args.question_dir, "val_all_questions.json")

        output_path = os.path.join(args.output_dir, "trainval_all_programs.json")
        preprocess(raw_data, output_path)
    elif args.process == "create_balanced_programs":
        train_bal_path = os.path.join(args.question_dir, "train_balanced_questions.json")
        val_bal_path = os.path.join(args.question_dir, "val_balanced_questions.json")
        testdev_bal_path = os.path.join(args.question_dir, "testdev_balanced_questions.json")

        with open(train_bal_path, "r") as f:
            raw_data = json.load(f)
        with open(val_bal_path, "r") as f:
            raw_data.update(json.load(f))

        output_path = os.path.join(args.output_dir, "trainval_balanced_programs.json")
        preprocess(raw_data, output_path)

        with open(testdev_bal_path, "r") as f:
            raw_dev_data = json.load(f)
        output_path = os.path.join(args.output_dir, "testdev_balanced_programs.json")
        preprocess(raw_dev_data, output_path)
    elif args.process == "create_all_inputs":
        trainval_path = os.path.join(args.preprocess_dir, "trainval_all_programs.json")
        output_path = os.path.join(args.output_dir, "trainval_all_inputs.json")
        create_inputs([trainval_path], output_path)
    elif args.process == "create_inputs":
        trainval_bal_path = os.path.join(args.preprocess_dir, "trainval_balanced_programs.json")
        testdev_bal_path = os.path.join(args.preprocess_dir, "testdev_balanced_programs.json")

        output_path = os.path.join(args.output_dir, "trainval_balanced_inputs.json")
        create_inputs([trainval_bal_path], output_path)
        output_path = os.path.join(args.output_dir, "testdev_balanced_inputs.json")
        create_inputs([testdev_bal_path], output_path)
    elif args.process == "create_ontology":
        trainval_bal_path = os.path.join(args.preprocess_dir, "trainval_balanced_programs.json")
        programs = json.load(open(trainval_bal_path, "r"))
        ontology_dict = dict()
        scene_dict = dict()

        for i, entry in tqdm(enumerate(programs)):
            program = entry[2]
            _, func, arguments = Constants.parse_program(program[-1])
            if func == "query":
                if arguments[1] not in ontology_dict:
                    ontology_dict[arguments[1]] = {entry[-1], }
                else:
                    ontology_dict[arguments[1]].add(entry[-1])
            elif func == "query_f":
                if arguments[0] not in scene_dict:
                    scene_dict[arguments[0]] = {entry[-1], }
                else:
                    scene_dict[arguments[0]].add(entry[-1])
            elif func == "choose_attr":
                if arguments[1] not in ontology_dict:
                    ontology_dict[arguments[1]] = {entry[-1], arguments[2], arguments[3]}
                else:
                    ontology_dict[arguments[1]].add(entry[-1])

        for k, v in ontology_dict.items():
            if k in Constants.ONTOLOGY:
                ontology_dict[k] = list(v - set(Constants.ONTOLOGY[k]))
            else:
                ontology_dict[k] = list(v)

        for k, v in scene_dict.items():
            if k in Constants.ONTOLOGY:
                scene_dict[k] = list(v - set(Constants.ONTOLOGY[k]))
            else:
                scene_dict[k] = list(v)

        save_path = os.path.join(args.output_dir, "ontology.json")
        with open(save_path, "w") as fp:
            json.dump(
                ontology_dict,
                fp,
                indent=4
            )
        save_path = os.path.join(args.output_dir, "scene.json")
        with open(save_path, "w") as fp:
            json.dump(
                scene_dict,
                fp,
                indent=4
            )
    elif args.process == "create_vqa_dataset":
        trainval_bal_path = os.path.join(args.preprocess_dir, "trainval_balanced_programs.json")
        testdev_bal_path = os.path.join(args.preprocess_dir, "testdev_balanced_programs.json")

        tracer = GQASceneGraphTrace()
        tracer.load_gqa_sg(args.sg_dir)
        tracer.transform_to_sg()

        programs = json.load(open(trainval_bal_path, "r"))
        outputs = []
        for program in tqdm(programs):
            prog_new = tracer.tracer(program)
            outputs.append(json.dumps(prog_new))
        output_path = os.path.join(args.output_dir, "trainval_bal_inputs.json")
        json.dump(outputs, open(output_path, "w"), indent=2)

        programs = json.load(open(testdev_bal_path, "r"))
        outputs = []
        for program in tqdm(programs):
            prog_new = tracer.tracer(program)
            outputs.append(json.dumps(prog_new))
        output_path = os.path.join(args.output_dir, "testdev_bal_inputs.json")
        json.dump(outputs, open(output_path, "w"), indent=2)
    elif args.process == "create_glove_emb":
        def loadGloveModel(gloveFile):
            print("Loading Glove Model")
            f = open(gloveFile, 'r')
            model = {}
            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
            print("Done.", len(model), " words loaded!")
            return model

        emb = loadGloveModel(args.glove_path)

        vocab_path = os.path.join(args.meta_dir, "full_vocab.json")
        with open(vocab_path, "r") as fp:
            vocab = json.load(fp)
        found, miss = 0, 0
        en_emb = np.random.randn(len(vocab), 300)
        for w, i in vocab.items():
            if w.lower() in emb:
                en_emb[i] = emb[w.lower()]
                found += 1
            elif ' ' in w:
                for w_elem in w.split(' '):
                    if w_elem.lower() in emb:
                        en_emb[i] += emb[w_elem.lower()]
            else:
                print(w)
                miss += 1
        print("found = {}, miss = {}".format(found, miss))

        output_path = os.path.join(args.output_dir, "en_emb.npy")
        np.save(output_path, en_emb)
    elif args.process == "create_image_features":
        h5_list = [fn for fn in os.listdir(args.image_feature_dir) if fn.endswith(".h5")]
        print("there are {} h5 files".format(len(h5_list)))
        object_info = json.load(open(os.path.join(args.image_feature_dir, "gqa_objects_info.json"), "r"))
        """
        {
            "[iid]": {
                "width": 500, 
                "objectsNum": 24, 
                "idx": 4550, 
                "height": 333, 
                "file": 9
            }
        }
        """

        file2feature = dict()
        for iid, info in tqdm(object_info.items()):
            file = info["file"]
            idx = info["idx"]
            if file not in file2feature:
                file2feature[file] = dict()

            info["iid"] = iid
            file2feature[file][idx] = info

        for h5 in h5_list:
            h5_path = os.path.join(args.image_feature_dir, h5)
            file = int(h5.split(".")[0].split("_")[-1])
            print("process h5 file: {}".format(h5_path))
            with h5py.File(h5_path, "r") as fp:
                for i in tqdm(range(len(fp["features"]))):
                    objectsNum = file2feature[file][i]["objectsNum"]
                    iid = file2feature[file][i]["iid"]

                    feature = fp["features"][i][:objectsNum]  # [num_objects, 2048]
                    bbox = fp["bboxes"][i][:objectsNum]  # [num_objects, 4], (x1, y1, x2, y2)

                    np.savez(os.path.join(args.output_dir, "{}.npz".format(iid)),
                             feature=feature,
                             bbox=bbox,
                             width=file2feature[file][i]["width"],
                             height=file2feature[file][i]["height"])
    elif args.process == "create_mmn_features":
        import zipfile

        zip_file = os.path.join(args.image_feature_dir, "gqa_visual_features.zip")
        fp = zipfile.ZipFile(zip_file, "r")
        for fn in tqdm(fp.namelist()):
            if fn.endswith("npz"):
                fp.extract(fn, args.output_dir)
                data = np.load(os.path.join(args.output_dir, fn))
                data = dict(data)
                data.pop("soft_labels")
                np.savez(os.path.join(args.output_dir, fn), **data)
        fp.close()
