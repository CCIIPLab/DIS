# -*- coding: utf-8 -*-

import os
import sys
root_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
print("append project dir {} to environment".format(root_dir))
sys.path.append(root_dir)
import json
import argparse

from tqdm import tqdm
from nltk import WordNetLemmatizer
wnl = WordNetLemmatizer()

import Constants

"""
==> Scene graph in GQA:
{
    "[iid]": {
        "width": int,
        "height": int,
        "location": str or None,
        "weather": str or None,
        "objects": {
            "[oid]": {
                "name": str,
                "x": int,
                "y": int,
                "w": int,
                "h": int,
                "attributes": [str],
                "relations": [
                    {"object": "[oid]", "name": str},
                    {"object": "[oid]", "name": str},
                    ...
                ]
            }
        }
    }
}
==> Program format:
[
    [
        "[iid]",
        "Is the sky dark?",
        [
            "[2486325]=select(sky)",
            "?=verify([0], dark)"
        ],
        "[qid]",
        "[answer]"
    ],
    ...
]
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_trainval", default=False, action="store_true")
    parser.add_argument("--do_testdev", default=False, action="store_true")
    parser.add_argument("--do_submission", default=False, action="store_true")

    parser.add_argument("--sg_dir", default="data/sceneGraphs", type=str)
    parser.add_argument("--data_path", default="data/gqa_program_t5", type=str)

    args = parser.parse_args()

    return args

"""
用于存储场景图节点的类
节点属性包括：
==> name: str
==> attributes: list
==> bbox: [x, y, w, h]
==> id: str
为了使节点之间能够相互连接，每个节点会持有一个list，用于保存该节点
和其它节点的关系，这个list的元素是一个元组：
==> relations_out: [(relation_name, node_ptr), ...]
==> relations_in: [(relation_name, node_ptr), ...]
"""
class SGNode(object):
    def __init__(self, **kwargs):
        assert "name" in kwargs and \
            "attributes" in kwargs and \
            "bbox" in kwargs and \
            "id" in kwargs

        self.name = kwargs["name"]
        self.attributes = kwargs["attributes"]
        self.bbox = kwargs["bbox"]
        self.id = kwargs["id"]

        self.relations_out = []
        self.relations_in = []

    def __str__(self):
        print_keys = ["name", "attributes", "bbox", "id"]
        params = []
        for k in print_keys:
            v = getattr(self, k)
            params.append(k)
            params.append(v)
        return "SGNode({}='{}', {}={}, {}={}, {}='{}')".format(
            *params
        )

    def find_outgoing_nodes(self, relation_name: str,
                            out_names=None):
        ret = []
        for rel in self.relations_out:
            if relation_name == rel[0]:
                if out_names is not None:
                    if rel[1].name in out_names:
                        ret.append(rel[1])
                else:
                    ret.append(rel[1])

        return ret

    def find_ingoing_nodes(self, relation_name: str,
                           in_names=None):
        ret = []
        for rel in self.relations_in:
            if relation_name == rel[0]:
                if in_names is not None:
                    if rel[1].name in in_names:
                        ret.append(rel[1])
                else:
                    ret.append(rel[1])

        return ret

"""
SG对象用于保存整个Scene Graph的信息。
属性包括：
==> location: None or str
==> weather: None or str
==> height: int
==> width: int
==> id: str
为了记录所有的节点信息，每个SG会持有一个字典，用于保存节点id到节点
对象的映射。
==> node_dict: dict
"""
class SG(object):
    def __init__(self, **kwargs):
        assert "location" in kwargs and \
            "weather" in kwargs and \
            "height" in kwargs and \
            "width" in kwargs and \
            "id" in kwargs

        self.location = kwargs["location"]
        self.weather = kwargs["weather"]
        self.height = kwargs["height"]
        self.width = kwargs["width"]
        self.id = kwargs["id"]

        self.node_dict = dict()

    def select_object_from_candidates(self, candidates: set):
        ret = []
        for n in self.node_dict.values():
            n: SGNode
            if n.name in candidates:
                ret.append(n)

        return ret

    def get_all_names(self):
        ret = {self.weather, self.location}
        for node in self.node_dict.values():
            ret.add(node.name)
        return ret

    # 根据GQA的场景图字典实例化一个SG对象；
    @classmethod
    def instance(cls, iid: str, sg_dict: dict):
        # 初始化一个SG对象
        if "location" not in sg_dict:
            sg_dict["location"] = None
        if "weather" not in sg_dict:
            sg_dict["weather"] = None
        sg_dict["id"] = iid
        sg = cls(**sg_dict)

        # 然后加入节点
        for oid, odict in sg_dict["objects"].items():
            local_node = SGNode(id=oid,
                                name=odict["name"],
                                bbox=[odict["x"], odict["y"], odict["w"], odict["h"]],
                                attributes=odict["attributes"])
            sg.node_dict[oid] = local_node

        # 最后加入关系
        for oid, odict in sg_dict["objects"].items():
            local_node = sg.node_dict[oid]
            local_node: SGNode

            for rdict in odict["relations"]:
                local_node.relations_out.append(
                    (rdict["name"], sg.node_dict[rdict["object"]])
                )
                sg.node_dict[rdict["object"]].relations_in.append(
                    (rdict["name"], local_node)
                )

        return sg

"""
推理的逻辑函数；
"""
def get_index(index: str):
    return int(index.lstrip("[").rstrip("]"))

def get_all_candidates(arg: set,
                       from_hypernym=False,
                       from_ontology=False):
    for a in arg.copy():
        if from_hypernym and a in Constants.hypernym:
            arg.update(Constants.hypernym[a])
        if from_ontology and a in Constants.ONTOLOGY:
            arg.update(Constants.ONTOLOGY[a])

    return arg

def filter_with_attribute(node_list, attribute):
    if isinstance(attribute, str):
        attribute = {attribute, }
    return list(filter(lambda x: len(attribute & set(x.attributes)) > 0,
                       node_list))

def filter_without_attribute(node_list, attribute):
    # 这里应该找到和attribute同属于一个ontology的其它属性
    # candidates = set()
    # for k, v in Constants.ONTOLOGY.items():
    #     if attribute in v:
    #         candidates.update(v)
    # candidates.remove(attribute)
    # res = list(filter(lambda x: len(candidates & set(x.attributes)) > 0,
    #                    node_list))

    res = list(filter(lambda x: attribute not in x.attributes,
                          node_list))

    return res

def get_all_candidates_in_question(arg: set, question:str,
                                   from_hypernym=False, from_ontology=False):
    for a in arg.copy():
        if from_hypernym and a in Constants.hypernym:
            for i in Constants.hypernym[a]:
                if i in question:
                    arg.add(i)
        if from_ontology and a in Constants.ONTOLOGY:
            for i in Constants.hypernym[a]:
                if i in question:
                    arg.add(i)

    return arg

def get_center(node: SGNode):
    return (node.bbox[0]+node.bbox[2]/2,
            node.bbox[1]+node.bbox[3]/2)

def filter_node_with_direction(node_list, sg, direction: str):
    assert direction in {"left", "right", "top", "bottom", "middle"}

    center_coors = [get_center(n) for n in node_list]
    if direction == "left":
        index = list(sorted(enumerate(center_coors), key=lambda x: x[1][0]))[0][0]
    elif direction == "right":
        index = list(sorted(enumerate(center_coors), key=lambda x: x[1][0]))[-1][0]
    elif direction == "middle":
        index = list(sorted(enumerate(center_coors),
                            key=lambda x: abs(x[1][0] - sg.width / 2)))[0][0]
    elif direction == "top":
        index = list(sorted(enumerate(center_coors), key=lambda x: x[1][1]))[0][0]
    elif direction == "bottom":
        index = list(sorted(enumerate(center_coors), key=lambda x: x[1][1]))[-1][0]
    else:
        raise ValueError("Invalid direction : {}".format(direction))

    return node_list[index]

def filter_node_with_names(node_list, names: set):
    ret = []
    for node in node_list:
        node: SGNode
        if node.name in names:
            ret.append(node)

    return ret

def find_node_with_out_relation(node_list, relation_name):
    ret = []
    for node in node_list:
        node: SGNode
        ret.extend(node.find_outgoing_nodes(relation_name))
    return ret

def find_node_with_in_relation(node_list, relation_name):
    ret = []
    for node in node_list:
        node: SGNode
        ret.extend(node.find_ingoing_nodes(relation_name))
    return ret

def filter_attributes_of_aspect(node: SGNode, aspect: str):
    r1, r2 = aspect.split()
    if r1 == "same":
        return set(node.attributes) & set(Constants.ONTOLOGY[r2])
    elif r1 == "different":
        return set(Constants.ONTOLOGY[r2]) - set(node.attributes)
    else:
        raise ValueError("Invalid aspect: {}".format(aspect))

def get_h_for_node(node: SGNode, sg: SG):
    center = get_center(node)
    center_sg = (sg.width / 2, sg.height / 2)

    if center[0] < center_sg[0]:
        return "left"
    else:
        return "right"

def get_v_for_node(node: SGNode, sg: SG):
    center = get_center(node)
    center_sg = (sg.width / 2, sg.height / 2)

    if center[1] < center_sg[1]:
        return "top"
    else:
        return "bottom"

def get_attributes_from_node_list(node_list):
    res = set()
    for node in node_list:
        node: SGNode
        res.update(node.attributes)
    return res

def get_gt_nodes(sg: SG, gt_str: str):
    gt_list = gt_str.lstrip("[").rstrip("]").split(",")
    if "" in gt_list:
        gt_list = []
    gt_nodes = [sg.node_dict[i] for i in gt_list]
    return gt_nodes

def merge_nodes_from_gt(node_list, sg:SG, gt_str: str):
    if gt_str == "?": return node_list
    gt_list = gt_str.lstrip("[").rstrip("]").split(",")
    if "" in gt_list:
        gt_list = []

    current_nodes = set(n.id for n in node_list)
    for i in gt_list:
        if i not in current_nodes:
            node_list.append(sg.node_dict[i])

    return node_list

"""
根据Program在场景图中进行推理；
"""
class GQASceneGraphTrace(object):
    def __init__(self):
        pass

    # 载入GQA的训练集和验证集场景图
    def load_gqa_sg(self, sg_dir):
        sg_names = ["train_sceneGraphs.json", "val_sceneGraphs.json"]

        self.sg_dict = dict()
        for sn in sg_names:
            sg_path = os.path.join(sg_dir, sn)
            assert os.path.exists(sg_path)
            print("loading scene graph : {}".format(sn))
            with open(sg_path, "r") as fp:
                self.sg_dict.update(json.load(fp))
            print("now {} scene graphs in total.".format(len(self.sg_dict)))

    # 将GQA场景图转换为SG对象
    def transform_to_sg(self):
        assert hasattr(self, "sg_dict") and \
            len(self.sg_dict) > 0

        print("transform scene graph dict to SG class")
        self.gqa_sg = dict()
        for iid, sg_dict in tqdm(self.sg_dict.items()):
            self.gqa_sg[iid] = SG.instance(iid, sg_dict)

    # 在GQA场景图上根据Program进行trace
    def tracer(self, entry):
        def find_all_nums(strings):
            nums = []
            for s in strings:
                if '[' in s and ']' in s:
                    nums.append(int(s[1:-1]))
            return nums

        # 首先获取SG对象，如果存在ground truth的场景图，那么进行推理获取中间结果；
        # 如果不存在，那么直接赋值为空
        if entry[0] in self.gqa_sg:
            sg = self.gqa_sg[entry[0]]
            sg: SG
            question = entry[1].strip("?").lower()

            # 主要有三种类型的results：
            # ==> list of objects, xywh
            # ==> true/false
            # ==> answer
            results = []
            results_gt = []
            for i, prog in enumerate(entry[2]):
                _, func, args = Constants.parse_program(prog)

                # 第一个function只有四种可能
                # {'choose_f', 'query_f', 'select', 'verify_f'}
                if func == "select":
                    candidates = get_all_candidates({args[0]}, from_hypernym=True)
                    res = sg.select_object_from_candidates(candidates)

                    res = merge_nodes_from_gt(res, sg, _)
                    results.append(res)
                    results_gt.append(_)
                elif func == "verify":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]
                    res = "false"
                    for node in dep_res:
                        node: SGNode
                        if args[1] in node.attributes:
                            res = "true"
                            break
                    results.append(res)
                    results_gt.append(entry[-1])
                elif func == "filter":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]
                    res = filter_with_attribute(dep_res, args[1])

                    res = merge_nodes_from_gt(res, sg, results_gt[dep_index])
                    # 使用Ground truth填充
                    # if len(res) == 0:
                    #     res = get_gt_nodes(sg, results_gt[dep_index])
                    results.append(res)
                    results_gt.append(results_gt[dep_index])
                elif func == "filter_not":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]
                    res = filter_without_attribute(dep_res, args[1])

                    res = merge_nodes_from_gt(res, sg, results_gt[dep_index])
                    results.append(res)
                    results_gt.append(results_gt[dep_index])
                elif func == "filter_h":
                    # {"left", "right", "top", "middle", "bottom"}
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]
                    try:
                        res = filter_node_with_direction(dep_res, sg, args[1])

                        res = merge_nodes_from_gt([res], sg, results_gt[dep_index])
                    except:
                        res = dep_res

                    results.append(res)
                    results_gt.append(results_gt[dep_index])
                elif func == "filter_v":
                    raise ValueError("[{}]: not implemented!".format(func))
                elif func == "relate":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]
                    res = find_node_with_out_relation(dep_res, args[1])

                    res = merge_nodes_from_gt(res, sg, _)
                    results.append(res)
                    results_gt.append(_)
                elif func == "relate_inv":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]
                    res = find_node_with_in_relation(dep_res, args[1])

                    res = merge_nodes_from_gt(res, sg, _)
                    results.append(res)
                    results_gt.append(_)
                elif func == "relate_attr":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]

                    res = []
                    for node in dep_res:
                        node: SGNode
                        attribute_cands = filter_attributes_of_aspect(node, args[1])
                        target_cands = sg.select_object_from_candidates(
                            get_all_candidates({args[2], }, True)
                        )
                        target = filter_with_attribute(target_cands, attribute_cands)
                        # 过滤node本身
                        target = [t for t in target if t.id != node.id]
                        res.extend(target)

                    res = merge_nodes_from_gt(res, sg, _)
                    results.append(res)
                    results_gt.append(_)
                elif func == "relate_name":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]
                    try:
                        target_cands = find_node_with_out_relation(dep_res, args[1])
                        res = filter_node_with_names(target_cands,
                                                     get_all_candidates({args[2], }, True))
                    except:
                        res = dep_res

                    res = merge_nodes_from_gt(res, sg, _)
                    results.append(res)
                    results_gt.append(_)
                elif func == "relate_inv_name":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]
                    target_cands = find_node_with_in_relation(dep_res, args[1])
                    res = filter_node_with_names(target_cands,
                                                 get_all_candidates({args[2], }, True))
                    # 直接加入Ground truth结果
                    res = merge_nodes_from_gt(res, sg, _)
                    results.append(res)
                    results_gt.append(_)
                elif func == "verify_h":
                    # {"left", "right"}
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]

                    res = "false"
                    for node in dep_res:
                        node: SGNode
                        if get_h_for_node(node, sg) == args[1]:
                            res = "true"
                            break
                    results.append(res)
                    results_gt.append(entry[-1])
                elif func == "verify_f":
                    all_names = sg.get_all_names()
                    if args[0] in all_names:
                        results.append("true")
                    else:
                        results.append("false")
                    results_gt.append(entry[-1])
                elif func == "verify_rel":
                    dep_index = get_index(args[0])  # [3362152]=verify_rel([0], to the left of, food)
                    dep_res = results[dep_index]

                    try:
                        target_cands = get_all_candidates_in_question(
                            {args[2], },
                            question,
                            True
                        )
                        res = []
                        for node in dep_res:
                            node: SGNode
                            res.extend(node.find_outgoing_nodes(args[1], target_cands))

                        if len(res) > 0:
                            results.append("true")
                        else:
                            results.append("false")
                    except:
                        results.append("true" if entry[-1] == "yes" else "false")
                    results_gt.append(entry[-1])
                elif func == "verify_rel_inv":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]

                    target_cands = get_all_candidates_in_question(
                        {args[2], },
                        question,
                        True
                    )
                    res = []
                    for node in dep_res:
                        node: SGNode
                        res.extend(node.find_ingoing_nodes(args[1], target_cands))

                    if len(res) > 0:
                        results.append("true")
                    else:
                        results.append("false")
                    results_gt.append(entry[-1])
                elif func == "verify_v":
                    # {"bottom", "top"}
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]

                    res = "false"
                    for node in dep_res:
                        node: SGNode
                        if get_v_for_node(node, sg) == args[1]:
                            res = "true"
                            break
                    results.append(res)
                    results_gt.append(entry[-1])
                elif func == "query_n":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]

                    # 担心前面没有找到
                    if len(dep_res) == 0:
                        results.append(entry[-1])
                    else:
                        results.append(dep_res[0].name)
                    results_gt.append(entry[-1])
                elif func == "choose_n":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]
                    names = set(r.name for r in dep_res)
                    candidates = get_all_candidates(
                        {args[1], args[2]},
                        True
                    )
                    res = (names & candidates)
                    if len(res) == 0:
                        results.append(entry[-1])
                    else:
                        results.append(res.pop())
                    results_gt.append(entry[-1])
                elif func == "query_h" or func == "choose_h":
                    # {"left", "right"}
                    try:
                        dep_index = get_index(args[0])
                        dep_res = results[dep_index]
                        res = get_h_for_node(dep_res[0], sg)
                    except:
                        res = None

                    if res is None:
                        results.append(entry[-1])
                    else:
                        results.append(res)
                    results_gt.append(entry[-1])
                elif func == "query_v" or func == "choose_v":
                    # {"top", "bottom"}
                    try:
                        dep_index = get_index(args[0])
                        dep_res = results[dep_index]
                        res = get_v_for_node(dep_res[0], sg)
                    except:
                        res = entry[-1]
                    results.append(res)
                    results_gt.append(entry[-1])
                elif func == "query":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]

                    attributes = get_attributes_from_node_list(dep_res)
                    if args[1] not in Constants.ONTOLOGY:
                        res = entry[-1]
                    else:
                        attribute_cands = set(Constants.ONTOLOGY[args[1]])
                        res = (attributes & attribute_cands)
                        if len(res) > 0:
                            res = res.pop()
                        else:
                            res = entry[-1]
                    results.append(res)
                    results_gt.append(entry[-1])
                elif func == "query_f":
                    # 直接使用答案算了
                    results.append(entry[-1])
                    results_gt.append(entry[-1])
                elif func == "choose":
                    try:
                        # TODO: 依赖多个object问题
                        dep_index = get_index(args[0])
                        dep_res = results[dep_index]

                        attributes = set(dep_res[0].attributes)
                        attribute_cands = {args[1], args[2]}
                        res = (attributes & attribute_cands)
                    except:
                        res = set()

                    if len(res) == 0:
                        results.append(entry[-1])
                    else:
                        results.append(res.pop())
                    results_gt.append(entry[-1])
                elif func == "choose_subj":
                    # 直接用答案把
                    results.append(entry[-1])
                    results_gt.append(entry[-1])
                elif func == "choose_attr":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]

                    try:
                        attributes = set()
                        for node in dep_res:
                            node: SGNode
                            attributes.update(node.attributes)
                        candidates = {args[2], args[3]}

                        res = (attributes & candidates)
                    except:
                        res = set()
                    if len(res) == 0:
                        results.append(entry[-1])
                    else:
                        results.append(res.pop())
                    results_gt.append(entry[-1])
                elif func == "choose_f":
                    names = sg.get_all_names()
                    candidates = {args[0], args[1]}
                    res = (names & candidates)

                    if len(res) == 0:
                        res = entry[-1]
                    else:
                        res = res.pop()
                    results.append(res)
                    results_gt.append(entry[-1])
                elif func == "choose_rel_inv":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]

                    names = get_all_candidates(
                        {args[1], },
                        True
                    )

                    res = []
                    for node in dep_res:
                        node: SGNode
                        res.extend([(args[2], i) for i in node.find_ingoing_nodes(args[2], names)])
                        res.extend([(args[3], i) for i in node.find_ingoing_nodes(args[3], names)])
                    if len(res) == 0:
                        results.append(entry[-1])
                    else:
                        res = res[0][0]
                        rets = {"left", "right", "front", "behind"}
                        for r in rets:
                            if r in res:
                                results.append(r)
                    results_gt.append(entry[-1])
                elif func == "same_attr":
                    try:
                        obj1 = results[get_index(args[0])]
                        obj2 = results[get_index(args[1])]

                        obj1_attrs = get_attributes_from_node_list(obj1)
                        obj2_attrs = get_attributes_from_node_list(obj2)

                        candidates = set(Constants.ONTOLOGY[args[2]])

                        if len(obj1_attrs & obj2_attrs & candidates) > 0:
                            results.append("true")
                        else:
                            results.append("false")
                    except:
                        results.append("true" if entry[-1] == "yes" else "false")
                    results_gt.append(entry[-1])
                elif func == "different_attr":
                    obj1 = results[get_index(args[0])]
                    obj2 = results[get_index(args[1])]

                    obj1_attrs = get_attributes_from_node_list(obj1)
                    obj2_attrs = get_attributes_from_node_list(obj2)

                    candidates = set(Constants.ONTOLOGY[args[2]])

                    if len(obj1_attrs & obj2_attrs & candidates) > 0:
                        results.append("false")
                    else:
                        results.append("true")
                    results_gt.append(entry[-1])
                elif func == "exist":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]
                    if len(dep_res) > 0:
                        results.append('true')
                    else:
                        results.append('false')

                    res_gt = results_gt[dep_index]
                    if res_gt == "[]":
                        results_gt.append("false")
                    else:
                        results_gt.append("true")
                elif func == "or":
                    dep_res1 = results[get_index(args[0])]
                    dep_res2 = results[get_index(args[1])]

                    res_dep1 = True if dep_res1 == "true" else False
                    res_dep2 = True if dep_res2 == "true" else False

                    if res_dep1 or res_dep2:
                        results.append("true")
                    else:
                        results.append("false")
                    results_gt.append(entry[-1])
                elif func == "and":
                    try:
                        dep_res1 = results[get_index(args[0])]
                        dep_res2 = results[get_index(args[1])]

                        res_dep1 = True if dep_res1 == "true" else False
                        res_dep2 = True if dep_res2 == "true" else False

                        if res_dep1 and res_dep2:
                            results.append("true")
                        else:
                            results.append("false")
                    except:
                        results.append("true" if entry[-1] == "yes" else "false")
                    results_gt.append(entry[-1])
                elif func == "different":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]

                    names = set(wnl.lemmatize(r.name) for r in dep_res)
                    if len(names) > 1:
                        results.append("true")
                    else:
                        results.append("false")
                    results_gt.append(entry[-1])
                elif func == "same":
                    dep_index = get_index(args[0])
                    dep_res = results[dep_index]

                    names = set(wnl.lemmatize(r.name) for r in dep_res)
                    if len(names) > 1:
                        results.append("false")
                    else:
                        results.append("true")
                    results_gt.append(entry[-1])
                elif func == "common":
                    obj1 = results[get_index(args[0])]
                    obj2 = results[get_index(args[1])]

                    obj1_attrs = get_attributes_from_node_list(obj1)
                    obj2_attrs = get_attributes_from_node_list(obj2)
                    common_attrs = obj1_attrs & obj2_attrs
                    res = set()
                    for c in common_attrs:
                        if c in Constants.BBOX_ATTRIBUTES:
                            res.add(Constants.BBOX_ATTR[Constants.BBOX_ATTRIBUTES[c][0][0]])
                    if len(res) == 0:
                        results.append(entry[-1])
                    else:
                        res = res.pop()
                        results.append(res)
                    results_gt.append(entry[-1])
                else:
                    raise ValueError("Not implemented operation: {}".format(func))
        else:
            results = []

        # 构建成输入的格式
        inputs = []

        cur_depth = 0
        tmp = []
        depth = {}
        tmp_connection = []
        connection = []
        rounds = []
        for i, program in enumerate(entry[2]):
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
                    if len(args) == 2:
                        args = [args[0]] + args[1].split()
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
                print(program, entry)
                raise ValueError("Program parse error!")
                # inputs.append([func, None, None, None, None, None, None, None])

            assert len(inputs[-1]) == 8
            if len(find_all_nums(args)) == 0:
                tmp.append(program)
                depth[i] = cur_depth
                tmp_connection.append([i, i])

        connection.append(tmp_connection)
        cur_depth += 1
        rounds.append(tmp)

        while len(depth) < len(entry[2]):
            tmp = []
            tmp_depth = {}
            tmp_connection = []
            for i, program in enumerate(entry[2]):
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

        # 处理中间结果
        new_results = []
        for r in results:
            if isinstance(r, list):
                r = [i.bbox for i in r]
            new_results.append(r)
        result_names = []
        for r in results:
            if isinstance(r, list):
                r = [i.name for i in r]
            result_names.append(r)

        new_entry = [
            entry[0],
            entry[1],
            new_results,
            result_names,
            inputs,
            connection,
            entry[-2],
            entry[-1]
        ]

        return new_entry

if __name__ == "__main__":
    args = parse_args()

    sg_dir = args.sg_dir
    tracer = GQASceneGraphTrace()
    tracer.load_gqa_sg(sg_dir)
    tracer.transform_to_sg()

    if args.do_trainval:
        all_programs_path = os.path.join(args.data_path, "trainval_pred_programs.json")
    elif args.do_testdev:
        all_programs_path = os.path.join(args.data_path, "testdev_pred_programs.json")
    elif args.do_submission:
        all_programs_path = os.path.join(args.data_path, "submission_programs.json")
    else:
        raise ValueError("Flag do_trainval|do_testdev|do_submission should be set.")

    with open(all_programs_path, "r") as fp:
        all_programs = json.load(fp)

    outputs = []
    error_ids = []
    for entry in tqdm(all_programs):
        try:
            output = tracer.tracer(entry)
        except:
            print(entry)
            error_ids.append(entry[-2])
            continue
        outputs.append(json.dumps(output))

    print(len(outputs))
    if args.do_trainval:
        save_path = os.path.join(args.data_path, "trainval_balance_inputs.json")
    elif args.do_testdev:
        save_path = os.path.join(args.data_path, "testdev_balance_inputs.json")
    elif args.do_submission:
        save_path = os.path.join(args.data_path, "submission_inputs.json")
    with open(save_path, "w") as fp:
        json.dump(outputs, fp)