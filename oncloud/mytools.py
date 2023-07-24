import copy

import torch
import numpy as np
import cv2
from typing import Union
import time
def _layeridx2multmodelidx(layeridx, block_idx, block_choice_idx, layer_len_list):  # layeridx = 2, 3, 4
    multiblockidx = [0, block_choice_idx]
    for i in range(2, layeridx):
        multiblockidx[0] += layer_len_list[i - 2]
    multiblockidx[0] += block_idx
    multiblockidxstr = "multiblocks." + str(multiblockidx[0]) + "." + str(multiblockidx[1]) + "."
    return multiblockidxstr


def _headlayer2multiResDetLayer(layeridx, block_idx):
    newlayeridx = 2 if layeridx == 1 else layeridx
    newblockidx = block_idx if layeridx != 2 else block_idx + 3
    new_key = "model.backbone." + "layer" + str(newlayeridx) + "." + str(newblockidx) + ".0."
    return new_key


def no_new_subnet(subnet):
    no_new = True
    for i in range(len(subnet)):
        for j in range(len(subnet[i])):
            if subnet[i][j] != 0:
                no_new = False
    return no_new


def load_to_multimodel(path, multimodel, part="head", freeze_head=False):
    pretrained_model_dict = torch.load(path, map_location=torch.device('cpu'))
    multimodel_dict = multimodel.state_dict()
    state = {}
    if part == "head":  # load the head and the original main subnet, freeze this subnet and the head
        key_list = ["fpn", "class_net", "box"]
        for k, v in pretrained_model_dict.items():
            head_flag = False
            for _k in key_list:
                if _k in k:
                    key = "model." + k
                    state[key] = v
                    head_flag = True
            if not head_flag:
                k_list = k.split('.')
                if len(k_list) <= 3:
                    key = "model." + k
                else:
                    k_list = k.split('.', 3)
                    key = _headlayer2multiResDetLayer(int(k_list[1][-1]), int(k_list[2]))
                    key += k_list[3]
                state[key] = v
        # for k, v in state.items():
        #     print("headkeys:", k)
        multimodel_dict.update(state)
        multimodel.load_state_dict(multimodel_dict)
        if freeze_head:
            for (name, parameter) in multimodel.named_parameters():
                if name in state:
                    parameter.requires_grad = False
        return multimodel, state
    else:
        # multimodel_dict.items():model.backbone.conv1.weight, model.backbone.layer2.0.0.conv1.weight...
        # pretrained_model_dict.items():conv1.weight,multiblocks.0.0.conv1.weight...
        for k, v in multimodel_dict.items():
            if k == "anchors.boxes":
                continue
            k_list = k.split('.')
            if k_list[2] == "conv1" or k_list[2] == "bn1":
                state[k] = pretrained_model_dict[k[15:]]
            else:
                k_list = k.split('.', 5)
                if k_list[1] != "fpn" and k_list[1] != "class_net" and k_list[1] != "box_net":
                    multiblockidxstr = _layeridx2multmodelidx(layeridx=int(k_list[2][-1]), block_idx=int(k_list[3]),
                                                              block_choice_idx=int(k_list[4]), layer_len_list=[7, 6, 3])
                    multiblockkey = multiblockidxstr + k_list[5]
                    state[k] = pretrained_model_dict[multiblockkey]
        # for k, v in state.items():
        #     print("multimodelkeys:", k)
        multimodel_dict.update(state)
        multimodel.load_state_dict(multimodel_dict)
        return multimodel


def freeze_main_subnet(multimodel, state):
    for (k, v) in multimodel.named_parameters():
        # print(k)
        if k in state:
            # print(";;;", k)
            v.requires_grad = False
    multimodel.model.backbone.conv1.eval()
    multimodel.model.backbone.bn1.eval()
    for blockidx in range(len(multimodel.model.backbone.layer2)):
        multimodel.model.backbone.layer2[blockidx][0].eval()
    for blockidx in range(len(multimodel.model.backbone.layer3)):
        multimodel.model.backbone.layer3[blockidx][0].eval()
    for blockidx in range(len(multimodel.model.backbone.layer4)):
        multimodel.model.backbone.layer4[blockidx][0].eval()
    multimodel.model.fpn.eval()
    multimodel.model.class_net.eval()
    multimodel.model.box_net.eval()


def freeze_bn(multimodel):
    # for layer in multimodel.model.modules():
    #     if isinstance(layer, torch.nn.BatchNorm2d):
    #         layer.eval()
    multimodel.model.backbone.bn1.eval()
    for blockidx in range(len(multimodel.model.backbone.layer2)):
        multimodel.model.backbone.layer2[blockidx][0].bn1.eval()
        multimodel.model.backbone.layer2[blockidx][0].bn2.eval()
        multimodel.model.backbone.layer2[blockidx][0].bn3.eval()
        if multimodel.model.backbone.layer2[blockidx][0].downsample is not None:
            multimodel.model.backbone.layer2[blockidx][0].downsample[1].eval()
    for blockidx in range(len(multimodel.model.backbone.layer3)):
        multimodel.model.backbone.layer3[blockidx][0].bn1.eval()
        multimodel.model.backbone.layer3[blockidx][0].bn2.eval()
        multimodel.model.backbone.layer3[blockidx][0].bn3.eval()
        if multimodel.model.backbone.layer3[blockidx][0].downsample is not None:
            multimodel.model.backbone.layer3[blockidx][0].downsample[1].eval()
    for blockidx in range(len(multimodel.model.backbone.layer4)):
        multimodel.model.backbone.layer4[blockidx][0].bn1.eval()
        multimodel.model.backbone.layer4[blockidx][0].bn2.eval()
        multimodel.model.backbone.layer4[blockidx][0].bn3.eval()
        if multimodel.model.backbone.layer4[blockidx][0].downsample is not None:
            multimodel.model.backbone.layer4[blockidx][0].downsample[1].eval()
    for layer in multimodel.model.fpn.modules():
        # if isinstance(layer, torch.nn.BatchNorm2d):
        layer.eval()
    for layer in multimodel.model.class_net.modules():
        # if isinstance(layer, torch.nn.BatchNorm2d):
        layer.eval()
    for layer in multimodel.model.box_net.modules():
        # print(layer)
        # if isinstance(layer, torch.nn.BatchNorm2d):
        layer.eval()


def get_val_subnets(model):
    # subnet_demo = model.generate_random_subnet()
    subnet_demo = [[2, 99, 99, 2, 99, 99, 0], [2, 99, 99, 2, 99, 99], [2, 99, 99]]
    subnet_demo1 = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
    skip_rate = [0.05 + 0.095 * i for i in range(11)]
    distill_next_rates = [0. + 0.2 * i for i in range(6)]
    # for _ in range(4):
    subnets = [subnet_demo1]
    for i in range(len(skip_rate)):
        for distill_next_rate in distill_next_rates:
            for test_time in range(10):
                subnet = []
                for layeridx in range(len(subnet_demo)):
                    blockidx = 0
                    layer_choice = []
                    while blockidx < len(subnet_demo[layeridx]):
                        choices = [0]
                        if blockidx < len(subnet_demo[layeridx]) - 1:
                            choices.append(1)
                        if blockidx < len(subnet_demo[layeridx]) - 2:
                            choices.append(2)

                        if blockidx < len(subnet_demo[layeridx]) - 2:
                            choice = np.random.choice(choices, p=[(1 - skip_rate[i]), skip_rate[i] * distill_next_rate,
                                                                  skip_rate[i] * (1 - distill_next_rate)])

                        elif len(choices) == 2:
                            choice = np.random.choice(choices, p=[(1 - skip_rate[i]), skip_rate[i]])

                        else:
                            choice = 0
                        if choice == 0:
                            layer_choice += [0]
                        elif choice == 1:
                            layer_choice += [1, 99]
                        else:
                            layer_choice += [2, 99, 99]
                        blockidx += (choice + 1)
                    subnet.append(layer_choice)
                subnets.append(subnet)
    subnets.append(subnet_demo)
    return subnets


def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)


obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']
import webcolors


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

color_list = standard_to_bgr(STANDARD_COLORS)

compound_coef = 0


def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index

def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,

def preprocess(*image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas

def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[int(round(preds[i]['class_ids'][j])) - 1]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)
        # cv2.putText(imgs[i], 'latency:0.0156s, model size:51.34MB', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 0.5)
        # cv2.putText(imgs[i], 'model:[2, -1, -1, 2, -1, 99, 0, 2, -1, -1, 2, -1, -1, 2, -1, -1]', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        #             (0, 0, 255), 0.5)
        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])


def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        # import pdb;pdb.set_trace()
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds
def test_lat(block, input, test_times):
    # import pdb;pdb.set_trace()
    lats = []
    for i in range(test_times):
        t1 = time.time()
        y = block(input)
        torch.cuda.synchronize()
        t2 = time.time() - t1
        if i > 200:
            lats.append(t2)
        del y
    # del x
    return np.mean(lats)
def test_fpn(block, input, test_times):
    lats = []
    s1, s2, s3 = input[0].shape, input[1].shape, input[2].shape
    for i in range(test_times):
        x_inf = [torch.rand(s1).cuda(), torch.rand(s2).cuda(), torch.rand(s3).cuda()]
        t1 = time.time()
        y = block(x_inf)
        torch.cuda.synchronize()
        t2 = time.time() - t1
        if i > 20:
            lats.append(t2)
        del y
    # del x
    return np.mean(lats)
def test_resnet_layer(layer, x):
    lats = []
    for blockidx in range(len(layer)):
        lat_choices = []
        for choiceidx in range(len(layer[blockidx])):
            lat_choices.append(test_lat(layer[blockidx][choiceidx], x, 1000))
        x = layer[blockidx][0](x)
        lats.append(lat_choices)
    return lats, x
def get_lats(model):
    lats = []
    test_model = model.model
    x = torch.rand(4, 3, 640, 640).cuda()
    f_layers = [test_model.backbone.conv1, test_model.backbone.bn1, test_model.backbone.act1, test_model.backbone.maxpool]
    block = torch.nn.Sequential(*f_layers).cuda()
    lats.append(test_lat(block, x, 1000))
    x = block(x)
    block = test_model.backbone.layer2.cuda()
    layer2lats, x1 = test_resnet_layer(block, x)
    block = test_model.backbone.layer3.cuda()
    layer3lats, x2 = test_resnet_layer(block, x1)
    block = test_model.backbone.layer4.cuda()
    layer4lats, x3 = test_resnet_layer(block, x2)
    lats += [layer2lats, layer3lats, layer4lats]

    print(x1.shape, x2.shape, x3.shape)
    block = test_model.fpn.cuda()
    x = [x1, x2, x3]
    latter_lat = test_fpn(block, x, 1000)
    x = block(x)
    block = test_model.class_net.cuda()
    latter_lat += test_lat(block, x, 1000)
    block = test_model.box_net.cuda()
    latter_lat += test_lat(block, x, 1000)
    lats.append(latter_lat)
    return lats



if __name__ == "__main__":
    s = get_val_subnets(model=None)
    print(len(s))
