
#ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py
import contextlib
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
import re
from typing import Dict, List, Optional, Tuple, Union

from DeepDataMiningLearning.detection.modules.block import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                    Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d,
                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, RepC3, RepConv, MP, SPPCSPC )
from DeepDataMiningLearning.detection.modules.utils import extract_filename, LOGGER, make_divisible, non_max_suppression, scale_boxes #colorstr, 
from DeepDataMiningLearning.detection.modules.head import Detect, IDetect, Classify, Pose, RTDETRDecoder, Segment
#Detect, Classify, Pose, RTDETRDecoder, Segment
from DeepDataMiningLearning.detection.modules.lossv8 import myv8DetectionLoss
from DeepDataMiningLearning.detection.modules.lossv7 import myv7DetectionLoss
from DeepDataMiningLearning.detection.modules.anchor import check_anchor_order



def yaml_load(file='data.yaml', append_filename=True):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in ('.yaml', '.yml'), f'Attempting to load non-YAML file {file} with yaml_load()'
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data['yaml_file'] = str(file)
        return data

#ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py
class YoloDetectionModel(nn.Module):
    #scale from nsmlx
    def __init__(self, cfg='yolov8n.yaml', scale='n', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)  # cfg dict, nc=80, 'scales', 'backbone', 'head'
        self.yaml['scale'] = scale
        self.modelname=extract_filename(cfg)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels, ch=3
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override YAML value

        # get FasterRCNN model and preprocess function
        self.model, self.preprocess = get_torchvision_detection_models('fasterrcnn_resnet50_fpn', box_score_thresh=0.9)

        #self.model.eval()

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)): 
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride #[ 8., 16., 32.]
            m.bias_init()  # only run once
        elif isinstance(m, IDetect): #added yolov7's IDetect
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            #self._initialize_biases()  # only run once
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
    
    #ref BaseModel forward in ultralytics\nn\tasks.py
    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale.
        Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        #model.train() self.training=True
        #model.eval() self.training=False
        if isinstance(x, dict):# for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        elif self.training:
            preds = self._predict_once(x) #tensor input
            return preds #training mode, direct output x (three items)
        else: #inference mode
            preds, xtensors = self._predict_once(x) #tensor input #[1, 3, 256, 256]
            #y,x output in inference mode, training mode, direct output x (three items),
            return preds

    def forward(self, images, targets=None):
        if self.training:
            # training mode
            images, targets = self.preprocess(images, targets)
            loss_dict = self.model(images, targets)
            return sum(loss for loss in loss_dict.values())
        else:
            # inference mode
            images = self.preprocess(images)
            outputs = self.model(images)
            return outputs

    def postprocess(self, outputs, original_image_sizes):
        results = []
        for output, image_size in zip(outputs, original_image_sizes):
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            boxes = boxes * torch.tensor([image_size[1], image_size[0], image_size[1], image_size[0]])

            for box, score, label in zip(boxes, scores, labels):
                result = {
                    'box': box.tolist(),
                    'score': score,
                    'label': self.config.id2label[label]
                }
                results.append(result)

        return results

    
    def _predict_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        #y, dt = [], []  # outputs, dt used in profile
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # if profile:
            #     self._profile_one_layer(m, x, dt)
            
            x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            # if visualize:
            #     feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

twoargs_blocks=[nn.Conv2d, Classify, Conv, ConvTranspose, GhostConv, RepConv, Bottleneck, GhostBottleneck, SPP, SPPF, SPPCSPC, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3]

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]
    elif all(key in d for key in ('depth_multiple', 'width_multiple')): #for yolov7:https://github.com/lkk688/myyolov7/blob/main/models/yolo.py
        depth = d['depth_multiple']
        width = d['width_multiple']
    
    if "anchors" in d.keys():
        anchors = d['anchors']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    else: #anchor-free
        no = nc

   
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"activation: {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in twoargs_blocks:
            c1, c2 = ch[f], args[0]
            #if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
            if c2 != no:
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (IDetect, Detect, Segment, Pose):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
    
def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


# def create_yolomodel(modelname,num_classes):
#     cfgpath='./DeepDataMiningLearning/detection/modules/'
#     cfgfile=os.path.join(cfgpath, modelname+'.yaml')
#     yolomodel=YoloDetectionModel(cfg=cfgfile, ch = 3, nc=num_classes)
#     return yolomodel

def load_defaultcfgs(cfgPath):
    DEFAULT_CFG_PATH = cfgPath #ROOT / 'cfg/default.yaml'
    DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
    for k, v in DEFAULT_CFG_DICT.items():
        if isinstance(v, str) and v.lower() == 'none':
            DEFAULT_CFG_DICT[k] = None
    DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
    #DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    return DEFAULT_CFG_DICT

def load_checkpoint(model, ckpt_file, fp16=False):
    
    #ModuleNotFoundError: No module named 'models'
    ckpt=torch.load(ckpt_file, map_location='cpu')
    nn_module = isinstance(ckpt, torch.nn.Module)
    print(ckpt.keys()) #'0.conv.weight', '0.bn.weight', '0.bn.bias'
    currentmodel_statedict = model.state_dict()#starts with 'model.'
    #rename ckpt for yolov7
    for key in list(ckpt.keys()):
        if not key.startswith("model"):
            ckpt['model.'+key] = ckpt.pop(key)

    csd = intersect_dicts(ckpt, currentmodel_statedict)  # intersect
    model.load_state_dict(ckpt, strict=False)
    print(f'Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights')
    #524/566 items
    #names = ckpt.module.names if hasattr(ckpt, 'module') else ckpt.names  # get class names
    model.half() if fp16 else model.float()
    return model

import DeepDataMiningLearning.detection.transforms as T
from DeepDataMiningLearning.detection.modules.yolotransform import LetterBox
from PIL import Image
def get_transformsimple(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ToDtype(torch.float, scale=True))
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def preprocess_img(imagepath, opencvread=True, fp16=False):
    if opencvread:
        im0 = cv2.imread(imagepath) #(1080, 810, 3)
    else:
        im0 = Image.open(imagepath).convert('RGB')
        transfunc=get_transformsimple(False)
        im0=transfunc(im0)
    im0s = [im0] #image list
    #preprocess(im0s)
    letterbox = LetterBox((640, 640), auto=True, stride=32) #only support cv2
    processedimgs=[letterbox(image=x) for x in im0s] #list of (640, 480, 3)
    im = np.stack(processedimgs) #(1, 640, 480, 3)
    if opencvread:
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im)
    #im = im.to(device)
    im = im.half() if fp16 else im.float()  # uint8 to fp16/32
    im /= 255
    return im #(1, 640, 480, 3) tensor

import os
from collections import OrderedDict
import cv2
from DeepDataMiningLearning.detection.modules.yolotransform import YoloTransform
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision

def create_yolomodel(modelname, num_classes = None, ckpt_file = None, fp16 = False, device = 'cuda:0', scale='n'):
    
    modelcfg_file=os.path.join('./DeepDataMiningLearning/detection/modules', modelname+'.yaml')
    cfgPath='./DeepDataMiningLearning/detection/modules/default.yaml'
    myyolo = None
    preprocess =None
    classesList = None
    if os.path.exists(modelcfg_file) and os.path.exists(cfgPath):
        DEFAULT_CFG_DICT = load_defaultcfgs(cfgPath)
        classes=DEFAULT_CFG_DICT['names']
        nc=len(classes) #80
        classesList = list(classes.values()) #class name list
        myyolo=YoloDetectionModel(cfg=modelcfg_file, scale=scale, ch=3, nc =num_classes)
        if ckpt_file is not None and os.path.exists(ckpt_file):
            myyolo=load_checkpoint(myyolo, ckpt_file)
        myyolo=myyolo.to(device).eval()
        stride = max(int(myyolo.stride.max()), 32)  # model stride
        names = myyolo.module.names if hasattr(myyolo, 'module') else myyolo.names  # get class names
        #model = model.fuse(verbose=verbose) if fuse else model
        myyolo = myyolo.half() if fp16 else myyolo.float()

        preprocess = YoloTransform(min_size=640, max_size=640, device=device, fp16=fp16, cfgs=DEFAULT_CFG_DICT)
        return myyolo, preprocess, classesList
    else:
        print("Config file not found")
        return myyolo, preprocess, classesList

def freeze_yolomodel(model, freeze=[]):
    # Freeze layers
    freeze_list = freeze if isinstance(
        freeze, list) else range(freeze) if isinstance(freeze, int) else []
    always_freeze_names = ['.dfl']  # always freeze these layers
    freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
    for k, v in model.named_parameters():
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze_layer_names):
            LOGGER.info(f"Freezing layer '{k}'")
            v.requires_grad = False
        elif not v.requires_grad:
            LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                        'See ultralytics.engine.trainer for customization of frozen layers.')
            v.requires_grad = True
    return model

def test_yolov7weights():
    myyolov7=YoloDetectionModel(cfg='./DeepDataMiningLearning/detection/modules/yolov7.yaml', ch=3) #nc =80
    print(myyolov7)
    img = torch.rand(1, 3, 640, 640)
    myyolov7.eval()
    preds = myyolov7(img)
    #y,x = myyolov7(img)#tuple output, first item is tensor [1, 25200, 85], second item is list of three, [1, 3, 80, 80, 85], [1, 3, 40, 40, 85], [1, 3, 20, 20, 85]
    #print(y.shape) #[1, 25200, 85]
    #print(len(x)) #3
    #print(x[0].shape) #[1, 3, 80, 80, 85]

    ckpt_file = '/data/cmpe249-fa23/modelzoo/yolov7_statedicts.pt'
    #ModuleNotFoundError: No module named 'models'
    ckpt=torch.load(ckpt_file, map_location='cpu')
    print(ckpt.keys()) #'0.conv.weight', '0.bn.weight', '0.bn.bias'
    newckpt = {}
    for key in ckpt.keys():
        newkey='model.'+key
        newckpt[newkey] = ckpt[key]#change key name
    currentmodel_statedict = myyolov7.state_dict()
    csd = intersect_dicts(newckpt, currentmodel_statedict)  # intersect
    myyolov7.load_state_dict(newckpt, strict=False)
    print(f'Transferred {len(csd)}/{len(myyolov7.state_dict())} items from pretrained weights')
    #524/566 items
    myyolov7.eval()

if __name__ == "__main__":
    test_yolov7weights()

    cfgPath='./DeepDataMiningLearning/detection/modules/default.yaml'
    DEFAULT_CFG_DICT = load_defaultcfgs(cfgPath)
    images=[]
    images.append(torch.rand(3, 640, 480))

    print(os.getcwd())
    modelcfg_file='./DeepDataMiningLearning/detection/modules/yolov8.yaml'
    imagepath='./sampledata/bus.jpg'
    #im=preprocess_img(imagepath, opencvread=True) #[1, 3, 640, 480]
    myyolo=YoloDetectionModel(cfg=modelcfg_file, scale='n', ch=3) #nc =80
    print(myyolo.modelname)

    ckpt_file = '/data/cmpe249-fa23/modelzoo/yolov8n_statedicts.pt'
    device = 'cuda:3'
    fp16 = False
    myyolo=load_checkpoint(myyolo, ckpt_file)
    myyolo=myyolo.to(device).eval()
    stride = max(int(myyolo.stride.max()), 32)  # model stride [ 8., 16., 32.]
    names = myyolo.module.names if hasattr(myyolo, 'module') else myyolo.names  # no module, get class names, dict 0~79
    #model = model.fuse(verbose=verbose) if fuse else model
    myyolo = myyolo.half() if fp16 else myyolo.float()

    #inference
    im0 = cv2.imread(imagepath) #(1080, 810, 3)
    imgs = [im0]
    origimageshapes=[img.shape for img in imgs] #(height, width, c)
    yoyotrans = YoloTransform(min_size=640, max_size=640, device=device, fp16=fp16, cfgs=DEFAULT_CFG_DICT)
    imgtensors = yoyotrans(imgs) #[1, 3, 640, 480]
    preds = myyolo(imgtensors) #inference od [1, 84, 6300], 84=4(boxes)+80(classes)
    imgsize = imgtensors.shape[2:] #640, 480 HW
    detections = yoyotrans.postprocess(preds, imgsize, origimageshapes)
    print(detections)  
    #list of resdict["boxes"], resdict["scores"], resdict["labels"]
    #bounding boxes in (xmin, ymin, xmax, ymax) format

    onedetection=detections[0]
    #labels = [names[i] for i in detections["labels"]] #classes[i]
    #img=im0.copy() #HWC (1080, 810, 3)
    img_trans=im0[..., ::-1].transpose((2,0,1))  # BGR to RGB, HWC to CHW
    imgtensor = torch.from_numpy(img_trans.copy()) #[3, 1080, 810]
    #pred_bbox_tensor=torchvision.ops.box_convert(torch.from_numpy(onedetection["boxes"]), 'xywh', 'xyxy')
    pred_bbox_tensor=onedetection["boxes"] #torch.from_numpy(onedetection["boxes"])
    print(pred_bbox_tensor)
    pred_labels = onedetection["labels"].numpy().astype(str).tolist()
    #img: Tensor of shape (C x H x W) and dtype uint8.
    #box: Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format.
    #labels: Optional[List[str]]
    box = draw_bounding_boxes(imgtensor, boxes=pred_bbox_tensor,
                            labels=pred_labels,
                            colors="red",
                            width=4, font_size=40)
    im = to_pil_image(box.detach())
    # save a image using extension
    im = im.save("results.jpg")
