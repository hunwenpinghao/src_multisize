# -*- coding: utf-8 -*-
import os
import codecs
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import visdom
viz = visdom.Visdom()
import model as modelZoo
from os.path import join
# from model_service.pytorch_model_service import PTServingBaseService
#
# import time
# from metric.metrics_manager import MetricsManager
# import log
# logger = log.getLogger(__name__)


class ImageClassificationService():
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

        # self.model = models.__dict__['resnet50'](num_classes=54)
        # model = modelZoo.EfficientNet_CBAM('efficientnet-b5')

        self.model = modelZoo.model_efficientnet.MultiNet_infer()

        # Net = getattr(modelZoo, 'EfficientNet')
        # self.model = Net.from_pretrained('efficientnet-b5')
        # self.model._fc = nn.Linear(self.model._fc.in_features, 54)


        self.use_cuda = True
        if torch.cuda.is_available():
            print('Using GPU for inference')
            self.use_cuda = True
            checkpoint = torch.load(self.model_path)
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            print('Using CPU for inference')
            checkpoint = torch.load(self.model_path, map_location='cpu')
            state_dict = OrderedDict()
            # 训练脚本 main.py 中保存了'epoch', 'arch', 'state_dict', 'best_acc1', 'optimizer'五个key值，
            # 其中'state_dict'对应的value才是模型的参数。
            # 训练脚本 main.py 中创建模型时用了torch.nn.DataParallel，因此模型保存时的dict都会有‘module.’的前缀，
            # 下面 tmp = key[7:] 这行代码的作用就是去掉‘module.’前缀
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                state_dict[tmp] = value
            self.model.load_state_dict(state_dict)

        self.model.eval()

        self.idx_to_class = checkpoint['idx_to_class']
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transforms = transforms.Compose([
            transforms.Resize(456),
            transforms.CenterCrop(456),
            transforms.ToTensor(),
            self.normalize
        ])

        self.label_id_name_dict = \
            {
                "0": "工艺品/仿唐三彩",
                "1": "工艺品/仿宋木叶盏",
                "2": "工艺品/布贴绣",
                "3": "工艺品/景泰蓝",
                "4": "工艺品/木马勺脸谱",
                "5": "工艺品/柳编",
                "6": "工艺品/葡萄花鸟纹银香囊",
                "7": "工艺品/西安剪纸",
                "8": "工艺品/陕历博唐妞系列",
                "9": "景点/关中书院",
                "10": "景点/兵马俑",
                "11": "景点/南五台",
                "12": "景点/大兴善寺",
                "13": "景点/大观楼",
                "14": "景点/大雁塔",
                "15": "景点/小雁塔",
                "16": "景点/未央宫城墙遗址",
                "17": "景点/水陆庵壁塑",
                "18": "景点/汉长安城遗址",
                "19": "景点/西安城墙",
                "20": "景点/钟楼",
                "21": "景点/长安华严寺",
                "22": "景点/阿房宫遗址",
                "23": "民俗/唢呐",
                "24": "民俗/皮影",
                "25": "特产/临潼火晶柿子",
                "26": "特产/山茱萸",
                "27": "特产/玉器",
                "28": "特产/阎良甜瓜",
                "29": "特产/陕北红小豆",
                "30": "特产/高陵冬枣",
                "31": "美食/八宝玫瑰镜糕",
                "32": "美食/凉皮",
                "33": "美食/凉鱼",
                "34": "美食/德懋恭水晶饼",
                "35": "美食/搅团",
                "36": "美食/枸杞炖银耳",
                "37": "美食/柿子饼",
                "38": "美食/浆水面",
                "39": "美食/灌汤包",
                "40": "美食/烧肘子",
                "41": "美食/石子饼",
                "42": "美食/神仙粉",
                "43": "美食/粉汤羊血",
                "44": "美食/羊肉泡馍",
                "45": "美食/肉夹馍",
                "46": "美食/荞面饸饹",
                "47": "美食/菠菜面",
                "48": "美食/蜂蜜凉粽子",
                "49": "美食/蜜饯张口酥饺",
                "50": "美食/西安油茶",
                "51": "美食/贵妃鸡翅",
                "52": "美食/醪糟",
                "53": "美食/金线油塔"
            }

        self.features = []
        self.handle = self.model.register_backward_hook(self.hook)

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.transforms(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def hook(self, module, input, output):
        print(module)
        for val in input:
            print('input val', val)
        for out_val in output:
            print('output_val', out_val)
        self.features.append(output.data.cpu().numpy())

    def _inference(self, data):
        img = data["input_img"]
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred_score,_ = self.model(img)
            self.handle.remove()
            pred_score = F.softmax(pred_score.data, dim=1)
            if pred_score is not None:
                pred_label = torch.argsort(pred_score[0], descending=True)[:1][0].item()
                pred_label = self.idx_to_class[int(pred_label)]
                result = {'result': self.label_id_name_dict[str(pred_label)]}
            else:
                result = {'result': 'predict score is None'}

        return result

    def _postprocess(self, data):
        return data




def infer_on_dataset(img_dir, label_dir, model_path, obj, single_img=None):
    if not os.path.exists(img_dir):
        print('img_dir: %s is not exist' % img_dir)
        return None
    if not os.path.exists(label_dir):
        print('label_dir: %s is not exist' % label_dir)
        return None
    if not os.path.exists(model_path):
        print('model_path: %s is not exist' % model_path)
        return None
    output_dir = model_path + '_output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    infer = ImageClassificationService('', model_path)
    files = os.listdir(img_dir)
    error_results = []
    right_count = 0
    total_count = 0
    if single_img:
        files = [single_img.split('/')[-1]]
    for file_name in files:
        if not file_name.endswith('jpg'):
            continue

        with codecs.open(os.path.join(label_dir, file_name.split('.jpg')[0] + '.txt'), 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(', ')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_name.split('.jpg')[0] + '.txt'))
            continue
        gt_label = infer.label_id_name_dict[line_split[1]]
        # gt_label = "工艺品/仿唐三彩"

        img_path = os.path.join(img_dir, file_name)
        print('img_path:', img_path)
        img = Image.open(img_path)
        img = infer.transforms(img)
        result = infer._inference({"input_img": img})
        pred_label = result.get('result', 'error')

        for i,feature in enumerate(infer.features):
            print('feature:', feature)
            viz.heatmap(feature, win='feature{}'.format(i), opts={'title': 'headmaps{}'.format(i)})

        feature = img
        feature = feature.unsqueeze(0)
        feature = feature.cuda(0, non_blocking=True)
        for name, module in infer.model._modules['module']._modules['img_model']._modules.items():
            print('name:', name)
            feature = module(feature)
            if name == '_bn0':
                break

        for i in range(len(infer.model._modules['module']._modules['img_model']._modules['_blocks'])):
            for name, module in infer.model._modules['module']._modules['img_model']._modules['_blocks'][i]._modules.items():
                feature = module(feature)
                if name == '_bn2':
                    print('layer '+ str(i), name + ' feature shape:', feature.shape)
                    feature_show = torch.sum(feature[0], 0)
                    viz.heatmap(feature_show.data, win='feature{}'.format(i), opts={'title': 'headmaps{}'.format(i)})


        total_count += 1
        if pred_label == gt_label:
            right_count += 1
        else:
            error_results.append(', '.join([img_path, gt_label, pred_label]) + '\n')

    acc = float(right_count) / total_count
    result_file_path = os.path.join(output_dir, 'accuracy_{}.txt'.format(obj))
    with codecs.open(result_file_path, 'w', 'utf-8') as f:
        f.write('# predict error files\n')
        f.write('####################################\n')
        f.write('file_name, gt_label, pred_label\n')
        f.writelines(error_results)
        f.write('####################################\n')
        # f.write('accuracy: %s\n' % acc)
    print('accuracy result file saved as %s' % result_file_path)
    print('class: %s, accuracy: %0.4f' % (infer.label_id_name_dict[str(obj)], acc))
    return acc, result_file_path


if __name__ == '__main__':
    testdata = 'test(val)'
    one_img = 1
    if one_img:
        obj = 27
        img_dir = r'../dataset/baseline/{}/{}'.format(testdata, obj)
        single_img = join(img_dir, 'img_2163.jpg')
        label_dir = r'../dataset/baseline/{}/{}'.format(testdata, obj)
        model_path = r'./test_weight/epoch_55_98.3.pth'
        infer_on_dataset(img_dir, label_dir, model_path, obj, single_img)
    else:
        for obj in range(54):
            # if obj == 40:
                img_dir = r'../dataset/baseline/{}/{}'.format(testdata, obj)
                label_dir = r'../dataset/baseline/{}/{}'.format(testdata, obj)
                model_path = r'./test_weight/epoch_80_98.4.pth'
                infer_on_dataset(img_dir, label_dir, model_path, obj)