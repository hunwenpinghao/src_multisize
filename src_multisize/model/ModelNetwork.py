import torch
import torch.nn as nn
import pretrainedmodels
from collections import OrderedDict
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import model.senet as senet

class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class se_resnext101_32x4d(nn.Module):
    def __init__(self, backbone1, drop, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            Net = getattr(senet, backbone1)
            img_model = Net(num_classes=1000, pretrained='imagenet')  # seresnext101
        else:
            Net = getattr(senet, backbone1)
            img_model = Net(num_classes=1000, pretrained=None)

        self.img_encoder = list(img_model.children())[:-2]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder = nn.Sequential(*self.img_encoder)

        if drop > 0:
            self.img_fc = nn.Sequential(FCViewer(),
                                        # nn.BatchNorm1d(img_model.last_linear.in_features),
                                        nn.Dropout(drop),
                                        nn.Linear(img_model.last_linear.in_features, 256),
                                        # nn.ReLU(),
                                        # nn.BatchNorm1d(2048),
                                        # nn.Dropout(drop)
                                        )

        else:
            self.img_fc = nn.Sequential(
                FCViewer(),
                nn.Linear(img_model.last_linear.in_features, 256)
            )

        self.cls = nn.Linear(256, num_classes)

    def forward(self, x_img):
        x_img = self.img_encoder(x_img)
        x_img = self.img_fc(x_img)
        x_last = self.cls(x_img)
        return x_last

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    model_path = '../checkpoints/epoch_50_99.6.pth'
    valdir = '../../dataset/val'

    model = se_resnext101_32x4d('se_resnext101_32x4d', "dpn92", 0.5, 54)

    use_cuda = True
    if use_cuda:
        print('Using GPU for inference')
        checkpoint = torch.load(model_path)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('Using CPU for inference')
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = OrderedDict()
        # 训练脚本 main.py 中保存了'epoch', 'arch', 'state_dict', 'best_acc1', 'optimizer'五个key值，
        # 其中'state_dict'对应的value才是模型的参数。
        # 训练脚本 main.py 中创建模型时用了torch.nn.DataParallel，因此模型保存时的dict都会有‘module.’的前缀，
        # 下面 tmp = key[7:] 这行代码的作用就是去掉‘module.’前缀
        for key, value in checkpoint['state_dict'].items():
            tmp = key[7:]
            state_dict[tmp] = value
        model.load_state_dict(state_dict)

    model.eval()

    # load dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=54, shuffle=False,
        num_workers=4, pin_memory=True)

    # excute val
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            print('inferring:', i, '/', len(val_loader))

            if use_cuda:
                images = images.cuda(0, non_blocking=True)
                target = target.cuda(0, non_blocking=True)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            print('acc1: %0.4f, acc5:%0.4f' % (acc1, acc5))






