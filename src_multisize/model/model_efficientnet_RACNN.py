import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
# from utils import CBAM
import torchvision.models as models
import pretrainedmodels
import visdom
viz = visdom.Visdom(env='img')
from .utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-','_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


def EfficientNet_CBAM(arch):
    model = EfficientNet.from_pretrained(arch)

    ## add cbam at every block layer
    # for i in range(len(model._blocks)):
    #     model._blocks[i]._bn1 = nn.Sequential(
    #         CBAM.SpatialGate(),
    #         model._blocks[i]._bn1,
    #     )


    ## add cbam at the fist conv layer
    # cbam_conv = CBAM(model._conv_stem.out_channels)
    # model._conv_stem = nn.Sequential(
    #     model._conv_stem,
    #     cbam_conv,
    # )

    return model

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        h = lambda x: 1 / (1 + torch.exp((-10 * x).float()))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size)
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()

        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]
            tx = tx if tx > (in_size / 3) else in_size / 3
            tx = tx if (in_size / 3 * 2) < tx else (in_size / 3 * 2)
            ty = ty if ty > (in_size / 3) else in_size / 3
            ty = ty if (in_size / 3 * 2) < ty else (in_size / 3 * 2)
            tl = tl if tl > (in_size / 3) else in_size / 3

            w_off = int(tx - tl) if (tx - tl) > 0 else 0
            h_off = int(ty - tl) if (ty - tl) > 0 else 0
            w_end = int(tx + tl) if (tx + tl) < in_size else in_size
            h_end = int(ty + tl) if (ty + tl) < in_size else in_size

            mk = (h(x - w_off) - h(x - w_end)) * (h(y - h_off) - h(y - h_end))
            xatt = images[i] * mk

            xatt_cropped = xatt[:, h_off: h_end, w_off: w_end]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            xamp = F.upsample(before_upsample, size=(224, 224), mode='bilinear', align_corners=True)
            ret.append(xamp.data.squeeze())

        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor = self.saved_variables[0], self.saved_variables[1]
        in_size = 224
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)

        #         show_image(inputs.cpu().data[0])
        #         show_image(ret_tensor.cpu().data[0])
        #         plt.imshow(norm[0].cpu().numpy(), cmap='gray')

        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size / 3 * 2)
        short_size = (in_size / 3)
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x < short_size) + (x >= long_size) + (y < short_size) + (y >= long_size)) > 0).float() * 2 - 1

        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))

        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()

        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret

class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """
    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)

class MultiNet(nn.Module):
    def __init__(self, backbone='efficientnet-b5', drop=0.2, num_classes=54):
        super().__init__()
        self.img_model = EfficientNet.from_pretrained(backbone)
        self.img_model._fc = nn.Linear(self.img_model._fc.in_features, num_classes)
        self.zoom_model = models.resnet18(pretrained=True)
        self.zoom_model.fc = nn.Linear(self.zoom_model.fc.in_features, num_classes)

        self.num_classes = num_classes
        self.tolayer = 4

        self.ca = ChannelAttention(self.img_model._bn0.num_features)
        self.sa = SpatialAttention()

        self.atten_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.apn1 = nn.Sequential(
            nn.Linear((176) * 14 * 14, 512),
            nn.Tanh(),
            nn.Linear(512, 3),
            nn.Sigmoid(),
        )
        self.crop_resize = AttentionCropLayer()

        self.img_stem = nn.Sequential(*list(self.img_model.children())[:-4])
        self._blocks_tolayer = nn.Sequential(*list(self.img_model._blocks.children())[:self.tolayer])

        # self.last_linear = nn.Sequential(
        #     nn.Conv2d(in_channels=self.img_model._blocks[16]._bn2.num_features + 3*self.img_model._blocks[21]._bn2.num_features,
        #               out_channels=256, kernel_size=5, stride=2, padding=0, dilation=1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d(1),
        #     FCViewer(),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(drop),
        #     # nn.Linear(256, num_classes),
        #     # nn.Linear(self.img_model._blocks[16]._bn2.num_features + 3*self.img_model._blocks[21]._bn2.num_features, num_classes)
        # )

        self.fc = nn.Linear(2048+256, self.num_classes)

    def forward(self, input):
        # x = self.img_model(x_img)

        # extract_feature
        x_img = self.img_model._conv_stem(input)
        x_img = relu_fn(self.img_model._bn0(x_img))
        x_img = self.sa(x_img) * x_img

        for idx, block in enumerate(self.img_model._blocks):
            drop_connect_rate = self.img_model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.img_model._blocks)
            x_img = block(x_img, drop_connect_rate=drop_connect_rate)

            # if idx == 16:
            #     feature_16 = x_img
            # if idx == 21:
            #     feature_21 = x_img
            # if idx == 22:
            #     feature_22 = x_img
            # if idx == 23:
            #     feature_23 = x_img
            if idx == 26:
                feature_26 = x_img

        x_img = self.img_model._conv_head(x_img)
        x_img = relu_fn(self.img_model._bn1(x_img))

        # Pooling and final linear layer
        x_img = F.adaptive_avg_pool2d(x_img, 1).squeeze(-1).squeeze(-1)
        if self.img_model._dropout:
            x_img = F.dropout(x_img, p=self.img_model._dropout, training=self.img_model.training)
        x = self.img_model._fc(x_img)

        # concat
        # x_vis = torch.cat((feature_16,feature_21,feature_22,feature_23),dim=1)
        # x_vis = self.last_linear(x_vis)
        #
        # x_cat = torch.cat((x_vis, x_img), dim=1)

        # MODEL 2
        # x_hook = torch.cat((feature_21), dim=1)
        atten1 = self.apn1(self.atten_pool(feature_26).view(-1, (176) * 14 * 14))
        scaledA_x = self.crop_resize(input, atten1 * input.size()[2]) # input 456

        x_zoomout = self.zoom_model(scaledA_x)

        # x = self.fc(x_cat)

        return x, x_zoomout, scaledA_x


class MultiNet_infer(nn.Module):
    def __init__(self, backbone='efficientnet-b5', drop=0.4, num_classes=54):
        super().__init__()
        # self.img_model = EfficientNet.from_pretrained(backbone)
        self.img_model = EfficientNet.from_name(backbone, override_params={'num_classes': 1000})

        self.num_classes = num_classes
        self.tolayer = 4

        self.img_stem = nn.Sequential(*list(self.img_model.children())[:-4])
        self._blocks_tolayer = nn.Sequential(*list(self.img_model._blocks.children())[:self.tolayer])

        self.last_linear = nn.Sequential(
            nn.Conv2d(in_channels=self.img_model._blocks[16]._bn2.num_features + 3*self.img_model._blocks[21]._bn2.num_features,
                      out_channels=256, kernel_size=5, stride=2, padding=0, dilation=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            FCViewer(),
            nn.BatchNorm1d(256),
            nn.Dropout(drop),
            nn.Linear(256, num_classes),
            # nn.Linear(self.img_model._blocks[16]._bn2.num_features + 3*self.img_model._blocks[21]._bn2.num_features, num_classes)
        )

        self.img_model._fc = nn.Linear(self.img_model._fc.in_features, num_classes)

    def forward(self, x_img):
        # x = self.img_model(x_img)

        # extract_feature
        x_img = self.img_model._conv_stem(x_img)
        x_img = relu_fn(self.img_model._bn0(x_img))

        for idx, block in enumerate(self.img_model._blocks):
            drop_connect_rate = self.img_model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.img_model._blocks)
            x_img = block(x_img, drop_connect_rate=drop_connect_rate)

            if idx == 16:
                feature_16 = x_img
            if idx == 21:
                feature_21 = x_img
            if idx == 22:
                feature_22 = x_img
            if idx == 23:
                feature_23 = x_img

        x_img = self.img_model._conv_head(x_img)
        x_img = relu_fn(self.img_model._bn1(x_img))

        # Pooling and final linear layer
        x_img = F.adaptive_avg_pool2d(x_img, 1).squeeze(-1).squeeze(-1)
        if self.img_model._dropout:
            x_img = F.dropout(x_img, p=self.img_model._dropout, training=self.img_model.training)
        x = self.img_model._fc(x_img)

        # concat
        x_cat = torch.cat((feature_16,feature_21,feature_22,feature_23),dim=1)
        x_cat = self.last_linear(x_cat)

        return x, x_cat

if __name__ == '__main__':
    # input = torch.ones((2, 3, 456, 456))
    # input = input.cuda()
    # Net = MultiNet()
    # Net.cuda()
    # out, x_cat = Net(input)
    # print('out shape:', out.shape)

    input = torch.ones((2, 3, 7, 7))
    Net = MultiNet()
    out = Net.upsample(input, 2)
    print('output shape:', out.size())