import torch
import torch.nn as nn
import visdom
viz = visdom.Visdom()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 2)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 2)).cpu() * iteration,
        Y=torch.Tensor([loc, conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 2)).cpu(),
            Y=torch.Tensor([loc, conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )

def update_acc_plot(iteration, loc, conf, window1, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 2)).cpu() * iteration,
        Y=torch.Tensor([loc, conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            # 目前不展示全连接层
            if "fc" in name:
                x = x.view(x.size(0), -1)
            print(module)
            x = module(x)
            print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs
