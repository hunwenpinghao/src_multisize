# src_multisize
这个工程是我在参加华为2019创新大赛的工程文件，里面有各种模型的backbone,例如比赛中常用到的
resnet系列：
se_resnext101_32x4d, resnet50, pnasnet5large, se_resnet152, resnext101_32x48d_wsl
efficientnet系列：
efficientnet b1-b7等

另外，该工程中还有一些小技巧，例如label smooth, cutmix, CBAM, focalloss, mish loss, OHEM, 各种数据增强， 各种learning scheduler
等等，对了还有多尺度训练，multisize就是指的多尺度训练的意思，由于这个是最后加的，所以名字就改为这个了。

另外对于损失函数也做了更改，加入了多尺度的特征，就是对多个尺度上的特征层进行融合，然后，将融合后的特征加上softmax再与groundtruth形成一个
loss2 ，把这个loss2加到原来的loss上，当时这个操作提升了将近一个点，还是挺有效的。

总结一下，最有效的还是数据增强和多尺度这两个方法，cutmix也属于数据增强。focalloss，注意力啥的对于这个数据集好像作用不是很大，很多trick
就是这样的，不同数据集的表现不同。

性能最好的backbone当然是efficientnet系列，不过，b7跑不动，显卡不够，我只能跑个b5，这我的batch_size都只能设置到6个，我用的是tantai-v.
次之的backbone是resnext101, 但是好像se_resnext101_32x4d 和 resnext101_32x48d_wsl我试了差不多，可能也和数据集有关.
