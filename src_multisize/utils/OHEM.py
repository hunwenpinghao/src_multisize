import torch
import torch.nn.functional as F

def ohem_loss(
        batch_size, cls_pred, cls_target
):
    """
    Arguments:
        batch_size (int): number of sampled rois for bbox head training
        loc_pred (FloatTensor): [R, 4], location of positive rois
        loc_target (FloatTensor): [R, 4], location of positive rois
        pos_mask (FloatTensor): [R], binary mask for sampled positive rois
        cls_pred (FloatTensor): [R, C]
        cls_target (LongTensor): [R]
    Returns:
        cls_loss, loc_loss (FloatTensor)
    """
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)
    # 这里先暂存下正常的分类loss

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    # 再对loss进行降序排列
    keep_num = min(sorted_ohem_loss.size()[0], batch_size)
    # 得到需要保留的loss数量
    if keep_num < sorted_ohem_loss.size()[0]:
        # 这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留
        keep_idx_cuda = idx[:keep_num]
        # 保留到需要keep的数目
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        # 分类保留相同的数目
    cls_loss = ohem_cls_loss.sum() / keep_num
    # 然后分别对分类loss求均值
    return cls_loss
