import torch
import numpy as np
from PIL import Image

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_mean_std_seg(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adaptive_instance_normalization_seg(content_feat, style_feat, cont_seg, styl_seg, device):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    label_set, label_indicator = compute_label_info(cont_seg, styl_seg)
    size = content_feat.size()
    cont_c, cont_h, cont_w = content_feat.size(0), content_feat.size(1), content_feat.size(2)
    styl_c, styl_h, styl_w = style_feat.size(0), style_feat.size(1), style_feat.size(2)
    cont_feat_view = content_feat.view(cont_c, -1).clone()
    styl_feat_view = style_feat.view(styl_c, -1).clone()

    target_feature = content_feat.view(cont_c, -1).clone()
    t_cont_seg = np.asarray(Image.fromarray(cont_seg).resize((cont_w, cont_h), Image.NEAREST))
    t_styl_seg = np.asarray(Image.fromarray(styl_seg).resize((styl_w, styl_h), Image.NEAREST))
    for l in label_set:
        if label_indicator[l] == 0:
            continue
        cont_mask = np.where(t_cont_seg.reshape(t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l)
        styl_mask = np.where(t_styl_seg.reshape(t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l)
        if cont_mask[0].size <= 0 or styl_mask[0].size <= 0:
            continue
        cont_indi = torch.LongTensor(cont_mask[0])
        styl_indi = torch.LongTensor(styl_mask[0])
        cont_indi = cont_indi.to(device)
        styl_indi = styl_indi.to(device)
        cFFG = torch.index_select(cont_feat_view, 1, cont_indi)
        sFFG = torch.index_select(styl_feat_view, 1, styl_indi)
        style_mean, style_std = calc_mean_std(sFFG)
        content_mean, content_std = calc_mean_std(cFFG)
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        tmp_target_feature = normalized_feat * style_std.expand(size) + style_mean.expand(size)
        # print(tmp_target_feature.size())
        if torch.__version__ >= "0.4.0":
            new_target_feature = torch.transpose(target_feature, 1, 0)
            new_target_feature.index_copy_(0, cont_indi, \
                                       torch.transpose(tmp_target_feature, 1, 0))
            target_feature = torch.transpose(new_target_feature, 1, 0)
        else:
            target_feature.index_copy_(1, cont_indi, tmp_target_feature)
    target_feature = target_feature.view_as(content_feat)
    return target_feature


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

def compute_label_info(cont_seg, styl_seg):
    if cont_seg.size == False or styl_seg.size == False:
        return
    max_label = np.max(cont_seg) + 1
    label_set = np.unique(cont_seg)
    label_indicator = np.zeros(max_label)
    for l in label_set:
        # if l==0:
        #   continue
        is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
        o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
        o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
        label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)
    return label_set, label_indicator
