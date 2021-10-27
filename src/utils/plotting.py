import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import torch


def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=color[i], linewidth=1)
                     for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def _make_evaluation_figure(data, b_id, alpha='dynamic'):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)

    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)

    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]

    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text)
    return figure


def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()


def _make_anchors_figure(data: dict, b_id: int, i_id, dpi=75):
    # 绘制 anchor 点位置，如果存在 transform_matrix 则也绘制 warp 后的 anchor 点和 图像
    # 注意单应变换矩阵并不是尺度不变的，所以只能在 coarse 尺度上进行 warp

    hw0_i, hw1_i = data['hw0_i'], data['hw1_i']
    hw0_c, hw1_c = data['hw0_c'], data['hw1_c']

    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    anchors_i = data['anchors_i'][i_id][b_id].cpu().numpy()
    anchors0 = anchors_i[:, 0, :]  # [anchor_num, 2]
    anchors1 = anchors_i[:, 1, :]

    scale0 = np.array([int(hw0_i[0]) / int(hw0_c[0]), int(hw0_i[1]) / int(hw0_c[1])])  # 2
    scale1 = np.array([int(hw1_i[0]) / int(hw1_c[0]), int(hw1_i[1]) / int(hw1_c[1])])

    anchors0 = (anchors0 * scale0[None, :]).round().astype(np.int32)
    anchors1 = (anchors1 * scale1[None, :]).round().astype(np.int32)

    if 'transform_matrix' in data:
        fig, axes = plt.subplots(1, 3, figsize=(10, 6), dpi=dpi)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(len(axes)):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    axes[0].scatter(anchors0[:, 1], anchors0[:, 0], c='r', s=4)
    axes[1].scatter(anchors1[:, 1], anchors1[:, 0], c='r', s=4)

    if 'transform_matrix' in data:
        transform_matrix = data['transform_matrix'][b_id].cpu().numpy()
        anchors0_warped = data['anchors0_warped'][b_id].cpu().numpy()
        anchors0_warped = anchors0_warped * scale1[None, :]
        img0_c = cv2.resize(img0, dsize=(int(hw0_c[1]), int(hw0_c[0])))
        img0_c_warped = cv2.warpPerspective(img0_c, transform_matrix, dsize=(int(hw1_c[1]), int(hw1_c[0])))
        img0_warped = cv2.resize(img0_c_warped, dsize=(int(hw0_i[1]), int(hw0_i[0])))
        axes[2].imshow(img0_warped, cmap='gray')
        axes[2].scatter(anchors0_warped[:, 1], anchors0_warped[:, 0], c='r', s=4)

    return fig


def make_matching_figures(data, config, mode='evaluation'):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA)
            figures[mode].append(fig)
            if 'anchors_i' in data:
                for i in range(len(data['anchors_i'])):
                    fig = _make_anchors_figure(data, b_id, i)
                figures[mode].append(fig)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
            figures[mode].append(fig)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
    # figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
            milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x) * alpha], -1), 0, 1)
