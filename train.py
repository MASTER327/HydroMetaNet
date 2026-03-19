import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torch import optim
from datasets import get_dataset, HyperX
from utils_HSI import sample_gt, metrics, seed_worker
import torch.nn.functional as F
from torch import einsum

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./datasets/Wetland/')

parser.add_argument('--source_name', type=str, default='ZY1-02D_HHK_2020',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='ZY1-02D_Yancheng_A',
                    help='the name of the test dir')
parser.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

parser.add_argument('--dim1', type=int, default=32)
parser.add_argument('--dim2', type=int, default=8)

group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13,
                         help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)")

group_train.add_argument('--lr', type=float, default=0.001,
                         help="Learning rate, set by the model if not specified.")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=256,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--pro_dim', type=int, default=128)
group_train.add_argument('--dim', type=int, default=512)
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=233,
                    help='random seed ')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=1,  # PaviaU-1 Houston13-5，图像扩充倍数
                    help='multiple of of data augmentation')
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--log_interval', type=int, default=5)
parser.add_argument('--lambda_1', type=float, default=0.1)
parser.add_argument('--lambda_2', type=float, default=0.1)
parser.add_argument('--lr_scheduler', type=str, default='none')

group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=False,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")
args = parser.parse_args()


def \
        evaluate(net, val_loader, gpu, tgt=False):
    ps = []
    ys = []
    t3 = time.time()
    for i, (x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(gpu)
            p1 = net(x1, mode='test')
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys == ps) * 100
    t4 = time.time()
    if tgt:
        results = metrics(ps, ys, n_classes=ys.max() + 1)  # metrics 和 show_results 均是可直接使用的HSI计算工具
        print(results['Confusion_matrix'], '\n', 'TPR:', np.round(results['TPR'] * 100, 2), '\n', 'OA:',
              np.round(results['Accuracy'], 2), 'AA:', np.round(100 * results['AA'], 2), 'Kappa:',
              np.round(100 * results["Kappa"], 2))
        print(f'testing time {t4 - t3:.2f}')
    return acc


def evaluate_tgt(cls_net, gpu, loader, modelpath):
    saved_weight = torch.load(modelpath)
    cls_net.load_state_dict(saved_weight['Discriminator'])
    cls_net.eval()
    teacc = evaluate(cls_net, loader, gpu, tgt=True)
    return teacc


# \subsection{Datasets}
# In the experiments, we perform the DG on real wetland scenes.
#
# ZY1-02D Yancheng data and GF-5 Yancheng data: These two data were acquired by the AHSI aboard on China’s ZY1-02D and GF-5 satellites, respectively. The image sizes are 1398 × 942 and 1175 × 585, respectively. The ZY1-02D and GF-5 Yancheng data are selected as source and target domains, respectively. For DA, 147 bands and seven common categories (i.e., Architecture, River, Reed, Paddy, Fallow land, Sea, and Offshore water) are chosen from two data sets for classification. The number of samples is shown in Table 2. The pseudocolor composite image and GT maps are shown in Fig. 3. This DA task is referred to as YC–YC task.
#
# ZY1-02D Huanghekou data: Two ZY1-02D Huanghekou data were acquired by the ZY1-02D-AHSI in June 28, 2020 and September 29, 2021, respectively. The image sizes are 1147 × 1600 and 1050 × 1219, respectively. The former is set as source domain and the latter is the target domain. For DA, 108 common bands and eight common categories: Reed, Salt Flat Filtration Pond, Salt Flat Evaporation Pond, Salt Flat, Suaeda, River, Sea, and Tide Ditch are used for classification. The number of samples and the categories of the data set are shown in Table 3. The pseudocolor composite image and GT maps are shown in Fig. 4. This DA task is referred to as HHK–HHK task.
#
# Based on the above Yancheng and Huanghekou data sets, we also construct a YC–HHK task. The YC–HHK task uses the ZY1-02D Yancheng data as the source domain, and ZY1-02D Huanghekou data acquired in June 28, 2020 as the target domain. 119 common bands and six common categories: Architecture, Paddy, Fallow land, Fish pond, Sea, and Salt Pond are chosen for DA. The number of samples and the categories of the data set are shown in Table 4. The GT maps are shown in Fig. 5.

def experiment():
    settings = locals().copy()
    print(settings)
    hyperparams = vars(args)
    print(hyperparams)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.source_name + 'to' + args.target_name)
    log_dir = os.path.join(root, str(args.lr) + '_dim' + str(args.pro_dim) +
                           '_pt' + str(args.patch_size) + '_bs' + str(args.batch_size) + '_' + time_str)
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    df = pd.DataFrame([args])
    df.to_csv(os.path.join(log_dir, 'params.txt'))

    seed_worker(args.seed)
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                                        args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                        args.data_path)
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])

    tmp = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')  # 划分训练集和验证集保持了类别比例
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src
    if tmp < 1:  # 如果预计增广后的训练样本数量少于测试样本数量，才真的对训练样本+验证样本增广
        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   worker_init_fn=seed_worker,
                                   generator=g,
                                   shuffle=True, )
    val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 pin_memory=True,
                                 batch_size=hyperparams['batch_size'])
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  batch_size=hyperparams['batch_size'])
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    # =================鉴别器模型配置=================
    config = {
        'patch_size': hyperparams['patch_size'],
        'num_bands': N_BANDS,
        'num_classes': num_classes,
        'num_domains': 1,
    }

    # ==================== 模型构建 ====================
    class SSEA(nn.Module):
        def __init__(self, in_channels, reduction=16):
            super().__init__()
            self.spectral_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(in_channels, in_channels // reduction, 1),
                nn.ReLU(),
                nn.Conv1d(in_channels // reduction, in_channels, 1),
                nn.Sigmoid()
            )
            self.spatial_att = nn.Sequential(
                nn.Conv2d(in_channels, 1, 7, padding=3),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, h, w = x.shape
            spectral = self.spectral_att(x.view(b, c, -1)).view(b, c, 1, 1)
            x_spectral = x * spectral
            spatial = self.spatial_att(x_spectral)
            return x_spectral * spatial + x

        # ------------等效----------
        #     def forward(self, x):
        #         # 输入尺寸: (B, C, H, W)
        #         b, c, h, w = x.shape
        #
        #         # 光谱增强
        #         spectral = self.spectral_att(x.view(b, c, -1)).view(b, c, 1, 1)
        #         x_spectral = x * spectral
        #
        #         # 空间增强
        #         spatial = self.spatial_att(x_spectral)
        #         x_spatial = x_spectral * spatial
        #
        #         return x_spatial + x  # 残差连接

    class InvContrastLoss(nn.Module):
        """本征差异对比损失"""

        def __init__(self, temperature=0.07, margin=1.0):
            super().__init__()
            self.temp = temperature
            self.margin = margin

        def forward(self, feats, labels):
            """
            输入:
                feats: [b,d] 特征向量
                labels: [b] 类别标签
            """
            # 计算样本间相似度矩阵
            sim_matrix = torch.matmul(
                F.normalize(feats, dim=1),
                F.normalize(feats, dim=1).T
            ) / self.temp  # [b,b]

            # 构建同类/异类掩码
            same_class = labels.unsqueeze(1) == labels.unsqueeze(0)  # [b,b]
            diff_class = labels.unsqueeze(1) != labels.unsqueeze(0)

            # 对齐损失（拉近同类样本）
            pos_pairs = -sim_matrix[same_class].mean()

            # 对比损失（推远异类样本）
            neg_pairs = F.relu(
                sim_matrix[diff_class] + self.margin
            ).mean()

            return (pos_pairs + neg_pairs) / 2

    class MetaHead(nn.Module):
        def __init__(self, feat_dim, num_classes):
            super().__init__()
            self.base_classifier = nn.Linear(feat_dim, num_classes)
            self.meta_weights = nn.Parameter(torch.randn(num_classes, feat_dim))
            self.meta_scale = nn.Parameter(torch.ones(1))

        def forward(self, x, labels=None):
            # 标准分类
            base_logits = self.base_classifier(x)  # [b, num_classes]

            # 元学习分类
            x_norm = F.normalize(x, dim=1)  # [b, feat_dim]
            meta_norm = F.normalize(self.meta_weights, dim=1)  # [num_classes, feat_dim]
            meta_logits = einsum('b d, c d -> b c', x_norm, meta_norm) * self.meta_scale

            # 训练时返回结果
            if self.training:
                # 修复：显式将 labels 转为 long 类型索引
                Meta_contrast_loss = -F.cosine_similarity(
                    x_norm,
                    self.meta_weights[labels.to(torch.long)],  # 关键修复
                    dim=1
                ).mean()

                return {
                    'logits': (base_logits + meta_logits) / 2,
                    'base_logits': base_logits,
                    'meta_logits': meta_logits,
                    'Meta_contrast_loss': Meta_contrast_loss
                }
            else:
                return (base_logits + meta_logits) / 2

                # 非元学习改成这个
                # return base_logits

    class WSDN(nn.Module):
        def __init__(self, num_domains):
            super().__init__()
            self.num_domains = num_domains
            self.shared_encoder = nn.Sequential(
                nn.Conv2d(config['num_bands'], 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                SSEA(64)
            )
            self.domain_specific = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.InstanceNorm2d(64),
                    nn.LeakyReLU(0.2)
                ) for _ in range(num_domains)
            ])
            self.domain_invariant = nn.Sequential(
                nn.Conv2d(64, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )

        def forward(self, x, domain_idx):
            shared_feat = self.shared_encoder(x)

            batch_size = shared_feat.size(0)
            # 初始化域特定特征
            domain_specific_feats = torch.zeros(batch_size, 64, *shared_feat.shape[2:],
                                                device=shared_feat.device)

            for i in range(batch_size):
                if domain_idx.dim() == 0:  # 标量情况
                    dom_idx = domain_idx.item()
                    if dom_idx == -1:
                        dom_idx = i % self.num_domains  # 默认轮询分配
                else:  # batch处理
                    dom_idx = domain_idx[i].item()

                domain_specific_feats[i] = self.domain_specific[dom_idx](shared_feat[i:i + 1])
                # [i:i+1] 切片保留维度，确保后续卷积层正常运算

            invariant_feat = self.domain_invariant(domain_specific_feats)
            return {
                'domain_specific': domain_specific_feats,
                'invariant': invariant_feat
            }

    class HydroFeatureEnhancer(nn.Module):
        """基于水文特征的湿地场景特化增强模块"""

        def __init__(self, channels):
            super().__init__()
            # 水文特征检测器（模拟NDWI）
            self.water_detector = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1),
                nn.Tanh()  # 输出[-1,1]模拟归一化指数
            )

            # 水文特征增强路径
            self.hydro_path = nn.Sequential(
                nn.Conv2d(channels + 1, channels, 3, padding=1),  # 输入+水掩膜
                nn.InstanceNorm2d(channels),
                nn.LeakyReLU(0.1),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.InstanceNorm2d(channels),
                nn.LeakyReLU(0.1)
            )

            # 门控融合机制
            self.gate = nn.Conv2d(1, channels, 3, padding=1)

        def forward(self, x):
            # 水文特征提取（模拟NDWI）
            water_map = self.water_detector(x)  # [b,1,h,w]

            # 特征增强
            hydro_feat = self.hydro_path(
                torch.cat([x, water_map], dim=1)
            )

            # 自适应融合（学习水体征的重要性）
            gate_weight = torch.sigmoid(self.gate(water_map))
            return x * (1 - gate_weight) + hydro_feat * gate_weight

    class WMFP(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.branch3x3 = nn.Conv2d(in_channels, 64, 3, padding=1)
            self.branch5x5 = nn.Conv2d(in_channels, 64, 5, padding=2)
            self.branch_dilated = nn.Conv2d(in_channels, 64, 3, padding=4, dilation=4)
            self.merge = nn.Conv2d(64 * 3, 256, 1)
            self.bn = nn.BatchNorm2d(256)
            self.act = nn.GELU()

        def forward(self, x):
            b3 = self.branch3x3(x)  # 精细纹理（草本细节）
            b5 = self.branch5x5(x)  # 中等结构（灌木轮廓）
            bd = self.branch_dilated(x)  # 大范围区域（水体边界）
            merged = torch.cat([b3, b5, bd], dim=1)
            return self.act(self.bn(self.merge(merged)))

    class WetlandGeneralizationNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.wsdn = WSDN(config['num_domains'])
            self.wmfp = WMFP(64)
            self.HFE = HydroFeatureEnhancer(256)

            # 元学习分类头
            self.classifier = MetaHead(256, config['num_classes'])

            # 损失函数
            self.ce_loss = nn.CrossEntropyLoss()
            self.inv_con_loss = InvContrastLoss()

        def forward(self, x, labels=None, domain_idx=None, mode='test'):
            if domain_idx is None:
                domain_idx = torch.tensor(-1, device=x.device)

            features = self.wsdn(x, domain_idx)
            features = self.wmfp(features['invariant'])
            features = self.HFE(features)

            pooled = nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            outputs = self.classifier(pooled, labels)

            # 训练阶段返回损失
            if mode == 'train':
                # 分类损失（非元学习）
                # cls_loss = self.ce_loss(outputs['base_logits'], labels)

                # 元学习
                cls_loss = self.ce_loss(outputs['logits'], labels)

                # 本征对比损失（在域不变特征上）
                contrast_loss = self.inv_con_loss(
                    features.flatten(1), labels
                )

                # 总损失（非元学习）
                # total_loss = cls_loss + args.lambda_1 * contrast_loss

                # 元学习
                total_loss = cls_loss + args.lambda_1 * contrast_loss + args.lambda_2 * outputs['Meta_contrast_loss']

                return {
                    'total_loss': total_loss,
                    'classification': cls_loss,
                    'contrast': contrast_loss,
                }
            else:

                return outputs

    D_net = WetlandGeneralizationNet().to(args.gpu)
    D_opt = optim.AdamW(D_net.parameters(), args.lr)
    cls_criterion = nn.CrossEntropyLoss()

    # =================模型LOSS与训练优化配置=================

    total_trainable_params_D = sum(p.numel() for p in D_net.parameters() if p.requires_grad)
    print(f'{total_trainable_params_D / (1024 * 1024):.2f}M training parameters.')

    # 计算鉴别器FLOPs
    from thop import profile
    input_sample = torch.randn(hyperparams['batch_size'], N_BANDS, hyperparams['patch_size'],
                               hyperparams['patch_size']).to(args.gpu)
    flops_D, _ = profile(D_net, inputs=(input_sample,))
    print(f"D_net FLOPs: {flops_D / 1e9:.2f} GFLOPs")
    # =================开始训练=================
    best_acc = 0
    best_loss = 100
    taracc, taracc_list = 0, []
    for epoch in range(1, args.max_epoch + 1):

        t1 = time.time()
        loss_list = []
        D_net.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.gpu), y.to(args.gpu)
            y = y - 1

            D_opt.zero_grad()
            outputs = D_net(x, y, mode='train')
            outputs['total_loss'].backward()
            D_opt.step()

            loss_list.append(
                [outputs['total_loss'].item(), outputs['classification'].item(), outputs['contrast'].item()])

        total_loss, classification, contrast = np.mean(loss_list, 0)
        D_net.eval()
        teacc = evaluate(D_net, val_loader, args.gpu)

        # 用teacc评价模型泛化能力
        if best_acc <= teacc:
            best_acc = teacc
            # 保存参数的时候不要同时运行另一个调用该参数的程序，否则不会被成功保存
            torch.save({'Discriminator': D_net.state_dict()}, os.path.join(log_dir, f'best.pkl'))

        t2 = time.time()
        print(
            f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2 - t1:.2f}, total_loss {total_loss:.4f}, classification {classification:.4f}, contrast {contrast:.4f} /// val {len(val_loader.dataset)}, teacc {teacc:2.2f}')
        writer.add_scalar('src_cls_loss', total_loss, epoch)
        writer.add_scalar('teacc', teacc, epoch)

        if epoch % args.log_interval == 0:
            pklpath = f'{log_dir}/best.pkl'
            taracc = evaluate_tgt(D_net, args.gpu, test_loader, pklpath)
            taracc_list.append(round(taracc, 2))
            print(f'load pth, target sample number {len(test_loader.dataset)}, max taracc {max(taracc_list):2.2f}')
    writer.close()


if __name__ == '__main__':
    experiment()
