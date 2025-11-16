#!/usr/bin/env python
import itertools
import os
import sys
import time
import random
from typing import Tuple

import cv2
import kornia
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision
from tensorboardX import SummaryWriter

sys.path.append("/opt/data/xiaobin/AIDN")
from base.baseTrainer import poly_learning_rate, load_state_dict, save_checkpoint_imp
from base.utilities import get_parser, get_logger, main_process, AverageMeter
from dataset.torch_bicubic import imresize
from metrics import psnr, ssim
from metrics.loss import *  # noqa: F401,F403
from metrics.perceptual import PerceptualLoss
from models import get_model
from models.RevealNet import RevealNet
from models.imp_subnet_DeepMIH import ImpMapBlock
from models.modules.Unet_common import DWT
from torch.optim.lr_scheduler import StepLR
from random import choices

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

population = [i / 10.0 for i in range(11, 41)]
weights = [i ** 2 for i in population]
weights_np = np.array(weights)
weights_np_sum = np.sum(weights_np)
weights = [i / weights_np_sum for i in weights]


def get_train_stage(epoch: int, cfg) -> str:
    """
    Decide training stage for current epoch (two-stage schedule).

    根据当前 epoch 决定训练阶段（简化为 2 个阶段）：
      - 前 no_im_epochs 轮：联合训练 SR/IH 与嵌入解密网络，但关闭 IM（阶段名 "hide_reveal"）
      - 之后：端到端训练 IH + IM + reveal（阶段名 "full"）

    其中 no_im_epochs 默认由 sr_pretrain_epochs + hide_pretrain_epochs 给出，
    你也可以通过 im_start_epoch 显式指定。
    """
    # 显式指定 IM 启用起始 epoch 优先
    im_start_explicit = getattr(cfg, "im_start_epoch", None)
    if im_start_explicit is not None:
        no_im_epochs = int(im_start_explicit)
    else:
        sr_pre = int(getattr(cfg, "sr_pretrain_epochs", 0))
        hide_pre = int(getattr(cfg, "hide_pretrain_epochs", 0))
        no_im_epochs = sr_pre + hide_pre

    if epoch < no_im_epochs:
        return "hide_reveal"  # 只训练 IH + reveal，IM 关闭
    return "full"  # IH + IM + reveal 端到端


def main() -> None:
    cfg = get_parser()

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        # cfg.train_gpu is a list like [0]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.train_gpu)

    cudnn.benchmark = True

    if getattr(cfg, "manual_seed", None) is not None:
        random.seed(cfg.manual_seed)
        np.random.seed(cfg.manual_seed)
        torch.manual_seed(cfg.manual_seed)
        torch.cuda.manual_seed(cfg.manual_seed)
        torch.cuda.manual_seed_all(cfg.manual_seed)

    # Single-GPU training for simplicity; original multi-GPU script is train_1.5.py
    if isinstance(cfg.train_gpu, (list, tuple)):
        cfg.gpu = cfg.train_gpu[0]
    else:
        cfg.gpu = int(cfg.train_gpu)

    # Disable distributed path for this multi-stage trainer
    cfg.distributed = False
    cfg.multiprocessing_distributed = False
    cfg.world_size = 1
    cfg.rank = 0

    main_worker(cfg)


def worker_init_fn(worker_id: int) -> None:
    manual_seed = 131
    random.seed(manual_seed + worker_id)
    np.random.seed(manual_seed + worker_id)
    torch.manual_seed(manual_seed + worker_id)
    torch.cuda.manual_seed(manual_seed + worker_id)
    torch.cuda.manual_seed_all(manual_seed + worker_id)


def main_worker(cfg) -> None:
    best_metric = 1e10

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.save_path)

    # ####################### Model ####################### #
    model = get_model(cfg, logger)
    revealNet = RevealNet(input_nc=3, output_nc=3, cfg=cfg)
    revealNet_2 = RevealNet(input_nc=3, output_nc=3, cfg=cfg)
    imp_net = ImpMapBlock()

    if cfg.sync_bn:
        logger.info("using SyncBatchNorm")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        revealNet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(revealNet)
        revealNet_2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(revealNet_2)
        imp_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(imp_net)

    if main_process(cfg):
        logger.info(cfg)
        logger.info("=> creating model ...")
        model.summary(logger, writer)

    torch.cuda.set_device(cfg.gpu)
    model = model.cuda(cfg.gpu)
    revealNet = revealNet.cuda(cfg.gpu)
    revealNet_2 = revealNet_2.cuda(cfg.gpu)
    imp_net = imp_net.cuda(cfg.gpu)

    # ####################### Loss ####################### #
    loss_fn_lr = nn.MSELoss()
    loss_fn_hr = nn.L1Loss()
    loss_fns = [loss_fn_lr, loss_fn_hr]

    # ####################### Optimizer ####################### #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(
            itertools.chain(
                model.parameters(),
                revealNet.parameters(),
                revealNet_2.parameters(),
                imp_net.parameters(),
            ),
            lr=cfg.base_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        optimizer_imp = torch.optim.SGD(
            imp_net.parameters(),
            lr=cfg.base_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            itertools.chain(
                model.parameters(),
                revealNet.parameters(),
                revealNet_2.parameters(),
                imp_net.parameters(),
            ),
            lr=cfg.base_lr,
        )
        optimizer_imp = torch.optim.Adam(imp_net.parameters(), lr=cfg.base_lr)

    # Optionally initialize from a pre-trained checkpoint (e.g., DIV2K)
    if getattr(cfg, "weight", None):
        if os.path.isfile(cfg.weight):
            if main_process(cfg):
                logger.info("=> loading weight '%s'", cfg.weight)
            checkpoint = torch.load(cfg.weight, map_location=torch.device("cpu"))
            load_state_dict(model, checkpoint["state_dict"], strict=False)
            load_state_dict(revealNet, checkpoint["reveal"], strict=False)
            load_state_dict(revealNet_2, checkpoint["reveal_2"], strict=False)
            load_state_dict(imp_net, checkpoint["imp_net"], strict=False)
            if main_process(cfg):
                logger.info("=> loaded weight '%s'", cfg.weight)
        else:
            if main_process(cfg):
                logger.info("=> no weight found at '%s'", cfg.weight)

    if cfg.StepLR:
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
        scheduler_imp = StepLR(optimizer_imp, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        scheduler = None
        scheduler_imp = None

    # Resume full checkpoint (multi-stage aware only through epochs)
    if getattr(cfg, "resume", None):
        if os.path.isfile(cfg.resume):
            if main_process(cfg):
                logger.info("=> loading checkpoint '%s'", cfg.resume)
            checkpoint = torch.load(cfg.resume, map_location=torch.device("cpu"))
            load_state_dict(model, checkpoint["state_dict"])
            load_state_dict(revealNet, checkpoint["reveal"])
            load_state_dict(revealNet_2, checkpoint["reveal_2"])
            load_state_dict(imp_net, checkpoint["imp_net"])
            cfg.start_epoch = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_metric = checkpoint["best_metric"]
            if cfg.StepLR and "scheduler" in checkpoint:
                scheduler = StepLR(
                    optimizer,
                    step_size=cfg.step_size,
                    gamma=cfg.gamma,
                    last_epoch=cfg.start_epoch - 1,
                )
                scheduler.load_state_dict(checkpoint["scheduler"])
            if main_process(cfg):
                logger.info(
                    "=> loaded checkpoint '%s' (epoch %d)", cfg.resume, checkpoint["epoch"]
                )
        else:
            if main_process(cfg):
                logger.info("=> no checkpoint found at '%s'", cfg.resume)

    # ####################### Data Loader ####################### #
    if cfg.data_name == "DIV2K":
        from dataset.div2k import DIV2K

        train_list = getattr(cfg, "train_set", None)
        val_list = getattr(cfg, "val_set", None)

        if train_list:
            if not os.path.isabs(train_list):
                train_list = os.path.join(cfg.data_root, train_list)
        else:
            train_list = os.path.join(cfg.data_root, "list/train.txt")

        if cfg.evaluate:
            if val_list:
                if not os.path.isabs(val_list):
                    val_list = os.path.join(cfg.data_root, val_list)
            else:
                val_list = os.path.join(cfg.data_root, "list/valid.txt")

        train_data = DIV2K(data_list=train_list, training=True, cfg=cfg)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        if cfg.evaluate:
            val_data = DIV2K(data_list=val_list, training=False, cfg=cfg)
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=cfg.batch_size_val,
                shuffle=False,
                num_workers=cfg.workers,
                pin_memory=True,
                drop_last=False,
                worker_init_fn=worker_init_fn,
            )
        else:
            val_loader = None
    else:
        raise Exception("Dataset not supported yet: {}".format(cfg.data_name))

    # ####################### Train ####################### #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        train_stage = get_train_stage(epoch, cfg)

        if main_process(cfg):
            logger.info(
                "====> Epoch %d / %d, train_stage=%s",
                epoch + 1,
                cfg.epochs,
                train_stage,
            )

        loss_train, hr_loss, _ = train(
            train_loader,
            model,
            revealNet,
            revealNet_2,
            imp_net,
            loss_fns,
            optimizer,
            optimizer_imp,
            epoch,
            cfg,
            train_stage=train_stage,
        )

        epoch_log = epoch + 1

        # Step LR schedulers according to stage
        if cfg.StepLR:
            if train_stage == "imp_warmup" and scheduler_imp is not None:
                scheduler_imp.step()
            elif scheduler is not None:
                scheduler.step()

        if main_process(cfg):
            logger.info(
                "TRAIN Epoch: %d loss_train: %f loss_hr: %f",
                epoch_log,
                float(loss_train),
                float(hr_loss),
            )
            for m_val, tag in zip(
                [loss_train, hr_loss],
                ["train/loss", "train/loss_hr"],
            ):
                writer.add_scalar(tag, m_val, epoch_log)

        is_best = False
        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0) and val_loader is not None:
            loss_val, hr_val_loss, _, psnr_vals, ssim_vals = validate(
                val_loader,
                model,
                revealNet,
                revealNet_2,
                imp_net,
                loss_fns,
                epoch,
                cfg,
            )
            if main_process(cfg):
                logger.info(
                    "VAL Epoch: %d loss_val: %.6f loss_hr: %.6f "
                    "PSNR: %.2f,%.2f,%.2f,%.2f SSIM: %.4f,%.4f,%.4f,%.4f",
                    epoch_log,
                    loss_val,
                    hr_val_loss,
                    *psnr_vals,
                    *ssim_vals,
                )
                for m_val, tag in zip(
                    [loss_val, hr_val_loss, *psnr_vals, *ssim_vals],
                    [
                        "val/loss",
                        "val/loss_hr",
                        "val/PSNR_lr",
                        "val/PSNR_hr",
                        "val/SSIM_lr",
                        "val/SSIM_hr",
                        "val/PSNR_lr_2",
                        "val/PSNR_hr_2",
                        "val/SSIM_lr_2",
                        "val/SSIM_hr_2",
                    ],
                ):
                    writer.add_scalar(tag, m_val, epoch_log)

            is_best = hr_val_loss < best_metric
            best_metric = min(best_metric, hr_val_loss)

        if (epoch_log % cfg.save_freq == 0) and main_process(cfg):
            other_state = {
                "epoch": epoch_log,
                "state_dict": model.state_dict(),
                "reveal": revealNet.state_dict(),
                "reveal_2": revealNet_2.state_dict(),
                "imp_net": imp_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_metric,
            }
            if cfg.StepLR and scheduler is not None:
                other_state["scheduler"] = scheduler.state_dict()

            save_checkpoint_imp(
                model,
                revealNet,
                revealNet_2,
                imp_net,
                other_state=other_state,
                sav_path=os.path.join(cfg.save_path, "model"),
                is_best=is_best,
            )


def train(
    train_loader,
    model,
    revealNet,
    revealNet_2,
    imp_net,
    loss_fn,
    optimizer,
    optimizer_imp,
    epoch: int,
    cfg,
    train_stage: str,
) -> Tuple[float, float, float]:
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_hr_meter = AverageMeter()
    loss_hr_meter2 = AverageMeter()
    loss_sec_meter = AverageMeter()
    loss_sec_meter2 = AverageMeter()
    lfreq1_meter = AverageMeter()
    lfreq2_meter = AverageMeter()
    lperc1_meter = AverageMeter()
    lperc2_meter = AverageMeter()
    limp_meter = AverageMeter()

    model.train()
    revealNet.train()
    revealNet_2.train()
    imp_net.train()

    dwt = DWT()
    lambda_freq = float(getattr(cfg, "lambda_freq", 0.0))
    lambda_perc = float(getattr(cfg, "lambda_perc", 0.0))
    lambda_imp = float(getattr(cfg, "lambda_imp", 1.0))

    perc_crit = None
    if lambda_perc != 0.0:
        perc_crit = PerceptualLoss().cuda(cfg.gpu)

    end = time.time()
    max_iter = cfg.epochs * len(train_loader)

    for i, batch in enumerate(train_loader):
        if cfg.fixed_scale:
            scale = cfg.scale
        else:
            if epoch == 0:
                scale = 1.5
            else:
                scale = 1.5
                # if cfg.balanceS:
                #     scale = choices(population, weights)[0]
                # else:
                #     scale = random.randint(11, cfg.scale * 10) / 10.0

        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)

        hr, sec = batch["img_gt"], batch["img_sec"]
        sec_2 = batch["img_sec_2"]

        hr = hr.cuda(cfg.gpu, non_blocking=True)
        sec_gt = sec.cuda(cfg.gpu, non_blocking=True)
        sec_gt2 = sec_2.cuda(cfg.gpu, non_blocking=True)

        lr_1_4 = imresize(hr, scale=1.0 / (scale * scale)).detach()
        lr_1_2 = imresize(hr, scale=1.0 / scale).detach()

        sec = imresize(sec_gt, scale=1.0 / (scale * scale)).detach()
        sec_2 = imresize(sec_gt2, scale=1.0 / scale).detach()

        # Initialise metrics containers
        loss_imp = torch.tensor(0.0, device=hr.device)
        l_freq_1 = 0.0
        l_freq_2 = 0.0
        l_perc_1 = 0.0
        l_perc_2 = 0.0

        if train_stage == "hide_reveal":
            # 阶段1：联合训练 SR/IH 与嵌入解密网络，但关闭 IM（x_imp = 0）。
            for p in model.parameters():
                p.requires_grad = True
            for p in revealNet.parameters():
                p.requires_grad = True
            for p in revealNet_2.parameters():
                p.requires_grad = True
            for p in imp_net.parameters():
                p.requires_grad = False

            zero_imp = torch.zeros_like(lr_1_2)
            restored_hr, restored_hr2 = model(lr_1_4, sec, sec_2, zero_imp, scale)
            recovered = revealNet(restored_hr, scale)
            recovered_2 = revealNet_2(restored_hr2, scale)

            _, _, w, h = restored_hr.shape
            dist = F.interpolate(
                restored_hr2, [w, h], mode="bilinear", align_corners=False
            )
            rev_dist = revealNet(dist, scale)

            loss_hr = loss_fn[1](restored_hr, lr_1_2)
            loss_hr_2 = loss_fn[1](restored_hr2, hr)
            loss_sec = loss_fn[1](sec, recovered)
            loss_sec_2 = loss_fn[1](sec_2, recovered_2)

            if lambda_freq != 0.0:
                ll_stego_1 = dwt(restored_hr).narrow(1, 0, restored_hr.shape[1])
                ll_cover_1 = dwt(lr_1_2).narrow(1, 0, lr_1_2.shape[1])
                l_freq_1 = torch.nn.functional.l1_loss(ll_stego_1, ll_cover_1)

                ll_stego_2 = dwt(restored_hr2).narrow(1, 0, restored_hr2.shape[1])
                ll_cover_2 = dwt(hr).narrow(1, 0, hr.shape[1])
                l_freq_2 = torch.nn.functional.l1_loss(ll_stego_2, ll_cover_2)

            if lambda_perc != 0.0 and perc_crit is not None:
                l_perc_1 = perc_crit(restored_hr, lr_1_2)
                l_perc_2 = perc_crit(restored_hr2, hr)

            loss_dist = loss_fn[1](dist, restored_hr)
            loss_rev_dist = loss_fn[1](rev_dist, sec)

            loss_stage_1 = (
                loss_hr
                + loss_sec
                + (lambda_freq * l_freq_1 if isinstance(l_freq_1, torch.Tensor) else 0.0)
                + (lambda_perc * l_perc_1 if isinstance(l_perc_1, torch.Tensor) else 0.0)
            )
            loss_stage_2 = (
                loss_hr_2
                + loss_sec_2
                + (lambda_freq * l_freq_2 if isinstance(l_freq_2, torch.Tensor) else 0.0)
                + (lambda_perc * l_perc_2 if isinstance(l_perc_2, torch.Tensor) else 0.0)
            )

            loss = loss_stage_1 + loss_stage_2 + loss_dist + loss_rev_dist
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            # 阶段2：joint end-to-end training with IH + IM + reveal.
            for p in model.parameters():
                p.requires_grad = True
            for p in revealNet.parameters():
                p.requires_grad = True
            for p in revealNet_2.parameters():
                p.requires_grad = True
            for p in imp_net.parameters():
                p.requires_grad = True

            zero_imp = torch.zeros_like(lr_1_2)
            sr_1_tmp, _ = model(lr_1_4, sec, sec_2, zero_imp, scale)
            imp_map = imp_net(lr_1_2, sec_2, sr_1_tmp.detach())
            restored_hr, restored_hr2 = model(lr_1_4, sec, sec_2, imp_map, scale)

            recovered = revealNet(restored_hr, scale)
            recovered_2 = revealNet_2(restored_hr2, scale)

            _, _, w, h = restored_hr.shape
            dist = F.interpolate(restored_hr2, [w, h], mode="bilinear", align_corners=False)
            rev_dist = revealNet(dist, scale)

            loss_hr = loss_fn[1](restored_hr, lr_1_2)
            loss_hr_2 = loss_fn[1](restored_hr2, hr)
            loss_sec = loss_fn[1](sec, recovered)
            loss_sec_2 = loss_fn[1](sec_2, recovered_2)

            if lambda_freq != 0.0:
                ll_stego_1 = dwt(restored_hr).narrow(1, 0, restored_hr.shape[1])
                ll_cover_1 = dwt(lr_1_2).narrow(1, 0, lr_1_2.shape[1])
                l_freq_1 = torch.nn.functional.l1_loss(ll_stego_1, ll_cover_1)

                ll_stego_2 = dwt(restored_hr2).narrow(1, 0, restored_hr2.shape[1])
                ll_cover_2 = dwt(hr).narrow(1, 0, hr.shape[1])
                l_freq_2 = torch.nn.functional.l1_loss(ll_stego_2, ll_cover_2)

            if lambda_perc != 0.0 and perc_crit is not None:
                l_perc_1 = perc_crit(restored_hr, lr_1_2)
                l_perc_2 = perc_crit(restored_hr2, hr)

            loss_dist = loss_fn[1](dist, restored_hr)
            loss_rev_dist = loss_fn[1](rev_dist, sec)

            loss_stage_1 = (
                loss_hr
                + loss_sec
                + (lambda_freq * l_freq_1 if isinstance(l_freq_1, torch.Tensor) else 0.0)
                + (lambda_perc * l_perc_1 if isinstance(l_perc_1, torch.Tensor) else 0.0)
            )
            loss_stage_2 = (
                loss_hr_2
                + loss_sec_2
                + (lambda_freq * l_freq_2 if isinstance(l_freq_2, torch.Tensor) else 0.0)
                + (lambda_perc * l_perc_2 if isinstance(l_perc_2, torch.Tensor) else 0.0)
            )

            loss = loss_stage_1 + loss_stage_2 + loss_dist + loss_rev_dist
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Common logging part
        batch_time.update(time.time() - end)
        end = time.time()

        for meter, value in zip(
            [
                loss_meter,
                loss_hr_meter,
                loss_hr_meter2,
                loss_sec_meter,
                loss_sec_meter2,
                lfreq1_meter,
                lfreq2_meter,
                lperc1_meter,
                lperc2_meter,
                limp_meter,
            ],
            [
                loss,
                loss_hr,
                loss_hr_2,
                loss_sec,
                loss_sec_2,
                l_freq_1,
                l_freq_2,
                l_perc_1,
                l_perc_2,
                loss_imp,
            ],
        ):
            val_x = value.item() if isinstance(value, torch.Tensor) else float(value)
            meter.update(val_x, lr_1_4.shape[0])

        # Adjust LR per-iteration if using poly schedule
        if cfg.poly_lr:
            current_lr = poly_learning_rate(
                cfg.base_lr, current_iter, max_iter, power=cfg.power
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        with torch.no_grad():
            batch_enc_psnr = abs(kornia.losses.psnr_loss(restored_hr, lr_1_2, 1))
            batch_dec_psnr = abs(kornia.losses.psnr_loss(recovered, sec, 1))
            batch_enc_ssim = 1 - abs(
                kornia.losses.ssim_loss(
                    restored_hr.detach(),
                    lr_1_2,
                    window_size=5,
                    reduction="mean",
                )
            )
            batch_dec_ssim = 1 - abs(
                kornia.losses.ssim_loss(
                    recovered.detach(), sec, window_size=5, reduction="mean"
                )
            )

            batch_enc_psnr_2 = abs(kornia.losses.psnr_loss(restored_hr2, hr, 1))
            batch_dec_psnr_2 = abs(kornia.losses.psnr_loss(recovered_2, sec_2, 1))
            batch_enc_ssim_2 = 1 - abs(
                kornia.losses.ssim_loss(
                    restored_hr2.detach(), hr, window_size=5, reduction="mean"
                )
            )
            batch_dec_ssim_2 = 1 - abs(
                kornia.losses.ssim_loss(
                    recovered_2.detach(), sec_2, window_size=5, reduction="mean"
                )
            )

            batch_dist_psnr = abs(kornia.losses.psnr_loss(dist, restored_hr, 1))
            batch_redist_psnr = abs(kornia.losses.psnr_loss(rev_dist, sec, 1))
            batch_dist_ssim = 1 - abs(
                kornia.losses.ssim_loss(
                    dist.detach(), restored_hr, window_size=5, reduction="mean"
                )
            )
            batch_redist_ssim = 1 - abs(
                kornia.losses.ssim_loss(
                    rev_dist.detach(), sec, window_size=5, reduction="mean"
                )
            )

            data_result_info = (
                "1/4 SR == psnr_enc:{}, psnr_dec:{}, ssim_enc:{}, ssim_dec:{} "
                "1/2 SR == psnr_enc2:{}, psnr_dec2:{}, ssim_enc2:{}, ssim_dec2:{}"
                "Dist == psnr_dist:{}, psnr_redist:{}, ssim_dist:{}, ssim_redist:{}"
            ).format(
                batch_enc_psnr,
                batch_dec_psnr,
                batch_enc_ssim,
                batch_dec_ssim,
                batch_enc_psnr_2,
                batch_dec_psnr_2,
                batch_enc_ssim_2,
                batch_dec_ssim_2,
                batch_dist_psnr,
                batch_redist_psnr,
                batch_dist_ssim,
                batch_redist_ssim,
            )

        # Remaining time estimation
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time_str = "{:02d}:{:02d}:{:02d}".format(
            int(t_h), int(t_m), int(t_s)
        )

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            logger.info(
                "Epoch: [%d/%d][%d/%d] "
                "Data: %.3f (%.3f) "
                "Batch: %.3f (%.3f) "
                "Remain: %s "
                "Loss: %.4f "
                "Loss_hr: %.4f "
                "Loss_sec: %.4f "
                "Loss_hr2: %.4f "
                "Loss_sec2: %.4f "
                "L_imp: %.4f "
                "L_freq1: %.4f "
                "L_freq2: %.4f "
                "L_perc1: %.4f "
                "L_perc2: %.4f "
                "data_info: %s ",
                epoch + 1,
                cfg.epochs,
                i + 1,
                len(train_loader),
                data_time.val,
                data_time.avg,
                batch_time.val,
                batch_time.avg,
                remain_time_str,
                loss_meter.val,
                loss_hr_meter.val,
                loss_sec_meter.val,
                loss_hr_meter2.val,
                loss_sec_meter2.val,
                limp_meter.val,
                lfreq1_meter.val,
                lfreq2_meter.val,
                lperc1_meter.val,
                lperc2_meter.val,
                data_result_info,
            )
            for meter, tag in zip(
                [
                    loss_meter,
                    loss_hr_meter,
                    loss_sec_meter,
                    loss_hr_meter2,
                    loss_sec_meter2,
                    limp_meter,
                    lfreq1_meter,
                    lfreq2_meter,
                    lperc1_meter,
                    lperc2_meter,
                ],
                [
                    "train_batch/loss",
                    "train_batch/loss_hr",
                    "train_batch/loss_sec",
                    "train_batch/loss_hr2",
                    "train_batch/loss_sec2",
                    "train_batch/l_imp",
                    "train_batch/l_freq1",
                    "train_batch/l_freq2",
                    "train_batch/l_perc1",
                    "train_batch/l_perc2",
                ],
            ):
                writer.add_scalar(tag, meter.val, current_iter)
            writer.add_scalar("learning_rate", current_lr, current_iter)
            writer.add_histogram("train_batch/scale", scale, current_iter)

    if main_process(cfg):
        writer.add_scalar("train/l_freq1", lfreq1_meter.avg, epoch + 1)
        writer.add_scalar("train/l_freq2", lfreq2_meter.avg, epoch + 1)
        writer.add_scalar("train/l_perc1", lperc1_meter.avg, epoch + 1)
        writer.add_scalar("train/l_perc2", lperc2_meter.avg, epoch + 1)
        writer.add_scalar("train/l_imp", limp_meter.avg, epoch + 1)

    return loss_meter.avg, loss_hr_meter.avg, loss_sec_meter.avg


def validate(val_loader, model, revealNet, revealNet_2, imp_net, loss_fn, epoch: int, cfg):
    loss_meter = AverageMeter()
    loss_hr_meter = AverageMeter()
    loss_sec_meter = AverageMeter()
    psnr_meter = [AverageMeter() for _ in range(4)]
    ssim_meter = [AverageMeter() for _ in range(4)]
    lfreq1_meter = AverageMeter()
    lfreq2_meter = AverageMeter()
    lperc1_meter = AverageMeter()
    lperc2_meter = AverageMeter()

    psnr_calculator, ssim_calculator = psnr.PSNR(), ssim.SSIM()

    model.eval()
    revealNet.eval()
    revealNet_2.eval()
    imp_net.eval()

    dwt = DWT()
    lambda_freq = float(getattr(cfg, "lambda_freq", 0.0))
    lambda_perc = float(getattr(cfg, "lambda_perc", 0.0))

    perc_crit = None
    if lambda_perc != 0.0:
        perc_crit = PerceptualLoss().cuda(cfg.gpu)

    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            scale = cfg.scale
            hr, sec = batch["img_gt"], batch["img_sec"]
            sec_2 = batch["img_sec_2"]

            hr = hr.cuda(cfg.gpu, non_blocking=True)
            sec_gt = sec.cuda(cfg.gpu, non_blocking=True)
            sec_gt2 = sec_2.cuda(cfg.gpu, non_blocking=True)

            lr_1_4 = imresize(hr, scale=1.0 / (scale * scale)).detach()
            lr_1_2 = imresize(hr, scale=1.0 / scale).detach()

            sec = imresize(sec_gt, scale=1.0 / (scale * scale)).detach()
            sec_2 = imresize(sec_gt2, scale=1.0 / scale).detach()

            zero_imp = torch.zeros_like(lr_1_2)
            sr_1_tmp, _ = model(lr_1_4, sec, sec_2, zero_imp, scale)
            imp_map = imp_net(lr_1_2, sec_2, sr_1_tmp)
            restored_hr, restored_hr2 = model(lr_1_4, sec, sec_2, imp_map, scale)

            recovered = revealNet(restored_hr, scale)
            recovered_2 = revealNet_2(restored_hr2, scale)

            _, _, w, h = restored_hr.shape
            dist = F.interpolate(restored_hr2, [w, h], mode="bilinear", align_corners=False)
            rev_dist = revealNet(dist, scale)

            loss_hr = loss_fn[1](restored_hr, lr_1_2)
            loss_hr_2 = loss_fn[1](restored_hr2, hr)
            loss_sec = loss_fn[1](sec, recovered)
            loss_sec_2 = loss_fn[1](sec_2, recovered_2)

            loss_dist = loss_fn[1](dist, restored_hr)
            loss_rev_dist = loss_fn[1](rev_dist, sec)

            if lambda_freq != 0.0:
                ll_stego_1 = dwt(restored_hr).narrow(1, 0, restored_hr.shape[1])
                ll_cover_1 = dwt(lr_1_2).narrow(1, 0, lr_1_2.shape[1])
                l_freq_1 = torch.nn.functional.l1_loss(ll_stego_1, ll_cover_1)

                ll_stego_2 = dwt(restored_hr2).narrow(1, 0, restored_hr2.shape[1])
                ll_cover_2 = dwt(hr).narrow(1, 0, hr.shape[1])
                l_freq_2 = torch.nn.functional.l1_loss(ll_stego_2, ll_cover_2)
            else:
                l_freq_1 = 0.0
                l_freq_2 = 0.0

            if lambda_perc != 0.0 and perc_crit is not None:
                l_perc_1 = perc_crit(restored_hr, lr_1_2)
                l_perc_2 = perc_crit(restored_hr2, hr)
            else:
                l_perc_1 = 0.0
                l_perc_2 = 0.0

            loss_stage_1 = (
                loss_hr
                + loss_sec
                + (lambda_freq * l_freq_1 if isinstance(l_freq_1, torch.Tensor) else 0.0)
                + (lambda_perc * l_perc_1 if isinstance(l_perc_1, torch.Tensor) else 0.0)
            )
            loss_stage_2 = (
                loss_hr_2
                + loss_sec_2
                + (lambda_freq * l_freq_2 if isinstance(l_freq_2, torch.Tensor) else 0.0)
                + (lambda_perc * l_perc_2 if isinstance(l_perc_2, torch.Tensor) else 0.0)
            )

            loss = loss_stage_1 + loss_stage_2 + loss_dist + loss_rev_dist

            psnr_lr, psnr_hr = (
                psnr_calculator(recovered, sec),
                psnr_calculator(restored_hr, lr_1_2),
            )
            ssim_lr, ssim_hr = (
                ssim_calculator(recovered, sec),
                ssim_calculator(restored_hr, lr_1_2),
            )

            psnr_lr_2, psnr_hr_2 = (
                psnr_calculator(recovered_2, sec_2),
                psnr_calculator(restored_hr2, hr),
            )
            ssim_lr_2, ssim_hr_2 = (
                ssim_calculator(recovered_2, sec_2),
                ssim_calculator(restored_hr2, hr),
            )

            for meter, value in zip(
                [
                    loss_meter,
                    loss_hr_meter,
                    loss_sec_meter,
                    *psnr_meter,
                    *ssim_meter,
                    lfreq1_meter,
                    lfreq2_meter,
                    lperc1_meter,
                    lperc2_meter,
                ],
                [
                    loss,
                    loss_hr,
                    loss_sec,
                    psnr_lr,
                    psnr_hr,
                    psnr_lr_2,
                    psnr_hr_2,
                    ssim_lr,
                    ssim_hr,
                    ssim_lr_2,
                    ssim_hr_2,
                    l_freq_1,
                    l_freq_2,
                    l_perc_1,
                    l_perc_2,
                ],
            ):
                val_x = value.item() if isinstance(value, torch.Tensor) else float(value)
                meter.update(val_x, hr.shape[0])

        if main_process(cfg):
            sample_lr = torchvision.utils.make_grid(recovered.clamp(0.0, 1.0))
            sample_hr = torchvision.utils.make_grid(restored_hr.clamp(0.0, 1.0))
            writer.add_image("sample_results/res_lr", sample_lr, epoch + 1)
            writer.add_image("sample_results/res_hr", sample_hr, epoch + 1)
            writer.add_scalar("val/l_freq1", lfreq1_meter.avg, epoch + 1)
            writer.add_scalar("val/l_freq2", lfreq2_meter.avg, epoch + 1)
            writer.add_scalar("val/l_perc1", lperc1_meter.avg, epoch + 1)
            writer.add_scalar("val/l_perc2", lperc2_meter.avg, epoch + 1)

    return (
        loss_meter.avg,
        loss_hr_meter.avg,
        loss_sec_meter.avg,
        [m.avg for m in psnr_meter],
        [m.avg for m in ssim_meter],
    )


if __name__ == "__main__":
    main()
