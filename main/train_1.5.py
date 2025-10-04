#!/usr/bin/env python
import itertools
import os
import sys
import time
import random

import kornia
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import cv2

sys.path.append("/opt/data/xiaobin/AIDN")
from base.baseTrainer import poly_learning_rate, reduce_tensor, save_checkpoint, load_state_dict, save_checkpoint_imp
from base.utilities import get_parser, get_logger, main_process, AverageMeter
from models.RevealNet import RevealNet
from models.imp_subnet_DeepMIH import ImpMapBlock
from models.modules.Unet_common import DWT
from models import get_model
from metrics.loss import *
from metrics.perceptual import PerceptualLoss
from metrics import psnr, ssim
from dataset.torch_bicubic import imresize
from torch.optim.lr_scheduler import StepLR
from random import choices

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

population = [i / 10.0 for i in range(11, 41)]
weights = [i ** 2 for i in population]

weights_np = np.array(weights)
weights_np_sum = np.sum(weights_np)
weights = [i / weights_np_sum for i in weights]


def main():
    args = get_parser()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    cudnn.benchmark = True

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.train_gpu = args.train_gpu[0]
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def worker_init_fn(worker_id):
    manual_seed = 131
    random.seed(manual_seed + worker_id)
    np.random.seed(manual_seed + worker_id)
    torch.manual_seed(manual_seed + worker_id)
    torch.cuda.manual_seed(manual_seed + worker_id)
    torch.cuda.manual_seed_all(manual_seed + worker_id)


def main_worker(gpu, ngpus_per_node, args):
    cfg = args
    cfg.gpu = gpu
    best_metric = 1e10
    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size,
                                rank=cfg.rank)
    # ####################### Model ####################### #
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.save_path)
    model = get_model(cfg, logger)
    revealNet = RevealNet(input_nc=3, output_nc=3, cfg=cfg)
    revealNet_2 = RevealNet(input_nc=3, output_nc=3, cfg=cfg)
    imp_net = ImpMapBlock()

    if cfg.sync_bn:
        logger.info("using DDP synced BN")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        revealNet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(revealNet)
        revealNet_2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(revealNet_2)
        imp_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(imp_net)
    if main_process(cfg):
        logger.info(cfg)
        logger.info("=> creating model ...")
        model.summary(logger, writer)
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        cfg.workers = int(cfg.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(gpu), device_ids=[gpu])
        revealNet = torch.nn.parallel.DistributedDataParallel(revealNet.cuda(gpu), device_ids=[gpu])
        revealNet_2 = torch.nn.parallel.DistributedDataParallel(revealNet_2.cuda(gpu), device_ids=[gpu])
        imp_net = torch.nn.parallel.DistributedDataParallel(imp_net.cuda(gpu), device_ids=[gpu])
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda()
        revealNet = revealNet.cuda()
        revealNet_2 = revealNet_2.cuda()
        imp_net = imp_net.cuda()
        # model = torch.nn.DataParallel(model.cuda(), device_ids=gpu)
    # ####################### Loss ####################### #
    loss_fn_lr = nn.MSELoss()
    loss_fn_hr = nn.L1Loss()
    loss = [loss_fn_lr, loss_fn_hr]

    # ####################### Optimizer ####################### #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(
            itertools.chain(model.parameters(), revealNet.parameters(), revealNet_2.parameters(), imp_net.parameters()), lr=cfg.base_lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay)
        optimizer_imp = torch.optim.SGD(imp_net.parameters(), lr=cfg.base_lr,
                                        momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            itertools.chain(model.parameters(), revealNet.parameters(), revealNet_2.parameters(), imp_net.parameters()), lr=cfg.base_lr)
        optimizer_imp = torch.optim.Adam(imp_net.parameters(), lr=cfg.base_lr)

    if cfg.weight:
        if os.path.isfile(cfg.weight):
            if main_process(cfg):
                logger.info("=> loading weight '{}'".format(cfg.weight))
            checkpoint = torch.load(cfg.weight, map_location=torch.device('cpu'))

            load_state_dict(model, checkpoint['state_dict'], strict=False)
            load_state_dict(revealNet, checkpoint['reveal'], strict=False)
            load_state_dict(revealNet_2, checkpoint['reveal_2'], strict=False)
            load_state_dict(imp_net, checkpoint['imp_net'], strict=False)

            if main_process(cfg):
                logger.info("=> loaded weight '{}'".format(cfg.weight))
        else:
            if main_process(cfg):
                logger.info("=> no weight found at '{}'".format(cfg.weight))
    if cfg.StepLR:
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
        scheduler_imp = StepLR(optimizer_imp, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        scheduler = None
        scheduler_imp = None
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            if main_process(cfg):
                logger.info("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
            load_state_dict(model, checkpoint['state_dict'])
            load_state_dict(revealNet, checkpoint['reveal'])
            load_state_dict(revealNet_2, checkpoint['reveal_2'])
            load_state_dict(imp_net, checkpoint['imp_net'])
            cfg.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_metric = checkpoint['best_metric']
            if cfg.StepLR:
                scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma, last_epoch=cfg.start_epoch - 1)
                scheduler.load_state_dict(checkpoint['scheduler'])

            if main_process(cfg):
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.resume, checkpoint['epoch']))
        else:
            if main_process(cfg):
                logger.info("=> no checkpoint found at '{}'".format(cfg.resume))

    # ####################### Data Loader ####################### #
    if cfg.data_name == 'DIV2K':
        from dataset.div2k import DIV2K
        # Prefer explicit list files from config when provided; fallback to default under data_root/list
        train_list = getattr(cfg, 'train_set', None)
        val_list = getattr(cfg, 'val_set', None)
        if not train_list:
            train_list = os.path.join(cfg.data_root, 'list/train.txt')
        if cfg.evaluate and not val_list:
            val_list = os.path.join(cfg.data_root, 'list/valid.txt')

        train_data = DIV2K(data_list=train_list, training=True, cfg=cfg)
        val_data = DIV2K(data_list=val_list, training=False, cfg=cfg) if cfg.evaluate else None

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if cfg.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=cfg.workers, pin_memory=True,
                                                   sampler=train_sampler,
                                                   worker_init_fn=worker_init_fn)
        if cfg.evaluate:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if cfg.distributed else None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.batch_size_val,
                                                     shuffle=False, num_workers=cfg.workers, pin_memory=True,
                                                     drop_last=False,
                                                     worker_init_fn=worker_init_fn, sampler=val_sampler)
    else:
        raise Exception('Dataset not supported yet'.format(cfg.data_name))

    # ####################### Train ####################### #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
            if cfg.evaluate:
                val_sampler.set_epoch(epoch)

        loss_train, hr_loss, _ = train(train_loader, model, revealNet, revealNet_2, imp_net, loss, optimizer, optimizer_imp, epoch, cfg)
        epoch_log = epoch + 1
        # # Adaptive LR
        if cfg.StepLR:
            scheduler.step()
            if scheduler_imp is not None:
                scheduler_imp.step()
        if main_process(cfg):
            logger.info('TRAIN Epoch: {} '
                        'loss_train: {} '
                        'loss_hr: {} '
                        .format(epoch_log, loss_train, hr_loss)
                        )
            for m, s in zip([loss_train, hr_loss],
                            ["train/loss", "train/loss_hr"]):
                writer.add_scalar(s, m, epoch_log)

        is_best = False
        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            loss_val, hr_loss, _, PSNR, SSIM = \
                validate(val_loader, model, revealNet, revealNet_2, imp_net, loss, epoch, cfg)
            if main_process(cfg):
                logger.info('VAL Epoch: {} '
                            'loss_val: {:.6} '
                            'loss_hr: {:.6} '
                            'PSNR: {:.2},{:.2},{:.2},{:.2} '
                            'SSIM: {:.4},{:.4},{:.4},{:.4}'
                            .format(epoch_log, loss_val, hr_loss, *PSNR, *SSIM)
                            )
                for m, s in zip([loss_val, hr_loss, *PSNR, *SSIM],
                                ["val/loss", "val/loss_hr", "val/PSNR_lr", "val/PSNR_hr", "val/SSIM_lr",
                                 "val/SSIM_hr", "val/PSNR_lr_2", "val/PSNR_hr_2", "val/SSIM_lr_2",
                                 "val/SSIM_hr_2"]):
                    writer.add_scalar(s, m, epoch_log)

            # remember best iou and save checkpoint
            is_best = hr_loss < best_metric
            best_metric = min(best_metric, hr_loss)
        if (epoch_log % cfg.save_freq == 0) and main_process(cfg):
            save_checkpoint_imp(model,
                            revealNet,
                            revealNet_2,
                            imp_net,
                            other_state={
                                'epoch': epoch_log,
                                'state_dict': model.state_dict(),
                                'reveal': revealNet.state_dict(),
                                'reveal_2': revealNet_2.state_dict(),
                                'imp_net': imp_net.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_metric': best_metric},
                            sav_path=os.path.join(cfg.save_path, 'model'),
                            is_best=is_best
                            )


def train(train_loader, model, revealNet, revealNet_2, imp_net, loss_fn, optimizer, optimizer_imp, epoch, cfg):
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
    # Haar-DWT for LL subband loss
    dwt = DWT()
    lambda_freq = getattr(cfg, 'lambda_freq', 0.0)
    # Perceptual loss (VGG16 conv3_3)
    lambda_perc = getattr(cfg, 'lambda_perc', 0.0)
    # Warm-start for IM
    warmup_imp_epochs = int(getattr(cfg, 'warmup_imp_epochs', 0))
    lambda_imp = float(getattr(cfg, 'lambda_imp', 1.0))
    perc_crit = None
    if lambda_perc != 0.0:
        perc_crit = PerceptualLoss()
        perc_crit = perc_crit.cuda(cfg.gpu)

    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    for i, batch in enumerate(train_loader):
        # pdb.set_trace()
        if cfg.fixed_scale:  # if training with fixed_scale
            scale = cfg.scale
        else:
            if epoch == 0:
                scale = 1.5 #random.randint(2, cfg.scale)
            else:
                scale = 1.5
                # if cfg.balanceS:
                #     scale = choices(population, weights)[0]
                # else:
                #     scale = random.randint(11, cfg.scale * 10) / 10.0

        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)
        hr, sec = batch['img_gt'], batch['img_sec']
        sec_2 = batch['img_sec_2']

        hr = hr.cuda(cfg.gpu, non_blocking=True)
        sec_gt = sec.cuda(cfg.gpu, non_blocking=True)  # size = hr/scale
        sec_gt2 = sec_2.cuda(cfg.gpu, non_blocking=True)  # size = hr / (scale*2)

        lr_1_4 = imresize(hr, scale=1.0 / (scale * scale)).detach()
        lr_1_2 = imresize(hr, scale=1.0 / scale).detach()

        sec = imresize(sec_gt, scale=1.0 / (scale * scale)).detach()
        sec_2 = imresize(sec_gt2, scale=1.0 / scale).detach()

        # Warm-start phase: only train IM to fit residual (sr_1 - cover)
        if epoch < warmup_imp_epochs:
            # Freeze main networks during warmup
            for p in model.parameters():
                p.requires_grad = False
            for p in revealNet.parameters():
                p.requires_grad = False
            for p in revealNet_2.parameters():
                p.requires_grad = False
            for p in imp_net.parameters():
                p.requires_grad = True

            # Get sr_1 with zero imp map just to form target residual
            zero_imp = torch.zeros_like(lr_1_2)
            sr_1_tmp, _ = model(lr_1_4, sec, sec_2, zero_imp, scale)
            # Target residual: stego_(t-1) - cover
            res_target = (sr_1_tmp.detach() - lr_1_2)
            imp_map = imp_net(lr_1_2, sec_2, sr_1_tmp.detach())
            loss_imp = torch.nn.functional.l1_loss(imp_map, res_target)

            optimizer_imp.zero_grad()
            (lambda_imp * loss_imp).backward()
            optimizer_imp.step()

            # For logging consistency, set other losses to zeros
            restored_hr = sr_1_tmp.detach()
            restored_hr2 = restored_hr.detach()  # placeholder
            recovered = restored_hr.detach()
            recovered_2 = restored_hr.detach()
            dist = restored_hr.detach()
            rev_dist = recovered

            loss_hr = torch.tensor(0.0, device=hr.device)
            loss_hr_2 = torch.tensor(0.0, device=hr.device)
            loss_sec = torch.tensor(0.0, device=hr.device)
            loss_sec_2 = torch.tensor(0.0, device=hr.device)
            l_freq_1 = 0.0
            l_freq_2 = 0.0
            l_perc_1 = 0.0
            l_perc_2 = 0.0
            loss_dist = torch.tensor(0.0, device=hr.device)
            loss_rev_dist = torch.tensor(0.0, device=hr.device)
            loss = lambda_imp * loss_imp
        else:
            # Joint stage: compute imp_map from (cover, stego_(t-1), secret_t), then train end-to-end
            for p in model.parameters():
                p.requires_grad = True
            for p in revealNet.parameters():
                p.requires_grad = True
            for p in revealNet_2.parameters():
                p.requires_grad = True
            for p in imp_net.parameters():
                p.requires_grad = True

            # First pass to get sr_1 (no IM guidance)
            zero_imp = torch.zeros_like(lr_1_2)
            sr_1_tmp, _ = model(lr_1_4, sec, sec_2, zero_imp, scale)
            # Compute importance map using correct DeepMIH inputs: (cover, secret_t, stego_(t-1))
            imp_map = imp_net(lr_1_2, sec_2, sr_1_tmp.detach())
            # Second pass with actual imp_map
            restored_hr, restored_hr2 = model(lr_1_4, sec, sec_2, imp_map, scale)
            recovered = revealNet(restored_hr, scale)
            recovered_2 = revealNet_2(restored_hr2, scale)

            _, _, w, h = restored_hr.shape
            dist = nn.functional.interpolate(restored_hr2, [w, h], mode="bilinear")
            rev_dist = revealNet(dist, scale)

            # Reconstruction losses
            loss_hr = loss_fn[1](restored_hr, lr_1_2)
            loss_hr_2 = loss_fn[1](restored_hr2, hr)
            loss_sec = loss_fn[1](sec, recovered)
            loss_sec_2 = loss_fn[1](sec_2, recovered_2)

            # Low-frequency consistency (Haar-DWT, LL subband)
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

            # Perceptual consistency on stego vs. cover for both stages
            if lambda_perc != 0.0 and perc_crit is not None:
                l_perc_1 = perc_crit(restored_hr, lr_1_2)
                l_perc_2 = perc_crit(restored_hr2, hr)
            else:
                l_perc_1 = 0.0
                l_perc_2 = 0.0

            loss_dist = loss_fn[1](dist, restored_hr)
            loss_rev_dist = loss_fn[1](rev_dist, sec)

            # Stage-wise loss with optional low-frequency and perceptual terms
            loss_stage_1 = loss_hr + loss_sec \
                            + (lambda_freq * l_freq_1 if isinstance(l_freq_1, torch.Tensor) else 0.0) \
                            + (lambda_perc * l_perc_1 if isinstance(l_perc_1, torch.Tensor) else 0.0)
            loss_stage_2 = loss_hr_2 + loss_sec_2 \
                            + (lambda_freq * l_freq_2 if isinstance(l_freq_2, torch.Tensor) else 0.0) \
                            + (lambda_perc * l_perc_2 if isinstance(l_perc_2, torch.Tensor) else 0.0)

            loss = loss_stage_1 + loss_stage_2 + loss_dist + loss_rev_dist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        # Ensure loss_imp is always defined for logging
        if epoch >= warmup_imp_epochs:
            loss_imp = torch.tensor(0.0, device=hr.device)
        for m, x in zip([loss_meter, loss_hr_meter, loss_hr_meter2, loss_sec_meter, loss_sec_meter2,
                         lfreq1_meter, lfreq2_meter, lperc1_meter, lperc2_meter, limp_meter],
                        [loss, loss_hr, loss_hr_2, loss_sec, loss_sec_2,
                         l_freq_1, l_freq_2, l_perc_1, l_perc_2, loss_imp]):
            val_x = x.item() if isinstance(x, torch.Tensor) else float(x)
            m.update(val_x, lr_1_4.shape[0])
        # Adjust lr
        if epoch < warmup_imp_epochs:
            # Track LR of imp optimizer during warmup
            if cfg.poly_lr:
                current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
                for param_group in optimizer_imp.param_groups:
                    param_group['lr'] = current_lr
            else:
                current_lr = optimizer_imp.param_groups[0]['lr']
        else:
            if cfg.poly_lr:
                current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                current_lr = optimizer.param_groups[0]['lr']

        with torch.no_grad():
            batch_enc_psnr = abs(kornia.losses.psnr_loss(restored_hr, lr_1_2, 1))
            batch_dec_psnr = abs(kornia.losses.psnr_loss(recovered, sec, 1))
            batch_enc_ssim = 1 - abs(kornia.losses.ssim_loss(restored_hr.detach(), lr_1_2, window_size=5, reduction="mean"))
            batch_dec_ssim = 1 - abs(kornia.losses.ssim_loss(recovered.detach(), sec, window_size=5, reduction="mean"))

            batch_enc_psnr_2 = abs(kornia.losses.psnr_loss(restored_hr2, hr, 1))
            batch_dec_psnr_2 = abs(kornia.losses.psnr_loss(recovered_2, sec_2, 1))
            batch_enc_ssim_2 = 1 - abs(kornia.losses.ssim_loss(restored_hr2.detach(), hr, window_size=5, reduction="mean"))
            batch_dec_ssim_2 = 1 - abs(
                kornia.losses.ssim_loss(recovered_2.detach(), sec_2, window_size=5, reduction="mean"))

            batch_dist_psnr = abs(kornia.losses.psnr_loss(dist, restored_hr, 1))
            batch_redist_psnr = abs(kornia.losses.psnr_loss(rev_dist, sec, 1))
            batch_dist_ssim = 1 - abs(kornia.losses.ssim_loss(dist.detach(), restored_hr, window_size=5, reduction="mean"))
            batch_redist_ssim = 1 - abs(kornia.losses.ssim_loss(rev_dist.detach(), sec, window_size=5, reduction="mean"))


            data_result_info = ('1/4 SR == psnr_enc:{}, psnr_dec:{}, ssim_enc:{}, ssim_dec:{} '
                                '1/2 SR == psnr_enc2:{}, psnr_dec2:{}, ssim_enc2:{}, ssim_dec2:{}'
                                'Dist == psnr_dist:{}, psnr_redist:{}, ssim_dist:{}, ssim_redist:{}'
                                ).format(batch_enc_psnr, batch_dec_psnr, batch_enc_ssim, batch_dec_ssim,
                                         batch_enc_psnr_2, batch_dec_psnr_2, batch_enc_ssim_2, batch_dec_ssim_2,
                                         batch_dist_psnr, batch_redist_psnr, batch_dist_ssim, batch_redist_ssim)

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain: {remain_time} '
                        'Loss: {loss_meter.val:.4f} '
                        'Loss_hr: {loss_hr_meter.val:.4f} '
                        'Loss_sec: {loss_sec_meter.val:.4f} '
                        'Loss_hr2: {loss_hr_meter2.val:.4f} '
                        'Loss_sec2: {loss_sec_meter2.val:.4f} '
                        'L_imp: {limp_meter.val:.4f} '
                        'L_freq1: {lfreq1_meter.val:.4f} '
                        'L_freq2: {lfreq2_meter.val:.4f} '
                        'L_perc1: {lperc1_meter.val:.4f} '
                        'L_perc2: {lperc2_meter.val:.4f} '
                        'data_info: {data_result_info} '
                        .format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
                                batch_time=batch_time, data_time=data_time,
                                remain_time=remain_time,
                                loss_meter=loss_meter,
                                loss_hr_meter=loss_hr_meter,
                                loss_sec_meter=loss_sec_meter,
                                loss_hr_meter2=loss_hr_meter2,
                                loss_sec_meter2=loss_sec_meter2,
                                limp_meter=limp_meter,
                                lfreq1_meter=lfreq1_meter,
                                lfreq2_meter=lfreq2_meter,
                                lperc1_meter=lperc1_meter,
                                lperc2_meter=lperc2_meter,
                                data_result_info=data_result_info
                                ))
            for m, s in zip([loss_meter, loss_hr_meter, loss_sec_meter, loss_hr_meter2, loss_sec_meter2,
                             limp_meter, lfreq1_meter, lfreq2_meter, lperc1_meter, lperc2_meter],
                            ["train_batch/loss", "train_batch/loss_hr", "train_batch/loss_sec", "train_batch/loss_hr2",
                             "train_batch/loss_sec2", "train_batch/l_imp", "train_batch/l_freq1", "train_batch/l_freq2",
                             "train_batch/l_perc1", "train_batch/l_perc2"]):
                writer.add_scalar(s, m.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)
            writer.add_histogram('train_batch/scale', scale, current_iter)
    # Epoch-level logging of averages for easier tuning
    if main_process(cfg):
        writer.add_scalar('train/l_freq1', lfreq1_meter.avg, epoch + 1)
        writer.add_scalar('train/l_freq2', lfreq2_meter.avg, epoch + 1)
        writer.add_scalar('train/l_perc1', lperc1_meter.avg, epoch + 1)
        writer.add_scalar('train/l_perc2', lperc2_meter.avg, epoch + 1)
        writer.add_scalar('train/l_imp', limp_meter.avg, epoch + 1)
    return loss_meter.avg, loss_hr_meter.avg, loss_sec_meter.avg


def validate(val_loader, model, revealNet, revealNet_2, imp_net, loss_fn, epoch, cfg):
    loss_meter = AverageMeter()
    loss_hr_meter = AverageMeter()
    loss_sec_meter = AverageMeter()
    psnr_meter, ssim_meter = [AverageMeter() for _ in range(4)], [AverageMeter() for _ in range(4)]
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
    lambda_freq = getattr(cfg, 'lambda_freq', 0.0)
    lambda_perc = getattr(cfg, 'lambda_perc', 0.0)
    perc_crit = None
    if lambda_perc != 0.0:
        perc_crit = PerceptualLoss()
        perc_crit = perc_crit.cuda(cfg.gpu)
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            scale = cfg.scale  # 4
            hr, sec = batch['img_gt'], batch['img_sec']
            sec_2 = batch['img_sec_2']

            hr = hr.cuda(cfg.gpu, non_blocking=True)
            sec_gt = sec.cuda(cfg.gpu, non_blocking=True)  # size = hr/scale
            sec_gt2 = sec_2.cuda(cfg.gpu, non_blocking=True)  # size = hr / (scale*2)

            lr_1_4 = imresize(hr, scale=1.0 / (scale * scale)).detach()
            lr_1_2 = imresize(hr, scale=1.0 / scale).detach()

            sec = imresize(sec_gt, scale=1.0 / (scale * scale)).detach()
            sec_2 = imresize(sec_gt2, scale=1.0 / scale).detach()

            # Two-pass evaluation: get sr_1, then compute imp_map and final outputs
            zero_imp = torch.zeros_like(lr_1_2)
            sr_1_tmp, _ = model(lr_1_4, sec, sec_2, zero_imp, scale)
            imp_map = imp_net(lr_1_2, sec_2, sr_1_tmp)
            restored_hr, restored_hr2 = model(lr_1_4, sec, sec_2, imp_map, scale)

            recovered = revealNet(restored_hr, scale)
            recovered_2 = revealNet_2(restored_hr2, scale)

            _, _, w, h = restored_hr.shape
            dist = nn.functional.interpolate(restored_hr2, [w, h], mode="bilinear")

            rev_dist = revealNet(dist, scale)

            # LOSS
            loss_hr = loss_fn[1](restored_hr, lr_1_2)
            loss_hr_2 = loss_fn[1](restored_hr2, hr)
            loss_sec = loss_fn[1](sec, recovered)
            loss_sec_2 = loss_fn[1](sec_2, recovered_2)

            loss_dist = loss_fn[1](dist, restored_hr)
            loss_rev_dist = loss_fn[1](rev_dist, sec)

            # Low-frequency consistency (Haar-DWT, LL subband)
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

            # Perceptual consistency on stego vs. cover for both stages
            if lambda_perc != 0.0 and perc_crit is not None:
                l_perc_1 = perc_crit(restored_hr, lr_1_2)
                l_perc_2 = perc_crit(restored_hr2, hr)
            else:
                l_perc_1 = 0.0
                l_perc_2 = 0.0

            loss_stage_1 = loss_hr + loss_sec \
                            + (lambda_freq * l_freq_1 if isinstance(l_freq_1, torch.Tensor) else 0.0) \
                            + (lambda_perc * l_perc_1 if isinstance(l_perc_1, torch.Tensor) else 0.0)
            loss_stage_2 = loss_hr_2 + loss_sec_2 \
                            + (lambda_freq * l_freq_2 if isinstance(l_freq_2, torch.Tensor) else 0.0) \
                            + (lambda_perc * l_perc_2 if isinstance(l_perc_2, torch.Tensor) else 0.0)

            loss = loss_stage_1 + loss_stage_2 + loss_dist + loss_rev_dist

            psnr_lr, psnr_hr = \
                psnr_calculator(recovered, sec), psnr_calculator(restored_hr, lr_1_2)
            ssim_lr, ssim_hr = \
                ssim_calculator(recovered, sec), ssim_calculator(restored_hr, lr_1_2)

            psnr_lr_2, psnr_hr_2 = \
                psnr_calculator(recovered_2, sec_2), psnr_calculator(restored_hr2, hr)
            ssim_lr_2, ssim_hr_2 = \
                ssim_calculator(recovered_2, sec_2), ssim_calculator(restored_hr2, hr)

            # psnr_dist, psnr_redist = \
            #     psnr_calculator(dist, restored_hr), psnr_calculator(rev_dist, sec)
            # ssim_dist, ssim_redist = \
            #     ssim_calculator(dist, restored_hr), ssim_calculator(rev_dist, sec)

            if cfg.distributed:
                loss = reduce_tensor(loss, cfg)
                loss_hr = reduce_tensor(loss_hr, cfg)

                loss_sec = reduce_tensor(loss_sec, cfg)

                psnr_lr = reduce_tensor(psnr_lr, cfg)
                psnr_hr = reduce_tensor(psnr_hr, cfg)
                ssim_lr = reduce_tensor(ssim_lr, cfg)
                ssim_hr = reduce_tensor(ssim_hr, cfg)

                psnr_lr_2 = reduce_tensor(psnr_lr_2, cfg)
                psnr_hr_2 = reduce_tensor(psnr_hr_2, cfg)
                ssim_lr_2 = reduce_tensor(ssim_lr_2, cfg)
                ssim_hr_2 = reduce_tensor(ssim_hr_2, cfg)


            for m, x in zip([loss_meter, loss_hr_meter, loss_sec_meter, *psnr_meter, *ssim_meter,
                              lfreq1_meter, lfreq2_meter, lperc1_meter, lperc2_meter],
                            [loss, loss_hr, loss_sec, psnr_lr, psnr_hr, psnr_lr_2, psnr_hr_2,
                             ssim_lr, ssim_hr, ssim_lr_2, ssim_hr_2,
                             l_freq_1, l_freq_2, l_perc_1, l_perc_2]):
                val_x = x.item() if isinstance(x, torch.Tensor) else float(x)
                m.update(val_x, hr.shape[0])

            # Visualize after validation
        if main_process(cfg):
            sample_lr = torchvision.utils.make_grid(recovered.clamp(0.0, 1.0))
            sample_hr = torchvision.utils.make_grid(restored_hr.clamp(0.0, 1.0))
            writer.add_image('sample_results/res_lr', sample_lr, epoch + 1)
            writer.add_image('sample_results/res_hr', sample_hr, epoch + 1)
            # Log validation averages for freq/perceptual
            writer.add_scalar('val/l_freq1', lfreq1_meter.avg, epoch + 1)
            writer.add_scalar('val/l_freq2', lfreq2_meter.avg, epoch + 1)
            writer.add_scalar('val/l_perc1', lperc1_meter.avg, epoch + 1)
            writer.add_scalar('val/l_perc2', lperc2_meter.avg, epoch + 1)

    return loss_meter.avg, loss_hr_meter.avg, loss_sec_meter.avg, [m.avg for m in psnr_meter], [m.avg for m in ssim_meter]


if __name__ == '__main__':
    main()
