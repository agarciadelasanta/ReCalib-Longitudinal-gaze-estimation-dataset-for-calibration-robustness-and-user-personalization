import os
import time
import shutil
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, angular_error
from model import gaze_network


class Trainer:
    def __init__(self, config, data_loader, val_loader=None):
        self.config = config

        # data
        self.train_loader = data_loader if config.is_train else None
        self.test_loader = None if config.is_train else data_loader
        self.val_loader = val_loader  # optional

        self.num_train = len(self.train_loader.dataset) if self.train_loader else 0
        self.num_test = len(self.test_loader.dataset) if self.test_loader else 0
        self.batch_size = config.batch_size

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.lr = config.init_lr
        self.lr_patience = config.lr_patience
        self.lr_decay_factor = config.lr_decay_factor

        # misc
        self.ckpt_dir = config.ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.print_freq = config.print_freq
        self.train_iter = 0
        self.pre_trained_model_path = getattr(config, "pre_trained_model_path", None)

        # device
        use_gpu = getattr(config, "use_gpu", True)
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

        # AMP
        self.use_amp = bool(getattr(config, "use_amp", True)) and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # tensorboard
        log_dir = './logs/' + os.path.basename(os.getcwd())
        if os.path.exists(log_dir) and os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

        # model
        self.model = gaze_network().to(self.device)

        # multi-GPU (simple)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)

        print('[*] Number of model parameters: {:,}'.format(
            sum(p.numel() for p in self.model.parameters())
        ))

        # optimizer + scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=self.lr_patience, gamma=self.lr_decay_factor)

        # resume
        resume = getattr(config, "resume_ckpt", None)
        if resume:
            self.load_checkpoint(resume, is_strict=True)

    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def train(self):
        print(f"\n[*] Train on {self.num_train} samples")
        best_val = float("inf")

        for epoch in range(self.start_epoch, self.epochs):
            print(f"\nEpoch: {epoch+1}/{self.epochs} - lr: {self.current_lr():.6f}")

            self.model.train()
            train_err, train_loss = self.train_one_epoch(epoch, self.train_loader)

            # optional validation
            val_err = None
            if self.val_loader is not None:
                val_err = self.validate(epoch, self.val_loader)

            # save every epoch
            self.save_checkpoint({
                "epoch": epoch + 1,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "scheule_state": self.scheduler.state_dict(),
                "train_iter": self.train_iter,
            }, add=f"epoch_{epoch}")

            # save best (by val error if available else train error)
            score = val_err if val_err is not None else train_err
            if score is not None and score < best_val:
                best_val = score
                self.save_checkpoint({
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "scheule_state": self.scheduler.state_dict(),
                    "train_iter": self.train_iter,
                }, add="best")

            self.scheduler.step()

        self.writer.close()

    def train_one_epoch(self, epoch, data_loader):
        batch_time = AverageMeter()
        errors = AverageMeter()
        losses = AverageMeter()

        tic = time.time()

        for i, (input_img, target) in enumerate(data_loader):
            input_var = input_img.float().to(self.device, non_blocking=True)
            target_var = target.float().to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(enabled=self.use_amp):
                pred_gaze = self.model(input_var)
                loss = F.l1_loss(pred_gaze, target_var)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # metric on CPU numpy
            gaze_err = np.mean(angular_error(
                pred_gaze.detach().cpu().numpy(),
                target_var.detach().cpu().numpy()
            ))
            errors.update(float(gaze_err), input_var.size(0))
            losses.update(float(loss.item()), input_var.size(0))

            if i % self.print_freq == 0:
                self.writer.add_scalar("Loss/train", losses.avg, self.train_iter)
                self.writer.add_scalar("Error/train", errors.avg, self.train_iter)
                self.writer.add_scalar("LR", self.current_lr(), self.train_iter)

            if i % self.print_freq == 0 and i != 0:
                toc = time.time()
                batch_time.update(toc - tic)
                tic = time.time()

                est_mins = (self.epochs - epoch) * (self.num_train / self.batch_size) * batch_time.avg / 60.0
                print("--------------------------------------------------------------------")
                print(f"train error: {errors.avg:.3f} - loss_gaze: {losses.avg:.5f}")
                print(f"iter {self.train_iter} | batch_time_avg {batch_time.avg:.3f}s | ETA ~ {est_mins:.0f} mins")

                # reset per-print window meters if you want “windowed” stats
                errors.reset()
                losses.reset()

            self.train_iter += 1

        return errors.avg, losses.avg

    @torch.no_grad()
    def validate(self, epoch, data_loader):
        self.model.eval()
        errors = AverageMeter()

        for i, (input_img, target) in enumerate(data_loader):
            input_var = input_img.float().to(self.device, non_blocking=True)
            target_var = target.float().to(self.device, non_blocking=True)

            pred = self.model(input_var)

            gaze_err = np.mean(angular_error(
                pred.detach().cpu().numpy(),
                target_var.detach().cpu().numpy()
            ))
            errors.update(float(gaze_err), input_var.size(0))

        self.writer.add_scalar("Error/val", errors.avg, epoch)
        print(f"[VAL] angular error: {errors.avg:.3f}")
        self.model.train()
        return errors.avg

    @torch.no_grad()
    def test(self):
        print("We are now doing the final test")
        self.model.eval()

        if not self.pre_trained_model_path:
            raise ValueError("config.pre_trained_model_path is required for test()")

        self.load_checkpoint(self.pre_trained_model_path, is_strict=False)

        pred_gaze_all = np.zeros((self.num_test, 2), dtype=np.float32)
        save_index = 0

        print("Testing on", self.num_test, "samples")
        for i, input_img in enumerate(self.test_loader):
            input_var = input_img.float().to(self.device, non_blocking=True)
            pred = self.model(input_var).detach().cpu().numpy()

            bs = pred.shape[0]
            pred_gaze_all[save_index:save_index + bs, :] = pred
            save_index += bs

        if save_index != self.num_test:
            print("WARNING: saved", save_index, "!= num_test", self.num_test)

        np.savetxt("./temp/test_results.txt", pred_gaze_all, delimiter=",")

    def save_checkpoint(self, state, add=None):
        filename = f"{add}_ckpt.pth.tar" if add else "ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        print("Saved:", ckpt_path)

    def load_checkpoint(self, input_file_path, is_strict=True):
        print("Loading checkpoint:", input_file_path)
        ckpt = torch.load(input_file_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state"], strict=is_strict)
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.scheduler.load_state_dict(ckpt["scheule_state"])
        self.start_epoch = ckpt["epoch"] - 1
        self.train_iter = ckpt.get("train_iter", 0)

        print(f"[*] Loaded checkpoint @ epoch {ckpt['epoch']}")