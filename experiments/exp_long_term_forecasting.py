from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')



class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -self.args.c_out if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def vali_pgd(self, vali_data, vali_loader, criterion,
                 eta_ratio=0.5, alpha_ratio=0.1, num_iter=10):
        """在对抗样本上计算 validation loss"""
        total_loss = []
        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

            # 生成对抗样本
            batch_x_adv = self.pgd_attack(
                batch_x.clone().detach(), batch_x_mark,
                batch_y.clone().detach(), batch_y_mark,
                eta_ratio=eta_ratio, alpha_ratio=alpha_ratio, num_iter=num_iter
            )

            # 在对抗样本上预测
            with torch.no_grad():
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -self.args.c_out if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -self.args.c_out if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -self.args.c_out if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model



    # ============================================================================
    # PGD 攻击 — 动态 epsilon = eta_ratio * max|x|
    # ============================================================================

    def pgd_attack(self, batch_x, batch_x_mark, batch_y, batch_y_mark,
               eta_ratio=0.1, alpha_ratio=0.1, num_iter=10):
        """
        PGD 对抗攻击 — per-element epsilon

        每个元素的扰动上界: |δ[t,i]| ≤ eta_ratio * |x[t,i]|
        即扰动不超过当前数值的 ±eta_ratio (默认 10%)

        参数:
            eta_ratio:   扰动占 |x| 的比例，默认 0.1 (10%)
            alpha_ratio: 每步步长占 eps 的比例，默认 0.1
            num_iter:    PGD 迭代次数
        """
        batch_x_orig = batch_x.detach().clone()
        batch_x_adv = batch_x.detach().clone()

        # per-element epsilon: [B, T, C]
        eps = eta_ratio * batch_x_orig.abs().clamp(min=1e-8)
        alpha = alpha_ratio * eps

        self.model.eval()

        for _ in range(num_iter):
            x_adv_var = batch_x_adv.clone()
            x_adv_var.requires_grad_(True)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().detach()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                dim=1).float().to(self.device).detach()

            if self.args.output_attention:
                outputs = self.model(x_adv_var, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(x_adv_var, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -self.args.c_out if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].detach()

            loss = nn.MSELoss()(outputs, batch_y_target)
            loss.backward()

            with torch.no_grad():
                if x_adv_var.grad is not None:
                    grad_sign = x_adv_var.grad.sign()
                    batch_x_adv = batch_x_adv + alpha * grad_sign
                    perturbation = torch.clamp(batch_x_adv - batch_x_orig, -eps, eps)
                    batch_x_adv = batch_x_orig + perturbation

        return batch_x_adv.detach()



    def train_pgd(self, setting, eta_ratio=0.1, alpha_ratio=0.1, num_iter=10, adv_weight=0.5):
        """
        PGD对抗训练：同时在干净样本和对抗样本上训练

        参数:
            setting:     实验名称
            eta_ratio:   扰动上界占 max|x| 的比例
            alpha_ratio: 每步步长占 eps 的比例
            num_iter:    PGD 迭代次数
            adv_weight:  对抗损失的权重 (0~1)
                loss = (1 - adv_weight) * loss_clean + adv_weight * loss_adv
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # ========== 1. 生成PGD对抗样本 ==========
                batch_x_adv = self.pgd_attack(
                    batch_x.clone().detach(), batch_x_mark, batch_y.clone().detach(), batch_y_mark,
                    eta_ratio=eta_ratio, alpha_ratio=alpha_ratio, num_iter=num_iter
                )
                # pgd_attack会设model.eval()，需要恢复
                self.model.train()

                # ========== 2. 计算干净样本损失 ==========
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs_clean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs_clean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -self.args.c_out if self.args.features == 'MS' else 0
                        outputs_clean = outputs_clean[:, -self.args.pred_len:, f_dim:]
                        batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss_clean = criterion(outputs_clean, batch_y_target)

                        # ========== 3. 计算对抗样本损失 ==========
                        if self.args.output_attention:
                            outputs_adv = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs_adv = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)

                        outputs_adv = outputs_adv[:, -self.args.pred_len:, f_dim:]
                        loss_adv = criterion(outputs_adv, batch_y_target)

                        # ========== 4. 联合损失 ==========
                        loss = (1 - adv_weight) * loss_clean + adv_weight * loss_adv
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs_clean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs_clean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -self.args.c_out if self.args.features == 'MS' else 0
                    outputs_clean = outputs_clean[:, -self.args.pred_len:, f_dim:]
                    batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_clean = criterion(outputs_clean, batch_y_target)

                    # ========== 3. 计算对抗样本损失 ==========
                    if self.args.output_attention:
                        outputs_adv = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs_adv = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)

                    outputs_adv = outputs_adv[:, -self.args.pred_len:, f_dim:]
                    loss_adv = criterion(outputs_adv, batch_y_target)

                    # ========== 4. 联合损失 ==========
                    loss = (1 - adv_weight) * loss_clean + adv_weight * loss_adv
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} (clean: {3:.7f}, adv: {4:.7f})".format(
                        i + 1, epoch + 1, loss.item(), loss_clean.item(), loss_adv.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            test_loss = self.vali_pgd(test_data, test_loader, criterion,
                                      eta_ratio=eta_ratio, alpha_ratio=alpha_ratio, num_iter=num_iter)
            vali_loss = test_loss  # 在对抗样本上验证
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test_pgd(self, setting, test=0, eta_ratio=0.5, alpha_ratio=0.1, num_iter=10):
        """
        PGD 对抗攻击测试

        epsilon 按每个样本的 max|x| * eta_ratio 动态计算
        """
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('Loading model...')
            self.model.load_state_dict(torch.load(
                os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds_clean = []
        preds_pgd = []
        trues = []

        folder_path = './test_results/' + setting + '_pgd/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

            # ========== 干净预测 ==========
            if i % 50 == 0:
                print(f'Batch {i}/{len(test_loader)}: ', end='', flush=True)

            with torch.no_grad():
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -self.args.c_out if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs_clean = outputs.detach().cpu().numpy().copy()
                batch_y_np = batch_y_target.detach().cpu().numpy().copy()

                if test_data.scale and self.args.inverse:
                    shape = outputs_clean.shape
                    outputs_clean = test_data.inverse_transform(outputs_clean.squeeze(0)).reshape(shape)
                    batch_y_np = test_data.inverse_transform(batch_y_np.squeeze(0)).reshape(shape)

            pred_clean = outputs_clean.copy()

            # ========== PGD攻击 ==========
            batch_x_adv = self.pgd_attack(
                batch_x.clone().detach(), batch_x_mark, batch_y.clone().detach(), batch_y_mark,
                eta_ratio=eta_ratio, alpha_ratio=alpha_ratio, num_iter=num_iter
            )

            # 对抗样本预测
            with torch.no_grad():
                dec_inp_adv = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp_adv = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp_adv], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs_adv = self.model(batch_x_adv, batch_x_mark, dec_inp_adv, batch_y_mark)[0]
                        else:
                            outputs_adv = self.model(batch_x_adv, batch_x_mark, dec_inp_adv, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs_adv = self.model(batch_x_adv, batch_x_mark, dec_inp_adv, batch_y_mark)[0]
                    else:
                        outputs_adv = self.model(batch_x_adv, batch_x_mark, dec_inp_adv, batch_y_mark)

                outputs_adv = outputs_adv[:, -self.args.pred_len:, f_dim:]
                outputs_adv = outputs_adv.detach().cpu().numpy().copy()

                if test_data.scale and self.args.inverse:
                    shape = outputs_adv.shape
                    outputs_adv = test_data.inverse_transform(outputs_adv.squeeze(0)).reshape(shape)

            pred_pgd = outputs_adv.copy()
            true = batch_y_np.copy()

            preds_clean.append(pred_clean)
            preds_pgd.append(pred_pgd)
            trues.append(true)

            # 保存可视化
            if i % 20 == 0:
                with torch.no_grad():
                    input_np = batch_x.detach().cpu().numpy().copy()
                    if test_data.scale and self.args.inverse:
                        shape = input_np.shape
                        input_np = test_data.inverse_transform(input_np.squeeze(0)).reshape(shape)

                    gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                    pd_clean = np.concatenate((input_np[0, :, -1], pred_clean[0, :, -1]), axis=0)
                    pd_pgd = np.concatenate((input_np[0, :, -1], pred_pgd[0, :, -1]), axis=0)

                    visual(gt, pd_clean, os.path.join(folder_path, f'{i}_clean.pdf'))
                    visual(gt, pd_pgd, os.path.join(folder_path, f'{i}_pgd.pdf'))

        # 整理结果
        preds_clean = np.array(preds_clean).reshape(-1, preds_clean[0].shape[-2], preds_clean[0].shape[-1])
        preds_pgd = np.array(preds_pgd).reshape(-1, preds_pgd[0].shape[-2], preds_pgd[0].shape[-1])
        trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])

        # ========== 计算指标 ==========
        mae_c, mse_c, rmse_c, mape_c, mspe_c = metric(preds_clean, trues)
        mae_p, mse_p, rmse_p, mape_p, mspe_p = metric(preds_pgd, trues)

        # 打印结果
        print('\n' + '='*80)
        print('CLEAN PREDICTIONS:')
        print(f'  MAE={mae_c:.6f}, MSE={mse_c:.6f}, RMSE={rmse_c:.6f}')

        print(f'\nPGD ATTACK PREDICTIONS (eta_ratio={eta_ratio}, alpha_ratio={alpha_ratio}, iter={num_iter}):')
        print(f'  MAE={mae_p:.6f}, MSE={mse_p:.6f}, RMSE={rmse_p:.6f}')

        mae_deg = (mae_p - mae_c) / mae_c * 100
        mse_deg = (mse_p - mse_c) / mse_c * 100
        rmse_deg = (rmse_p - rmse_c) / rmse_c * 100

        print(f'\nPERFORMANCE DEGRADATION:')
        print(f'  MAE: {mae_deg:+.2f}% | MSE: {mse_deg:+.2f}% | RMSE: {rmse_deg:+.2f}%')
        print('='*80 + '\n')

        # 保存结果
        result_folder = './results/' + setting + '_pgd/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        np.save(result_folder + 'metrics_clean.npy',
                np.array([mae_c, mse_c, rmse_c, mape_c, mspe_c]))
        np.save(result_folder + 'metrics_pgd.npy',
                np.array([mae_p, mse_p, rmse_p, mape_p, mspe_p]))
        np.save(result_folder + 'pred_clean.npy', preds_clean)
        np.save(result_folder + 'pred_pgd.npy', preds_pgd)
        np.save(result_folder + 'true.npy', trues)

        # 保存到文本文件
        with open('result_pgd_attack.txt', 'a') as f:
            f.write(f'{setting} (eta_ratio={eta_ratio}, alpha_ratio={alpha_ratio}, iter={num_iter})\n')
            f.write(f'Clean: MAE={mae_c:.6f}, MSE={mse_c:.6f}\n')
            f.write(f'PGD:   MAE={mae_p:.6f}, MSE={mse_p:.6f}\n')
            f.write(f'Degrade: MAE {mae_deg:+.2f}%, MSE {mse_deg:+.2f}%\n\n')

        return


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -self.args.c_out if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def visualize_pgd_freq(self, setting, eta_ratio=0.5, alpha_ratio=0.1, num_iter=10,
                           n_windows=4, feature_idx=-1):
        """clean model 遭受 PGD 攻击前后的时域/频域对比图"""
        import matplotlib.pyplot as plt

        test_data, test_loader = self._get_data(flag='test')
        self.model.load_state_dict(torch.load(
            os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'),
            map_location=self.device))
        self.model.eval()

        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        batch_x_adv = self.pgd_attack(
            batch_x.clone().detach(), batch_x_mark,
            batch_y.clone().detach(), batch_y_mark,
            eta_ratio=eta_ratio, alpha_ratio=alpha_ratio, num_iter=num_iter)

        x_clean = batch_x.detach().cpu().numpy()
        x_adv = batch_x_adv.detach().cpu().numpy()
        delta = x_adv - x_clean
        seq_len = x_clean.shape[1]
        freqs = np.fft.rfftfreq(seq_len)

        # 3 列：时域叠加 | 频域叠加 | δ频谱
        fig, axes = plt.subplots(n_windows, 3, figsize=(15, 3 * n_windows))

        for k in range(n_windows):
            c = x_clean[k, :, feature_idx]
            a = x_adv[k, :, feature_idx]
            d = delta[k, :, feature_idx]

            # 时域叠加
            axes[k, 0].plot(c, color='royalblue', lw=1.2, label='clean')
            axes[k, 0].plot(a, color='crimson', lw=1.0, alpha=0.7, label='adv')
            axes[k, 0].set_ylabel(f'window {k}')
            if k == 0:
                axes[k, 0].set_title('Time domain')
                axes[k, 0].legend()

            # 频域叠加
            spec_c = np.abs(np.fft.rfft(c))[1:]
            spec_a = np.abs(np.fft.rfft(a))[1:]
            axes[k, 1].plot(freqs[1:], spec_c, color='royalblue', lw=1.2, label='clean')
            axes[k, 1].plot(freqs[1:], spec_a, color='crimson', lw=1.0, alpha=0.7, label='adv')
            axes[k, 1].set_yscale('log')
            if k == 0:
                axes[k, 1].set_title('Amplitude spectrum')
                axes[k, 1].legend()

            # δ 频谱
            spec_d = np.abs(np.fft.rfft(d))[1:]
            axes[k, 2].plot(freqs[1:], spec_d, color='darkgreen', lw=1.2)
            axes[k, 2].set_yscale('log')
            if k == 0:
                axes[k, 2].set_title('Perturbation δ spectrum')

        plt.tight_layout()
        out = f'./test_results/{setting}_pgd_freq_f{feature_idx}.pdf'
        plt.savefig(out, dpi=150)
        print(f'saved: {out}')

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return