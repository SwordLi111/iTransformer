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


class Exp_Variant_Weight_Forecast(Exp_Basic):
    """
    Vulnerability-Weighted Adversarial Training

    每个 window 都参与 adversarial training，但根据 vulnerability 分配不同的 loss weight。

    流程：
    1. 对 batch 中每个 window 计算 MSE_clean 和 MSE_adv
    2. v_i = max(0, MSE_adv_i - MSE_clean_i)
    3. 标准化: z_i = (v_i - μ_v) / (σ_v + ε)
    4. Soft weight: w_i = exp(α * z_i) / mean(exp(α * z_j))
    5. Loss = (1/B) * Σ w_i * MSE_adv_i
    """

    def __init__(self, args):
        super(Exp_Variant_Weight_Forecast, self).__init__(args)

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

    # =====================================================================
    # FGSM 攻击
    # =====================================================================
    def fgsm_attack(self, batch_x, batch_x_mark, batch_y, batch_y_mark, epsilon=8/255):
        batch_x_adv = batch_x.clone().detach().requires_grad_(True)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().detach()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device).detach()

        if self.args.output_attention:
            outputs = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].detach()

        loss = nn.MSELoss()(outputs, batch_y_target)
        loss.backward()

        with torch.no_grad():
            grad_sign = batch_x_adv.grad.sign()
            batch_x_adv = batch_x_adv + epsilon * grad_sign
            batch_x_adv = torch.clamp(batch_x_adv, min=0)

        return batch_x_adv.detach()

    # =====================================================================
    # PGD 攻击
    # =====================================================================
    def pgd_attack(self, batch_x, batch_x_mark, batch_y, batch_y_mark,
                   epsilon=8/255, alpha=2/255, num_iter=10):
        batch_x_orig = batch_x.detach().clone()
        batch_x_adv = batch_x.detach().clone()

        self.model.eval()

        for _ in range(num_iter):
            x_adv_var = batch_x_adv.clone().requires_grad_(True)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().detach()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device).detach()

            if self.args.output_attention:
                outputs = self.model(x_adv_var, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(x_adv_var, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].detach()

            loss = nn.MSELoss()(outputs, batch_y_target)
            loss.backward()

            with torch.no_grad():
                grad_sign = x_adv_var.grad.sign()
                batch_x_adv = batch_x_adv + alpha * grad_sign
                perturbation = torch.clamp(batch_x_adv - batch_x_orig, -epsilon, epsilon)
                batch_x_adv = batch_x_orig + perturbation
                batch_x_adv = torch.clamp(batch_x_adv, min=0)

        return batch_x_adv.detach()

    # =====================================================================
    # 训练
    # =====================================================================
    def train_pgd(self, setting, epsilon=8/255, alpha=2/255, num_iter=10, weight_alpha=1.0):
        """
        Vulnerability-Weighted Adversarial Training

        参数:
            setting: 实验名称
            epsilon: PGD 扰动幅度
            alpha: PGD 步长
            num_iter: PGD 迭代次数
            weight_alpha: 控制权重平滑程度的参数 α
                α 小 → 权重更平滑（接近均匀）
                α 大 → 更强调 vulnerable windows
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
        criterion_mean = nn.MSELoss()           # 用于 vali
        criterion_none = nn.MSELoss(reduction='none')  # 用于逐样本计算

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

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0

                # ========== 1. Clean MSE (per sample, no grad) ==========
                with torch.no_grad():
                    if self.args.output_attention:
                        out_clean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        out_clean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    out_clean = out_clean[:, -self.args.pred_len:, f_dim:]
                    batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:]
                    # [batch_size]
                    mse_clean = criterion_none(out_clean, batch_y_target).mean(dim=(1, 2))

                # ========== 2. pgd 生成对抗样本 ==========
                batch_x_adv = self.pgd_attack(
                        batch_x.clone().detach(), batch_x_mark,
                        batch_y.clone().detach(), batch_y_mark,
                        epsilon=epsilon, alpha=alpha, num_iter=num_iter
                    )
                self.model.train()

                # ========== 3. Adversarial forward (需要梯度，用于训练) ==========
                if self.args.output_attention:
                    out_adv = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    out_adv = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)
                out_adv = out_adv[:, -self.args.pred_len:, f_dim:]

                # 逐样本 MSE: [batch_size]
                mse_adv_per_sample = criterion_none(out_adv, batch_y_target).mean(dim=(1, 2))

                # ========== 4. 计算 vulnerability 和 soft weight ==========
                with torch.no_grad():
                    # v_i = max(0, MSE_adv - MSE_clean)
                    v = torch.clamp(mse_adv_per_sample.detach() - mse_clean, min=0)

                    # 标准化: z_i = (v_i - μ) / (σ + ε)
                    v_mean = v.mean()
                    v_std = v.std() + 1e-8
                    z = (v - v_mean) / v_std

                    # soft weight: w_i = exp(α * z_i)
                    w = torch.exp(weight_alpha * z)

                    # 归一化: 使 mean(w) ≈ 1
                    w = w / (w.mean() + 1e-8)

                # ========== 5. Weighted loss ==========
                # L = (1/B) * Σ w_i * MSE_adv_i
                loss = (w * mse_adv_per_sample).mean()
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | "
                          "w_max: {3:.2f}, w_min: {4:.2f}, w_mean: {5:.2f}".format(
                        i + 1, epoch + 1, loss.item(),
                        w.max().item(), w.min().item(), w.mean().item()))
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

            print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion_mean)
            test_loss = self.vali(test_data, test_loader, criterion_mean)

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

    # =====================================================================
    # Validation
    # =====================================================================
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

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # =====================================================================
    # Test（干净测试）
    # =====================================================================
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

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

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                preds.append(outputs)
                trues.append(batch_y)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

    # =====================================================================
    # Test PGD（对抗测试）
    # =====================================================================
    def test_pgd(self, setting, test=0, epsilon=8/255, alpha=2/255, num_iter=10):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('Loading model...')
            self.model.load_state_dict(torch.load(
                os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

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

            if i % 50 == 0:
                print(f'Batch {i}/{len(test_loader)}', flush=True)

            f_dim = -1 if self.args.features == 'MS' else 0

            with torch.no_grad():
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:]

                outputs_clean = outputs.detach().cpu().numpy().copy()
                batch_y_np = batch_y_target.detach().cpu().numpy().copy()

                if test_data.scale and self.args.inverse:
                    shape = outputs_clean.shape
                    outputs_clean = test_data.inverse_transform(outputs_clean.squeeze(0)).reshape(shape)
                    batch_y_np = test_data.inverse_transform(batch_y_np.squeeze(0)).reshape(shape)

            batch_x_adv = self.pgd_attack(
                batch_x.clone().detach(), batch_x_mark,
                batch_y.clone().detach(), batch_y_mark,
                epsilon=epsilon, alpha=alpha, num_iter=num_iter
            )

            with torch.no_grad():
                dec_inp_adv = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp_adv = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp_adv], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs_adv = self.model(batch_x_adv, batch_x_mark, dec_inp_adv, batch_y_mark)[0]
                else:
                    outputs_adv = self.model(batch_x_adv, batch_x_mark, dec_inp_adv, batch_y_mark)

                outputs_adv = outputs_adv[:, -self.args.pred_len:, f_dim:]
                outputs_adv = outputs_adv.detach().cpu().numpy().copy()

                if test_data.scale and self.args.inverse:
                    shape = outputs_adv.shape
                    outputs_adv = test_data.inverse_transform(outputs_adv.squeeze(0)).reshape(shape)

            preds_clean.append(outputs_clean.copy())
            preds_pgd.append(outputs_adv.copy())
            trues.append(batch_y_np.copy())

            if i % 20 == 0:
                with torch.no_grad():
                    input_np = batch_x.detach().cpu().numpy().copy()
                    if test_data.scale and self.args.inverse:
                        shape = input_np.shape
                        input_np = test_data.inverse_transform(input_np.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input_np[0, :, -1], batch_y_np[0, :, -1]), axis=0)
                    pd_c = np.concatenate((input_np[0, :, -1], outputs_clean[0, :, -1]), axis=0)
                    pd_a = np.concatenate((input_np[0, :, -1], outputs_adv[0, :, -1]), axis=0)
                    visual(gt, pd_c, os.path.join(folder_path, f'{i}_clean.pdf'))
                    visual(gt, pd_a, os.path.join(folder_path, f'{i}_pgd.pdf'))

        preds_clean = np.array(preds_clean).reshape(-1, preds_clean[0].shape[-2], preds_clean[0].shape[-1])
        preds_pgd = np.array(preds_pgd).reshape(-1, preds_pgd[0].shape[-2], preds_pgd[0].shape[-1])
        trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])

        mae_c, mse_c, rmse_c, mape_c, mspe_c = metric(preds_clean, trues)
        mae_p, mse_p, rmse_p, mape_p, mspe_p = metric(preds_pgd, trues)

        print('\n' + '=' * 80)
        print('CLEAN:')
        print(f'  MAE={mae_c:.6f}, MSE={mse_c:.6f}, RMSE={rmse_c:.6f}')
        print(f'\nPGD (ε={epsilon}, α={alpha}, iter={num_iter}):')
        print(f'  MAE={mae_p:.6f}, MSE={mse_p:.6f}, RMSE={rmse_p:.6f}')
        print(f'\nDEGRADATION:')
        print(f'  MAE: {(mae_p-mae_c)/mae_c*100:+.2f}% | MSE: {(mse_p-mse_c)/mse_c*100:+.2f}%')
        print('=' * 80 + '\n')

        result_folder = './results/' + setting + '_pgd/'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        np.save(result_folder + 'metrics_clean.npy', np.array([mae_c, mse_c, rmse_c, mape_c, mspe_c]))
        np.save(result_folder + 'metrics_pgd.npy', np.array([mae_p, mse_p, rmse_p, mape_p, mspe_p]))
        np.save(result_folder + 'pred_clean.npy', preds_clean)
        np.save(result_folder + 'pred_pgd.npy', preds_pgd)
        np.save(result_folder + 'true.npy', trues)

        with open('result_pgd_attack.txt', 'a') as f:
            f.write(f'{setting} (ε={epsilon}, α={alpha}, iter={num_iter})\n')
            f.write(f'Clean: MAE={mae_c:.6f}, MSE={mse_c:.6f}\n')
            f.write(f'PGD:   MAE={mae_p:.6f}, MSE={mse_p:.6f}\n')
            f.write(f'Degrade: MAE {(mae_p-mae_c)/mae_c*100:+.2f}%, MSE {(mse_p-mse_c)/mse_c*100:+.2f}%\n\n')

        return

    # =====================================================================
    # Predict
    # =====================================================================
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)
        return