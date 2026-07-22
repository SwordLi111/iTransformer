from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import torch
import torch.nn as nn
import numpy as np
import os
from scipy.stats import pearsonr


class Exp_Overlap_Pilot(Exp_Long_Term_Forecast):
    """
    Pilot 实验：相邻 sliding window 的 adversarial vulnerability
    是否比 clean loss 更强相关？

    继承 Exp_Long_Term_Forecast，直接复用 pgd_attack。
    只做诊断，不训练。
    """

    def __init__(self, args):
        super().__init__(args)

    def eval_pilot(self, setting, eta_ratio=0.1, alpha_ratio=0.1, num_iter=10,
                   lags=(1, 4, 16, 64), flag='val'):
        """
        在 val/test split 上计算 per-window clean/adv loss，
        比较 vulnerability 和 clean loss 的 lag correlation。

        必须保证 dataloader 是 shuffle=False，否则相邻 index 不再对应
        时间上相邻的 window。iTransformer 的 val/test loader 默认满足。

        参数:
            setting:     实验名 (用于加载 checkpoint)
            eta_ratio:   PGD 每元素扰动比例，默认 0.1 匹配训练配置
            alpha_ratio: PGD 步长比例
            num_iter:    PGD 迭代次数
            lags:        要检查的 window lag
            flag:        'val' 或 'test'
        """
        # ---- 加载 checkpoint ----
        ckpt = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(ckpt):
            print(f'Loading checkpoint: {ckpt}')
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        else:
            print(f'[WARN] No checkpoint at {ckpt}, using current weights')

        data_set, data_loader = self._get_data(flag=flag)
        criterion_none = nn.MSELoss(reduction='none')

        clean_losses, adv_losses = [], []
        self.model.eval()

        print(f'Running PGD pilot on {flag} split '
              f'(eta_ratio={eta_ratio}, num_iter={num_iter}, N_batches={len(data_loader)})')

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

            # ---- 生成对抗样本（pgd_attack 内部会临时切 eval，无副作用）----
            batch_x_adv = self.pgd_attack(
                batch_x.clone().detach(), batch_x_mark,
                batch_y.clone().detach(), batch_y_mark,
                eta_ratio=eta_ratio, alpha_ratio=alpha_ratio, num_iter=num_iter
            )

            # ---- 两次 forward: clean 和 adv ----
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                dim=1).float().to(self.device)

            with torch.no_grad():
                if self.args.output_attention:
                    out_clean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    out_adv = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    out_clean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    out_adv = self.model(batch_x_adv, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -self.args.c_out if self.args.features == 'MS' else 0
                out_clean = out_clean[:, -self.args.pred_len:, f_dim:]
                out_adv = out_adv[:, -self.args.pred_len:, f_dim:]
                y_true = batch_y[:, -self.args.pred_len:, f_dim:]

                # per-sample MSE: [B]
                mse_clean = criterion_none(out_clean, y_true).mean(dim=(1, 2))
                mse_adv = criterion_none(out_adv, y_true).mean(dim=(1, 2))

            clean_losses.append(mse_clean.cpu().numpy())
            adv_losses.append(mse_adv.cpu().numpy())

            if (i + 1) % 20 == 0:
                print(f'  batch {i+1}/{len(data_loader)}')

        clean_losses = np.concatenate(clean_losses)
        adv_losses = np.concatenate(adv_losses)
        v = adv_losses - clean_losses  # vulnerability gap
        c = clean_losses               # baseline

        # ---- Lag correlation ----
        print('\n' + '=' * 66)
        print(f'N windows = {len(v)}')
        print(f'clean_loss: mean={c.mean():.6f}  std={c.std():.6f}')
        print(f'adv_loss  : mean={adv_losses.mean():.6f}  std={adv_losses.std():.6f}')
        print(f'vuln_gap  : mean={v.mean():.6f}  std={v.std():.6f}')
        print('=' * 66)
        print(f'{"lag":>5} | {"vuln_corr":>10} | {"clean_corr":>10} | {"diff":>10}')
        print('-' * 66)

        correlations = {}
        for lag in lags:
            if lag >= len(v):
                continue
            r_v, _ = pearsonr(v[:-lag], v[lag:])
            r_c, _ = pearsonr(c[:-lag], c[lag:])
            correlations[lag] = {'vuln_corr': float(r_v),
                                 'clean_corr': float(r_c),
                                 'diff': float(r_v - r_c)}
            print(f'{lag:>5} | {r_v:>10.4f} | {r_c:>10.4f} | {r_v - r_c:>+10.4f}')

        print('=' * 66)
        print('Decision guide (diff = vuln_corr - clean_corr):')
        print('  diff ~ 0        -> 无 adversarial-specific 冗余，砍掉 idea')
        print('  diff > 0.1      -> 相邻窗口共享 adversarial 结构，值得跟进')
        print('  diff 随 lag 衰减 -> 效应集中在短 lag，方向正确')
        print('=' * 66)

        # ---- 保存 ----
        out_dir = os.path.join('./results', setting)
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, 'overlap_pilot.npz'),
                 clean_losses=clean_losses,
                 adv_losses=adv_losses,
                 vulnerabilities=v,
                 lags=np.array(list(correlations.keys())),
                 vuln_corrs=np.array([correlations[l]['vuln_corr'] for l in correlations]),
                 clean_corrs=np.array([correlations[l]['clean_corr'] for l in correlations]))
        print(f'Saved: {out_dir}/overlap_pilot.npz')

        return {
            'clean_losses': clean_losses,
            'adv_losses': adv_losses,
            'vulnerabilities': v,
            'correlations': correlations,
        }