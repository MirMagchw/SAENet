import torch, math, torchaudio
import torch.nn as nn
import torch.nn.functional as F

class LossMeter:
    def __init__(self):
        self.losses = {}
        self.count = 0
        
    def reset(self):
        self.losses = {}
        self.count = 0
        
    def add(self, loss_dict):
        for name, value in loss_dict.items():
            if name not in self.losses:
                self.losses[name] = {'sum': 0, 'count': 0}
            self.losses[name]['sum'] += value.item() if hasattr(value, 'item') else value
            self.losses[name]['count'] += 1
        self.count += 1
        
    def value(self):
        result = {}
        for name, data in self.losses.items():
            result[name] = data['sum'] / data['count']
        return result

class CoherenceLoss(nn.Module):
    def __init__(self, n_fft=8192, hop=4096, device="cuda"):
        super().__init__()
        self.device = device
        self.n_fft = n_fft
        self.hop = hop

    def apply_stft(self, x, n_fft, hop_length, return_complex=True):
        # x:(B,1,T)
        spec = torch.stft(
            x.squeeze(1),
            n_fft,
            hop_length,
            window=torch.hann_window(n_fft).to(x.device),
            onesided=True,
            return_complex=return_complex,
        )
        return spec
        
    def coherence(self, x, y, n_fft, hop_length):
        X = self.apply_stft(x, n_fft=n_fft, hop_length=hop_length)
        Y = self.apply_stft(y, n_fft=n_fft, hop_length=hop_length)

        num = torch.sum(X * torch.conj(Y), dim=-1)              # ∑ X Y*
        den = torch.sqrt(torch.sum(torch.abs(X)**2, dim=-1)) * \
              torch.sqrt(torch.sum(torch.abs(Y)**2, dim=-1))

        coh = torch.abs(num) / (den + 1e-8)                     # coherence
        return 1 - coh.mean()                                   # 1 - coherence

    def calc_sdr_torch(self, estimation, origin):
        """
        batch-wise SDR caculation for one audio file on pytorch Variables.
        estimation: (batch, nsample)
        origin: (batch, nsample)
        mask: an optional mask for sequence masking. This is for cases where zero-padding was applied at the end and should not be consider for SDR calculation.
        """
        if estimation.dim() == 3:
            if estimation.size(1) == 1:
                estimation = estimation.squeeze(1)
                origin = origin.squeeze(1)
        def calculate(estimation, origin):
            origin_power = torch.pow(origin, 2).sum(1, keepdim=True) + 1e-8  # (batch, 1)
            scale = torch.sum(origin*estimation, 1, keepdim=True) / origin_power  # (batch, 1)

            est_true = scale * origin  # (batch, nsample)
            est_res = estimation - est_true  # (batch, nsample)

            true_power = torch.pow(est_true, 2).sum(1) + 1e-8
            res_power = torch.pow(est_res, 2).sum(1) + 1e-8

            return 10*torch.log10(true_power) - 10*torch.log10(res_power)  # (batch, )
        
        best_sdr = calculate(estimation, origin)
        
        return -best_sdr.mean()
    
    def forward(self, G_results):
        l1 = F.l1_loss(G_results['est'], G_results['tgt'])
        coh1 = self.coherence(G_results['est'], G_results['tgt'], n_fft=self.n_fft, hop_length=self.hop)
        coh2 = self.coherence(G_results['est'], G_results['tgt'], n_fft=512, hop_length=256)
        return {
            'l1':l1,
            'coh1':coh1,
            'coh2':coh2,
        }


