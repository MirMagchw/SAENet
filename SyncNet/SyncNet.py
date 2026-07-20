import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import Parameter
import torch.nn.init as init

class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        assert x.ndim == self.num_dims, print(
            "Expect x to have {} dimensions, but got {}".format(self.num_dims, x.ndim))

        mu_ = x.mean(dim=self.stat_dims, keepdim=True)  # [B,1,F,T]
        std_ = torch.sqrt(
            x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,F,T]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat

# =========================
# Depthwise Separable Conv
# =========================
class DSConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.pad_left = padding * 2
        self.pad_freq = padding

        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, 
                                   stride=stride, groups=in_ch)

        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = F.pad(x, (self.pad_left, 0, self.pad_freq, self.pad_freq))
        return self.pointwise(self.depthwise(x))

# =========================
# LiteResidual 2D CNN
# =========================
class ResidualBlock2D_Lite(nn.Module):
    def __init__(self, in_ch, out_ch, n_fft=512):
        super().__init__()

        d = n_fft // 4 + 1
        self.conv1 = DSConv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = DSConv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.n1 = CustomLayerNorm((1, d), stat_dims=(1,2), num_dims=4)
        self.n2 = CustomLayerNorm((1, d), stat_dims=(1,2), num_dims=4)
        self.prelu = nn.PReLU()

        self.down = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        out = self.prelu(self.n1(self.conv1(x)))
        out = self.prelu(self.n2(self.conv2(out)))
        return out + res

class DSConvTranspose2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=(2,2), padding=1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size,
            padding=padding, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def zero_insert(self, x):
        """
        Zero-insert upsampling
        """
        B, C, H, W = x.shape
        sH, sW = self.stride

        H_new = H * sH - 1
        W_new = W * sW - 1

        out = torch.zeros(B, C, H_new, W_new, device=x.device, dtype=x.dtype)

        out[:, :, ::sH, ::sW] = x

        return out

    def forward(self, x):
        x = self.zero_insert(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
# =========================
# DPRNN BLOCK 
# =========================
class DPRNNBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # time RNN
        self.intra_rnn = nn.GRU(dim, hidden_dim, batch_first=True, bidirectional=False)
        self.intra_linear = nn.Linear(hidden_dim, dim)
        self.intra_norm = nn.LayerNorm(dim)

        # freq RNN
        self.inter_rnn = nn.GRU(dim, hidden_dim, batch_first=True, bidirectional=True)
        self.inter_linear = nn.Linear(hidden_dim * 2, dim)
        self.inter_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, F, T]
        B, C, F, T = x.shape
        # ===== (time) =====
        x = x.permute(0,2,3,1)        # [B,F,T,C]
        x_res = x
        x = self.intra_norm(x)
        x = x.reshape(B*F, T, C)
        out, _ = self.intra_rnn(x)
        out = self.intra_linear(out)
        out = out.reshape(B, F, T, C)
        out = out + x_res

        # ===== (freq) =====
        out = out.permute(0,2,1,3)     # [B,T,F,C]
        x_res = out
        out = self.inter_norm(out)
        x = out.reshape(B*T, F, C)
        out2, _ = self.inter_rnn(x)
        out2 = self.inter_linear(out2)
        out2 = out2.reshape(B, T, F, C)
        out2 = out2 + x_res
        out = out2.permute(0,3,2,1)    # [B,C,F,T]

        return out

# =========================
# MAIN NETWORK
# =========================
class SyncNet(nn.Module):
    def __init__(self, n_fft=512, hop=256, hidden_dim=64, num_dprnn=2):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop

        # input: real1, imag1, real2, imag2, IPD
        in_ch = 5

        self.encoder = nn.Sequential(
            # Downsample
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            CustomLayerNorm((1, self.n_fft//4+1), stat_dims=(1,2), num_dims=4),
            nn.PReLU(),

            ResidualBlock2D_Lite(32, 64),
            ResidualBlock2D_Lite(64, hidden_dim),
        )

        self.dprnn = nn.Sequential(
            *[DPRNNBlock(hidden_dim, hidden_dim) for _ in range(num_dprnn)]
        )

        self.decoder = nn.Sequential(
            # Upsample
            DSConvTranspose2d(hidden_dim, 64, kernel_size=3, stride=(2,2), padding=1),
            CustomLayerNorm((1, self.n_fft//2+1), stat_dims=(1,2), num_dims=4),
            nn.PReLU(),

            DSConv2d(64, 32, kernel_size=3, padding=1),
            CustomLayerNorm((1, self.n_fft//2+1), stat_dims=(1,2), num_dims=4),
            nn.PReLU(),

            nn.Conv2d(32, 2, kernel_size=1)  # real + imag
        )

    def apply_stft(self, x):
        return torch.stft(
            x, self.n_fft, self.hop,
            onesided=True,
            window=torch.hann_window(self.n_fft).to(x.device),
            return_complex=True,
        ) 

    def apply_istft(self, x, length):
        return torch.istft(
            x, self.n_fft, self.hop,
            onesided=True,
            window=torch.hann_window(self.n_fft).to(x.device),
            length=length,
        ) 

    def build_input(self, mic1, mic2):
        X1 = self.apply_stft(mic1.squeeze(1)) # [B,F,T]
        X2 = self.apply_stft(mic2.squeeze(1)) # [B,F,T]

        real1, imag1 = X1.real, X1.imag
        real2, imag2 = X2.real, X2.imag

        # IPD
        ipd = torch.angle(X1) - torch.angle(X2)
        ipd = torch.atan2(torch.sin(ipd), torch.cos(ipd))
        inp = torch.stack([real1, imag1, real2, imag2, ipd], dim=1)

        return inp, X2

    def forward(self, mic1, mic2, tgt=None):
        if tgt is None:
            tgt = mic1
        B, _, T = mic1.shape
        hop = self.hop

        T_orig = (T + hop - 1) // hop

        if T_orig % 2 == 1:
            T_pad = T_orig + 1
            L_pad = T_pad * hop
            pad_len = L_pad - T
            mic1_pad = F.pad(mic1, (0, pad_len))
            mic2_pad = F.pad(mic2, (0, pad_len))
            x, _ = self.build_input(mic1_pad, mic2_pad)  
        else:
            x, _ = self.build_input(mic1, mic2)          

        # Forward Pass
        enc = self.encoder(x)      
        feat = self.dprnn(enc)     
        feat = feat + enc          
        out = self.decoder(feat)   

        if T_orig % 2 == 1:
            out = out[..., :T_orig]       

        Y_real = out[:,0]
        Y_imag = out[:,1]
        est_spec = torch.complex(Y_real, Y_imag) 
        est = self.apply_istft(est_spec, length=T) 

        tgt_spec = self.apply_stft(tgt.squeeze(1))
        stft_len = min(est_spec.shape[2], tgt_spec.shape[2])
        tgt_mag = torch.abs(tgt_spec)
        est_mag = torch.abs(est_spec)
        return {
            "est": est.unsqueeze(1),  
            "tgt": tgt, 
            "est_spec": est_spec[:,:,:stft_len], 
            "tgt_spec": tgt_spec[:,:,:stft_len],
            "est_mag": est_mag[:,:,:stft_len],
            "tgt_mag": tgt_mag[:,:,:stft_len],
        }

if __name__ == '__main__':

    print("=== SyncNet Test ===\n")

    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    T = 16000  
    batch_size = 1
    
    model = SyncNet().to(device)

    # input tensors
    mic1 = torch.randn(batch_size, 1, T).to(device)
    mic2 = torch.randn(batch_size, 1, T).to(device)

    # forward pass
    with torch.no_grad():
        output = model(mic1, mic2)
    print(f"\nInput shape: mic1 {mic1.shape}, mic2 {mic2.shape}")
    print(f"Output shape: {output['est'].shape}")


