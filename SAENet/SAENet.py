import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import init
from torch.nn.parameter import Parameter

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

        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_ch)

        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# =========================
# Residual 2D CNN
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

# -------------------------
# Enhancement Network for Multi-Mic Fusion
# -------------------------
class AttentionBlock(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_heads=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(emb_dim)
        self.norm_kv = nn.LayerNorm(emb_dim)
        self.Wq = nn.Linear(emb_dim, hidden_dim)
        self.Wk = nn.Linear(emb_dim, hidden_dim)
        self.Wv = nn.Linear(emb_dim, hidden_dim)
        self.Wo = nn.Linear(hidden_dim, emb_dim)
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        # x: [B, M, C, F, T]
        B, M, C, F, T = x.size()

        x = x.permute(0, 3, 4, 1, 2) # [B,F,T,M,C]
        
        x_ref = x[..., :1, :] # [B,F,T,1,C]

        q = self.Wq(self.norm_q(x_ref)).reshape(B, F, T, 1, self.n_heads, self.head_dim).transpose(-2, -3)  # (b,f,t,h,1,c/h)
        x = self.norm_kv(x)
        k = self.Wk(x).reshape(B, F, T, M, self.n_heads, self.head_dim).transpose(-2, -3)  # (b,f,t,h,m,c/h)
        v = self.Wv(x).reshape(B, F, T, M, self.n_heads, self.head_dim).transpose(-2, -3)  # (b,f,t,h,m,c/h)

        attn = torch.matmul(q, k.transpose(-2, -1)) # (b,f,t,h,1,m)
        attn = torch.mul(attn, self.scale)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (b,f,t,h,1,c/h)

        out = out.transpose(-2, -3).flatten(-2, -1) # (b,f,t,1,c)
        out = self.Wo(out)  # (b,f,t,1,c)
        
        attn_out = out + x_ref # [B,F,T,1,C]
        # print(f"Attention output shape: {attn_out.shape}") 
        return attn_out.permute(0, 3, 4, 1, 2) # [B, 1, C, F, T]

class SEDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
  
        self.decoder = nn.Sequential(
            ResidualBlock2D_Lite(hidden_dim, 64),
            ResidualBlock2D_Lite(64, 64),
            DSConvTranspose2d(64, 64, kernel_size=3, stride=(2,2), padding=1),
            CustomLayerNorm((1, 257), stat_dims=(1,2), num_dims=4),
            nn.PReLU(),
            DSConv2d(64, 32),
            CustomLayerNorm((1, 257), stat_dims=(1,2), num_dims=4),
            nn.PReLU(),
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, 1, D, F, T]
        x = x.squeeze(1) # [B, D, F, T]
        out = self.decoder(x) # [B, 2, F, T]
        return out

class SyncFrontend(nn.Module):
    def __init__(self, pretrained_path=None, device='cuda', freeze=True, **kwargs):
        super().__init__()

        self.model = SyncNet(**kwargs)

        if pretrained_path:
            print(f"Loading SyncNet frontend from {pretrained_path}...")
            state_dict = torch.load(pretrained_path, map_location=device)
            model_state = self.model.state_dict()

            for k, v in state_dict.items():
                name = k.replace("module.", "") if "module." in k else k

                if "G." in name: 
                    name = name.replace("G.", "")
                    if name not in model_state:
                        print(f"Skipping {name}, not found in model.")
                        continue
                    if model_state[name].size() != v.size():
                        print(f"Size mismatch for {name}, model: {model_state[name].size()}, loaded: {v.size()}")
                        continue
                    model_state[name].copy_(v)
                    print(f"Loading weight for: {name}")

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        else:
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.train()

    def forward(self, ref_mic, target_mic):
        """
        ref_mic: [B, 1, T] 
        target_mic: [B, 1, T] 
        return: [B, bib*hidden_dim, T'']
        """

        B, _, T = ref_mic.shape
        hop = 256

        T_orig = (T + hop - 1) // hop

        if T_orig % 2 == 1:
            T_pad = T_orig + 1
            L_pad = T_pad * hop
            pad_len = L_pad - T
            ref_mic_pad = F.pad(ref_mic, (0, pad_len))
            target_mic_pad = F.pad(target_mic, (0, pad_len))
            x, _ = self.model.build_input(ref_mic_pad, target_mic_pad)  
        else:
            x, _ = self.model.build_input(ref_mic, target_mic)    
        enc = self.model.encoder(x)                  # [B, 64, F', T']
        feat = self.model.dprnn(enc)             # [B, 64, F', T']
        
        feat = feat + enc
        
        return feat, T_orig

class JointMultiMicSE(nn.Module):
    def __init__(self, sync_model_path, device='cuda:0', freeze_frontend=True):
        super().__init__()
        self.device = device
        
        self.frontend = SyncFrontend(
            pretrained_path=sync_model_path, 
            device=device, 
            freeze=freeze_frontend,
            hidden_dim=64, num_dprnn=2
        )
        
        frontend_out_dim = 64
        
        self.fusion = AttentionBlock(emb_dim=frontend_out_dim, hidden_dim=frontend_out_dim, n_heads=4)
        
        self.backend = SEDecoder(in_channels=frontend_out_dim, hidden_dim=64)

    def forward(self, mics, tgt=None):
        """
        mics: [B, M, T]
        """
        if tgt is None:
            tgt = mics[:,0:1]

        B, M, T = mics.shape
        
        ref_mic = mics[:, 0:1, :]
        ref_mic_expanded = ref_mic.expand(-1, M, -1) 
        
        target_mics = mics
        
        ref_flat = ref_mic_expanded.reshape(B * M, 1, T)
        target_flat = target_mics.reshape(B * M, 1, T)
        
        sync_features_flat, T_orig = self.frontend(ref_flat, target_flat) # output: [B*M, C, F', T']
        
        _, C, F_down, T_down = sync_features_flat.shape

        sync_features = sync_features_flat.view(B, M, C, F_down, T_down) # [B, M, C, F', T']

        fused_feature = self.fusion(sync_features).squeeze(1) # [B, C, F', T']
        
        out = self.backend(fused_feature) # [B, 2, F, T]

        if T_orig % 2 == 1:
            out = out[..., :T_orig]       
        Y_real = out[:,0]
        Y_imag = out[:,1]
        est_spec = torch.complex(Y_real, Y_imag) 
        est = self.frontend.model.apply_istft(est_spec, length=T) 

        tgt_spec = self.frontend.model.apply_stft(tgt.squeeze(1))
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
    print("=== SAENet Test ===\n")

    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    T = 16000   
    batch_size = 8

    model = JointMultiMicSE(sync_model_path=None, device=device, freeze_frontend=False).to(device)

    # input: [B, M, T]
    mics = torch.randn(batch_size, 16, T).to(device)
    tgt = torch.randn(batch_size, 1, T).to(device)

    # forward pass
    with torch.no_grad():
        output = model(mics)
    print(f"\nInput shape: mics {mics.shape}")
    print(f"Output shape: {output['est'].shape}")
