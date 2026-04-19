class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FPNDecoder(nn.Module):
    def __init__(self, in_chs=(64, 128, 256, 512), out_ch=128):
        super().__init__()
        self.l1 = ConvBNAct(in_chs[0], out_ch, k=1, p=0)
        self.l2 = ConvBNAct(in_chs[1], out_ch, k=1, p=0)
        self.l3 = ConvBNAct(in_chs[2], out_ch, k=1, p=0)
        self.l4 = ConvBNAct(in_chs[3], out_ch, k=1, p=0)

        self.s1 = ConvBNAct(out_ch, out_ch, k=3, p=1)
        self.s2 = ConvBNAct(out_ch, out_ch, k=3, p=1)
        self.s3 = ConvBNAct(out_ch, out_ch, k=3, p=1)
        self.s4 = ConvBNAct(out_ch, out_ch, k=3, p=1)

    def forward(self, feats):
        f1, f2, f3, f4 = feats
        p4 = self.s4(self.l4(f4))
        p3 = self.s3(self.l3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.s2(self.l2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.s1(self.l1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False))
        return p1, {"p2": p2, "p3": p3, "p4": p4}


class D4SpectralPrior(nn.Module):
    def __init__(self, in_ch=512, proj_ch=128, tau=0.03, k_eig=8, gain=0.20, allow_eig_grad: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, proj_ch, kernel_size=1)
        self.prior_to_gate = nn.Sequential(
            ConvBNAct(1, 16, k=1, p=0),
            nn.Conv2d(16, in_ch, kernel_size=1),
        )
        self.tau = tau
        self.k_eig = k_eig
        self.gain = nn.Parameter(torch.tensor(float(gain)))
        self.gain_max = 0.30
        self.allow_eig_grad = allow_eig_grad

    def build_prior(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        n = h * w
        with torch.amp.autocast(device_type=feat.device.type, enabled=False):
            feat32 = torch.nan_to_num(feat.float(), nan=0.0, posinf=1.0, neginf=-1.0)
            x = self.proj(feat32).flatten(2).transpose(1, 2)
            x = F.normalize(x, p=2, dim=-1)
            sim = torch.bmm(x, x.transpose(1, 2)) / self.tau
            sim = torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=-40.0)
            sim = sim - sim.max(dim=-1, keepdim=True).values
            sim = sim.clamp(min=-40.0, max=0.0)
            aff = torch.exp(sim).clamp_min(1e-6)
            aff = torch.nan_to_num(aff, nan=1e-6, posinf=1.0, neginf=1e-6)
            eye = torch.eye(n, device=feat.device, dtype=aff.dtype).unsqueeze(0)
            aff = aff * (1.0 - eye)
            deg = aff.sum(dim=-1).clamp_min(1e-6)
            inv = deg.rsqrt()
            a_norm = inv.unsqueeze(-1) * aff * inv.unsqueeze(-2)
            lap = eye - a_norm + 1e-3 * eye
            lap = 0.5 * (lap + lap.transpose(1, 2))
            lap = torch.nan_to_num(lap, nan=0.0, posinf=1.0, neginf=-1.0)
            k_eff = max(1, min(self.k_eig, n - 1))

        mode_list = []
        lap_eig = lap if self.allow_eig_grad else lap.detach()
        eye_n = torch.eye(n, device=feat.device, dtype=lap.dtype)
        for bi in range(b):
            lbi = lap_eig[bi]
            eigvecs = None
            for jitter in (0.0, 1e-4, 1e-3, 1e-2):
                try:
                    l_try = lbi + jitter * eye_n
                    _, ev = torch.linalg.eigh(l_try.double())
                    eigvecs = ev.float()
                    if not torch.isfinite(eigvecs).all():
                        eigvecs = None
                    break
                except RuntimeError:
                    continue

            if eigvecs is None:
                mode_list.append(aff[bi].std(dim=-1).detach())
            else:
                evk = eigvecs[:, 1 : 1 + k_eff]
                mode_list.append(evk.abs().mean(dim=-1))

        modes = torch.stack(mode_list, dim=0)
        prior = modes.view(b, 1, h, w)
        prior = prior / (prior.flatten(1).amax(dim=1, keepdim=True).view(b, 1, 1, 1) + 1e-6)
        prior = torch.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return prior

    def forward(self, feat: torch.Tensor):
        prior = self.build_prior(feat)
        gate = torch.sigmoid(self.prior_to_gate(prior))
        gain = self.gain.clamp(0.0, self.gain_max)
        out = feat * (1.0 + gain * gate)
        return out, prior


class B1Net(nn.Module):
    # B1 = SMT backbone + D4 spectral prior + FPN decoder
    def __init__(self, tau=0.03, k_eig=8):
        super().__init__()
        self.backbone = smt_t()
        self.d4 = D4SpectralPrior(in_ch=512, proj_ch=128, tau=tau, k_eig=k_eig, gain=0.20)
        self.decoder = FPNDecoder()
        self.head = nn.Sequential(
            ConvBNAct(128, 64, k=3, p=1),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        # Boundary refinement head
        self.boundary_head = nn.Sequential(
            ConvBNAct(128, 64, k=3, p=1),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)
        f4, prior4 = self.d4(f4)
        p1, _ = self.decoder([f1, f2, f3, f4])

        logits_96 = self.head(p1)
        logits = F.interpolate(logits_96, size=x.shape[-2:], mode="bilinear", align_corners=False)

        boundary_96 = self.boundary_head(p1)
        boundary_logits = F.interpolate(boundary_96, size=x.shape[-2:], mode="bilinear", align_corners=False)

        spectral_prior = F.interpolate(prior4, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return {
            "logits": logits,
            "boundary_logits": boundary_logits,
            "spectral_prior": spectral_prior,
        }


def load_smt_pretrained(backbone: nn.Module, ckpt_path: str):
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"SMT checkpoint not found: {ckpt_path}")
    try:
        # PyTorch 2.6+ defaults to weights_only=True; keep explicit for clarity.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Trusted legacy checkpoints may contain objects (e.g., yacs CfgNode).
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    clean = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        if nk.startswith("backbone."):
            nk = nk[len("backbone."):]
        if nk.startswith("head."):
            continue
        clean[nk] = v
    missing, unexpected = backbone.load_state_dict(clean, strict=False)
    print(f"Loaded SMT weights: missing={len(missing)} unexpected={len(unexpected)}")
