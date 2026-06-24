import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ScoreNetwork(nn.Module):
    def __init__(self, x_dim, cond_dim, hidden_dim):
        super(ScoreNetwork, self).__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(x_dim),
            nn.Linear(x_dim, x_dim),
            nn.SiLU()
        )
        
        self.input_dim = x_dim + x_dim + x_dim 
        
        # ResNet-style MLP
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim) # Output: Score / Noise
        )

    def forward(self, x, t, condition):
        # x: (B, d)
        # t: (B,)
        # condition: (B, d)
        
        t_emb = self.time_mlp(t)
        inp = torch.cat([x, t_emb, condition], dim=-1)
        return self.net(inp)

class CPDiff(nn.Module):
    """
    Module 3: Conditional Preference Diffusion (CP-Diff)
    Strictly follows '扩散模型.md'
    """
    def __init__(self, config, static_dim, dynamic_dim):
        super(CPDiff, self).__init__()
        self.config = config
        self.d = config.embed_dim
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(static_dim + dynamic_dim, self.d),
            nn.Tanh() # Common activation for fusion
        )
        
        # --- 3. SDE Setup (VP-SDE) ---
        self.num_steps = config.diff_steps
        self.beta_min = config.beta_start
        self.beta_max = config.beta_end
        
        # Linear Beta Schedule
        betas = torch.linspace(self.beta_min, self.beta_max, self.num_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # --- 4. Score Network ---
        self.score_net = ScoreNetwork(self.d, self.d, config.hidden_dim)
        
    def get_loss(self, x_0, p_intrinsic, h_adaptive, time_emb):
        """
        Training Step (Doc Section 4.2)
        Objective: L_simple = || eps_theta - eps ||^2
        """
        B = x_0.size(0)
        
        # 1. Condition C
        # [UPDATED] Added time_emb to condition
        cond = self.cond_mlp(torch.cat([p_intrinsic, h_adaptive, time_emb], dim=-1))
        
        # 2. Sample Time t
        t = torch.randint(0, self.num_steps, (B,), device=x_0.device)
        
        # 3. Forward SDE (Add Noise)
        noise = torch.randn_like(x_0)
        alpha_bar = self.alphas_cumprod[t].view(B, 1)
        
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * eps
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        
        # 4. Predict Noise (Conditional Score)
        noise_pred = self.score_net(x_t, t, cond)
        
        # 5. MSE Loss
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    @torch.no_grad()
    def sample(self, p_intrinsic, h_adaptive, time_emb, verbose=False):
        """
        Inference Step (Doc Section 5.1)
        Executes T steps reverse sampling to generate x_0
        """
        B = p_intrinsic.size(0)
        device = p_intrinsic.device
        
        # 1. Condition
        # [UPDATED] Added time_emb
        cond = self.cond_mlp(torch.cat([p_intrinsic, h_adaptive, time_emb], dim=-1))
        
        # 2. Initial Noise x_T ~ N(0, I)
        x = torch.randn(B, self.d, device=device)
        
        # 3. Reverse Process (Euler-Maruyama / DDPM Ancestral)
        # Using standard DDPM scheduler which is equivalent to discretized VP-SDE
        for i in reversed(range(self.num_steps)):
            t = torch.tensor([i] * B, device=device)
            
            # Predict noise/score
            noise_pred = self.score_net(x, t, cond)
            
            # Step Parameters
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]
            
            if i > 0:
                z = torch.randn_like(x)
            else:
                z = 0
            
            # Update Formula
            # mu = 1/sqrt(alpha) * (x - beta/sqrt(1-alpha_bar) * eps)
            term1 = 1 / torch.sqrt(alpha)
            term2 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            
            mu = term1 * (x - term2 * noise_pred)
            sigma = torch.sqrt(beta)
            
            x = mu + sigma * z
            
        return x # This is hat_v_next
        
    def match_candidates(self, generated_vector, candidate_embs, temperature=1.0):
        """
        Step 5.2: Next POI Scoring
        Cosine Similarity Matching
        
        Inputs:
        - generated_vector: (B, d)
        - candidate_embs: (Num_Candidates, d) or (B, Num_Candidates, d)
        - temperature: float
        
        Returns:
        - probs: (B, Num_Candidates)
        """
        # Normalize Generated
        gen_norm = F.normalize(generated_vector, p=2, dim=-1) # (B, d)
        
        # Normalize Candidates
        cand_norm = F.normalize(candidate_embs, p=2, dim=-1) # (N, d)
        
        if cand_norm.dim() == 2:
            # Shared candidates for all batch (e.g. all OOT POIs)
            # (B, d) @ (d, N) -> (B, N)
            logits = torch.matmul(gen_norm, cand_norm.t())
        else:
            # Per-user candidates
            # (B, 1, d) @ (B, d, N) -> (B, 1, N)
            logits = torch.bmm(gen_norm.unsqueeze(1), cand_norm.transpose(1, 2)).squeeze(1)
            
        # Temperature
        logits = logits / temperature
        
        return F.softmax(logits, dim=-1)
