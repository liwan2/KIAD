import torch
import torch.nn as nn
import torch.nn.functional as F

class STAPM(nn.Module):
    def __init__(self, config):
        super(STAPM, self).__init__()
        self.config = config
        self.d = config.embed_dim
        self.k_stiefel = config.stiefel_k
        self.num_cats = getattr(config, 'num_cats', 0)
        
        # --- 1. Feature Encoding ---
        # 1.1 Time Encoding (Step 1.1)
        # Input: Delta T (1) + Period Emb (3) = 4
        self.W_t = nn.Linear(4, self.d)
        self.b_t = nn.Parameter(torch.zeros(self.d))
        
        # 1.2 Space Encoding (Step 1.2)
        # Input: POI Cats (C) + Dist (1) = C + 1
        # If num_cats is 0 (missing config), we rely on robust forward handling
        self.input_dim_s = self.num_cats + 1
        self.W_s = nn.Linear(self.input_dim_s, self.d)
        self.b_s = nn.Parameter(torch.zeros(self.d))

        # --- 2. Bi-GRU (Step 2) ---
        # Input: v_emb (d) + t_feat (d) + s_feat (d) = 3d
        self.gru = nn.GRU(input_size=3 * self.d, 
                          hidden_size=self.d, 
                          num_layers=config.rnn_layers, 
                          batch_first=True, 
                          bidirectional=True)
        # Output: 2d (Bidirectional)

        # --- 3. Dynamic Graph (Module 2 Legacy / Stiefel) ---
        self.W_E = nn.Linear(2 * self.d, self.k_stiefel)
        self.G_kernel = nn.Parameter(torch.randn(self.k_stiefel, 2 * self.d))

        # --- 4. Gated Attention (Step 3 in Doc) ---
        # Document Step 3.1: W_a \cdot concat(gru, s)
        self.W_a = nn.Linear(2 * self.d + self.d, 2 * self.d) 
        self.b_a = nn.Parameter(torch.zeros(2 * self.d))
        
        # Doc Step 3.2: pad(s_feat) -> Done in forward

        # --- 5. Evolution (Step 4 in Doc) ---
        # Time Aware Gate
        # Input: Delta T (1) -> 2d
        self.W_g = nn.Linear(1, 2 * self.d)
        self.b_g = nn.Parameter(torch.zeros(2 * self.d))

    def _haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate Haversine distance between two points (lat, lon) in Scale of Earth Radius.
        Input: Degrees (Tensor)
        Output: Distance in km (Tensor)
        """
        R = 6371.0
        # Convert to radians
        lat1_rad = torch.deg2rad(lat1)
        lon1_rad = torch.deg2rad(lon1)
        lat2_rad = torch.deg2rad(lat2)
        lon2_rad = torch.deg2rad(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = torch.sin(dlat / 2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2)**2
        # Clamp a to [0, 1] to avoid nan in sqrt
        a = torch.clamp(a, 0.0, 1.0)
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        
        return R * c

    def forward(self, v_emb, timestamps, coords, poi_cats=None, local_hours=None):
        """
        Strict implementation of 自适应偏好 (Adaptive Preference)建模.md
        
        Inputs:
        - v_emb: (B, M, d)
        - timestamps: (B, M) - Unix Seconds (preferred)
        - coords: (B, M, 2) - [Lat, Lon]
        - poi_cats: (B, M, C) - OneHot Category (Optional, default 0 if missing)
        - local_hours: (B, M) - Local Hour of Day (0-23). If provided, used for Period.
        
        Returns:
        - H_refined: (B, M, 2d)
        - P_adaptive_seq: (B, M, 2d)
        """
        B, M, _ = v_emb.shape
        device = v_emb.device
        
        # --- Step 1.1: Time Feature Encoding ---
        # 1.1.1 Delta T
        # Assume timestamps are meaningful for delta
        delta_t = torch.zeros(B, M, 1, device=device)
        dt_vals = torch.zeros_like(timestamps)
        dt_vals[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
        dt_vals[:, 0] = 0
        # dt_vals is (B, M). delta_t[:,:,0] expects (B, M).
        delta_t[:, :, 0] = dt_vals
        
        # 1.1.2 Period Encoding (Early/Mid/Late)
        if local_hours is not None:
            # Use pre-computed local hours (Timezone Corrected)
            hour = local_hours.long()
            hour = torch.clamp(hour, 0, 23)
        else:
            # Fallback to Raw Timestamps (Legacy)
            # heuristic mod
            ts_mod = torch.fmod(timestamps, 86400.0)
            hour = torch.floor(ts_mod / 3600.0).long() # (B, M)
            hour = torch.clamp(hour, 0, 23)
        
        # 0=[0-7], 1=[8-16], 2=[17-23]
        period_idx = torch.zeros_like(hour)
        mask_mid = (hour >= 8) & (hour <= 16)
        mask_late = (hour >= 17)
        period_idx[mask_mid] = 1
        period_idx[mask_late] = 2
        
        period_emb = F.one_hot(period_idx, num_classes=3).float() # (B, M, 3)
        
        # 1.1.3 Fusion
        time_input = torch.cat([delta_t, period_emb], dim=-1) # (B, M, 4)
        t_feat = F.relu(self.W_t(time_input) + self.b_t) # (B, M, d)
        
        # --- Step 1.2: Space Feature Encoding ---
        # 1.2.1 Haversine Distance
        delta_dist = torch.zeros(B, M, 1, device=device)
        # Lat/Lon separation
        lat = coords[:, :, 0]
        lon = coords[:, :, 1]
        
        # dist[i] = dist(loc[i-1], loc[i])
        if M > 1:
            d_val = self._haversine(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
            delta_dist[:, 1:, 0] = d_val
        
        # 1.2.2 Fusion
        # Concat(poi_cats, dist)
        if poi_cats is None:
            # Fallback for missing data
            if self.num_cats > 0:
                poi_cats = torch.zeros(B, M, self.num_cats, device=device)
            else:
                poi_cats = torch.zeros(B, M, 0, device=device)
                
        space_input = torch.cat([poi_cats, delta_dist], dim=-1) # (B, M, C+1)
        
        # Dimension check
        if space_input.shape[-1] != self.input_dim_s:
            # If shape mismatch (e.g. config says 250, input is 0), we pad or slice
            curr_dim = space_input.shape[-1]
            if curr_dim < self.input_dim_s:
                pad = torch.zeros(B, M, self.input_dim_s - curr_dim, device=device)
                space_input = torch.cat([space_input, pad], dim=-1)
            else:
                space_input = space_input[:, :, :self.input_dim_s]
                
        s_feat = F.relu(self.W_s(space_input) + self.b_s) # (B, M, d)
        
        # --- Step 1.3: Total Input ---
        X = torch.cat([v_emb, t_feat, s_feat], dim=-1) # (B, M, 3d)
        
        # --- Step 2: Bi-GRU ---
        H_gru, _ = self.gru(X) # (B, M, 2d)
        
        # --- Step 3: Stiefel Dynamic Graph (Preserved Module 2) ---
        # 3.1 Construct A
        E = F.relu(self.W_E(H_gru))
        A_dynamic = torch.eye(M, device=device).unsqueeze(0) + torch.bmm(E, E.transpose(1, 2))
        
        # 3.2 Projection & Spectral Conv
        k = min(self.k_stiefel, M, 2 * self.d)
        try:
            U, S, V = torch.linalg.svd(H_gru, full_matrices=False)
            F_opt = U[:, :, :k]
        except:
            F_opt = torch.eye(M, k).unsqueeze(0).repeat(B, 1, 1).to(device)
            
        H_tilde = torch.bmm(F_opt.transpose(1, 2), H_gru)
        G_k = self.G_kernel[:k, :]
        H_spectral = H_tilde * G_k.unsqueeze(0)
        H_refined = torch.bmm(F_opt, H_spectral) # (B, M, 2d)
        
        # --- Step 4: Gated Attention Fusion (Doc Step 3) ---
        # Logic: a = Sigmoid(W[H, s] + b)
        # H is H_refined (2d). s is s_feat (d).
        
        combined_fusion = torch.cat([H_refined, s_feat], dim=-1) # (B, M, 3d)
        a_vals = torch.sigmoid(self.W_a(combined_fusion) + self.b_a) # (B, M, 2d)
        
        # Pad s_feat to 2d (append zeros)
        s_feat_pad = torch.cat([s_feat, torch.zeros_like(s_feat)], dim=-1) # (B, M, 2d)
        
        # Weighted Sum
        H_weighted = a_vals * H_refined + (1 - a_vals) * s_feat_pad
        
        # --- Step 5: Time-Aware Evolution (Doc Step 4) ---
        # g = SiLU(W dt + b)
        g_vals = F.silu(self.W_g(delta_t) + self.b_g) # (B, M, 2d)
        
        # Iterative Update: p_i = g p_i* + (1-g) p_{i-1}
        # Init p_prev as First H_weighted (Doc: p_0 = H_weighted_0)
        
        p_seq = []
        p_prev = H_weighted[:, 0, :] # Init from first element.
        p_seq.append(p_prev.unsqueeze(1))

        # Loop from 1 to M-1
        for i in range(1, M):
            h_curr = H_weighted[:, i, :] # (B, 2d)
            g_curr = g_vals[:, i, :]     # (B, 2d)
            
            p_curr = g_curr * h_curr + (1 - g_curr) * p_prev
            
            p_seq.append(p_curr.unsqueeze(1))
            p_prev = p_curr
            
        P_adaptive = torch.cat(p_seq, dim=1) # (B, M, 2d)
        
        # Return tuple (Context, P_adaptive_seq)
        # Model expects: _, p_adaptive_seq = stapm(...)
        # So we return (H_refined, P_adaptive) or similar
        return H_refined, P_adaptive
