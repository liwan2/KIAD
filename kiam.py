import torch
import torch.nn as nn
import torch.nn.functional as F

class KIAM(nn.Module):
    def __init__(self, config, transe_model):
        super(KIAM, self).__init__()
        self.config = config
        self.d = config.embed_dim
        
        self.entity_emb = transe_model.entity_emb
        self.relation_emb = transe_model.relation_emb
        
        # 2.1 Relation-aware Attention
        self.W_att = nn.Linear(2 * self.d, 1) # [e || r] -> score
        
        # 2.2 Noise Filtering Gate
        self.W_g1 = nn.Linear(self.d, self.d)
        self.W_g2 = nn.Linear(self.d, self.d)
        self.b_g = nn.Parameter(torch.zeros(self.d))
        
        # 3.1 Adaptive Aggregation
        self.W_agg = nn.Linear(self.d, self.d)
        self.V_agg = nn.Linear(self.d, 1, bias=False)
        self.b_agg = nn.Parameter(torch.zeros(self.d))
        
        # 4.1 Mapping Network (Residual)
        self.map_fc1 = nn.Linear(self.d, self.d)
        self.map_fc2 = nn.Linear(self.d, self.d)
        
    def adaptive_aggregate(self, seq_emb, mask=None):
        """
        Step 3: Adaptive Preference Aggregation
        seq_emb: (B, L, d)
        mask: (B, L) - 1 for valid, 0 for padding
        """
        # Attention weights w_m = V^T * tanh(W v + b)
        trans_v = torch.tanh(self.W_agg(seq_emb) + self.b_agg) # (B, L, d)
        w_m = self.V_agg(trans_v) # (B, L, 1)
        
        if mask is not None:
            # Mask out padding with very large negative value
            mask_expanded = mask.unsqueeze(-1) # (B, L, 1)
            w_m = w_m.masked_fill(mask_expanded == 0, -1e9)
            
        beta_m = F.softmax(w_m, dim=1) # (B, L, 1)
        
        # u: (B, d)
        u_agg = torch.sum(beta_m * seq_emb, dim=1)
        return u_agg

    def forward(self, poi_emb, neighbors, home_mask=None):
        """
        poi_emb: (B, M, d) - Initial ID-based embedding of POIs
        neighbors: (B, M, K, 2) - Each neighbor is (EntityID, RelationID)
        home_mask: (B, M) - Optional mask for home sequence
        """
        B, M, d = poi_emb.shape
        K = neighbors.shape[2]
        
        # --- Step 2: Gated Fusion ---
        
        # Get Neighbor Embeddings: e_k, r_k
        # neighbors[..., 0] is EntityID, [..., 1] is RelationID
        flat_ents = neighbors[..., 0].view(-1)
        flat_rels = neighbors[..., 1].view(-1)
        
        e_k = self.entity_emb(flat_ents).view(B, M, K, d)
        r_k = self.relation_emb(flat_rels).view(B, M, K, d)
        
        # Attention
        # feat_k: (B, M, K, 2d)
        feat_k = torch.cat([e_k, r_k], dim=-1)
        # score_k: (B, M, K, 1)
        score_k = F.leaky_relu(self.W_att(feat_k))
        alpha_k = F.softmax(score_k, dim=2) # Softmax over K neighbors
        
        # v_kg: (B, M, d)
        v_kg = torch.sum(alpha_k * e_k, dim=2)
        
        # Gate
        # g: (B, M, d)
        gate_val = torch.sigmoid(self.W_g1(poi_emb) + self.W_g2(v_kg) + self.b_g)
        
        # vec_v: (B, M, d) - Fused POI representation
        vec_v = gate_val * poi_emb + (1 - gate_val) * v_kg
        
        # --- Step 3: Adaptive Aggregation (for Home) ---
        u_home = self.adaptive_aggregate(vec_v, home_mask)
        
        # --- Step 4: Contrastive Alignment ---
        
        # Mapping to OOT space
        # Residual MLP
        res = F.relu(self.map_fc1(u_home))
        P_intrinsic = self.map_fc2(res) + u_home  # (B, d)
        
        return u_home, P_intrinsic

    def calculate_loss(self, P_intrinsic, u_oot_gt, temperature=0.1):
        """
        Step 4.2: Joint Loss (MSE + InfoNCE)
        P_intrinsic: (B, d) - Predicted preferences
        u_oot_gt: (B, d) - Ground truth aggregated OOT preferences (computed similarly to u_home)
        """
        # 1. MSE
        loss_mse = F.mse_loss(P_intrinsic, u_oot_gt)
        
        # 2. InfoNCE
        P_norm = F.normalize(P_intrinsic, dim=1)
        T_norm = F.normalize(u_oot_gt, dim=1)
        
        # Similarity matrix (B, B)
        logits = torch.matmul(P_norm, T_norm.T) / temperature
        
        # Labels: diagonal are positives
        labels = torch.arange(logits.size(0), device=logits.device)
        
        loss_cl = F.cross_entropy(logits, labels)
        
        return loss_mse, loss_cl
