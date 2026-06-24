import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transe import create_kg_model
from kiam import KIAM
from stapm import STAPM
from cpdiff import CPDiff
from base_replacements import (
    BaseStyleDynamicModule,
    BaseStyleDiffusionReplacement,
    BaseStyleStaticModule,
)

class KSDDiff(nn.Module):
    def __init__(self, config, dataset):
        super(KSDDiff, self).__init__()
        self.config = config
        
        # 0. Embeddings (ID)
        self.poi_embedding = nn.Embedding(dataset.num_pois, config.embed_dim, padding_idx=0)

        # [UPDATED] Use Config Dropout
        self.grad_dropout = nn.Dropout(getattr(config, 'dropout', 0.1))

        
        # Category Embedding (New: Sparsity Fix)
        # Try dataset first (dataset.cat2id len)
        num_cats_val = getattr(dataset, 'num_cats', getattr(config, 'num_cats', 0))
        if num_cats_val == 0 and hasattr(dataset, 'cat2id'):
             num_cats_val = len(dataset.cat2id)
        
        if hasattr(dataset, 'poi_cats'):
            self.register_buffer('all_poi_cats', dataset.poi_cats)
            # Register regions too for lookup
            if hasattr(dataset, 'poi_regions'):
                 self.register_buffer('all_poi_regions', dataset.poi_regions)
            
            # [NEW] Add POI Coords for Distance-Aware Reranking (Pairs-F1 Fix)
            if hasattr(dataset, 'poi_coords') and dataset.poi_coords is not None:
                if isinstance(dataset.poi_coords, np.ndarray):
                    self.register_buffer('poi_coords', torch.from_numpy(dataset.poi_coords).float())
                elif isinstance(dataset.poi_coords, torch.Tensor):
                    self.register_buffer('poi_coords', dataset.poi_coords.float())
            
            print(f"Model: Registered POI Categories buffer. Num Cats: {num_cats_val}")
            self.cat_embedding = nn.Embedding(num_cats_val + 1, config.embed_dim, padding_idx=0)
        else:
            self.cat_embedding = None
            print("Model: Warning - No POI Categories found.")

        # Region Embedding (New)
        # We might use lookup instead of passed argument
        n_reg = getattr(dataset, 'num_regions', 1)
        self.region_embedding = None
        if n_reg > 1:
            print(f"Model: Initializing Region Embeddings for {n_reg} regions.")
            self.region_embedding = nn.Embedding(n_reg, config.embed_dim, padding_idx=0)


        # KG embedding model (TransE by default, configurable to TransH).
        self.transe = create_kg_model(dataset.num_entities, dataset.num_relations, config)
        
        # 1. Modules
        self.kiam = KIAM(config, self.transe)
        self.stapm = STAPM(config)

        self.replace_module_a_with_base = bool(getattr(config, 'replace_module_a_with_base', False))
        self.replace_module_b_with_base = bool(getattr(config, 'replace_module_b_with_base', False))
        self.replace_module_c_with_base = bool(getattr(config, 'replace_module_c_with_base', False))

        self.base_static_module = None
        self.base_dynamic_module = None
        self.base_diffusion_module = None
        if self.replace_module_a_with_base:
            self.base_static_module = BaseStyleStaticModule(
                embed_dim=config.embed_dim,
                num_layers=2,
                num_heads=4,
                dropout=getattr(config, 'dropout', 0.1),
                poi_size=dataset.num_pois,
            )
        if self.replace_module_b_with_base:
            self.base_dynamic_module = BaseStyleDynamicModule(config.embed_dim)
        
        self.time_proj = nn.Linear(1, config.embed_dim)
        
        # [UPDATED] Dynamic Dim now includes STAPM state (2d) + Time Delta (d)
        self.cpdiff = CPDiff(config, static_dim=config.embed_dim, dynamic_dim=3*config.embed_dim)
        if self.replace_module_c_with_base:
            self.base_diffusion_module = BaseStyleDiffusionReplacement(config.embed_dim)

        # Direct output layer for recommendation.
        self.output_layer = nn.Linear(config.embed_dim, dataset.num_pois)

        # [NEW] Adaptive Fusion Gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * config.embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )
        
        self.to(config.device)
    
    def get_full_embedding(self, poi_ids, regions=None):
        """Helper to combine ID + Cat + Region"""
        emb = self.poi_embedding(poi_ids)
        
        # Enable Rich Feature Summation
        # Integrating Category and Region info to deal with Cold Start / Sparsity
        if self.cat_embedding is not None and hasattr(self, 'all_poi_cats'):
            all_poi_cats = self.all_poi_cats
            if not isinstance(all_poi_cats, torch.Tensor):
                all_poi_cats = torch.as_tensor(all_poi_cats, device=poi_ids.device)
            # Clip indices to safe range just in case
            safe_ids = poi_ids.long().clamp(0, all_poi_cats.shape[0] - 1)
            cats = all_poi_cats[safe_ids] # (B, N)
            emb = emb + self.cat_embedding(cats)
            
        if self.region_embedding is not None:
            if regions is None and hasattr(self, 'all_poi_regions'):
                 all_poi_regions = self.all_poi_regions
                 if not isinstance(all_poi_regions, torch.Tensor):
                     all_poi_regions = torch.as_tensor(all_poi_regions, device=poi_ids.device)
                 safe_ids = poi_ids.long().clamp(0, all_poi_regions.shape[0] - 1)
                 regions = all_poi_regions[safe_ids]
            
            if regions is not None:
                emb = emb + self.region_embedding(regions)
             
        return emb

    def forward(self, home_seq, context_seq, context_times, context_coords, target_poi_emb=None):
        pass # Not used directly in training loop which calls compute_loss

    def compute_loss(self, home_seq, home_seq_ents, home_neighbors, oot_seq, oot_times, oot_coords, oot_mask, oot_regions=None, home_hours=None, oot_hours=None):
        """
        Full Training Step
        """
        B, N = oot_seq.shape
        
        # --- 1. intrinsic preference (K-SAM) ---
        # Embed Home POIs (ID + Cat + Region)
        home_emb = self.get_full_embedding(home_seq) 
        home_mask = (home_seq != 0)
        
        # home_neighbors: (B, M, K, 2)
        if self.replace_module_a_with_base:
            home_times = torch.zeros_like(home_seq, dtype=torch.float32)
            home_coords = torch.zeros(home_seq.size(0), home_seq.size(1), 2, device=home_seq.device)
            p_intrinsic = self.base_static_module(home_seq, home_times, home_coords, home_mask)
        else:
            _, p_intrinsic = self.kiam(home_emb, home_neighbors, home_mask) # p_intrinsic: (B, d)
        if bool(getattr(self.config, 'ablate_module_a', False)):
            # Module A ablation: remove intrinsic preference signal.
            p_intrinsic = torch.zeros_like(p_intrinsic)
        
        # Auxiliary Alignment Loss
        # Embed OOT Sequence (ID + Cat + Region)
        oot_emb_all = self.get_full_embedding(oot_seq)
        
        # Adaptive Aggregation for OOT (Ground Truth)
        u_oot_gt = self.kiam.adaptive_aggregate(oot_emb_all, oot_mask)
        
        loss_mse, loss_cl = self.kiam.calculate_loss(p_intrinsic, u_oot_gt)
        
        
        pos_h = home_seq_ents.unsqueeze(2).expand(-1, -1, self.config.kg_k_hop).reshape(-1)
        pos_r = home_neighbors[..., 1].reshape(-1)
        pos_t = home_neighbors[..., 0].reshape(-1)
        
        # Filter padding
        valid_triplets = (pos_h != 0) & (pos_r != 0)
        pos_h = pos_h[valid_triplets]
        pos_r = pos_r[valid_triplets]
        pos_t = pos_t[valid_triplets]

        if pos_h.numel() > 0:
            # Negative Sampling (Corrupt Tail)
            # Shuffle tails in the batch
            idx = torch.randperm(pos_t.size(0))
            neg_t = pos_t[idx]
            
            # Or use global random entity?
            # Shuffle is efficient for batch.
            loss_kg = self.transe(pos_h, pos_r, pos_t, pos_h, pos_r, neg_t)
        else:
            loss_kg = torch.tensor(0.0, device=home_seq.device)

        # --- 2. adaptive preference (STAPM) ---
        # We want to predict oot_seq[t] given oot_seq[:t]
        input_seq = oot_seq[:, :-1] 
        input_times = oot_times[:, :-1]
        input_coords = oot_coords[:, :-1] # (B, N-1, 2)
        input_hours = oot_hours[:, :-1] if oot_hours is not None else None
        
        # Embed
        input_emb = oot_emb_all[:, :-1] # Use the already combined embeddings
        
        # Run STAPM
        # Returns: H_refined, P_adaptive (B, N-1, 2d)
        if self.replace_module_b_with_base:
            p_adaptive_seq = self.base_dynamic_module(input_emb, input_times, input_coords)
        else:
            _, p_adaptive_seq = self.stapm(input_emb, input_times, input_coords, local_hours=input_hours)
        if bool(getattr(self.config, 'ablate_module_b', False)):
            # Module B ablation: remove adaptive preference signal.
            p_adaptive_seq = torch.zeros_like(p_adaptive_seq)
        
        # --- 3. Diffusion And Recommendation ---
        target_embs = oot_emb_all[:, 1:]
        
        # [NEW] Calculate Time Delta (Next Time - Current Time)
        delta_t = (oot_times[:, 1:] - oot_times[:, :-1]).unsqueeze(-1).float()
        delta_t_emb = self.time_proj(delta_t) # (B, N-1, d)
        
        # We need to reshape for batch processing
        flat_p_intrinsic = p_intrinsic.unsqueeze(1).expand(-1, N-1, -1).reshape(-1, self.config.embed_dim)
        flat_dynamic = p_adaptive_seq.reshape(-1, 2 * self.config.embed_dim)
        flat_time = delta_t_emb.reshape(-1, self.config.embed_dim)
        flat_target = target_embs.reshape(-1, self.config.embed_dim)
        
        # Filter padding
        mask = oot_mask[:, 1:].reshape(-1).bool()
        
        # a. Diffusion Condition Construction (Intent)
        # [UPDATED] Condition on State + Time Delta
        diffusion_module = self.base_diffusion_module if self.replace_module_c_with_base else self.cpdiff
        intent_vec = diffusion_module.cond_mlp(torch.cat([flat_p_intrinsic, flat_dynamic, flat_time], dim=-1)) 

        # b. [New] Alignment Loss: Force Intent to match Target Embedding Feature-wise
        # This justifies fusing Intent + Diff_Sample at inference, as they target the same space
        loss_align = F.mse_loss(intent_vec[mask], flat_target[mask])

        # c. Recommendation (Classification)
        logits = self.output_layer(intent_vec)

        
        # 1. Positive Sample (Ideal Augmentation)
        residual = flat_target[mask] - intent_vec[mask] # This is what diffusion SHOULD generate
        gate_pos = self.fusion_gate(torch.cat([intent_vec[mask], residual], dim=-1))
        loss_gate_pos = F.mse_loss(gate_pos, torch.ones_like(gate_pos)) # Should open gate

        # 2. Negative Sample (Random Noise)
        noise = torch.randn_like(residual)
        gate_neg = self.fusion_gate(torch.cat([intent_vec[mask], noise], dim=-1))
        loss_gate_neg = F.mse_loss(gate_neg, torch.zeros_like(gate_neg)) # Should close gate
        
        loss_gate = (loss_gate_pos + loss_gate_neg) * 0.5

        # d. Diffusion Loss (Reconstruction)
        # Diffusion tries to reconstruct flat_target using intent_vec as condition
        loss_diff = diffusion_module.get_loss(flat_target[mask], flat_p_intrinsic[mask], flat_dynamic[mask], flat_time[mask])

        if bool(getattr(self.config, 'ablate_module_c', False)):
            # Module C ablation: remove diffusion branch contribution.
            loss_diff = torch.tensor(0.0, device=logits.device)
            loss_gate = torch.tensor(0.0, device=logits.device)

        # Targets
        flat_target_ids = oot_seq[:, 1:].reshape(-1)
        
        # [NEW] Region Masking for Loss Calculation (Crucial for convergence)
        if oot_regions is not None and hasattr(self, 'all_poi_regions'):
            all_poi_regions = self.all_poi_regions
            if not isinstance(all_poi_regions, torch.Tensor):
                all_poi_regions = torch.as_tensor(all_poi_regions, device=logits.device)
            target_region_ids = oot_regions[:, 1:].reshape(-1)
            # Create mask: (Batch, Num_POIs) where region matches
            # broadcasting: (Batch, 1) == (1, Num_POIs)
            valid_region_mask = (target_region_ids.unsqueeze(1) == all_poi_regions.unsqueeze(0))
            logits = logits.masked_fill(~valid_region_mask, -1e9)

        loss_rec = F.cross_entropy(logits, flat_target_ids, reduction='mean', ignore_index=0)

        # [NEW] Order Ranking Loss: Explicitly penalize staying/reversing.
        logits_seq = logits.view(B, N-1, -1)
        target_ids = oot_seq[:, 1:] # y_t
        prev_ids = oot_seq[:, :-1]  # y_{t-1}
        ranking_mask = (target_ids != 0) & (prev_ids != 0)

        if ranking_mask.sum() > 0:
            pos_scores = torch.gather(logits_seq, 2, target_ids.unsqueeze(-1)).squeeze(-1)
            neg_scores = torch.gather(logits_seq, 2, prev_ids.unsqueeze(-1)).squeeze(-1)
            loss_order = F.relu(2.0 - (pos_scores - neg_scores))
            loss_order = loss_order[ranking_mask].mean()
        else:
            pos_scores = torch.gather(logits_seq, 2, target_ids.unsqueeze(-1)).squeeze(-1)
            loss_order = torch.tensor(0.0, device=logits.device)

        # [NEW] Global order consistency across all valid step pairs.
        # Compute only when enabled to avoid unnecessary O(T^2) overhead.
        lambda_global_order = getattr(self.config, 'lambda_global_order', 0.0)
        if lambda_global_order > 0:
            global_margin = getattr(self.config, 'global_order_margin', 1.0)
            T = target_ids.size(1)
            global_losses = []
            for i in range(T - 1):
                ids_i = target_ids[:, i]
                logits_i = logits_seq[:, i, :]
                score_i_curr = pos_scores[:, i]
                for j in range(i + 1, T):
                    ids_j = target_ids[:, j]
                    pair_mask = (ids_i != 0) & (ids_j != 0) & (ids_i != ids_j)
                    if pair_mask.any().item():
                        logits_j = logits_seq[:, j, :]
                        score_j_curr = pos_scores[:, j]

                        score_i_j = torch.gather(logits_i, 1, ids_j.unsqueeze(-1)).squeeze(-1)
                        score_j_i = torch.gather(logits_j, 1, ids_i.unsqueeze(-1)).squeeze(-1)

                        loss_ij = F.relu(global_margin - (score_i_curr - score_i_j))
                        loss_ji = F.relu(global_margin - (score_j_curr - score_j_i))
                        global_losses.append(0.5 * (loss_ij[pair_mask] + loss_ji[pair_mask]))

            if len(global_losses) > 0:
                loss_order_global = torch.cat(global_losses).mean()
            else:
                loss_order_global = torch.tensor(0.0, device=logits.device)
        else:
            loss_order_global = torch.tensor(0.0, device=logits.device)

        # --- Total Loss ---
        total_loss = (self.config.lambda_diff * loss_diff) + \
                     (getattr(self.config, 'lambda_align', 0.1) * loss_align) + \
                     loss_mse + \
                     (self.config.lambda_cl * loss_cl) + \
                     (self.config.lambda_kg * loss_kg) + \
                     (getattr(self.config, 'lambda_rec', 1.0) * loss_rec) + \
                     (getattr(self.config, 'lambda_order', 1.0) * loss_order) + \
                     (lambda_global_order * loss_order_global) + \
                     (0.1 * loss_gate)
        
        return total_loss, {
            "diff": loss_diff.item(),
            "align": loss_align.item(),
            "rec": loss_rec.item(),
            "order": loss_order.item(),
            "order_global": loss_order_global.item(),
            "gate": loss_gate.item(),
            "mse": loss_mse.item(),
            "cl": loss_cl.item(),
            "kg": loss_kg.item()
        }

    def predict_next(self, home_seq, home_neighbors, context_seq, context_times, context_coords, target_time, target_region=None, context_hours=None, alpha=None):
        """
        Inference with Adaptive Diffusion Fusion
        """
        # Static
        home_emb = self.get_full_embedding(home_seq)
        home_mask = (home_seq != 0)
        if self.replace_module_a_with_base:
            home_times = torch.zeros_like(home_seq, dtype=torch.float32)
            home_coords = torch.zeros(home_seq.size(0), home_seq.size(1), 2, device=home_seq.device)
            p_intrinsic = self.base_static_module(home_seq, home_times, home_coords, home_mask)
        else:
            _, p_intrinsic = self.kiam(home_emb, home_neighbors, home_mask)
        if bool(getattr(self.config, 'ablate_module_a', False)):
            p_intrinsic = torch.zeros_like(p_intrinsic)
        
        # Dynamic
        ctx_emb = self.get_full_embedding(context_seq)
        if self.replace_module_b_with_base:
            p_dyn_seq = self.base_dynamic_module(ctx_emb, context_times, context_coords)
            h_last = p_dyn_seq[:, -1, :]
        else:
            _, p_dyn_seq = self.stapm(ctx_emb, context_times, context_coords, local_hours=context_hours)
            h_last = p_dyn_seq[:, -1, :] # Last step state (B, 2d)
        if bool(getattr(self.config, 'ablate_module_b', False)):
            h_last = torch.zeros_like(h_last)
        
        # [IMPROVED] Pairs-F1 Boosting: Add Order-Aware Loss
        # target_time: (B, 1) or (B,)
        last_time = context_times[:, -1:] # (B, 1)
        delta_t = (target_time - last_time).float()
        delta_t_emb = self.time_proj(delta_t).squeeze(1) # (B, d)
        
        diffusion_module = self.base_diffusion_module if self.replace_module_c_with_base else self.cpdiff
        intent_vec = diffusion_module.cond_mlp(torch.cat([p_intrinsic, h_last, delta_t_emb], dim=-1))
        
        # --- 2. Diffusion Generation (Noise) ---
        gen_vec = diffusion_module.sample(p_intrinsic, h_last, delta_t_emb)

        # --- 3. Adaptive Fusion ---
        # "Learn from noises": The Gate decides if this noise is helpful.
        # Concatenate Intent and Generated to decide weight
        gate_input = torch.cat([intent_vec, gen_vec], dim=-1)
        gate_val = self.fusion_gate(gate_input) # (B, 1) in [0, 1]
        
        # Weighted Residual Connection
        # Final = Intent + (Gate * Generated_Diff)
        if bool(getattr(self.config, 'ablate_module_c', False)):
            final_vec = intent_vec
        elif alpha is not None:
            final_vec = (alpha * intent_vec) + ((1 - alpha) * gen_vec)
        else:
            final_vec = intent_vec + (gate_val * gen_vec)

        # --- 4. Prediction ---
        logits_final = self.output_layer(final_vec)

        # --- Region Masking ---
        # [FIXED] Use TARGET Region for masking (Teacher Forcing Destination City)
        if target_region is not None and hasattr(self, 'all_poi_regions'):
            all_poi_regions = self.all_poi_regions
            if not isinstance(all_poi_regions, torch.Tensor):
                all_poi_regions = torch.as_tensor(all_poi_regions, device=logits_final.device)
            # target_region: (B, 1)
            target_rids = target_region.reshape(-1)
            valid_region_mask = (target_rids.unsqueeze(1) == all_poi_regions.unsqueeze(0))
            logits_final = logits_final.masked_fill(~valid_region_mask, -1e9)
        
        return logits_final
