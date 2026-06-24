import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import json
import importlib
from collections import defaultdict
import math
from tqdm import tqdm

from dataset import FoursquareDataset
from model import KSDDiff
from transe import train_kg_model
from utils import Metrics, top_np_recommendation, set_seed
from new_metrics import f1_score, pairs_f1_score, count_adjacent_repetition_rate # New Aligning Metrics


_CONFIG_CLASS = None


def _load_config_class(config_module_name: str):
    """Load Config class dynamically so renamed config files work without code edits."""
    module_candidates = [config_module_name]
    if config_module_name.endswith('.py'):
        module_candidates.append(config_module_name[:-3])

    tried = []
    for name in module_candidates:
        for mod_name in (name, f"KSD_Diff_Code.{name}"):
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "Config"):
                    return mod.Config
                if hasattr(mod, "load_config"):
                    cfg_obj = mod.load_config()
                    return cfg_obj.__class__
            except Exception as e:
                tried.append(f"{mod_name}: {e}")

    raise ImportError(
        "Unable to load Config class. Tried modules: " + " | ".join(tried)
    )


def get_config_class(config_module_name: str | None = None):
    global _CONFIG_CLASS
    if _CONFIG_CLASS is not None:
        return _CONFIG_CLASS

    module_name = config_module_name or os.environ.get("KSD_CONFIG_MODULE", "config")
    _CONFIG_CLASS = _load_config_class(module_name)
    return _CONFIG_CLASS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KSD-Diff training/evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("KSD_CONFIG_MODULE", "config"),
        help="Config module name, e.g. config or config_Foursquare",
    )
    return parser.parse_args()


class TrainHomeSparsityWrapper(torch.utils.data.Dataset):
    """Apply deterministic home-checkin sparsity on training samples only."""

    def __init__(self, subset, keep_ratio: float, seed: int):
        self.subset = subset
        self.keep_ratio = float(keep_ratio)
        self.seed = int(seed)

    def __len__(self):
        return len(self.subset)

    def _build_keep_mask(self, home_seq: torch.Tensor, item_idx: int) -> torch.Tensor:
        valid_idx = (home_seq != 0).nonzero(as_tuple=False).flatten()
        n_valid = int(valid_idx.numel())
        mask = torch.zeros_like(home_seq, dtype=torch.bool)
        if n_valid == 0:
            return mask

        if self.keep_ratio >= 1.0:
            mask[valid_idx] = True
            return mask

        n_keep = max(1, int(round(n_valid * self.keep_ratio)))
        # Deterministic per sample and seed.
        rs = np.random.RandomState(self.seed + int(item_idx))
        chosen = rs.choice(valid_idx.cpu().numpy(), size=n_keep, replace=False)
        mask[torch.as_tensor(chosen, dtype=torch.long)] = True
        return mask

    def __getitem__(self, idx):
        sample = self.subset[idx]
        home_seq, home_seq_ents, kg_data, oot_seq, oot_times, oot_coords, oot_mask, oot_regions, home_hours, oot_hours = sample

        keep_mask = self._build_keep_mask(home_seq, idx)
        drop_mask = ~keep_mask

        # Keep tensors consistent after sparsification.
        home_seq = home_seq.clone()
        home_seq_ents = home_seq_ents.clone()
        kg_data = kg_data.clone()
        home_hours = home_hours.clone()

        home_seq[drop_mask] = 0
        home_seq_ents[drop_mask] = 0
        kg_data[drop_mask] = 0
        home_hours[drop_mask] = 0

        return home_seq, home_seq_ents, kg_data, oot_seq, oot_times, oot_coords, oot_mask, oot_regions, home_hours, oot_hours

def train_one_epoch(model, dataloader, optimizer, config):
    model.train()
    total_loss = 0
    loss_dict = {}
    
    for batch in tqdm(dataloader, desc="Training"):
        # Unpack 10 items (Added home_hours, oot_hours)
        home_seq, home_seq_ents, kg_data, oot_seq, oot_times, oot_coords, oot_mask, oot_regions, home_hours, oot_hours = batch
        
        # Move to device
        home_seq = home_seq.to(config.device)
        home_seq_ents = home_seq_ents.to(config.device)
        kg_data = kg_data.to(config.device)
        oot_seq = oot_seq.to(config.device)
        oot_times = oot_times.to(config.device)
        oot_coords = oot_coords.to(config.device)
        oot_regions = oot_regions.to(config.device)
        oot_mask = oot_mask.to(config.device)
        home_hours = home_hours.to(config.device)
        oot_hours = oot_hours.to(config.device)
        
        optimizer.zero_grad()
        # Pass regions to model
        loss, loss_dict = model.compute_loss(home_seq, home_seq_ents, kg_data, oot_seq, oot_times, oot_coords, oot_mask, oot_regions, home_hours, oot_hours)
        loss.backward()
        optimizer.step()
        
        # Accumulate breakdown
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss, loss_dict # Return last batch breakdown for inspection


def build_transition_prior(full_dataset, train_indices):
    """Build sparse log-count transition prior from train OOT trajectories."""
    trans_counts = defaultdict(lambda: defaultdict(int))
    for idx in train_indices:
        sample = full_dataset.samples[idx]
        seq = sample['oot_seq'].tolist()
        # Keep valid non-padding prefix.
        valid = [int(x) for x in seq if int(x) != 0]
        for i in range(len(valid) - 1):
            a, b = valid[i], valid[i + 1]
            if a > 0 and b > 0:
                trans_counts[a][b] += 1

    # Use log(1+count) as stable sparse bonus.
    prior = {}
    for a, row in trans_counts.items():
        prior[a] = {b: math.log1p(c) for b, c in row.items()}
    return prior


def build_pair_aware_split_indices(full_dataset, seed, ratios=(0.8, 0.1, 0.1)):
    """Match base_code random_split behavior: split within each OD pair by 8:1:1."""
    pair_to_indices = defaultdict(list)
    for idx, sample in enumerate(full_dataset.samples):
        pair_to_indices[sample.get('od_pair', ('UNK', 'UNK'))].append(idx)

    np.random.seed(seed)
    train_idx, valid_idx, test_idx = [], [], []
    for _, idxs in pair_to_indices.items():
        idxs = idxs.copy()
        np.random.shuffle(idxs)
        n = len(idxs)
        train_offset = int(n * ratios[0])
        valid_offset = int(n * (ratios[0] + ratios[1]))
        train_idx.extend(idxs[:train_offset])
        valid_idx.extend(idxs[train_offset:valid_offset])
        test_idx.extend(idxs[valid_offset:])

    return train_idx, valid_idx, test_idx

def evaluate(model, dataloader, config, transition_prior=None):
    model.eval()
    
    batch_alt_f1 = []
    batch_alt_pairs_f1 = []
    batch_full_f1 = []      # [NEW]
    batch_full_pairs_f1 = [] # [NEW]
    repetition_list = []
    
    # Pre-fetch region map to device for lookups
    # (Assuming region logic is handled inside model per batch now)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Unpack 10 items
            home_seq, _, kg_data, oot_seq, oot_times, oot_coords, oot_mask, batch_regions_gt, home_hours, oot_hours = batch 
            
            # To device
            home_seq = home_seq.to(config.device)
            kg_data = kg_data.to(config.device)
            gt_seq = oot_seq.to(config.device)
            
            # --- Autoregressive Generation (Trajectory Level) ---
            # Start with the first token (Home/Start)
            # Must keep dimension [B, 1]
            curr_seq = gt_seq[:, :1] 
            
            # Initialize contexts with the first step properties
            curr_times = oot_times[:, :1].to(config.device)
            curr_coords = oot_coords[:, :1].to(config.device)
            curr_regions = batch_regions_gt[:, :1].to(config.device)
            curr_hours = oot_hours[:, :1].to(config.device)
            
            # List to store the generated POI IDs for this batch (keep as list of tensors [B,1])
            generated_trajectory = [curr_seq] 
            
            seq_len = gt_seq.size(1)
            # Generate T-1 steps (predict 2nd to End)
            for t in range(seq_len - 1):
                # [NEW] Get Target Time for Step t+1
                target_time_step = oot_times[:, t+1:t+2].to(config.device)
                
                # [NEW] Get Target Region for Step t+1 (For Masking)
                # Pass destination region to allow model to select POIs in that region ONLY
                target_region_step = batch_regions_gt[:, t+1:t+2].to(config.device)

                # 1. Predict Next Logits
                mc_samples = max(1, int(getattr(config, 'decode_mc_samples', 1)))
                if mc_samples == 1:
                    logits_tuple = model.predict_next(home_seq, kg_data, curr_seq, curr_times, curr_coords, target_time_step, target_region_step, curr_hours, alpha=None)
                    if isinstance(logits_tuple, tuple):
                        logits = logits_tuple[0]
                    else:
                        logits = logits_tuple
                else:
                    logits_list = []
                    for _ in range(mc_samples):
                        logits_tuple = model.predict_next(home_seq, kg_data, curr_seq, curr_times, curr_coords, target_time_step, target_region_step, curr_hours, alpha=None)
                        step_logits = logits_tuple[0] if isinstance(logits_tuple, tuple) else logits_tuple
                        logits_list.append(step_logits)
                    logits = torch.stack(logits_list, dim=0).mean(dim=0)
                
                # [OPTIMIZATION] Temperature Scaling for Stable Mean > 0.04
                # Making the distribution sharper before sampling reduces variance and "bad luck".
                temperature = getattr(config, 'decode_temperature', 0.4)
                logits = logits / temperature

                decode_mode = getattr(config, 'decode_mode', 'greedy')
                if decode_mode == 'topnp':
                    # Strictly align with base_code inference: top-k (k=seq_len) then top-np sampling.
                    k_align = min(gt_seq.size(1), logits.size(1))
                    cand_vals, cand_ids = torch.topk(logits, k=k_align, dim=1)
                    sampled = top_np_recommendation(
                        cand_ids.unsqueeze(1),
                        cand_vals.unsqueeze(1),
                        confidence=getattr(config, 'decode_np_confidence', 1.0),
                        threshold=getattr(config, 'decode_np_threshold', 0.8),
                    )
                    next_token_step = sampled[:, 0].unsqueeze(1).to(config.device)
                else:
                    # Greedy mode keeps the previous heuristic penalties.
                    for b_i in range(logits.size(0)):
                        history = curr_seq[b_i]
                        unique_hist = torch.unique(history)
                        logits[b_i, unique_hist] -= getattr(config, 'decode_hist_penalty', 10.0)

                    cur_pos = curr_coords[:, -1, :]
                    if hasattr(model, 'poi_coords'):
                        diff = cur_pos.unsqueeze(1) - model.poi_coords.unsqueeze(0)
                        dist_sq = (diff ** 2).sum(dim=-1)
                        logits -= (dist_sq * getattr(config, 'decode_dist_penalty', 20.0))

                    last_poi = curr_seq[:, -1]
                    for b_i in range(logits.size(0)):
                        logits[b_i, last_poi[b_i]] -= getattr(config, 'decode_last_penalty', 80.0)
                        if curr_seq.size(1) > 1:
                            second_last = curr_seq[b_i, -2]
                            if second_last != 0:
                                logits[b_i, second_last] -= getattr(config, 'decode_loop2_penalty', 40.0)

                    k_align = min(gt_seq.size(1), logits.size(1))
                    cand_vals, cand_ids = torch.topk(logits, k=k_align, dim=1)

                    # Optional sequence-level rerank with one-step lookahead.
                    rerank_enable = bool(getattr(config, 'decode_rerank_enable', False))
                    rerank_topk = int(getattr(config, 'decode_rerank_topk', 3))
                    rerank_weight = float(getattr(config, 'decode_rerank_weight', 0.15))
                    can_lookahead = rerank_enable and (t < seq_len - 2) and (rerank_topk > 1)

                    if can_lookahead:
                        K = min(rerank_topk, cand_ids.size(1))
                        topk_ids = cand_ids[:, :K]
                        topk_vals = cand_vals[:, :K]

                        B = topk_ids.size(0)
                        flat_ids = topk_ids.reshape(-1, 1)

                        # Expand current context to B*K and append candidate token.
                        home_exp = home_seq.repeat_interleave(K, dim=0)
                        kg_exp = kg_data.repeat_interleave(K, dim=0)

                        seq_exp = curr_seq.repeat_interleave(K, dim=0)
                        times_exp = curr_times.repeat_interleave(K, dim=0)
                        coords_exp = curr_coords.repeat_interleave(K, dim=0)
                        hours_exp = curr_hours.repeat_interleave(K, dim=0)

                        gt_time_step = oot_times[:, t+1:t+2].to(config.device).repeat_interleave(K, dim=0)
                        gt_coord_step = oot_coords[:, t+1:t+2].to(config.device).repeat_interleave(K, dim=0)
                        gt_hour_step = oot_hours[:, t+1:t+2].to(config.device).repeat_interleave(K, dim=0)

                        seq_next = torch.cat([seq_exp, flat_ids], dim=1)
                        times_next = torch.cat([times_exp, gt_time_step], dim=1)
                        coords_next = torch.cat([coords_exp, gt_coord_step], dim=1)
                        hours_next = torch.cat([hours_exp, gt_hour_step], dim=1)

                        next_target_time = oot_times[:, t+2:t+3].to(config.device).repeat_interleave(K, dim=0)
                        next_target_region = batch_regions_gt[:, t+2:t+3].to(config.device).repeat_interleave(K, dim=0)

                        lookahead_logits = model.predict_next(
                            home_exp,
                            kg_exp,
                            seq_next,
                            times_next,
                            coords_next,
                            next_target_time,
                            next_target_region,
                            hours_next,
                            alpha=None,
                        )
                        lookahead_logits = lookahead_logits / temperature
                        lookahead_best = lookahead_logits.max(dim=1).values.reshape(B, K)

                        rerank_score = topk_vals + (rerank_weight * lookahead_best)
                        best_idx = rerank_score.argmax(dim=1, keepdim=True)
                        next_token_step = torch.gather(topk_ids, 1, best_idx)
                    else:
                        # Default deterministic greedy decode + optional transition-prior rerank.
                        trans_w = float(getattr(config, 'decode_trans_weight', 0.0))
                        trans_topk = int(getattr(config, 'decode_trans_topk', 0))
                        if transition_prior is not None and trans_w > 0 and trans_topk > 1:
                            Kt = min(trans_topk, cand_ids.size(1))
                            top_ids = cand_ids[:, :Kt]
                            top_vals = cand_vals[:, :Kt]
                            bonus = torch.zeros_like(top_vals)

                            prev_tok = curr_seq[:, -1]
                            B = top_ids.size(0)
                            for b in range(B):
                                row = transition_prior.get(int(prev_tok[b].item()))
                                if row is None:
                                    continue
                                for k in range(Kt):
                                    cid = int(top_ids[b, k].item())
                                    bonus[b, k] = row.get(cid, 0.0)

                            rerank = top_vals + (trans_w * bonus)
                            best_idx = rerank.argmax(dim=1, keepdim=True)
                            next_token_step = torch.gather(top_ids, 1, best_idx)
                        else:
                            next_token_step = cand_ids[:, 0].unsqueeze(1)
                
                # next_token_step is [B, 1] (It's a tensor of IDs)
                next_token = next_token_step.to(config.device).long()
                
                # 3. Append to Sequence
                generated_trajectory.append(next_token)
                
                # Update Context for next step (Accumulate history)
                curr_seq = torch.cat([curr_seq, next_token], dim=1)
                
                # Use Ground Truth Context for next step (Teacher Forcing on environment)
                curr_regions = torch.cat([curr_regions, batch_regions_gt[:, t+1:t+2].to(config.device)], dim=1)
                curr_coords = torch.cat([curr_coords, oot_coords[:, t+1:t+2].to(config.device)], dim=1)
                curr_times = torch.cat([curr_times, oot_times[:, t+1:t+2].to(config.device)], dim=1)
                curr_hours = torch.cat([curr_hours, oot_hours[:, t+1:t+2].to(config.device)], dim=1)
        
            # Concatenate predictions: (B, L)
            full_pred = torch.cat(generated_trajectory, dim=1)
            
            # --- Strict Base Code Metric Logic ---
            B = gt_seq.size(0)
            for i in range(B):
                sample_pred = full_pred[i].cpu()
                sample_target = gt_seq[i].cpu()
                
                # 1. Exclude padded
                non_padded_indices = sample_target != 0
                sample_pred = sample_pred[non_padded_indices]
                sample_target = sample_target[non_padded_indices]
                
                # 2. [NEW] Full-Metrics (Evaluate on the entire sequence INCLUDING Start/End)
                if sample_target.numel() > 1:
                    # sample_pred contains [Start, P1, P2, ..., Pn]
                    # We want to form the full trajectory: Start -> P1...Pn -> End
                    # But generated_trajectory starts with [curr_seq (Start)] and predicts P1..Pn
                    # So sample_pred is ALREADY [Start, P1...Pn]
                    # The only missing part is the END token if we are not predicting it but teacher-forcing context.
                    # Wait, generated_trajectory loop ran for (seq_len - 1) times.
                    # If target is [S, A, B, E], len=4. Loop runs 3 times.
                    # Gen: [S] -> [S, A] -> [S, A, B] -> [S, A, B, E_pred]
                    # So full_pred SHOULD potentially naturally align if trained well.
                    # BUT our loop logic: start with curr_seq=gt[:, :1]. Loop t in range(seq_len-1).
                    # t=0: predict 2nd item (index 1). t=1: predict 3rd (index 2).
                    # So full_pred length is exactly seq_len.
                    
                    alt_sample_pred = torch.cat((sample_target[:1], sample_pred[1:-1], sample_target[-1:]), dim=0)
                else:
                    alt_sample_pred = sample_pred

                batch_full_f1.append(f1_score(sample_target.tolist(), alt_sample_pred.tolist()))
                batch_full_pairs_f1.append(pairs_f1_score(sample_target, alt_sample_pred))

                # 4. Calculate F1 / Pairs-F1 on [1:-1] (Intermediate Steps Only - "Strict")
                if sample_target.numel() > 2:
                    t_eval = sample_target[1:-1]
                    p_eval = sample_pred[1:-1]
                    s_f1 = f1_score(t_eval, p_eval)
                    s_pairs = pairs_f1_score(t_eval, p_eval)
                else:
                    s_f1 = 0.0
                    s_pairs = 0.0
                
                batch_alt_f1.append(s_f1)
                batch_alt_pairs_f1.append(s_pairs)
                
                # 5. Repetition Rate on Alt Sequence
                rep = count_adjacent_repetition_rate(alt_sample_pred)
                repetition_list.append(rep)

    avg_f1 = np.mean(batch_alt_f1) # Strict (Intermediate)
    avg_pairs_f1 = np.mean(batch_alt_pairs_f1) # Strict (Intermediate)
    
    # [NEW] Full Metrics
    avg_full_f1 = np.mean(batch_full_f1)
    avg_full_pairs_f1 = np.mean(batch_full_pairs_f1)
    
    avg_rep = np.mean(repetition_list)
    
    print(f"[Results] F1: {avg_f1:.4f} | Pairs-F1: {avg_pairs_f1:.4f} | Full-F1: {avg_full_f1:.4f} | Full-Pairs-F1: {avg_full_pairs_f1:.4f} | Repetition: {avg_rep:.4f}")

    return {
        "F1": avg_f1,
        "Pairs_F1": avg_pairs_f1,
        "Full_F1": avg_full_f1,
        "Full_Pairs_F1": avg_full_pairs_f1,
        "Repetition": avg_rep
    }

def main(config_module_name: str | None = None):
    Config = get_config_class(config_module_name)
    config = Config()

    # Allow external one-by-one ablation scripts to override config safely.
    overrides_json = os.environ.get("KSD_OVERRIDES_JSON", "").strip()
    if overrides_json:
        try:
            overrides = json.loads(overrides_json)
            if isinstance(overrides, dict):
                print(f"Applying overrides from KSD_OVERRIDES_JSON: {overrides}")
                config.update(overrides)
            else:
                print("[WARN] KSD_OVERRIDES_JSON is not a dict, ignored.")
        except Exception as e:
            print(f"[WARN] Failed to parse KSD_OVERRIDES_JSON: {e}")
    
    # 0. Set Seed for Reproducibility
    set_seed(config.seed if hasattr(config, 'seed') else 2050)
    
    # 1. Dataset
    full_dataset = FoursquareDataset(config)
    
    # Patch Config with Dataset Stats
    try:
        config.num_cats = len(full_dataset.cat2id)
        print(f"Config: num_cats set to {config.num_cats} from dataset")
    except:
        config.num_cats = 0
        print("Config: num_cats not found in dataset, defaulting to 0")

    # Split policy
    split_mode = getattr(config, 'split_mode', 'random')
    if split_mode == 'pair_aware':
        # Match base_code: always rebuild OD-pair 8:1:1 split and persist it.
        split_file = os.path.join(config.data_dir, 'split_indices_pair_aware.pt')
        print("Creating and saving new pair-aware 8:1:1 split (base_code-aligned)...")
        train_idx, valid_idx, test_idx = build_pair_aware_split_indices(
            full_dataset,
            seed=config.seed if hasattr(config, 'seed') else 2050,
            ratios=(0.8, 0.1, 0.1),
        )
        torch.save((train_idx, valid_idx, test_idx), split_file)
    else:
        # Keep backward compatibility for non-pair-aware mode.
        legacy_split = os.path.join(config.data_dir, 'split_indices.pt')
        if os.path.exists(legacy_split):
            split_file = legacy_split
        else:
            split_file = os.path.join(config.data_dir, 'split_indices_random.pt')

        if os.path.exists(split_file):
            print(f"Loading saved split indices from {split_file}...")
            train_idx, test_idx = torch.load(split_file)
        else:
            print("Creating and saving new random split...")
            train_size = int(0.8 * len(full_dataset))
            indices = torch.randperm(len(full_dataset)).tolist()
            train_idx = indices[:train_size]
            test_idx = indices[train_size:]
            torch.save((train_idx, test_idx), split_file)
        valid_idx = None
    
    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    keep_ratio = float(getattr(config, 'train_home_keep_ratio', 1.0))
    if keep_ratio < 1.0:
        print(f"Applying train-only home sparsity: keep_ratio={keep_ratio:.2f}")
        train_ds = TrainHomeSparsityWrapper(
            train_ds,
            keep_ratio=keep_ratio,
            seed=(config.seed if hasattr(config, 'seed') else 2050),
        )
    valid_ds = torch.utils.data.Subset(full_dataset, valid_idx) if split_mode == 'pair_aware' else None
    test_ds = torch.utils.data.Subset(full_dataset, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False) if split_mode == 'pair_aware' else None
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    if split_mode == 'pair_aware':
        print(f"Split sizes -> train: {len(train_idx)}, valid: {len(valid_idx)}, test: {len(test_idx)}")

    # Build sparse train-transition prior for pair-focused decode reranking.
    transition_prior = build_transition_prior(full_dataset, train_idx)
    
    # 2. KG pretraining (TransE / TransH by config)
    print(f">>> Phase 1: KG Pre-training ({getattr(config, 'kg_model', 'transE')})")
    transe_model = train_kg_model(full_dataset, config)
    
    # 3. KSD-Diff (Second Phase)
    print(">>> Phase 2: Diffusion Training")
    model = KSDDiff(config, full_dataset).to(config.device)
    
    # Load knowledge from KG model
    print(">>> Phase 2: Loading KG weights into Model...")
    # 1. Load the internal KG module (for Graph reasoning capability)
    model.transe.load_state_dict(transe_model.state_dict())
    
    # 2. Initialize POI Embeddings (DIRECT COPY)
    # Be aware: dataset.py now uses standard IDs from poi_id.pkl to align Model ID == KG entity ID.
    with torch.no_grad():
        trained_weights = transe_model.entity_emb.weight.data
        # Copy weights where IDs match
        limit = min(model.poi_embedding.weight.size(0), trained_weights.size(0))
        model.poi_embedding.weight.data[:limit] = trained_weights[:limit]
        print(f">>> Direct Weight Transfer: Initialized {limit} POI Embeddings from KG model.")

    # [UPDATED] Added Weight Decay for Regularization
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=getattr(config, 'weight_decay', 0.0))
    
    # [NEW] Learning Rate Scheduler
    # Dynamically reduce LR when F1 stops improving to find the loss minimum
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Early Stopping
    best_score = -1.0
    best_pairs_f1 = -1.0
    best_full_f1 = -1.0
    best_full_pairs = -1.0
    min_train_loss = float('inf') # Track minimum loss model
    patience = 0
    
    for epoch in range(config.epochs):
        train_loss, last_loss_dict = train_one_epoch(model, train_loader, optimizer, config)
        
        # Save Min Loss Model
        if train_loss < min_train_loss:
            min_train_loss = train_loss
            torch.save(model.state_dict(), 'best_loss_model.pth')
            print(f" >> [SAVE] New Min Loss: {min_train_loss:.4f} -> Overwriting 'best_loss_model.pth'")
        
        # [FIX] Handle dynamic loss keys safely
        rec_loss = last_loss_dict.get('rec', last_loss_dict.get('uni_ce', 0.0))
        align_loss = last_loss_dict.get('align', 0.0)
        gate_loss = last_loss_dict.get('gate', 0.0)
        
        print(f"Epoch {epoch+1} Loss: {train_loss:.4f} "
              f"[Diff: {last_loss_dict['diff']:.4f}, Align: {align_loss:.4f}, Gate: {gate_loss:.4f}, Rec: {rec_loss:.4f}]")
        
        # Validation (pair_aware uses explicit valid split like base_code).
        eval_loader = valid_loader if split_mode == 'pair_aware' else test_loader
        res = evaluate(model, eval_loader, config, transition_prior)
        print(f"Eval (Epoch {epoch+1}): F1={res['F1']:.4f}, Pairs-F1={res['Pairs_F1']:.4f}, Full-F1={res['Full_F1']:.4f}")
        
        # Use harmonic mean to favor balanced gains on both full metrics.
        full_f1 = res['Full_F1']
        full_pairs_f1 = res['Full_Pairs_F1']
        monitor_metric = (2.0 * full_f1 * full_pairs_f1) / (full_f1 + full_pairs_f1 + 1e-12)
        
        # [NEW] Step the scheduler
        scheduler.step(monitor_metric)
        
        non_loss_ckpt_improved = False

        if monitor_metric > best_score:
            best_score = monitor_metric
            non_loss_ckpt_improved = True
            torch.save(model.state_dict(), 'best_ksd_diff.pth')
            print(f" >> [SAVE] New Best Composite: {best_score:.4f} -> Overwriting 'best_ksd_diff.pth'")

        if res['Pairs_F1'] > best_pairs_f1:
            best_pairs_f1 = res['Pairs_F1']
            non_loss_ckpt_improved = True
            torch.save(model.state_dict(), 'best_pairs_f1.pth')
            print(f" >> [SAVE] New Best Pairs-F1: {best_pairs_f1:.4f} -> Overwriting 'best_pairs_f1.pth'")

        if full_f1 > best_full_f1:
            best_full_f1 = full_f1
            non_loss_ckpt_improved = True
            torch.save(model.state_dict(), 'best_full_f1.pth')
            print(f" >> [SAVE] New Best Full-F1: {best_full_f1:.4f} -> Overwriting 'best_full_f1.pth'")

        if full_pairs_f1 > best_full_pairs:
            best_full_pairs = full_pairs_f1
            non_loss_ckpt_improved = True
            torch.save(model.state_dict(), 'best_full_pairs.pth')
            print(f" >> [SAVE] New Best Full-Pairs: {best_full_pairs:.4f} -> Overwriting 'best_full_pairs.pth'")

        if non_loss_ckpt_improved:
            patience = 0
            print("Patience reset: non-minLoss checkpoint improved.")
        else:
            patience += 1
            print(f"No non-minLoss checkpoint improve. (Patience: {patience}/{config.patience})")
            
        if patience >= config.patience:
            print("Early Stopping Triggered.")
            break
            
    # Final candidate evaluation and target-aware model selection.
    eval_seed = int(config.seed)
    candidates = [
        ('best_ksd_diff.pth', 'Best Composite'),
        ('best_pairs_f1.pth', 'Best Pairs-F1'),
        ('best_full_pairs.pth', 'Best Full-Pairs'),
        ('best_full_f1.pth', 'Best Full-F1'),
        ('best_loss_model.pth', 'Min Loss'),
    ]

    candidate_stats = []
    print(f"\n>>> Final Candidate Evaluation (single seed={eval_seed})")
    for path, name in candidates:
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found")
            continue

        # Safe load: only load parameters whose names and shapes match the current model.
        raw_ckpt = torch.load(path)
        # support checkpoints that are either state_dict or wrapped dicts
        if isinstance(raw_ckpt, dict) and 'state_dict' in raw_ckpt:
            ckpt = raw_ckpt['state_dict']
        else:
            ckpt = raw_ckpt

        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in ckpt.items():
            if k in model_state:
                try:
                    if v.shape == model_state[k].shape:
                        filtered[k] = v
                    else:
                        skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                except Exception:
                    skipped.append((k, None, tuple(model_state[k].shape)))
            else:
                skipped.append((k, 'MISSING', None))

        if len(filtered) == 0:
            print(f"[WARN] No matching parameters found in checkpoint {path}. Skipping load.")
        else:
            model_state.update(filtered)
            model.load_state_dict(model_state)
            print(f"[INFO] Loaded {len(filtered)}/{len(ckpt)} params from {path} (others skipped).")

        if len(skipped) > 0:
            print(f"[DEBUG] Skipped {len(skipped)} parameters due to shape/name mismatch. Examples: {skipped[:5]}")
        set_seed(eval_seed)
        res = evaluate(model, test_loader, config, transition_prior)

        stat = {
            'name': name,
            'path': path,
            'seed': eval_seed,
            'F1': float(res['F1']),
            'Pairs': float(res['Pairs_F1']),
            'Full_F1': float(res['Full_F1']),
            'Full_Pairs': float(res['Full_Pairs_F1']),
        }
        stat['target_score'] = 0.45 * stat['Full_F1'] + 0.55 * stat['Full_Pairs']
        candidate_stats.append(stat)
        print(
            f"[{name}] F1={stat['F1']:.4f} | Pairs={stat['Pairs']:.4f} | "
            f"Full-F1={stat['Full_F1']:.4f} | Full-Pairs={stat['Full_Pairs']:.4f} | "
            f"Score={stat['target_score']:.4f}"
        )

    if len(candidate_stats) == 0:
        print("[WARN] No checkpoint candidates found for final selection.")
        return 0.0

    best_final = max(candidate_stats, key=lambda x: x['target_score'])
    best_by_pairs = max(candidate_stats, key=lambda x: x['Pairs'])
    print("\nSummary Comparison:")
    for s in candidate_stats:
        print(
            f"{s['name']} -> F1: {s['F1']:.4f}, Pairs: {s['Pairs']:.4f}, "
            f"Full-F1: {s['Full_F1']:.4f}, Full-Pairs: {s['Full_Pairs']:.4f}"
        )

    print(
        f"\nFinal Selected by Target Score: {best_final['name']} ({best_final['path']}) "
        f"with Seed={best_final['seed']}, Full-F1={best_final['Full_F1']:.4f}, Full-Pairs={best_final['Full_Pairs']:.4f}"
    )
    print(
        f"Final Selected by Pairs-F1: {best_by_pairs['name']} ({best_by_pairs['path']}) "
        f"with Seed={best_by_pairs['seed']}, Pairs={best_by_pairs['Pairs']:.4f}, F1={best_by_pairs['F1']:.4f}"
    )
    return best_final['F1']

def run_experiment(config):
    """
    Runs a complete training cycle with the given config.
    Returns: Best F1 Score achieved.
    """
    # 0. Set Seed
    set_seed(config.seed if hasattr(config, 'seed') else 2024)
    
    # 1. Dataset
    full_dataset = FoursquareDataset(config)
    try:
        config.num_cats = len(full_dataset.cat2id)
    except:
        config.num_cats = 0
    
    split_mode = getattr(config, 'split_mode', 'random')
    if split_mode == 'pair_aware':
        train_idx, valid_idx, test_idx = build_pair_aware_split_indices(
            full_dataset,
            seed=config.seed if hasattr(config, 'seed') else 2024,
            ratios=(0.8, 0.1, 0.1),
        )
        train_ds = torch.utils.data.Subset(full_dataset, train_idx)
        valid_ds = torch.utils.data.Subset(full_dataset, valid_idx)
        test_ds = torch.utils.data.Subset(full_dataset, test_idx)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    else:
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        valid_loader = None
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    
    # 2. KG pretraining (TransE / TransH)
    transe_model = train_kg_model(full_dataset, config)
    
    # 3. Model
    model = KSDDiff(config, full_dataset).to(config.device)
    model.transe.load_state_dict(transe_model.state_dict())
    
    # Init Weights
    with torch.no_grad():
        trained_weights = transe_model.entity_emb.weight.data
        limit = min(model.poi_embedding.weight.size(0), trained_weights.size(0))
        model.poi_embedding.weight.data[:limit] = trained_weights[:limit]

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=getattr(config, 'weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    best_f1 = -1.0
    patience = 0
    
    # Reduced epochs for search efficiency
    search_epochs = 50 
    
    for epoch in range(search_epochs):
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, config)
        eval_loader = valid_loader if split_mode == 'pair_aware' else test_loader
        res = evaluate(model, eval_loader, config)
        f1 = res['F1']
        
        scheduler.step(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            # Save strictly the best from search
            torch.save(model.state_dict(), 'best_search_temp.pth')
        else:
            patience += 1
            
        if patience >= 15: # Stricter early stopping for search
            break
            
    print(f"Experiment Finished. Best F1: {best_f1:.4f}")
    return best_f1

def run_random_search():
    import random
    
    # Define Search Space
    search_space = {
        'lr': [1e-3, 5e-4],
        'lambda_align': [0.5, 1.0, 5.0],
        'dropout': [0.3, 0.5],
        'hidden_dim': [512, 1024]
        # Keep lambda_diff=0.5, embed_dim=128 fixed as known goods
    }
    
    num_trials = 5
    best_score = -1.0
    best_params = {}
    
    print(f">>> STARTING RANDOM SEARCH ({num_trials} Trials) <<<")
    
    for i in range(num_trials):
        # Sample Config
        trial_params = {k: random.choice(v) for k, v in search_space.items()}
        print(f"\n=== Trial {i+1}/{num_trials} ===")
        print(f"Params: {trial_params}")
        
        # Instantiate Config and Update
        Config = get_config_class()
        conf = Config()
        conf.update(trial_params)
        
        score = run_experiment(conf)
        
        if score > best_score:
            best_score = score
            best_params = trial_params.copy()
            # Save the winner
            if os.path.exists('best_search_temp.pth'):
                if os.path.exists('best_search_winner.pth'):
                    os.remove('best_search_winner.pth')
                os.rename('best_search_temp.pth', 'best_search_winner.pth')
            print(f"--> New Global Best: {best_score:.4f}")
            
    print("\n" + "="*50)
    print(f"SEARCH COMPLETE.")
    print(f"Best Score: {best_score:.4f}")
    print(f"Best Params: {best_params}")
    print("Best model saved to 'best_search_winner.pth'")
    print("="*50)

if __name__ == "__main__":
    args = parse_args()
    main(args.config)