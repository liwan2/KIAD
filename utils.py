import torch
import torch.nn.functional as F
import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def random_choice_by_probability(probability_list):
    """
    Selects an index from a list of probabilities based on their values.
    """
    cumulative_probabilities = []
    cumulative_prob = 0
    for prob in probability_list:
        cumulative_prob += prob
        cumulative_probabilities.append(cumulative_prob)

    random_number = random.random()

    for i, cumulative_prob in enumerate(cumulative_probabilities):
        if random_number <= cumulative_prob:
            return i
    return len(probability_list) - 1

def select_top_p_indices(probabilities, threshold=0.8):
    """
    Selects indices that sum up to a cumulative probability threshold.
    Returns the index (in the sorted list) where the threshold is crossed.
    """
    sorted_indices = np.argsort(probabilities)[::-1]  # re-order the probability
    cumulative_prob = 0.0
    selected_indices = []

    for idx in sorted_indices:
        cumulative_prob += probabilities[idx]
        selected_indices.append(idx)
        if cumulative_prob >= threshold:
            break

    return selected_indices[-1]

def top_np_recommendation(batch_candidate, batch_similarity, confidence=0.5, threshold=0.8):
    """
    Top-NP Recommendation strategy from base_code.
    
    Args:
        batch_candidate: Tensor [B, L, K], containing POI indices for Top-K candidates.
        batch_similarity: Tensor [B, L, K], containing raw scores/logits for Top-K candidates.
        confidence: float, temperature scaling factor.
        threshold: float, probability mass threshold (Nucleus Sampling).
    
    Returns:
        top_candidates: Tensor [B, L], the recommended POI indices.
    """
    # Move to CPU for loop processing
    batch_candidate = batch_candidate.cpu()
    batch_similarity = batch_similarity.cpu()
    
    B, L, K = batch_candidate.shape
    top_candidates = torch.zeros((B, L), dtype=batch_candidate.dtype)

    for batch in range(B):
        for middle_index in range(L):
            
            # 1. Apply Temperature (Confidence) to Logits -> Probabilities
            probs = F.softmax(batch_similarity[batch, middle_index] * confidence, dim=0)
            batch_similarity[batch, middle_index] = probs

            # 2. Select Top-P Cutoff
            top_p_idx = select_top_p_indices(batch_similarity[batch, middle_index].tolist(), threshold)
            
            # 3. Renormalize the truncated distribution
            cutoff = top_p_idx + 1 
            subset = batch_similarity[batch, middle_index, :cutoff]
            # base_code re-softmax with confidence
            subset_new = F.softmax(subset * confidence, dim=0)
            
            batch_similarity[batch, middle_index, :cutoff] = subset_new
            batch_similarity[batch, middle_index, cutoff:] = 0.0

            # 4. Random Choice
            batch_probability_list = batch_similarity[batch, middle_index].tolist()
            nonzero_probability_list = [x for x in batch_probability_list if x != 0]

            new_top_p_index = random_choice_by_probability(nonzero_probability_list)
            
            top_candidates[batch, middle_index] = batch_candidate[batch, middle_index, new_top_p_index]

    return top_candidates

class Metrics:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.hits_k = {1: [], 5: [], 10: []}
        self.count = 0
        
    def update(self, pred_logits, target_idx):
        """
        Calculates Step-wise accuracy (Next POI Recommendation).
        pred_logits: (B, Num_POI) - Similarity Scores / Probabilities
        target_idx: (B,) - The true next POI ID
        """
        B = pred_logits.size(0)
        
        # Don't compute for padding (0)
        mask = (target_idx != 0)
        if mask.sum() == 0: return
        
        # Filter valid items
        valid_logits = pred_logits[mask]
        valid_target = target_idx[mask]
        
        # Get Top-K
        # We need up to K=10
        _, top_k = torch.topk(valid_logits, 10, dim=1) # (B_valid, 10)
        
        for k in [1, 5, 10]:
            # Check if target is in first k columns
            pred_k = top_k[:, :k]
            # valid_target: (B_valid,) -> (B_valid, 1)
            target_expanded = valid_target.view(-1, 1)
            
            # Hit: (B_valid, k) boolean
            hits = (pred_k == target_expanded).sum(dim=1) # Should be 0 or 1
            
            # Store 1.0 or 0.0 for each sample
            self.hits_k[k].extend(hits.cpu().tolist())
            
        self.count += mask.sum().item()

    def compute(self):
        
        metrics = {}
        for k in [1, 5, 10]:
            if len(self.hits_k[k]) == 0:
                metrics[f"Recall@{k}"] = 0.0
                metrics[f"F1@{k}"] = 0.0
                continue
                
            hits = np.sum(self.hits_k[k])
            total = len(self.hits_k[k])
            
            recall = hits / total
            precision = hits / (total * k)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
                
            metrics[f"Recall@{k}"] = recall
            metrics[f"F1@{k}"] = f1
            
        return metrics

def flatten_preds(logits, target):
    """Helper to flatten outputs for Loss calculation if needed"""
    return logits.view(-1, logits.size(-1)), target.view(-1)
