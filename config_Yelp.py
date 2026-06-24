import torch
from pathlib import Path

class Config:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent

        # Paths
        # Switch between datasets here. 
        # Option 1: project_root / 'Foursquare'
        # Option 2: project_root / 'Yelp'
        self.dataset_option = 'Yelp' 
        
        if self.dataset_option == 'Foursquare':
            self.data_dir = str(project_root / 'Foursquare')
        elif self.dataset_option == 'Yelp':
            self.data_dir = str(project_root / 'Yelp')
        else:
            # Change this to your second dataset path
            self.data_dir = str(project_root / 'YourSecondDataset')
            
        self.home_file = 'home.txt'
        self.oot_file = 'oot.txt'
        self.travel_file = 'travel.txt'
        self.kg_file = 'kg.txt'
        
        # Training Params
        self.batch_size = 128     
        self.epochs = 500         
        self.lr = 0.0008          # [BEST] Stable convergence near best full metrics,0.0008
        self.patience = 10        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 2024          

        # Model Params
        self.embed_dim = 128      
        self.hidden_dim = 1024    
        self.num_cats = 0
        self.dropout = 0.2        
        self.kg_k_hop = 1         
        self.max_k_neighbors = 10 

        # Ablation toggles (A/B/C). Keep all disabled by default.
        # A: Static module (KIAM output)
        # B: Dynamic module (STAPM output)
        # C: Diffusion enhancement branch (CPDiff sampling + gate residual)
        self.ablate_module_a = False
        self.ablate_module_b = False
        self.ablate_module_c = False

        # Replacement ablation toggles (A/B/C).
        # Replace the corresponding module with base_code-style counterpart.
        self.replace_module_a_with_base = False
        self.replace_module_b_with_base = False
        self.replace_module_c_with_base = False

        # Loss Weights (USER REQUEST: Short Seq + Strong Recon)
        # KG model switch: 'transE' (default) | 'transH'
        # Keep default as transE for one-line rollback.
        self.kg_model = 'transE'
        self.transe_margin = 1.0
        self.lambda_kg = 0.1      
        self.lambda_cl = 0.2      
        self.lambda_rec = 2.1     # Slightly stronger reconstruction to recover Full-F1
        self.lambda_order = 1.9   #1.9 Stable setting before sequence-level loss experiments
        self.lambda_global_order = 0.0  # Disable sequence-level order consistency by default
        self.global_order_margin = 1.0
        self.lambda_diff = 1.0    
        self.lambda_align = 3.0   # [BEST] Most stable alignment weight
        self.weight_decay = 1e-4  

        # STAPM
        self.stiefel_k = 64       # [USER REQUEST] 64 is optimal
        self.rnn_layers = 1       
        
        # CP-Diff
        self.diff_steps = 100     
        self.beta_start = 0.0001
        self.beta_end = 0.02
        
        # Data Params
        self.max_seq_len = 20     # [USER REQUEST] Short Context
        self.oot_seq_len = 12     # [ROLLBACK] 14 degraded full metrics on Yelp

        # Inference/Decoding Params (for one-by-one tuning)
        self.decode_temperature = 0.45#0.4 is good ,0.5 is a little worse;0.45 is a good balance for both datasets.
        self.decode_dist_penalty = 18.0#the best setting for distance penalty is 18.0.
        self.decode_hist_penalty = 5.0#5
        self.decode_last_penalty = 55.0#48
        self.decode_loop2_penalty = 10.0
        self.decode_trans_weight = 0.02
        self.decode_trans_topk = 12
        self.decode_mode = 'greedy'   # 'greedy' | 'topnp'
        self.decode_mc_samples = 1    # keep optional; default single-sample decode is stronger
        self.decode_rerank_enable = False
        self.decode_rerank_topk = 3
        self.decode_rerank_weight = 0.15
        self.decode_np_confidence = 1.0
        self.decode_np_threshold = 0.8

        # Data split mode aligned with base_code split policy.
        self.split_mode = 'pair_aware'    # 'pair_aware' | 'random'

        # Train-only home check-in sparsity control.
        # Keep ratio of non-padding home check-ins in training set only.
        self.train_home_keep_ratio = 1.0

    def update(self, params_dict):
        """Helper to update params from search"""
        for k, v in params_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
                print(f"Config Updated: {k} -> {v}")
            else:
                print(f"Warning: Config has no attribute {k}")
        self.beta_end = 0.02
