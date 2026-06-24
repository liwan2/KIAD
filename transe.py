import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim, margin=1.0):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.margin = margin
        
        self.entity_emb = nn.Embedding(num_entities + 1, embed_dim)
        self.relation_emb = nn.Embedding(num_relations + 1, embed_dim)
        
        # Init
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        
        # Margin Loss
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='mean') # Note: sum vs mean

    def _calc_score(self, h, r, t):
        # h + r - t
        # Norm L2
        return torch.norm(h + r - t, p=2, dim=-1)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """
        pos_triplets: (Batch, 3)
        neg_triplets: (Batch, 3)
        """
        ph = self.entity_emb(pos_h)
        pr = self.relation_emb(pos_r)
        pt = self.entity_emb(pos_t)
        
        nh = self.entity_emb(neg_h)
        nr = self.relation_emb(neg_r)
        nt = self.entity_emb(neg_t)
        
        pos_score = self._calc_score(ph, pr, pt)
        neg_score = self._calc_score(nh, nr, nt)
        
        # Loss: max(0, margin + pos - neg)
        # Target for MarginRankingLoss is -1 for (x1, x2) -> max(0, -y*(x1-x2) + margin)
        # If y=1, loss = max(0, -x1 + x2 + margin) = max(0, margin + x2 - x1) -> Bad?
        # Standard: L = max(0, margin + pos - neg). Pos should be LOW, Neg should be HIGH.
        # Form: margin + (pos_dist - neg_dist)
        # PyTorch: input1 (pos), input2 (neg), target (1 or -1).
        # loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
        # Use y=-1: max(0, 1 * (pos - neg) + margin)
        
        target = torch.ones_like(pos_score) * -1
        loss = self.criterion(pos_score, neg_score, target)
        
        return loss

    def predict(self, h, r, t):
        ph = self.entity_emb(h)
        pr = self.relation_emb(r)
        pt = self.entity_emb(t)
        return self._calc_score(ph, pr, pt)


class TransH(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim, margin=1.0):
        super(TransH, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embed_dim = embed_dim
        self.margin = margin

        self.entity_emb = nn.Embedding(num_entities + 1, embed_dim)
        self.relation_emb = nn.Embedding(num_relations + 1, embed_dim)
        self.normal_emb = nn.Embedding(num_relations + 1, embed_dim)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.normal_emb.weight)

        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='mean')

    def _project(self, e, w):
        # Project entity vector onto relation-specific hyperplane.
        w = F.normalize(w, p=2, dim=-1)
        return e - torch.sum(e * w, dim=-1, keepdim=True) * w

    def _calc_score(self, h, r, t, w):
        h_proj = self._project(h, w)
        t_proj = self._project(t, w)
        return torch.norm(h_proj + r - t_proj, p=2, dim=-1)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        ph = self.entity_emb(pos_h)
        pr = self.relation_emb(pos_r)
        pt = self.entity_emb(pos_t)
        pw = self.normal_emb(pos_r)

        nh = self.entity_emb(neg_h)
        nr = self.relation_emb(neg_r)
        nt = self.entity_emb(neg_t)
        nw = self.normal_emb(neg_r)

        pos_score = self._calc_score(ph, pr, pt, pw)
        neg_score = self._calc_score(nh, nr, nt, nw)

        target = torch.ones_like(pos_score) * -1
        loss = self.criterion(pos_score, neg_score, target)
        return loss

    def predict(self, h, r, t):
        eh = self.entity_emb(h)
        er = self.relation_emb(r)
        et = self.entity_emb(t)
        ew = self.normal_emb(r)
        return self._calc_score(eh, er, et, ew)


def _resolve_kg_model_name(config):
    return str(getattr(config, 'kg_model', 'transE')).strip().lower()


def create_kg_model(num_entities, num_relations, config):
    name = _resolve_kg_model_name(config)
    margin = getattr(config, 'transe_margin', 1.0)
    if name == 'transh':
        return TransH(num_entities, num_relations, config.embed_dim, margin)
    return TransE(num_entities, num_relations, config.embed_dim, margin)

def train_transe(dataset, config):
    """
    Offline training for TransE
    """
    model_name = _resolve_kg_model_name(config)
    print(f"Starting KG Training ({model_name})...")
    # Prepare KG triples buffer
    triples = []
    # dataset.kg_adj is {h: [(r, t), ...]}
    for h, rels in dataset.kg_adj.items():
        try:
            h_int = int(h) # Assuming int mapping
            for r, t in rels:
                triples.append((h_int, int(r), int(t)))
        except:
            continue
            
    if not triples:
        print("No valid KG triples found. Skipping KG training.")
        return create_kg_model(100, 10, config) # Dummy return

    # Model
    # Determine max IDs
    max_ent = max([max(t[0], t[2]) for t in triples]) + 1
    max_rel = max([t[1] for t in triples]) + 1
    
    # Update dataset num entities if needed (might be larger if mapped poorly)
    dataset.num_entities = max(dataset.num_entities, max_ent)
    
    kg_model = create_kg_model(dataset.num_entities, max_rel, config).to(config.device)
    optimizer = optim.Adam(kg_model.parameters(), lr=config.lr)
    
    bs = config.batch_size
    num_batches = len(triples) // bs
    
    for epoch in range(5): # Short training 5 epochs for demo
        random.shuffle(triples)
        total_loss = 0
        
        for i in range(num_batches):
            batch = triples[i*bs : (i+1)*bs]
            
            # Simple negative sampling: corrupt tail
            pos_h = torch.tensor([x[0] for x in batch], device=config.device)
            pos_r = torch.tensor([x[1] for x in batch], device=config.device)
            pos_t = torch.tensor([x[2] for x in batch], device=config.device)
            
            # Corrupt tail
            neg_t = torch.randint(0, dataset.num_entities, (bs,), device=config.device)
            
            optimizer.zero_grad()
            loss = kg_model(pos_h, pos_r, pos_t, pos_h, pos_r, neg_t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"KG Epoch {epoch+1}, Loss: {total_loss/num_batches:.4f}")
        
    return kg_model


def train_kg_model(dataset, config):
    """Backward-compatible alias for configurable KG pretraining."""
    return train_transe(dataset, config)

