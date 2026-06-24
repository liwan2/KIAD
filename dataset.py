import os
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random
import pickle
import numpy as np
import pytz
from datetime import datetime, timezone

class FoursquareDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        
        # --- Vocabularies ---
        # Internal Mappings for the Model
        self.user2id = {'<PAD>': 0}
        self.poi2id = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        self.id2poi = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}
        self.cat2id = {'<PAD>': 0}
        self.relation2id = {'<PAD>': 0}
        self.city2id = {'<PAD>': 0} # From Travel.txt
        
        # --- Feature Containers (Indexed by My_POI_ID) ---
        self.poi_coords = None   # FloatTensor (N, 2)
        self.poi_regions = None  # LongTensor  (N)
        self.num_regions = 1
        self.num_entities = 0 # Max Entity ID in KG
        
        # --- Raw Metadata Containers ---
        self.raw_poi_map = {} # StrID -> OrigIntID (from poi_id.pkl)
        self.city_tz = {}     # City -> Timezone (from city_tz_mapping.pkl)
        
        # --- Sequence Data (Indexed by Dataset Sample Index) ---
        self.samples = [] # List of tuples/dicts
        
        self.kg_adj = defaultdict(list)
        self.poi_to_kg_entity = {}

        # --- Loading Process ---
        print(f"[{mode.upper()}] Initializing Dataset...")
        
        # 1. Load Metadata FIRST (Need poi_id.pkl for consistent ID mapping)
        self._load_metadata()
        
        # 2. Build Vocab from files (Home, OOT, Travel) using existing IDs
        self._build_vocab()
        
        # 3. Load KG
        self._load_kg()

        # 4. Process Trips
        self._load_trips()
        
    def _load_metadata(self):
        print(">>> Loading External Metadata (Pickle Files)...")
        
        # Load poi_id.pkl (RawStr -> OrigInt)
        pid_path = os.path.join(self.config.data_dir, 'poi_id.pkl')
        if os.path.exists(pid_path):
            try:
                with open(pid_path, 'rb') as f:
                    self.raw_poi_map = pickle.load(f)
                # Ensure we track the max ID to size the embedding layer correctly
                max_orig = max(self.raw_poi_map.values()) if self.raw_poi_map else 0
                self.num_entities = max(self.num_entities, max_orig + 1)
                print(f"Loaded mapping for {len(self.raw_poi_map)} POIs. Max ID: {max_orig}")
            except: pass
            
        # Load city_tz_mapping.pkl
        tz_path = os.path.join(self.config.data_dir, 'city_tz_mapping.pkl')
        if os.path.exists(tz_path):
            try:
                with open(tz_path, 'rb') as f:
                    self.city_tz = dict(pickle.load(f))
                print(f"Loaded Timezones for {len(self.city_tz)} cities.")
            except: pass
            
    def _build_vocab(self):
        print(">>> Building vocabularies from txt files...")
        
        # 1. Scan Check-in Files (Home/OOT)
        idx_user, idx_cat = 1, 1
        
        for fname in [self.config.home_file, self.config.oot_file]:
            path = os.path.join(self.config.data_dir, fname)
            if not os.path.exists(path): continue
            
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 6: continue
                    # Format: TripID, UserID, CityName, POI_ID, Time, Type
                    uid, vid, cat = parts[1], parts[3], parts[5]
                    
                    if uid not in self.user2id:
                        self.user2id[uid] = idx_user
                        idx_user += 1
                    
                    # LINK: Use consistent ID from poi_id.pkl
                    if vid not in self.poi2id:
                        if vid in self.raw_poi_map:
                            self.poi2id[vid] = self.raw_poi_map[vid]
                            self.id2poi[self.raw_poi_map[vid]] = vid
                        else:
                            # Skip or Warning? For now skip to be safe/consistent with KG.
                            # Or assign Max+1? Let's skip to align with "Strict".
                            pass

                    if cat not in self.cat2id:
                        self.cat2id[cat] = idx_cat
                        idx_cat += 1

        # 2. Scan Travel File (Cities)
        idx_city = 1
        travel_path = os.path.join(self.config.data_dir, self.config.travel_file)
        if os.path.exists(travel_path):
            with open(travel_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 4: continue
                    orig, dest = parts[2], parts[3]
                    
                    if orig not in self.city2id:
                        self.city2id[orig] = idx_city
                        idx_city += 1
                    if dest not in self.city2id:
                        self.city2id[dest] = idx_city
                        idx_city += 1
                        
        self.num_users = len(self.user2id)
        # Num POIs is not len(poi2id) but Max ID + 1 to accomodate sparse IDs
        max_id = max(self.poi2id.values()) if self.poi2id else 0
        self.num_pois = max(max_id, self.num_entities) + 1
        self.num_cities = len(self.city2id)
        print(f"Vocab Stats: {self.num_users} Users, Max POI ID {self.num_pois-1}, {self.num_cities} Cities.")
        
        # Initialize Feature Tensors AFTER we know num_pois
        self._init_features()
        
        # Populate Category Features (Second Pass or during Build)
        # We need to re-scan or store in memory during build.
        # Let's simple re-scan for robustness.
        for fname in [self.config.home_file, self.config.oot_file]:
            path = os.path.join(self.config.data_dir, fname)
            if not os.path.exists(path): continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 6: continue
                    vid, cat = parts[3], parts[5]
                    if vid in self.raw_poi_map:
                         orig_id = self.raw_poi_map[vid]
                         if orig_id < self.num_pois and cat in self.cat2id:
                             self.poi_cats[orig_id] = self.cat2id[cat]

    def _init_features(self):
        self.poi_coords = torch.zeros((self.num_pois, 2), dtype=torch.float32)
        self.poi_regions = torch.zeros(self.num_pois, dtype=torch.long)
        self.poi_cats = torch.zeros(self.num_pois, dtype=torch.long)
        
        # Load Coordinates (OrigInt -> (Lat, Lon))
        coord_path = os.path.join(self.config.data_dir, 'poi_coord.pkl')
        if os.path.exists(coord_path):
            try:
                with open(coord_path, 'rb') as f:
                    orig2coord = pickle.load(f) 
                
                count = 0
                for orig_id, coords in orig2coord.items():
                    if orig_id < self.num_pois:
                        lat, lon = coords
                        self.poi_coords[orig_id] = torch.tensor([lat, lon])
                        count += 1
                
                # Normalize
                if count > 0:
                     max_val = self.poi_coords.max(dim=0)[0]
                     min_val = self.poi_coords.min(dim=0)[0]
                     denom = max_val - min_val
                     denom[denom == 0] = 1.0
                     self.poi_coords = (self.poi_coords - min_val) / denom
                     self.poi_coords = self.poi_coords * 2 - 1
                print(f"Loaded Coords for {count} POIs.")
            except: pass
            
        # Load Regions
        reg_path = os.path.join(self.config.data_dir, 'region_poi.pkl')
        max_reg = 0
        if os.path.exists(reg_path):
            try:
                with open(reg_path, 'rb') as f:
                    reg_dict = pickle.load(f)
                
                orig2reg = {}
                for rid, pset in reg_dict.items():
                    max_reg = max(max_reg, rid)
                    for pid in pset:
                        orig2reg[pid] = rid
                
                count = 0
                for orig_id, rid in orig2reg.items():
                    if orig_id < self.num_pois:
                        self.poi_regions[orig_id] = rid
                        count += 1
                            
                self.num_regions = max_reg + 1
                print(f"Loaded Regions for {count} POIs.")
            except: pass
            
    def _load_kg(self):
        # KG load is simpler now, IDs match directly
        print(">>> Loading Knowledge Graph...")
        path = os.path.join(self.config.data_dir, self.config.kg_file)
        if not os.path.exists(path): return

        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                h, r, t = parts[0], parts[1], parts[2]
                
                if r not in self.relation2id:
                    self.relation2id[r] = len(self.relation2id)
                
                try: 
                    h_idx, t_idx = int(h), int(t)
                    # Directly use these IDs
                    if h_idx < self.num_pois and t_idx < self.num_pois:
                        self.kg_adj[h_idx].append((self.relation2id[r], t_idx))
                except: continue
        
        # No need for poi_to_kg_entity map anymore, it is Identity
        self.num_relations = len(self.relation2id)
        print(f"KG Loaded. Relations: {self.num_relations}")

    def _get_local_hour(self, ts, city):
        """Convert UTC timestamp to Local Hour (0-23) based on city timezone."""
        if city not in self.city_tz:
            # Fallback to UTC hour if timezone unknown
            return int((ts % 86400) // 3600)
        
        tz_name = self.city_tz[city]
        try:
            ts_sec = int(ts)
            dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
            local_tz = pytz.timezone(tz_name)
            local_dt = dt_utc.astimezone(local_tz)
            return local_dt.hour
        except:
             return int((ts % 86400) // 3600)

    def _scan_max_lengths(self):
        """Scan data to compute actual max sequence lengths (dynamic)"""
        print(">>> Scanning data for actual max sequence lengths...")

        trip_travel = {}
        trip_home = defaultdict(list)
        trip_oot = defaultdict(list)

        # Load Travel Info
        tp = os.path.join(self.config.data_dir, self.config.travel_file)
        if os.path.exists(tp):
            with open(tp, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 4: continue
                    tid, uid, orig, dest = parts[0], parts[1], parts[2], parts[3]
                    trip_travel[tid] = {'uid': uid, 'orig': orig, 'dest': dest}

        # Load Home Checkins
        hp = os.path.join(self.config.data_dir, self.config.home_file)
        if os.path.exists(hp):
            with open(hp, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 6: continue
                    tid, uid, vid, ts = parts[0], parts[1], parts[3], float(parts[4])
                    if vid in self.poi2id:
                        trip_home[tid].append({'poi': self.poi2id[vid], 'time': ts})

        # Load OOT Checkins
        op = os.path.join(self.config.data_dir, self.config.oot_file)
        if os.path.exists(op):
            with open(op, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 6: continue
                    tid, uid, vid, ts = parts[0], parts[1], parts[3], float(parts[4])
                    if vid in self.poi2id:
                        trip_oot[tid].append({'poi': self.poi2id[vid], 'time': ts})

        # Compute actual max lengths
        max_home_len = 0
        max_oot_len = 0

        for tid, oot_visits in trip_oot.items():
            if tid not in trip_home: continue
            home_visits = trip_home[tid]

            if len(home_visits) < 5 or len(oot_visits) < 2: continue

            max_home_len = max(max_home_len, len(home_visits))
            max_oot_len = max(max_oot_len, len(oot_visits))

        # Update config with actual max lengths (but keep some buffer)
        self.config.max_seq_len = max(max_home_len, 20)  # At least 20
        self.config.oot_seq_len = max(max_oot_len, 12)   # At least 12

        print(f"    Dynamic Max Lengths: home_seq={self.config.max_seq_len}, oot_seq={self.config.oot_seq_len}")

    def _load_trips(self):
        print(">>> Processing Trip Sequences...")

        # Data Structures by TripID
        trip_travel = {} # TripID -> {orig, dest, user}
        trip_home = defaultdict(list)
        trip_oot = defaultdict(list)
        
        # 1. Load Travel Info
        tp = os.path.join(self.config.data_dir, self.config.travel_file)
        if os.path.exists(tp):
            with open(tp, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 4: continue
                    tid, uid, orig, dest = parts[0], parts[1], parts[2], parts[3]
                    trip_travel[tid] = {'uid': uid, 'orig': orig, 'dest': dest}

        # 2. Load Home Checkins
        hp = os.path.join(self.config.data_dir, self.config.home_file)
        if os.path.exists(hp):
            with open(hp, 'r', encoding='utf-8') as f:
                 for line in f:
                     parts = line.strip().split('\t')
                     if len(parts) < 6: continue
                     tid, uid, vid, ts = parts[0], parts[1], parts[3], float(parts[4])
                     if vid in self.poi2id:
                        trip_home[tid].append({'poi': self.poi2id[vid], 'time': ts})
        
        # 3. Load OOTA Checkins
        op = os.path.join(self.config.data_dir, self.config.oot_file)
        if os.path.exists(op):
            with open(op, 'r', encoding='utf-8') as f:
                 for line in f:
                     parts = line.strip().split('\t')
                     if len(parts) < 6: continue
                     tid, uid, vid, ts = parts[0], parts[1], parts[3], float(parts[4])
                     if vid in self.poi2id:
                        trip_oot[tid].append({'poi': self.poi2id[vid], 'time': ts})

        # 4. Construct Samples (Sliding Window per Trip)
        count = 0
        
        for tid, oot_visits in trip_oot.items():
            if tid not in trip_home: continue
            
            home_visits = trip_home[tid]
            # Time Sort
            home_visits.sort(key=lambda x: x['time'])
            oot_visits.sort(key=lambda x: x['time'])
            
            if len(home_visits) < 5 or len(oot_visits) < 2: continue
            
            # Static Home Seq (Last M)
            h_seq = [x['poi'] for x in home_visits[-self.config.max_seq_len:]]
            
            # Get Home City (Origin)
            home_city = trip_travel[tid]['orig']
            h_hours = [self._get_local_hour(x['time'], home_city) for x in home_visits[-self.config.max_seq_len:]]
            
            if len(h_seq) < self.config.max_seq_len:
                pad_len = self.config.max_seq_len - len(h_seq)
                h_seq = [0]*pad_len + h_seq
                h_hours = [0]*pad_len + h_hours
                
            h_tensor = torch.LongTensor(h_seq)
            h_hours_tensor = torch.FloatTensor(h_hours) # Float for embedding input compatibility
            
            # Sliding Window on OOT
            N = self.config.oot_seq_len
            
            # Get OOT City (Dest) - assuming trip only has one dest city for simplicity of 'travel.txt'
            # But in reality oot checks might be in multiple cities? 
            # We use 'dest' from trip info as approximation for timezone.
            oot_city = trip_travel[tid]['dest']
            
            stride = 5
            min_len = 2
            
            slices = []
            if len(oot_visits) <= N:
                slices = [oot_visits]
            else:
                for i in range(0, len(oot_visits), stride):
                    chunk = oot_visits[i : i+N]
                    if len(chunk) < min_len: break
                    slices.append(chunk)
            
            for chunk in slices:
                p_ids = [x['poi'] for x in chunk]
                p_times = [x['time'] for x in chunk]
                p_hours = [self._get_local_hour(t, oot_city) for t in p_times]
                
                start_t = p_times[0]
                norm_times = [(t - start_t)/3600.0 for t in p_times]
                
                mask = [1]*len(p_ids)
                if len(p_ids) < N:
                    pad = N - len(p_ids)
                    p_ids += [0]*pad
                    norm_times += [0.0]*pad
                    p_hours += [0]*pad
                    mask += [0]*pad
                else:
                    p_ids = p_ids[:N]
                    norm_times = norm_times[:N]
                    p_hours = p_hours[:N]
                    mask = mask[:N]
                
                self.samples.append({
                    'home_seq': h_tensor,
                    'home_hours': h_hours_tensor,
                    'oot_seq': torch.LongTensor(p_ids),
                    'oot_times': torch.FloatTensor(norm_times),
                    'oot_hours': torch.FloatTensor(p_hours),
                    'oot_mask': torch.FloatTensor(mask),
                    'trip_id': tid,
                    'od_pair': (trip_travel[tid]['orig'], trip_travel[tid]['dest']),
                })
                count += 1
                
        print(f"Generated {count} samples from {len(trip_travel)} trips.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        home_seq = sample['home_seq']
        
        # KG Context
        home_seq_ents = torch.zeros_like(home_seq)
        kg_data = torch.zeros((self.config.max_seq_len, self.config.kg_k_hop, 2), dtype=torch.long)
        for i, pid in enumerate(home_seq):
            if pid.item() == 0: continue
            
            # Map POI -> Entity
            node = self.poi_to_kg_entity.get(pid.item())
            if node is not None:
                home_seq_ents[i] = node
                if node in self.kg_adj:
                    nbrs = self.kg_adj[node]
                    if nbrs:
                        sel = random.sample(nbrs, min(len(nbrs), self.config.kg_k_hop))
                        for k, (r, t) in enumerate(sel):
                            kg_data[i, k, 0] = t
                            kg_data[i, k, 1] = r
                        
        oot_seq = sample['oot_seq']
        assert self.poi_regions is not None and self.poi_coords is not None
        oot_regions = self.poi_regions[oot_seq]
        oot_coords = self.poi_coords[oot_seq]
        
        # RETURN 9 ITEMS (Added home_hours, oot_hours)
        return home_seq, home_seq_ents, kg_data, oot_seq, sample['oot_times'], oot_coords, sample['oot_mask'], oot_regions, sample['home_hours'], sample['oot_hours']

def get_dataloader(config):
    ds = FoursquareDataset(config)
    return torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=True)
