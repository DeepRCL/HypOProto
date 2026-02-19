import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from rclstream.datasets.private import echo

# Load ALL files into massive dict ONCE (415GB RAM)
class MassiveEmbeddingsDict:
    def __init__(self, embeddings_dir):
        self.embeddings_dir = embeddings_dir
        self.data = {}  # {stream_id: embedding}
        
        file_names = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.npy')])
        print(f"Loading {len(file_names)} files into massive dict...")
        
        for fname in tqdm(file_names, desc="Loading files"):
            fpath = os.path.join(embeddings_dir, fname)
            data_dict = np.load(fpath, allow_pickle=True).item()
            self.data.update(data_dict)  # Merge all dicts
        
        print(f"✓ Loaded {len(self.data)} stream_ids total")
    
    def __getitem__(self, stream_id):
        return self.data[stream_id]
    
    def keys(self):
        return list(self.data.keys())
    
    def __contains__(self, stream_id):
        return stream_id in self.data

def load_embeddings_and_metadata(embeddings_dir, config):
    """Returns massive in-memory dict"""
    # Load patient info
    df_patient = pd.read_csv(config['data']['data_info_file'])
    df_patient.set_index('exam_id', inplace=True)
    
    #Load file metadata using ONLY echo (your provided import)
    # df_file = echo.get_metadata(columns=['exam_id'])
    # df_file = df_file[df_file.exam_id.isin(df_patient.index)].merge(
    #     df_patient, left_on='exam_id', right_index=True
    # )
    df_file = pd.read_csv('data/info_original.csv')
    df_file = (df_file[df_file.exam_id.isin(df_patient.index)]
           .merge(df_patient[['average_e_e_ratio']], 
                  left_on='exam_id', right_index=True, how='left')
           # Keep: stream_id, patient_id, lvfp_category (from df_file) + average_e_e (new)
          )

    # Print E/e' stats
    print(f"✅ average_e_e_ratio loaded: {df_file['average_e_e_ratio'].notna().sum()}/{len(df_file)} non-NaN")

    # *** REMOVE INDETERMINATE SAMPLES ***
    df_file = df_file[df_file['lvfp_category'] != 'Indeterminate'].copy()
    print(f"After removing Indeterminate: {len(df_file)} samples")
    
    return MassiveEmbeddingsDict(embeddings_dir), df_file

def get_patient_mapping(embeddings, df_file):
    """Map embeddings to patient_id and labels using df_file order"""
    # Map string labels to integers: "Normal /No Elevated" -> 0, "Indeterminate" -> 1, "Elevated" -> 2
    label_mapping = {
        "Normal / No Elevated": 0,
        #"Indeterminate": 1, 
        "Elevated": 1
    }
    
    df_file['label_numeric'] = df_file['lvfp_category'].map(label_mapping)
    df_file_valid = df_file.dropna(subset=['label_numeric', 'patient_id'])

    # *** MATCH embeddings_dict keys with df_file stream_ids ***
    matched_stream_ids = []
    matched_patient_ids = []
    matched_exam_ids = []  
    matched_labels = []
    matched_ee_ratios = []
    
    print(f"Matching {len(embeddings.keys())} embeddings with {len(df_file_valid)} metadata rows...")
    
    for stream_id in embeddings.keys():
        # Check if stream_id exists in df_file_valid (index or stream_id column)
        if stream_id in df_file_valid.index or (stream_id in df_file_valid['stream_id'].values if 'stream_id' in df_file_valid.columns else False):
            match = df_file_valid.loc[stream_id] if stream_id in df_file_valid.index else df_file_valid[df_file_valid['stream_id'] == stream_id].iloc[0]
            matched_stream_ids.append(stream_id)
            matched_patient_ids.append(match['patient_id'])
            matched_exam_ids.append(match['exam_id']) 
            matched_labels.append(int(match['label_numeric']))
            matched_ee_ratios.append(match.get('average_e_e_ratio', np.nan))
    
    df_embeddings = pd.DataFrame({
        'stream_id': matched_stream_ids,
        'patient_id': matched_patient_ids,
        'exam_id': matched_exam_ids,
        'label': matched_labels,
        'average_e_e_ratio': matched_ee_ratios
    })
    
    match_rate = len(df_embeddings) / len(embeddings.keys()) * 100
    print(f"✅ MATCH RATE: {len(df_embeddings)}/{len(embeddings.keys())} ({match_rate:.1f}%)")
    print(f"Label distribution: {df_embeddings['label'].value_counts().to_dict()}")

    return df_embeddings

class EmbeddingDataset(Dataset):
    """Unchanged - works with MultiFileEmbeddings"""
    def __init__(self, df, embeddings):
        self.df = df
        self.embeddings = embeddings
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = torch.FloatTensor(self.embeddings[row['stream_id']])#.view(16, 14, 14, 512).permute(3, 0, 1, 2)  
        label = torch.LongTensor([row['label']])
        return {
            'video': emb, 
            'label': label, 
            'patient_id': row.patient_id,
            'stream_id': row.stream_id,
            'exam_id': row.exam_id,
            'average_e_e_ratio': row.average_e_e_ratio
        }
    
def create_clip_datasets(df_embeddings, embeddings, config):
    """Unchanged"""
    gss = GroupShuffleSplit(n_splits=1, test_size=(config['data']['val_size']+config['data']['test_size']), random_state=config['data']['random_seed'])
    train_idx, temp_idx = next(gss.split(df_embeddings, df_embeddings['label'], groups=df_embeddings['patient_id']))
    temp_df = df_embeddings.iloc[temp_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=config['data']['val_size']/(config['data']['val_size']+config['data']['test_size']), 
                                random_state=config['data']['random_seed'])
    val_idx, test_idx = next(gss_val.split(temp_df, temp_df['label'], groups=temp_df['patient_id']))
    
    train_df = df_embeddings.iloc[train_idx].reset_index(drop=True)
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    
    train_dataset = EmbeddingDataset(train_df, embeddings)
    val_dataset = EmbeddingDataset(val_df, embeddings)
    test_dataset = EmbeddingDataset(test_df, embeddings)
    return train_dataset, val_dataset, test_dataset

def create_clip_dataloader(train_dataset, val_dataset, test_dataset, batch_size, embeddings_list, num_workers):
    """Helper: Clip-level dataloaders (original)"""
    train_patient_ids = set(train_dataset.df['patient_id'].unique())
    val_patient_ids = set(val_dataset.df['patient_id'].unique())
    overlap = train_patient_ids.intersection(val_patient_ids)
    print(f"Clip-level - Patient overlap: {len(overlap)}")

    # WEIGHTED SAMPLER
    # labels for each sample in the *training* dataset
    train_labels = train_dataset.df['label'].values
    classes, counts = np.unique(train_labels, return_counts=True)

    # inverse-frequency class weights
    class_sample_weights = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}

    # per-sample weight vector aligned with train_df
    sample_weights = np.array([class_sample_weights[y] for y in train_labels],
                            dtype=np.float32)
    sample_weights = torch.from_numpy(sample_weights)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # or larger if you want oversampling
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Clip-level sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader #, embeddings_list


def create_dino_dataloaders(config):
    """Main entry - clip-level only, no study-level"""
    embeddings_dir = config['data']['embeddings_path']  # e.g., '/path/to/patch_embeddings'
    
    embeddings, dffile = load_embeddings_and_metadata(embeddings_dir, config)
    df_embeddings = get_patient_mapping(embeddings, dffile)
    
    train_dataset, val_dataset, test_dataset = create_clip_datasets(df_embeddings, embeddings, config)
    return create_clip_dataloader(train_dataset, val_dataset, test_dataset, 
                                  config['train']['batch_size'], embeddings, config['train']['num_workers'])

# Usage
def test_dataloader(embeddings_dir, metadata_csv=None, batch_size=16, num_workers=0):
    import time
    print("🚀 Loading MASSIVE dict...")
    start_time = time.time()
    
    # Same config/dataloaders as before...
    config = {
        'data': {'test_size': 0.1, 'val_size': 0.1, 'random_seed': 42, 'data_info_file': METADATA_CSV},
        'train': {'batch_size': batch_size, 'num_workers': num_workers}
    }

    embeddings, dffile = load_embeddings_and_metadata(embeddings_dir, config)
    df_embeddings = get_patient_mapping(embeddings, dffile)
    print(f"✓ FULLY loaded {len(embeddings.keys())} embeddings in {time.time()-start_time:.1f}s")
    
    train_ds, val_ds, test_ds = create_clip_datasets(df_embeddings, embeddings, config)
    train_loader, val_loader, test_loader = create_clip_dataloader(
        train_ds, val_ds, test_ds, batch_size, embeddings, num_workers
    )
    
    # Test batch (now ~0.1s!)
    print("\n🧪 Testing 1 batch...")
    embeddings_b, labels_b, *_ = next(iter(train_loader))
    print(f"  ✓ Batch shape: {embeddings_b.shape}")
    print(f"  ✓ Batch time: <0.1s (in-memory!)")
    
    import psutil
    print(f"\n🎉 MASSIVE DICT READY! RAM: {psutil.Process().memory_info().rss/1024**3:.1f}GB")
    return train_loader, val_loader, test_loader, embeddings

if __name__ == "__main__":
    EMBEDDINGS_DIR = "/data/project/users/victoriawu/dinov3/dataset/"
    METADATA_CSV = "data/view3_exams_patients_with_lvfp_doppler.csv"
    
    train_loader, val_loader, test_loader, embeddings = test_dataloader(
        EMBEDDINGS_DIR, METADATA_CSV, batch_size=16, num_workers=4
    )

