import os
import io
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from rclstream.datasets.private import echo
from io import BytesIO
from functools import partial
# Assuming you have S3MapDataset available - adjust import as needed
from s3torchconnector.s3map_dataset import S3MapDataset, get_objects_from_uris, identity

import boto3
from botocore.config import Config

class S3EmbeddingsDataset(S3MapDataset):
    def __init__(self, s3_prefix, stream_ids):
        """s3_prefix: "s3://ra-puranga-1/stroke/stream_test/victoria/dino/lvfp/" """
        self.object_uris = [
            f"{s3_prefix.rstrip('/')}/{str(int(sid))}.npy"
            for sid in stream_ids
        ]
        super().__init__(
            "us-east-1",
            partial(get_objects_from_uris, self.object_uris),  # s3:// URIs!
            endpoint="https://chinook.arc.ubc.ca",
            transform=self._load_embedding,
        )
    
    def _load_embedding(self, item):
        #return np.load(BytesIO(item.read()))
        return torch.from_numpy(np.load(io.BytesIO(item.read()))).float()
    
    def __getitem__(self, idx):
        #item = super().__getitem__(idx)
        #emb_array = np.load(BytesIO(item.read()))
        #return torch.FloatTensor(item)  # Raw tensor first
        return super().__getitem__(idx) 

def load_embeddings_and_metadata_s3(embeddings_s3_prefix, config):
    """S3 version - NO MASSIVE DICT!"""
    # Load patient info (unchanged)
    df_patient = pd.read_csv(config['data']['data_info_file'])
    df_patient.set_index('exam_id', inplace=True)
    
    # Load file metadata (unchanged)
    df = echo.get_metadata(columns=["exam_id", "predicted_cardiac_view_id"])

    df = df[df.exam_id.isin(df_patient.index)]
    df = df.merge(df_patient, left_on="exam_id", right_index=True)

    df_file = df[df["predicted_cardiac_view_id"] == 3].copy()
    
    print(f"✅ average_e_e_ratio loaded: {df_file['average_e_e_ratio'].notna().sum()}/{len(df_file)} non-NaN")
    
    # Remove indeterminate (unchanged)
    df_file = df_file[df_file['lvfp_category'] != 'Indeterminate'].copy()
    print(f"After removing Indeterminate: {len(df_file)} samples")
    
    return embeddings_s3_prefix, df_file  # Just return prefix, no massive dict!

def get_patient_mapping_s3(df_file, sample_stream_ids=None):
    """FIXED: stream_id is df_file.index!"""
    label_mapping = {"Normal / No Elevated": 0, "Elevated": 1}
    df_file['label_numeric'] = df_file['lvfp_category'].map(label_mapping)
    df_file_valid = df_file.dropna(subset=['label_numeric', 'patient_id']).copy()
    
    # *** stream_id IS THE INDEX from echo.get_metadata() ***
    df_file_valid['stream_id'] = df_file_valid.index  # Extract from index!
    
    if sample_stream_ids:
        df_file_valid = df_file_valid[
            df_file_valid['stream_id'].isin(sample_stream_ids)
        ]
    
    # Now create df_embeddings
    df_embeddings = df_file_valid[['stream_id', 'patient_id', 'exam_id', 'label_numeric', 'average_e_e_ratio']].copy()
    df_embeddings.columns = ['stream_id', 'patient_id', 'exam_id', 'label', 'average_e_e_ratio']
    df_embeddings['label'] = df_embeddings['label'].astype(int)
    
    print(f"✅ S3-ready: {len(df_embeddings)} matched samples")
    print(f"Label distribution: {df_embeddings['label'].value_counts().to_dict()}")
    print(f"Stream_ids sample: {df_embeddings['stream_id'].head().tolist()}")
    
    return df_embeddings.reset_index(drop=True)  # Clean index


class EmbeddingDataset(Dataset):
    """MODIFIED: Now takes S3EmbeddingsDataset instead of massive dict"""
    def __init__(self, df, s3_embeddings_dataset):
        self.df = df.reset_index()  # Ensure stream_id is in index 0 TODO remove this
        self.s3_embeddings = s3_embeddings_dataset  # S3MapDataset instance
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb_array = self.s3_embeddings[idx]  # [T, N_patches, 512] from S3
        # *** FIXED shape for your DINOv3 embeddings ***
        emb = torch.FloatTensor(emb_array).view(16, 196, 4096)#.permute(0, 2, 1)  # [16, 512, 196]
        # Or if you need spatial: .view(16, 14, 14, 512).permute(3, 0, 1, 2)
        
        label = torch.LongTensor([row['label']])
        return {
            'video': emb, 
            'label': label, 
            'patient_id': row.patient_id,
            'stream_id': row.stream_id,
            'exam_id': row.exam_id,
            'average_e_e_ratio': row.average_e_e_ratio
        }


def create_clip_dataloader(train_dataset, val_dataset, test_dataset, batch_size, embeddings_list, num_workers):
    """Helper: Clip-level dataloaders (unchanged from original)"""
    if val_dataset:
        train_patient_ids = set(train_dataset.df['patient_id'].unique())
        val_patient_ids = set(val_dataset.df['patient_id'].unique())
        overlap = train_patient_ids.intersection(val_patient_ids)
        print(f"Clip-level - Patient overlap: {len(overlap)}")

    # WEIGHTED SAMPLER (only for train)
    if train_dataset:
        train_labels = train_dataset.df['label'].values
        classes, counts = np.unique(train_labels, return_counts=True)
        class_sample_weights = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
        sample_weights = np.array([class_sample_weights[y] for y in train_labels], dtype=np.float32)
        sample_weights = torch.from_numpy(sample_weights)
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, persistent_workers=False, pin_memory=False)
    else:
        train_loader = None

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False, pin_memory=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False, pin_memory=False) if test_dataset else None
    
    if train_dataset:
        print(f"Clip-level sizes - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}, Test: {len(test_dataset) if test_dataset else 0}")
    
    return train_loader, val_loader, test_loader

def create_clip_datasets_s3(df_embeddings, embeddings_s3_prefix, config):
    """Same splits, but return stream_ids needed for S3 dataset"""
    gss = GroupShuffleSplit(n_splits=1, test_size=config['data']['test_size'] + config['data']['val_size'], random_state=config['data']['random_seed'])
    train_idx, temp_idx = next(gss.split(df_embeddings, df_embeddings['label'], groups=df_embeddings['patient_id']))
    temp_df = df_embeddings.iloc[temp_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=config['data']['val_size'] / (config['data']['val_size'] + config['data']['test_size']), 
                                random_state=config['data']['random_seed'])
    val_idx, test_idx = next(gss_val.split(temp_df, temp_df['label'], groups=temp_df['patient_id']))
    
    train_df = df_embeddings.iloc[train_idx].reset_index(drop=True)
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    
    # Extract stream_ids for each split
    train_stream_ids = train_df['stream_id'].tolist()
    val_stream_ids = val_df['stream_id'].tolist()
    test_stream_ids = test_df['stream_id'].tolist()
    
    return train_df, val_df, test_df, train_stream_ids, val_stream_ids, test_stream_ids

def create_dino_dataloaders_s3(config):
    """UPDATED: bucket + key_prefix"""
    bucket = "ra-puranga-1"
    embeddings_s3_prefix = "stroke/stream_test/victoria/dino/lvfp/"  # From your creation script
    
    prefix, df_file = load_embeddings_and_metadata_s3(embeddings_s3_prefix, config)  # Pass key_prefix
    df_embeddings = get_patient_mapping_s3(df_file)
    
    train_df, val_df, test_df, train_stream_ids, val_stream_ids, test_stream_ids = create_clip_datasets_s3(
        df_embeddings, embeddings_s3_prefix, config
    )
    
    # Create S3 datasets (lazy loading!)
    train_s3_embeddings = S3EmbeddingsDataset(
        s3_prefix="s3://ra-puranga-1/stroke/stream_test/victoria/dino/lvfp/", 
        stream_ids=train_stream_ids
    )
    val_s3_embeddings = S3EmbeddingsDataset(
        s3_prefix="s3://ra-puranga-1/stroke/stream_test/victoria/dino/lvfp/", 
        stream_ids=val_stream_ids
    )
    test_s3_embeddings = S3EmbeddingsDataset(
        s3_prefix="s3://ra-puranga-1/stroke/stream_test/victoria/dino/lvfp/", 
        stream_ids=test_stream_ids
    )
    
    train_dataset = EmbeddingDataset(train_df, train_s3_embeddings)
    val_dataset = EmbeddingDataset(val_df, val_s3_embeddings)
    test_dataset = EmbeddingDataset(test_df, test_s3_embeddings)
    
    return create_clip_dataloader(train_dataset, val_dataset, test_dataset, 
                                  config['train']['batch_size'], None, config['train']['num_workers'])


def list_actual_s3_keys(prefix='stroke/stream_test/victoria/dino/lvfp/'):
    """Use YOUR working s3 client config"""
    from rclstream.config import get_s3_storage_options
    from botocore.config import Config
    
    s3_options = get_s3_storage_options()
    config = Config(request_checksum_calculation='when_required', response_checksum_validation='when_required')
    
    s3 = boto3.client("s3", **s3_options, config=config)
    bucket = "ra-puranga-1"
    
    print(f"📁 Listing s3://{bucket}/{prefix}")
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=20)
        if 'Contents' not in response:
            print("❌ No objects!")
            return []
        
        keys = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.npy')]
        for key in keys:
            size_mb = response['Contents'][keys.index(key)]['Size']/1024/1024
            print(f"✅ {key} ({size_mb:.1f}MB)")
        
        print(f"\n📊 Found {len(keys)} .npy files")
        return keys
    except Exception as e:
        print(f"❌ List failed: {e}")
        return []

# Updated test function
def test_dataloader_s3(embeddings_s3_prefix, metadata_csv=None, batch_size=16, num_workers=0, max_samples=1000):
    """Test with optional max_samples to limit S3 dataset size"""
    import time
    import psutil
    print("🚀 S3 Dataloader (lazy loading)...")
    start_time = time.time()
    
    config = {
        'data': {
            'test_size': 0.1, 
            'val_size': 0.1, 
            'random_seed': 42, 
            'data_info_file': metadata_csv,
            'embeddings_s3_prefix': embeddings_s3_prefix
        },
        'train': {'batch_size': batch_size, 'num_workers': num_workers}
    }
    
    prefix, df_file = load_embeddings_and_metadata_s3(embeddings_s3_prefix, config)
    df_embeddings = get_patient_mapping_s3(df_file)
    actual_keys = list_actual_s3_keys()
    
    # Limit for testing
    if max_samples:
        df_embeddings = df_embeddings.sample(n=min(max_samples, len(df_embeddings)), random_state=42)
    
    print(f"✓ Setup complete in {time.time()-start_time:.1f}s (RAM: {psutil.Process().memory_info().rss/1024**3:.1f}GB)")
    
    train_df, val_df, test_df, train_stream_ids, val_stream_ids, test_stream_ids = create_clip_datasets_s3(
        df_embeddings, embeddings_s3_prefix, config
    )
    
    train_s3_emb = S3EmbeddingsDataset(embeddings_s3_prefix, train_stream_ids)
    train_dataset = EmbeddingDataset(train_df, train_s3_emb)
    
    train_loader, _, _ = create_clip_dataloader(
        train_dataset, None, None, batch_size, None, num_workers
    )

    # === Print time for 1 batch ===
    print(f"\n🧪 Fetching one S3 batch (batch_size = {batch_size})...")
    batch_times = []

    for i in range(2):  # 1 real + 1 warm‑up can help if there’s connection overhead
        t0 = time.time()
        batch = next(iter(train_loader))
        t1 = time.time()
        batch_times.append(t1 - t0)

        print(f"  Batch {i+1} ready in {t1-t0:.3f}s | keys: {len(batch['video'])}")

    avg_batch_time = np.array(batch_times).mean()
    print(f"\n📊 Average batch fetch time (batch_size={batch_size}): {avg_batch_time:.3f}s")
    
    return train_loader, None, None, None  # Simplified for testing

def test_s3_embeddings_io(embeddings_s3_prefix, metadata_csv, max_samples=1000, batch_size=64, num_workers=0):
    """Isolate S3EmbeddingsDataset IO speed - 10 batches with tqdm"""
    import time
    import psutil
    import numpy as np
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    print("🚀 Testing raw S3EmbeddingsDataset IO (batch_size=64)...")
    start_time = time.time()
    
    # Minimal setup - just get some valid stream_ids (your exact code)
    config = {
        'data': {
            'test_size': 0.1, 
            'val_size': 0.1, 
            'random_seed': 42, 
            'data_info_file': metadata_csv,
            'embeddings_s3_prefix': embeddings_s3_prefix
        },
        'train': {'batch_size': batch_size, 'num_workers': num_workers}
    }
    
    _, df_file = load_embeddings_and_metadata_s3(embeddings_s3_prefix, config)
    df_embeddings = get_patient_mapping_s3(df_file)
    
    if max_samples:
        df_embeddings = df_embeddings.sample(n=min(max_samples, len(df_embeddings)), random_state=42)
    
    train_df, _, _, train_stream_ids, _, _ = create_clip_datasets_s3(
        df_embeddings, embeddings_s3_prefix, config
    )
    
    print(f"✓ Setup complete in {time.time()-start_time:.1f}s (RAM: {psutil.Process().memory_info().rss/1024**3:.1f}GB)")
    print(f"📦 Testing with {len(train_stream_ids)} valid stream_ids")
    
    # === 10 BATCH BENCHMARK with tqdm ===
    s3_dataset = S3EmbeddingsDataset(embeddings_s3_prefix, train_stream_ids)
    s3_loader = DataLoader(
        s3_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    
    print(f"\n🧪 10 batches (batch_size={batch_size}, workers={num_workers})...")
    batch_times = []
    
    # Warm-up batch (untracked)
    print("  ⚡ Warm-up...")
    _ = next(iter(s3_loader))
    
    # 10 tracked batches with tqdm
    for batch in tqdm(s3_loader, total=10, desc="Benchmark", unit="batch"):
        t0 = time.time()
        # Simulate minimal processing (your real code does more)
        _ = batch['embedding'].mean() if isinstance(batch, dict) else batch.mean()
        t1 = time.time()
        batch_times.append(t1 - t0)
        tqdm.write(f"  Batch time: {t1-t0:.2f}s | Running avg: {np.mean(batch_times[-5:]):.2f}s")
    
    avg_time = np.mean(batch_times)
    throughput = (10 * batch_size) / sum(batch_times)
    
    print(f"\n📊 10-Batch Results (batch_size={batch_size}):")
    print(f"   Avg time:     {avg_time:.2f}s/batch ± {np.std(batch_times):.2f}s")
    print(f"   Total time:   {sum(batch_times):.1f}s")
    print(f"   Throughput:   {throughput:.1f} samples/sec")
    print(f"   Peak RAM:     {psutil.Process().memory_info().rss/1024**3:.1f}GB")
    
    return s3_loader, {'avg_batch_time': avg_time, 'throughput': throughput}
# Usage:
# s3_loader = test_s3_embeddings_io(
#     EMBEDDINGS_S3_PREFIX, 
#     max_samples=2000,  # Adjust based on your needs
#     batch_size=64,
#     num_workers=0  # Test serial first, then try 4/8/16
# )

if __name__ == "__main__":
    EMBEDDINGS_S3_PREFIX = "s3://ra-puranga-1/stroke/stream_test/victoria/dino/lvfp/"  # UPDATE THIS
    METADATA_CSV = "/home/victoriawu/workspace/ProtoASNet/data/view3_exams_patients_with_lvfp_doppler.csv"
    
    # train_loader, _, _, _ = test_dataloader_s3(
    #     EMBEDDINGS_S3_PREFIX, METADATA_CSV, batch_size=64, num_workers=0, max_samples=1000
    # )

    s3_loader = test_s3_embeddings_io(
        EMBEDDINGS_S3_PREFIX, 
        METADATA_CSV,
        max_samples=2000,  # Adjust based on your needs
        batch_size=8,
        num_workers=0  # Test serial first, then try 4/8/16
    )

