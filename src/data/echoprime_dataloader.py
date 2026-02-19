import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from rclstream.datasets.private import echo  # ONLY this import
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit


class_labels = ["Normal", "Elevated"]

def load_embeddings_and_metadata(config):
    """Load embeddings and create metadata DataFrame"""
    # Load embeddings
    embeddings_dict = np.load(config['data']['embeddings_path'], allow_pickle=True).item()
    #embeddings = list(embeddings_dict.values())  # Convert dict values to list
    
    # Load patient info
    df_patient = pd.read_csv(config['data']['data_info_file'])
    df_patient.set_index('exam_id', inplace=True)
    
    # Load file metadata using ONLY echo (your provided import)
    df_file = echo.get_metadata(columns=['exam_id'])
    df_file = df_file[df_file.exam_id.isin(df_patient.index)].merge(
        df_patient, left_on='exam_id', right_index=True
    )

    # Print E/e' stats
    print(f"✅ average_e_e_ratio loaded: {df_file['average_e_e_ratio'].notna().sum()}/{len(df_file)} non-NaN")

    # *** REMOVE INDETERMINATE SAMPLES ***
    df_file = df_file[df_file['lvfp_category'] != 'Indeterminate'].copy()
    print(f"After removing Indeterminate: {len(df_file)} samples")
    
    return embeddings_dict, df_file

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
    
    print(f"Matching {len(embeddings)} embeddings with {len(df_file_valid)} metadata rows...")
    
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
    
    match_rate = len(df_embeddings) / len(embeddings) * 100
    print(f"✅ MATCH RATE: {len(df_embeddings)}/{len(embeddings)} ({match_rate:.1f}%)")
    print(f"Label distribution: {df_embeddings['label'].value_counts().to_dict()}")

    return df_embeddings

def get_patient_mapping_study(embeddings_dict, df_file):
    """Create study-level mapping [exam_id → [stream_ids]]"""
    label_mapping = {
        "Normal / No Elevated": 0,
        #"Indeterminate": 1, 
        "Elevated": 1
    }
    df_file['label_numeric'] = df_file['lvfp_category'].map(label_mapping)
    df_file_valid = df_file.dropna(subset=['label_numeric', 'patient_id'])
    
    # *** STUDY-LEVEL: Group by exam_id ***
    study_groups = df_file_valid.groupby('exam_id').agg({
        'patient_id': 'first',
        'label_numeric': 'first'
    }).reset_index()
    study_groups['label'] = study_groups['label_numeric'].astype(int)
    
    # Get stream_ids per study + matching embeddings
    study_data = []
    study_embeddings = []
    
    for _, study in study_groups.iterrows():
        exam_id = study['exam_id']
        # Get all stream_ids for this exam
        exam_streams = df_file_valid[df_file_valid['exam_id'] == exam_id].index.tolist()
        
        # Match with embeddings_dict
        valid_embs = []
        for stream_id in exam_streams:
            if stream_id in embeddings_dict:
                valid_embs.append(embeddings_dict[stream_id])
        
        if len(valid_embs) > 0:
            study_data.append({
                'exam_id': exam_id,
                'patient_id': study['patient_id'],
                'label': study['label'],
                'num_videos': len(valid_embs)
            })
            study_embeddings.append(np.stack(valid_embs))  # [seq_len, emb_dim]
    
    df_studies = pd.DataFrame(study_data)
    print(f"Study-level dataset: {len(df_studies)} studies")
    print(f"Label distribution: {df_studies['label'].value_counts().to_dict()}")
    print(f"Avg videos/study: {df_studies['num_videos'].mean():.1f}")
    
    return df_studies, np.array(study_embeddings, dtype=object)

class EmbeddingDataset(Dataset):
    def __init__(self, df, embeddings):
        self.df = df
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = torch.FloatTensor(self.embeddings[row.stream_id])
        label = torch.LongTensor([row.label])
        return {
            'video': emb, 
            'label': label, 
            'patient_id': row.patient_id,
            'stream_id': row.stream_id,
            'exam_id': row.exam_id,
            'average_e_e_ratio': row.average_e_e_ratio
        }

class StudyEmbeddingDataset(Dataset):
    def __init__(self, df, study_embeddings):
        self.df = df
        self.study_embeddings = study_embeddings
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = torch.FloatTensor(self.study_embeddings[idx])  # [seq_len, emb_dim]
        return {
            'video': emb, 
            'label': torch.tensor(row.label,dtype=torch.long),
            'patient_id': row.patient_id,
            'exam_id': row.exam_id,
            'average_e_e_ratio': row.average_e_e_ratio
        }

def collate_fn(batch):
    """Pad variable-length study sequences"""
    videos = [item['video'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    exam_ids = [item['exam_id'] for item in batch]
    
    max_len = max(v.shape[0] for v in videos)
    emb_dim = videos[0].shape[-1]
    padded_videos = torch.zeros(len(batch), max_len, emb_dim)
    
    for i, video in enumerate(videos):
        padded_videos[i, :video.shape[0]] = video
    
    return {
        'video': padded_videos,  # [B, max_seq_len, emb_dim]
        'label': labels.squeeze(),
        'patient_id': patient_ids,
        'exam_id': exam_ids
    }


def create_echoprime_dataloaders(config, study_level=False):
    """Create train/val/test dataloaders - configurable study vs clip level"""
    embeddings_dict, df_file = load_embeddings_and_metadata(config)
    
    if study_level:
        df_studies, study_embeddings = get_patient_mapping_study(embeddings_dict, df_file)
        train_dataset, val_dataset, test_dataset = _create_study_datasets(df_studies, study_embeddings, config)
        return _create_study_dataloaders(train_dataset, val_dataset, test_dataset, config['train']['batch_size'], study_embeddings, config['train']['num_workers'])
    else:
        df_embeddings = get_patient_mapping(embeddings_dict, df_file)
        train_dataset, val_dataset, test_dataset = _create_clip_datasets(df_embeddings, embeddings_dict, config)
        return _create_clip_dataloaders(train_dataset, val_dataset, test_dataset, config['train']['batch_size'], embeddings_dict, config['train']['num_workers'])

def _create_study_datasets(df_studies, study_embeddings, config):
    """Helper: Create study-level split"""
    gss = GroupShuffleSplit(n_splits=1, test_size=config['data']['test_size'], random_state=config['data']['random_seed'])
    train_idx, temp_idx = next(gss.split(df_studies, df_studies['label'], groups=df_studies['patient_id']))
    
    temp_df = df_studies.iloc[temp_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=config['data']['val_size']/(1-config['data']['test_size']), random_state=config['data']['random_seed'])
    val_idx, test_idx = next(gss_val.split(temp_df, temp_df['label'], groups=temp_df['patient_id']))
    
    train_df = df_studies.iloc[train_idx].reset_index(drop=True)
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    
    train_dataset = StudyEmbeddingDataset(train_df, study_embeddings[train_idx])
    val_dataset = StudyEmbeddingDataset(val_df, study_embeddings[temp_idx][val_idx])
    test_dataset = StudyEmbeddingDataset(test_df, study_embeddings[temp_idx][test_idx])
    
    return train_dataset, val_dataset, test_dataset

def _create_clip_datasets(df_embeddings, embeddings_dict, config):
    """Helper: Create clip-level split"""
    gss = GroupShuffleSplit(n_splits=1, test_size=config['data']['test_size'] + config['data']['val_size'], random_state=config['data']['random_seed'])
    train_idx, temp_idx = next(gss.split(df_embeddings, df_embeddings['label'], groups=df_embeddings['patient_id']))
    
    temp_df = df_embeddings.iloc[temp_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=config['data']['val_size'] / (config['data']['val_size'] + config['data']['test_size']), random_state=config['data']['random_seed'])
    val_idx, test_idx = next(gss_val.split(temp_df, temp_df['label'], groups=temp_df['patient_id']))
    
    train_df = df_embeddings.iloc[train_idx].reset_index(drop=True)
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    
    # Convert dict to list for indexing (maintains order)
    embeddings_list = [embeddings_dict[sid] for sid in df_embeddings['stream_id']]
    
    train_dataset = EmbeddingDataset(train_df, embeddings_dict)
    val_dataset = EmbeddingDataset(val_df, embeddings_dict)
    test_dataset = EmbeddingDataset(test_df, embeddings_dict)
    
    return train_dataset, val_dataset, test_dataset

def _create_study_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, study_embeddings, num_workers):
    """Helper: Study-level dataloaders with collate_fn"""
    train_patient_ids = set(train_dataset.df['patient_id'].unique())
    val_patient_ids = set(val_dataset.df['patient_id'].unique())
    overlap = train_patient_ids.intersection(val_patient_ids)
    print(f"Study-level - Patient overlap: {len(overlap)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, collate_fn=collate_fn)
    
    print(f"Study-level sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, study_embeddings

def _create_clip_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, embeddings_list, num_workers):
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

