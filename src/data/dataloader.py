import torch
import pandas as pd
import torchvision.transforms as T
from torchvision.transforms._transforms_video import RandomResizedCropVideo
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import GroupShuffleSplit
from rclstream.datasets.private import echo
from s3torchconnectorclient._mountpoint_s3_client import S3Exception
import numpy as np
import random

# -----------------------------
# Transforms
# -----------------------------

class_labels = ["Normal", "Elevated"]

class DownsampleTemporal:
    def __init__(self, num_frames, frame_stride, random_crop=True):
        self.num_frames = num_frames

        self.frames_to_take = num_frames
        self.frame_stride = frame_stride

        self.random_crop = random_crop

    def __call__(self, sample):

        x = sample["video"]  # (C, T, H, W)
        T, H, W = x.shape

        # ---- Pad temporally if needed ----
        if T < self.frames_to_take:
            pad_frames = self.frames_to_take - T
            padding = torch.zeros(
                (pad_frames, H, W),
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat((x, padding), dim=0)
            T = x.shape[0]

        # # ---- Temporal subsample ----
        # x = x[:self.frames_to_take : self.frame_stride, :, :]

         # ---- Random temporal crop (augmentation) ----
        if self.random_crop:
            max_start = T - self.frames_to_take
            start = random.randint(0, max_start)
        else:
            start = 0  # deterministic for val/test

        x = x[start : start + self.frames_to_take]

        # ---- Temporal subsample ----
        x = x[:: self.frame_stride]

        sample["video"] = x.contiguous().clone()
        return sample



class Resize:
    def __init__(self, size=(256, 256)):
        self.resize = T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)

    def __call__(self, sample):
        # video = torch.from_numpy(sample["video"]).unsqueeze(1).float()  # (T,1,H,W)
        # sample["video"] = self.resize(video)

        video = sample["video"]

        # video: (T,H,W) → (T,1,H,W)
        if video.ndim == 3:
            video = video.unsqueeze(1)

        video = video.float()
        video = self.resize(video)

        video = video.permute(1, 0, 2, 3).contiguous()
        sample["video"] = video  # (T,1,H,W)

        return sample

# =========================================================
# GLOBAL NORMALIZATION (computed from TRAIN SET ONLY)
# =========================================================
class GlobalVideoNormalize:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, video):
        # video: (C, T, H, W)
        return (video - self.mean) / (self.std + self.eps)

# -----------------------------
# Dataset
# -----------------------------

def per_video_normalize(video, eps=1e-6):
    """
    video: (C, T, H, W)
    """
    mean = video.mean()
    std = video.std()
    return (video - mean) / (std + eps)

def global_0_1_normalize(video):
    """
    video: (C, T, H, W) or any shape, values in [0,255]
    """
    return video.float() / 255.0

class LVFPEchoDataset(echo.EchoDataset):
    """
    Single, safe dataset:
    - idx-based tensor indexing
    - stream_id is metadata only
    """

    def __init__(self, df, num_frames, frame_stride, transform=None, video_transform=None, augment=False, global_norm=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.video_transform = video_transform

        self.num_frames=num_frames
        self.frame_stride = frame_stride
        self.height=256
        self.width=256

        # Pre-build tensors aligned to idx
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.ee = torch.tensor(df["average_e_e_ratio"].values, dtype=torch.float32)
        self.patient_ids = torch.tensor(df["patient_id"].values, dtype=torch.long)
        self.stream_ids = torch.tensor(df["stream_id"].values, dtype=torch.long)
        self.exam_ids = torch.tensor(df["exam_id"].values, dtype=torch.long)

        self.augment=augment
        self.global_norm=global_norm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # raw = super().__getitem__(idx)

        # video = raw["video"]
        # video = torch.from_numpy(video).contiguous()

        # sample = {
        #     "video": video,
        #     "label": self.labels[idx],
        #     "patient_id": self.patient_ids[idx],
        #     "stream_id": self.stream_ids[idx],
        #     "average_e_e_ratio": self.ee[idx],
        # }

        # if self.transform:
        #     sample = self.transform(sample)

        # return sample
        stream_id = int(self.stream_ids[idx])
    
        try:
            raw = super().__getitem__(stream_id)
            video = raw["video"]  # (T,H,W) numpy
            video = torch.from_numpy(video).float().contiguous()
            video = global_0_1_normalize(video)

            sample = {
                "video": video,
                "label": self.labels[idx],
                "patient_id": self.patient_ids[idx],
                "stream_id": self.stream_ids[idx],
                "average_e_e_ratio": self.ee[idx],
                "exam_id": self.exam_ids[idx],
            }

            if self.augment:
                if self.transform:
                    sample = self.transform(sample)

                video = sample["video"]
                # video: (T,H,W) → (T,1,H,W)
                if video.ndim == 3:
                    video = video.unsqueeze(1)

                # 2) Video-only transform
                if self.video_transform:
                    video = self.video_transform(video)  # expects tensor/clip only
                    video = video.permute(1, 0, 2, 3).contiguous()
                    #video = global_0_1_normalize(video)
                    sample["video"] = video
            else:
                sample = self.transform(sample)
                #sample['video'] = global_0_1_normalize(sample['video'])
            
            sample["video"] = self.gray_to_gray3(sample["video"])  # shape = (3,T,H,W), where T=1 if image
            sample["video"] = sample["video"].float()  # 3xTxHxW, where T=1 if image

            return sample

        except S3Exception as e:
            # Log once per failure
            print(f"[WARN] S3Exception at stream_id {self.stream_ids[idx]}: {e}. Returning dummy sample.")

            # Dummy zero sample; label=-1 will be ignored by loss
            dummy_video = torch.zeros(1, self.num_frames // self.frame_stride, self.height, self.width, dtype=torch.float32)
            return {
                "video": dummy_video,
                "label": torch.tensor(-1, dtype=torch.long),
                "patient_id": torch.tensor(-1, dtype=torch.long),
                "stream_id": torch.tensor(-1, dtype=torch.long),
                "exam_id": torch.tensor(-1, dtype=torch.long),
                "average_e_e_ratio": torch.tensor(0.0, dtype=torch.float32),
            }
        
        # expands one channel to 3 color channels, useful for some pretrained nets
    @staticmethod
    def gray_to_gray3(in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(3, -1, -1, -1)


# -----------------------------
# Metadata
# -----------------------------

def load_ap4_metadata(config):
    df_patient = pd.read_csv(config['data']['data_info_file'])
    df_patient.set_index("exam_id", inplace=True)

    # *** REMOVE INDETERMINATE SAMPLES ***
    df_patient = df_patient[df_patient['lvfp_category'] != 'Indeterminate'].copy()
    print(f"After removing Indeterminate: {len(df_patient)} samples")

    df = echo.get_metadata(columns=["exam_id", "predicted_cardiac_view_id"])
    df = df.reset_index()  # stream_id becomes column

    df = df[df.exam_id.isin(df_patient.index)]
    df = df.merge(df_patient, left_on="exam_id", right_index=True)

    df = df[df["predicted_cardiac_view_id"] == 3].copy()

    label_map = {
        "Normal / No Elevated": 0,
        #"Indeterminate": 1,
        "Elevated": 1,
    }

    df["label"] = df["lvfp_category"].map(label_map)
    df = df.dropna(subset=["label", "patient_id"]).copy()
    df["label"] = df["label"].astype(int)

    print(f"✅ AP4 samples: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    return df.reset_index(drop=True)



def collate_skip_invalid(batch):
    batch = [b for b in batch if b["label"].item() != -1]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# -----------------------------
# Public API
# -----------------------------

def create_video_dataloaders(config):
    print("Creating dataloader")
    batch_size = config['train']['batch_size']
    df = load_ap4_metadata(config)

    # Group-aware split
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=config['data']['test_size'] + config['data']['val_size'],
        random_state=config['data']['random_seed'],
    )
    train_idx, temp_idx = next(
        gss.split(df, df["label"], groups=df["patient_id"])
    )

    # CLASS WEIGHTS FROM TRAIN SPLIT ONLY (no leakage!)
    train_df = df.iloc[train_idx]
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_df['label']), 
        y=train_df['label'].values
    )
    class_weights = torch.FloatTensor(class_weights)
    print(f"Train-only class weights: {class_weights}")


    # WEIGHTED SAMPLER
    # labels for each sample in the *training* dataset
    train_labels = train_df['label'].values
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

    temp_df = df.iloc[temp_idx]
    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=config['data']['val_size'] / (config['data']['val_size'] + config['data']['test_size']),
        random_state=config['data']['random_seed'],
    )
    val_idx, test_idx = next(
        gss_val.split(temp_df, temp_df["label"], groups=temp_df["patient_id"])
    )

    transform_train = T.Compose([
        DownsampleTemporal(num_frames=config['data']['num_frames'], frame_stride=config['data']['frame_stride']),
    ])

    transform_val = T.Compose([
        DownsampleTemporal(num_frames=config['data']['num_frames'], frame_stride=config['data']['frame_stride'], random_crop=False),
        Resize((256, 256)),
    ])

    video_transform = T.Compose([
        RandomResizedCropVideo(size=(256, 256), scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2),
    ])

    #base_dataset = LVFPEchoDataset(df, transform=transform, video_transform=video_transform)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[temp_idx[val_idx]].reset_index(drop=True)
    test_df  = df.iloc[temp_idx[test_idx]].reset_index(drop=True)

    train_ds = LVFPEchoDataset(
        train_df,
        num_frames=config['data']['num_frames'],
        frame_stride=config['data']['frame_stride'],
        transform=transform_train,
        video_transform=video_transform,
        augment=True,   # 👈 only training augments
    )

    val_ds = LVFPEchoDataset(
        val_df,
        num_frames=config['data']['num_frames'],
        frame_stride=config['data']['frame_stride'],
        transform=transform_val,
        augment=False,
    )

    test_ds = LVFPEchoDataset(
        test_df,
        num_frames=config['data']['num_frames'],
        frame_stride=config['data']['frame_stride'],
        transform=transform_val,
        augment=False,
    )

    x = train_ds[0]["video"]
    print(x.min(), x.max(), x.mean(), x.std())

    # train_ds = Subset(base_dataset, train_idx)
    # val_ds = Subset(base_dataset, temp_idx[val_idx])
    # test_ds = Subset(base_dataset, temp_idx[test_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,   # 👈 instead of shuffle=True
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_skip_invalid,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_skip_invalid,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_skip_invalid,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    print(
        f"Splits — Train: {len(train_ds)}, "
        f"Val: {len(val_ds)}, Test: {len(test_ds)}"
    )

    return train_loader, val_loader, test_loader #, class_weights
