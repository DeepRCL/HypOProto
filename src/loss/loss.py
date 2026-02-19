import torch.nn as nn
import torch
from torchvision.ops import sigmoid_focal_loss
from torchvision.transforms.functional import affine, InterpolationMode
import torch.nn.functional as F
import random
import logging

from src.utils import lorentz as L


class MSE(object):
    def __init__(self, loss_weight=1, reduction="mean"):
        self.criterion = nn.MSELoss(reduction=reduction)
        self.loss_weight = loss_weight
        logging.info(f"setup MSE Loss with loss_weight:{loss_weight}, and reduction:{reduction}")

    def compute(self, pred, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        loss = self.criterion(input=pred, target=target)  # (Num_nodes, 4)
        return self.loss_weight * loss
    
class MAE(object):
    def __init__(self, loss_weight=1, reduction="mean"):
        self.criterion = nn.L1Loss(reduction=reduction)
        self.loss_weight = loss_weight
        logging.info(f"setup MAE Loss with loss_weight:{loss_weight}, and reduction:{reduction}")

    def compute(self, pred, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        loss = self.criterion(input=pred, target=target)  # (Num_nodes, 4)
        return self.loss_weight * loss


class CeLoss(object):
    def __init__(self, loss_weight=1, reduction="mean"):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)  # , weight=tbd)
        self.loss_weight = loss_weight
        logging.info(f"setup CE Loss with loss_weight:{loss_weight}, and reduction:{reduction}")

    def compute(self, logits, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        loss = self.criterion(input=logits, target=target)  # (Num_nodes, num_classes)
        return self.loss_weight * loss


class ClusterPatch(object):
    """
    Cluster cost based on ProtoPNet architecture, using distance between patches and prototypes
    """

    def __init__(self, loss_weight, num_classes=4, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        logging.info(
            f"setup Path-Based Cluster Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, "
            f"and reduction:{reduction}"
        )

    def compute(self, min_distances, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)  # shape (N, classes)
        min_distances = min_distances.reshape((min_distances.shape[0], self.num_classes, -1))
        class_specific_min_distances, _ = min_distances.min(dim=2)  # Shape = (N, classes)
        positives = class_specific_min_distances * target_one_hot  # shape (N, classes)

        if self.reduction == "mean":
            loss = positives.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = positives.sum()

        return self.loss_weight * loss


class SeparationPatch(object):
    """
    Cluster cost based on ProtoPNet architecture, using distance between patches and prototypes
    """

    def __init__(self, loss_weight, num_classes=4, reduction="mean"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        logging.info(
            f"setup Patch-Based Separation Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, "
            f"and reduction:{reduction}"
        )

    def compute(self, min_distances, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)
        min_distances = min_distances.reshape((min_distances.shape[0], self.num_classes, -1))
        class_specific_min_distances, _ = min_distances.min(dim=2)  # Shape = (N, classes)
        negatives = class_specific_min_distances * (1 - target_one_hot)  # shape (N, classes)

        if self.reduction == "mean":
            loss = -negatives.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = -negatives.sum()

        return self.loss_weight * loss


class ClusterRoiFeat(object):
    """
    Cluster cost based on XprotoNet architecture, using similarities between ROI features and prototypes
    """

    def __init__(self, loss_weight, num_classes=4, reduction="sum"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        logging.info(
            f"setup ROI-Based Cluster Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, "
            f"and reduction:{reduction}"
        )

    def compute(self, similarities, target):
        """
        compute loss given the similarity scores
        :param similarities: the cosine similarities calculated. shape (N, P). P=num_classes x numPrototypes
        :param target: labels, with shape of (N)
        :return: cluster loss using the similarities between the ROI features and prototypes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        # turning labels into one hot
        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)  # shape (N, classes)
        # reshaping similarities to group based on class they belong to. shape (N, classes, P_pre_classes)
        similarities = similarities.reshape((similarities.shape[0], self.num_classes, -1))
        # get largest prototype-ROIfeature similarity scores per class
        class_specific_max_similarity, _ = similarities.max(dim=2)  # Shape = (N, classes)
        # pick similarity scores of classes the input belongs to
        positives = class_specific_max_similarity * target_one_hot  # shape (N, classes)
        loss = -1 * positives  # loss is negative sign of similarity scores

        # aggregate loss values
        if self.reduction == "mean":  # average across batch size
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss


class SeparationRoiFeat(object):
    """
    Separation cost based on XprotoNet architecture, using similarities between ROI features and prototypes
    """

    def __init__(self, loss_weight, num_classes=4, reduction="sum", abstain_class=True):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.abstain_class = abstain_class
        logging.info(
            f"setup ROI-Based Separation Loss with loss_weight:{loss_weight}, for num_classes:{num_classes}, "
            f"and reduction:{reduction}"
        )

    def compute(self, similarities, target):
        """
        compute loss given the similarity scores
        :param similarities: the cosine similarities calculated. shape (N, P). P=num_classes x numPrototypes
        :param target: labels, with shape of (N)
        :return: separation loss using the similarities between the ROI features and prototypes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        # turning labels into one hot
        target_one_hot = nn.functional.one_hot(target, num_classes=self.num_classes)  # shape (N, classes)
        if self.abstain_class:
            # do not calculate the separation losses for the fifth class prototypes
            target_one_hot[:, -1] = 1
        # reshaping similarities to group based on class they belong to. shape (N, classes, P_pre_classes)
        similarities = similarities.reshape((similarities.shape[0], self.num_classes, -1))
        # get largest prototype-ROIfeature similarity scores per class
        class_specific_max_similarity, _ = similarities.max(dim=2)  # Shape = (N, classes)
        # pick similarity scores of classes the input belongs to
        negatives = class_specific_max_similarity * (1 - target_one_hot)  # shape (N, classes)
        loss = negatives

        # aggregate loss values
        if self.reduction == "mean":  # average across batch size
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        return self.loss_weight * loss


class OrthogonalityLoss(object):
    """
    orthogonality loss to encourage diversity in learned prototype vectors
    """

    def __init__(self, loss_weight, num_classes=4, mode="per_class"):
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.mode = mode  # one of 'per_class' or 'all'
        if mode == "per_class":
            self.cosine_similarity = nn.CosineSimilarity(dim=3)
        elif mode == "all":
            self.cosine_similarity = nn.CosineSimilarity(dim=2)
        logging.info(
            f"setup Orthogonality Loss with loss_weight:{loss_weight}, "
            f"for num_classes:{num_classes}, in {mode} mode"
        )

    def compute(self, prototype_vectors):
        """
        compute loss given the prototype_vectors
        :param prototype_vectors: shape (P, channel, 1, 1). P=num_classes x numPrototypes
        :return: orthogonality loss either across each class, summed (or averaged), or across all classes
        """
        if self.loss_weight == 0:
            return torch.tensor(0, device=prototype_vectors.device)

        # per class diversity
        if self.mode == "per_class":
            # reshape to (num_classes, num_prot_per_class, channel):
            prototype_vectors = prototype_vectors.reshape(self.num_classes, -1, prototype_vectors.shape[1])
            # shape of similarity matrix is (num_classes, num_prot_per_class, num_prot_per_class)
            sim_matrix = self.cosine_similarity(prototype_vectors.unsqueeze(1), prototype_vectors.unsqueeze(2))
        elif self.mode == "all":
            # shape of similarity matrix is (num_prot_per_class, num_prot_per_class)
            sim_matrix = self.cosine_similarity(
                prototype_vectors.squeeze().unsqueeze(1),
                prototype_vectors.squeeze().unsqueeze(0),
            )
        # use upper traingle elements of similarity matrix (excluding main diagonal)
        loss = torch.triu(sim_matrix, diagonal=1).sum()

        return self.loss_weight * loss


class L_norm(object):
    def __init__(self, mask=None, p=1, loss_weight=1e-4, reduction="sum"):
        self.mask = mask  # mask determines which elements of tensor to be used for Lnorm calculations
        self.p = p
        self.loss_weight = loss_weight
        self.reduction = reduction
        logging.info(f"setup L{p}-Norm Loss with loss_weight:{loss_weight}, with reduction:{reduction}")

    def compute(self, tensor, dim=None):
        if self.loss_weight == 0:
            return torch.tensor(0, device=tensor.device)

        if self.mask != None:
            loss = (self.mask.to(tensor.device) * tensor).norm(p=self.p, dim=dim)
        else:
            loss = tensor.norm(p=self.p, dim=dim)
        if self.reduction == "mean":
            loss = loss.mean(dim=0).sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss


def get_affine_config():
    config = {
        "angle": random.uniform(-20, 20),
        "translate": (
            0,
            0,
        ),
        "scale": random.uniform(0.6, 1.5),
        "shear": 0.0,
        "fill": 0,
        "interpolation": InterpolationMode.BILINEAR,
    }
    return config


class TransformLoss(object):
    """
    the loss applied on generated ROIs!
    """

    def __init__(self, loss_weight=1e-4, reduction="sum"):
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction="sum")
        self.reduction = reduction
        logging.info(f"setup Transformation Loss with loss_weight:{loss_weight}, with reduction:{reduction}")

    def compute(self, x, occurrence_map, model):
        if self.loss_weight == 0:
            return torch.tensor(0, device=x.device)

        if len(x.shape) == 5:
            video_based = True
        else:
            video_based = False

        # get the affine transform randomly sampled configuration
        config = get_affine_config()

        # transform input and get its new occurrence map
        # *** NEW: Handle DINOv3 patch embeddings ***
        if len(x.shape) == 4 and x.shape[-1] == 512:  # [B,T,N_patches,4096]
            # Project + reshape to spatial INSIDE model
            x_spatial = model.preprocess_embeddings(x)  # [B,512,T,14,14]
            video_based = True
            N, D, T, H, W = x_spatial.shape
            
            # Transform flattened spatial frames
            x_flat = x_spatial.permute(0, 2, 1, 3, 4).reshape(-1, D, H, W)  # [N*T, 512, 14, 14]
            transformed_x_flat = affine(x_flat, **config)
            
            # Reshape back to video
            transformed_x = transformed_x_flat.reshape(N, T, D, H, W).permute(0, 2, 1, 3, 4)
            # [N, D, T, H, W] ready for model.compute_occurence_map()
        elif len(x.shape) == 3 and x.shape[-1] == 384:  # [B,T,N_patches,4096]
            # Project + reshape to spatial INSIDE model
            x_spatial = model.preprocess_embeddings(x)  # [B,512,T,16,16]
            video_based = True
            N, D, T, H, W = x_spatial.shape
            
            # Transform flattened spatial frames
            x_flat = x_spatial.permute(0, 2, 1, 3, 4).reshape(-1, D, H, W)  # [N*T, 512, 14, 14]
            transformed_x_flat = affine(x_flat, **config)
            
            # Reshape back to video
            transformed_x = transformed_x_flat.reshape(N, T, D, H, W).permute(0, 2, 1, 3, 4)
            
        else:
            N, D, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, D, H, W)  # shape (NxT, D, H, W)
            transformed_x = affine(x, **config)  # shape (NxT, D, H, W), T is 1 for image-based
        if video_based:
            transformed_x = transformed_x.reshape(N, T, D, H, W).permute(0, 2, 1, 3, 4)  # shape (N, D, T, H, W)
        occurrence_map_transformed = model.compute_occurence_map(transformed_x, preprocess=False).squeeze(2)  # shape (N, P, (T), H, W)

        # transform initial occurence map
        occurrence_map = occurrence_map.squeeze(2)  # shape (N, P, H, W) or (N, P, T, H, W) for video-based
        if video_based:
            N, P, T, H, W = occurrence_map.shape
            occurrence_map = occurrence_map.permute(0, 2, 1, 3, 4).reshape(-1, P, H, W)  # shape (NxT, P, H, W)
        transformed_occurrence_map = affine(occurrence_map, **config)  # shape (NxT, P, H, W), T is 1 for image-based
        if video_based:
            transformed_occurrence_map = transformed_occurrence_map.reshape(N, T, P, H, W).permute(
                0, 2, 1, 3, 4
            )  # shape (N, P, T, H, W)

        # compute L1 loss
        loss = self.criterion(occurrence_map_transformed, transformed_occurrence_map)
        if self.reduction == "mean":
            loss = loss / (occurrence_map_transformed.shape[0] * occurrence_map_transformed.shape[1])

        return self.loss_weight * loss


class CeLossAbstain(object):
    """
    Cross-entropy-like loss. Introduces a K+1-th class, abstention, which is a
    learned estimate of aleatoric uncertainty. When the network abstains,
    the loss will be computed with the ground truth, but, the network incurs
    loss for using the abstension
    """

    def __init__(self, loss_weight=1, ab_weight=0.3, reduction="sum", ab_logitpath="joined"):
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.ab_weight = ab_weight
        self.ab_logitpath = ab_logitpath
        self.criterion = nn.NLLLoss(reduction=reduction)
        assert self.ab_logitpath == "joined" or self.ab_logitpath == "separate"
        logging.info(
            f"setup CE Abstain Loss with loss_weight:{loss_weight}, "
            + f"ab_penalty:{ab_weight}, ab_path:{ab_logitpath} and reduction:{reduction}"
        )

    def to(self, device):
        self.criterion = self.criterion.to(device)

    def compute(self, logits, target):
        if self.loss_weight == 0:
            return torch.tensor(0, device=target.device)

        B, K_including_abs = logits.shape  # (B, K+1)
        K = K_including_abs - 1
        assert K >= 2, "CeLossAbstain input must have >= 2 classes not including abstention"

        # virtual_pred = (1-alpha) * pred + alpha * target
        if self.ab_logitpath == "joined":
            abs_pred = logits.softmax(dim=1)[:, K : K + 1]  # (B, 1)
        elif self.ab_logitpath == "separate":
            abs_pred = logits.sigmoid()[:, K : K + 1]  # (B, 1)
        class_pred = logits[:, :K].softmax(dim=1)  # (B, K)
        target_oh = nn.functional.one_hot(target, num_classes=K)
        virtual_pred = (1 - abs_pred) * class_pred + abs_pred * target_oh  # (B, K)

        loss_pred = self.criterion(torch.log(virtual_pred), target)
        loss_abs = -torch.log(1 - abs_pred).squeeze()  # (B)

        if self.reduction == "mean":
            loss_abs = loss_abs.mean()
        elif self.reduction == "sum":
            loss_abs = loss_abs.sum()

        return self.loss_weight * (loss_pred + self.ab_weight * loss_abs)
    
class EEPrototypeLoss(object):
    def __init__(self, loss_weight=1.0, reduction="mean", temperature=0.5):
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction=reduction)
        self.temperature = temperature

    def compute(self, gt_ee, similarities, prototype_ee, valid_mask=None):
        if valid_mask is None:
            valid_mask = ~torch.isnan(gt_ee)
        
        # Only labeled samples
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=gt_ee.device)
        
        gt_valid = gt_ee[valid_mask]
        sim_valid = similarities[valid_mask]      # (M, P)
        proto_weights = F.softmax(sim_valid / self.temperature, dim=1)
        proto_ee_target = torch.matmul(proto_weights, prototype_ee)  # (M,)
        
        ee_loss = self.criterion(gt_valid, proto_ee_target)
        return self.loss_weight * ee_loss
    
class HyperbolicAngularSeparationLoss(object):
    """
    Enforces that prototypes of the same class have similar directions
    and prototypes of different classes have dissimilar directions.

    Hyperbolic-aware: angles are computed in the tangent space at origin.
    """
    def __init__(self, loss_weight=1.0, temperature=0.1, eps=1e-8):
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.eps = eps

    def compute(self, model):
        """
        model.prototype_vectors: (P, D) hyperbolic points (Lorentz or spatial part)
        model.proto_classes:     (P,) class id per prototype
        """
        # 1) Get hyperbolic prototype vectors
        P = model.prototype_vectors.shape[0]
        proto = model.get_prototype_vectors()  # (P, D) hyperbolic

        # 2) Log-map to tangent space at origin
        # You already have similar ops in L.*
        proto_tangent = L.log_map0(proto, model.curv.exp())  # (P, D)

        # 3) Normalize directions
        proto_tangent = proto_tangent / torch.clamp(
            torch.norm(proto_tangent, dim=1, keepdim=True),
            min=self.eps
        )

        # 4) Cosine similarity matrix between all prototypes
        sim = torch.matmul(proto_tangent, proto_tangent.t())  # (P, P)
        sim = torch.clamp(sim, -1 + self.eps, 1 - self.eps)

        # 5) Build class mask
        device = model.device()
        proto_class_id = model.get_prototype_class_identity()  # (P,)
        proto_classes = torch.argmax(proto_class_id, dim=1)     # (P,) ← FIX!
        same_class = (proto_classes.unsqueeze(0) == proto_classes.unsqueeze(1)).to(device)
        diff_class = (~same_class).to(device)

        # remove diagonal
        eye = torch.eye(P, device=model.device(), dtype=torch.bool)
        same_class = same_class & ~eye
        diff_class = diff_class & ~eye

        # 6) InfoNCE-style angular loss
        logits = sim / self.temperature

        # for each prototype i: positives = same-class, negatives = diff-class
        loss_terms = []
        for i in range(P):
            pos_mask = same_class[i]
            neg_mask = diff_class[i]

            if pos_mask.sum() == 0:
                continue  # skip if only one proto in class

            pos_logits = logits[i][pos_mask]        # (K,)
            neg_logits = logits[i][neg_mask]        # (M,)

            # log( sum exp(pos) / (sum exp(pos) + sum exp(neg)) )
            numerator = torch.exp(pos_logits).sum()
            denominator = numerator + torch.exp(neg_logits).sum()

            loss_i = -torch.log(numerator / torch.clamp(denominator, min=self.eps))
            loss_terms.append(loss_i)

        if len(loss_terms) == 0:
            return torch.tensor(0.0, device=proto.device)

        loss = torch.stack(loss_terms).mean()
        return self.loss_weight * loss

