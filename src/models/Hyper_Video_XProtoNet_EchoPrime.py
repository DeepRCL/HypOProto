import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.ProtoPNet import base_architecture_to_features
from src.models.Video_XProtoNet_EchoPrime import Video_XProtoNet_EchoPrime
from src.utils import lorentz as L
from src.utils.model_utils import get_prototype_class_identity

class Hyper_Video_XProtoNet(Video_XProtoNet_EchoPrime):
    def __init__(
        self,
        prototype_activation_function="linear",
        curv_init: float = 1.0,
        learn_curv: bool = False,
        init_weights=True,
        lift_prototypes: bool = True,
        feat_channels=384,
        feat_shape=(8, 14, 14),
        proto_layer_rf_info=None,
        # pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),  # TODO modify
        # pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),  # TODO modify
        **kwargs
    ):
        super(Hyper_Video_XProtoNet, self).__init__(init_weights=False, feat_channels=feat_channels, feat_shape=feat_shape, proto_layer_rf_info=proto_layer_rf_info, **kwargs)

        self.prototype_activation_function = prototype_activation_function

        # self.proj = nn.Linear(4096, feat_channels)
        # self.spatial_size = 14
        # self.embed_dim = feat_channels

        ###############################################################
        ########## MERU-based hyperbolic parameters ###############
        # # Initialize a learnable logit scale parameter.
        # TODO check if it is useful here. seems to be related to temperature used in cross entropy loss in CLIP/MERU
        # self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        # # Color mean/std to normalize image.
        # self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        # self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        # TODO include if DDP is used
        # # Get rank of current GPU process for gathering features.
        # self._rank = dist.get_rank()

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10),
        }

        # Learnable scalars to ensure that image features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(self.prototype_shape[1] ** -0.5).log())
        self.set_alpha = False  # to initialize the alpha value based on the extracted features! this flag is set at the beginning of the training once!
        # TODO maybe similar to MERU that had visual and text alphas,
        #  we need alphas for diff hierarchical layers?
        ##########################################################
        # to lift prototype vectors to hyperboloid (thus assuming they are in euclidean space)   OR
        # not to lift them, thus assuming they are already in hyperbolic space and are learnt to be on the hyperboloid
        self.lift_prototypes = lift_prototypes

        ####################################################################
        # Learnable radius for the prototypes to denote boundary cases
        self.prototype_ee = nn.Parameter(torch.zeros(self.num_prototypes))
        proto_class_id = self.get_prototype_class_identity()  # (P, C)

        # Class ranges: class 0 (1-13), class 1 (13-25)
        class_min = torch.tensor([1.0, 14.0], device=self.curv.device)
        class_max = torch.tensor([14.0, 25.0], device=self.curv.device)
        class_ranges = class_max - class_min  # (2,)

        # Assign each prototype a radius based on its class range
        for j in range(self.num_prototypes):
            c = torch.argmax(proto_class_id[j]).item()
            # Sample uniformly from class range + small noise for intra-class spread
            base_r = class_min[c] + torch.rand(1, device=self.curv.device) * class_ranges[c]
            self.prototype_ee.data[j] = base_r + 0.1 * torch.randn(1, device=self.curv.device)


        # ---------- Semi-supervised radius head ----------
        self.radius_head = nn.Sequential(
            nn.Linear(self.prototype_shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # ensures radius > 0
        )

        if init_weights:
            self._initialize_weights(self.add_on_layers)
            self._initialize_weights(self.occurrence_module)
            self.set_last_layer_incorrect_connection(incorrect_strength=0)

    def forward(self, x):
        (_, distances, occurrence_map, logits, pred_radius) = self.forward_detailed(x)
        similarities = self.distance_2_similarity(distances)
        return logits, similarities, occurrence_map, pred_radius
    
    def get_hyper_video_features(self, x):
        (feats, distances, _, _, _) = self.forward_detailed(x)
        similarities = self.distance_2_similarity(distances)
        return feats, similarities

    def distance_2_similarity(self, distances, max_distance=0):
        # TODO CHECK TO FIND WITH WHAT FORMULA THE HYPERBOLIC DISTANCE CAN BE CONVERTED TO SIMILARITY SCORE!
        if self.prototype_activation_function == "log":
            return torch.log((distances + 1) / (distances + self.epsilon))
        else:
            return -distances
        
    # @staticmethod
    # def ee_to_radius(
    #     ee_value: torch.Tensor,
    #     r_min: float = 0.0,
    #     r_max: float = 2.5,
    #     ee_min: float = 1.0,
    #     ee_max: float = 30.0,  # Clinical max
    #     outlier_cap: float = 50.0  # Your request
    # ):
    #     # Clamp outliers BEFORE normalization
    #     ee_clamped = torch.clamp(ee_value, min=ee_min, max=outlier_cap)
        
    #     # Normalize [ee_min, min(ee_max, outlier_cap)] → [0,1]
    #     ee_norm = (ee_clamped - ee_min) / (min(ee_max, outlier_cap) - ee_min)
    #     ee_norm = ee_norm.clamp(0.0, 1.0)
        
    #     # Confidence bowl: 0 at borderline (0.5), 1 at extremes
    #     confidence = (2.0 * ee_norm - 1.0) ** 2
        
    #     radius = r_min + confidence * (r_max - r_min)
    #     return radius

    @staticmethod
    def ee_to_radius(
        ee_value: torch.Tensor,
        r_min: float = 0.05,     # small but nonzero → keeps gradients alive
        r_max: float = 2.5,
        ee_center: float = 14.0,
        slope: float = 0.20,    # controls how fast radius grows away from center
        smooth: float = 0.5,    # >0 ensures nonzero gradient at center
    ):
        # Smooth distance from center (no kink at ee_center)
        dist = torch.sqrt((ee_value - ee_center) ** 2 + smooth ** 2)

        # Linear growth away from center
        radius_raw = slope * dist

        # Smooth saturation to r_max (no hard clamp)
        radius = r_min + (r_max - r_min) * torch.tanh(radius_raw / r_max)

        return radius    

    def forward_detailed(self, x):
        # x: [B, 16, 196, 4096]  -> project to 512
        x = self.preprocess_embeddings(x)

        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, D, T, H, W)
        occurrence_map = self.get_occurence_map(x)  # shape (N, P, 1, T, H, W)
        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3).sum(dim=3)  # shape (N, P, D)

        # print("[DBG] features_extracted shape:", features_extracted.shape)
        # print("[DBG] features_extracted norm mean:",
        #     torch.norm(features_extracted, dim=2).mean().item())
        
        #features_extracted = F.normalize(features_extracted, p=2, dim=-1)  # Unit norm
        #print(f"[NORM] features_extracted norm={torch.norm(features_extracted, dim=-1).mean():.3f}")

        ###############################################################

        ###############################################################
        ##### MERU-based operations for hyperbolic space analysis #####
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)

        # print("[DBG] curv (exp):", _curv.item())
        # print("[DBG] visual_alpha (exp):", self.visual_alpha.exp().item())

        # initialize the alpha value based on the features_extracted
        if self.set_alpha == True:
            # find the norm of the features_extracted and then use it to initialize the alpha value
            norm = torch.norm(features_extracted, dim=2).detach()
            self.visual_alpha.data = torch.log(1/(norm.mean()))

            norm_proto = torch.norm(self.prototype_vectors, dim=1).detach()
            self.prototype_vectors.data = self.prototype_vectors.data * (norm.mean() / norm_proto.mean())

            self.set_alpha = False

        ###############################################################
        ##### NEW: semi-supervised radius computation  #####
        ###############################################################

        # features_extracted: (N, P, D)
        N, P, D = features_extracted.shape

        # Pool over prototypes → sample-level descriptor
        pooled_feat = features_extracted.mean(dim=1)  # (N, D)
        # Learned radius prediction
        pred_radius = self.radius_head(pooled_feat).squeeze(-1)  # (N,)
        radius_from_ee = self.ee_to_radius(pred_radius)

        #radius = torch.zeros_like(radius_from_ee)

        # Expand for per-prototype lifting
        #radius = radius.unsqueeze(1)  # (N, 1)
        radius_per_proto = radius_from_ee.unsqueeze(1).expand(-1, P)  # (N,1) → (N,P)
        radius_per_proto = radius_per_proto.unsqueeze(-1) 

        # print("[DBG] pred_radius shape:", pred_radius.shape)
        # print("[DBG] pred_radius stats: mean",
        #   pred_radius.mean().item(), "std", pred_radius.std().item())
        
        # print("[DBG] radius_from_ee stats: min",
        #   radius_from_ee.min().item(), "max", radius_from_ee.max().item())

        ###########################################################################################################
        # lift features to hyperbolic space
        hyperbolic_feature_map = L.get_hyperbolic_feats_with_radius(features_extracted, radius_per_proto,
                                                         self.curv, self.device)  # shape (N, P, D)
        
        # TODO: Check if this is needed - self.prototype_ee.data = torch.clamp(self.prototype_ee.data, min=0.1, max=55.0)
        prototype_vectors = self.get_prototype_vectors()

        ###########################################################################################################
        # # TODO include if DDP is used
        # # Get features from all GPUs to increase negatives for contrastive loss.
        # # These will be lists of tensors with length = world size.
        # all_image_feats = dist.gather_across_processes(image_feats)
        # # shape: (batch_size * world_size, D), D=self.prototype_shape[1]
        #  all_image_feats = torch.cat(all_image_feats, dim=0)
        ###########################################################################################################

        #### Lorentz distance of each prototype from its corresponding extracted feature.  Shape (N, P)
        part_feat_prot_lorentz_distance = L.elementwise_dist(hyperbolic_feature_map,  # (N, P, D)
                                                             prototype_vectors,  # (P, D)
                                                             _curv)
        part_prototype_activations = self.distance_2_similarity(part_feat_prot_lorentz_distance)  # shape (N, P)

        # print("[DBG] dist stats: min",
        #   part_feat_prot_lorentz_distance.min().item(),
        #   "max", part_feat_prot_lorentz_distance.max().item())
        # print("[DBG] any NaN in dist:",
        #     torch.isnan(part_feat_prot_lorentz_distance).any().item())
        # print("[DBG] proto activations stats: mean",
        #   part_prototype_activations.mean().item(),
        #   "std", part_prototype_activations.std().item())
    
        # classification layer
        logits = self.last_layer(part_prototype_activations)  # shape (N, num_classes)
        # print("[DBG] logits stats: mean",
        #   logits.mean().item(), "std", logits.std().item())


        # TODO check what should be returned!
        if self.lift_prototypes:
        #### if Option 1 is selected to not lift the prototypes, return the hyperbolic features!
            features_to_track = hyperbolic_feature_map  # shape (N, P, D)
        #### if Option 2 is selected to lift the prototypes, return the euclidean features!
        else:
            # TODO or try logmap0 of the hyperbolic_feature_map!
            features_to_track = features_extracted  # shape (N, P, D)

        return (features_to_track, part_feat_prot_lorentz_distance, occurrence_map, logits, pred_radius)

    def preprocess_embeddings_dino(self, x):
        """
        Convert DINOv3 patch embeddings to spatial video format
        
        Args:
            x: [B, T, N_patches, D_in] where N_patches=196=14x14, D_in=4096
        
        Returns:
            [B, D_out, T, H, W] where D_out=512, H=W=14
        """
        B, T, N_patches, D_in = x.shape
        
        # Project DINOv3 dim to CNN dim
        x_proj = self.proj(x)  # [B, T, 196, 512]
        
        # Reshape flattened patches back to spatial
        H = W = int(N_patches ** 0.5)  # 14 for 196 patches
        x_spatial = x_proj.view(B, T, H, W, self.embed_dim)  # [B, T, 14, 14, 512]
        
        # To CNN format: [B, D, T, H, W]
        x_cnn_ready = x_spatial.permute(0, 4, 1, 2, 3)  # [B, 512, T, 14, 14]
        
        return x_cnn_ready
    
    def preprocess_embeddings(self, x):
        """
        Convert flattened patch embeddings to spatial video format
        
        Args:
            x: [B, 4096, 384] flattened spatial-temporal patches
            
        Returns:
            [B, 16, 16, 16, 384] spatial video format
        """
        B, T, flattened_patches, D = x.shape  # [B, 4096, 384]
        
        # Reshape: 4096 = 16x16x16 (spatial-temporal patches)
        x_reshaped = x.view(B, 16, 14, 14, D)  # [B, 16, 16, 16, 384]
        x_reshaped = x_reshaped.permute(0, 4, 1, 2, 3)  # [B, 384, 16, 16, 16]
        
        return x_reshaped

    def compute_occurence_map(self, x, preprocess=True):
        # Feature Extractor Layer
        if preprocess:
            x = self.preprocess_embeddings(x)
        occurrence_map = self.get_occurence_map(x)  # shape (N, P, 1, T, H, W)
        return occurrence_map


    def push_forward(self, x):
        """
        this method is needed for the pushing operation
        """
        #(features_extracted, part_feat_prot_lorentz_distance, occurrence_map,
        #        local_features_to_track, local_feat_prot_lorentz_distance, local_attn_map, logits) = self.forward_detailed(x)
        (features_extracted, distances, occurrence_map, logits, _) = self.forward_detailed(x)
        return features_extracted, distances, occurrence_map, logits

    def device(self) -> torch.device:
        return self.curv.device

    def get_prototype_vectors(self):
        if self.lift_prototypes:
            proto_rad = self.ee_to_radius(self.prototype_ee)
            # radius = torch.zeros_like(proto_rad)

            # # Expand for per-prototype lifting
            # #radius = radius.unsqueeze(1)  # (N, 1)
            # radius_per_proto = radius.unsqueeze(1).expand(-1, self.num_prototypes)  # (N,1) → (N,P)
            # radius_per_proto = radius_per_proto.unsqueeze(-1) 
            # print(self.prototype_vectors.shape)
            # print(radius_per_proto.shape)

            prototype_vectors = L.get_hyperbolic_feats_with_radius(self.prototype_vectors.squeeze(),
                                                       proto_rad.unsqueeze(1), self.curv, self.device)  # shape (P, D)
        else:
            prototype_vectors = self.prototype_vectors.squeeze()

        return prototype_vectors

    def __repr__(self):
        # Hyper_Video_XProtoNet(self, backbone, img_size, prototype_shape,
        # num_classes, init_weights=True):
        rep = (
            "Hyperbolic_Video_XProtoNet(\n"
            "\tcnn_backbone: {},\n"
            "\timg_size: {},\n"
            "\tnum_local_prototypes_per_class: {},\n"
            "\tprototype_shape: {},\n"
            "\tnum_classes: {},\n"
            "\tepsilon: {}\n"
            ")"
        )

        return rep.format(
            self.cnn_backbone,
            self.img_size,
            self.prototype_shape,
            self.num_classes,
            self.epsilon,
        )


def construct_Hyper_Video_XProtoNet_EchoPrime(
    img_size=224,
    prototype_shape=(30, 256, 1, 1, 1),
    num_classes=3,
    prototype_activation_function="linear",
    feat_range_type="Sigmoid",
    learn_curv=False,
    feat_channels=384,
    feat_shape=(8, 14, 14),
    ** kwargs
):
    return Hyper_Video_XProtoNet(
        img_size=img_size,
        prototype_shape=prototype_shape,
        num_classes=num_classes,
        learn_curv=learn_curv,
        init_weights=True,
        prototype_activation_function=prototype_activation_function,
        feat_range_type=feat_range_type,
        feat_channels=feat_channels,
        feat_shape=feat_shape,
        proto_layer_rf_info=None,
    )