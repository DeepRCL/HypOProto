import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.receptive_field import compute_proto_layer_rf_info_v2
from src.models.ProtoPNet import PPNet, base_architecture_to_features


class Video_XProtoNet_EchoPrime(PPNet):
    def __init__(
        self, feat_channels, feat_shape, img_size, prototype_shape, proto_layer_rf_info, num_classes, init_weights=True, **kwargs
    ):
        # super(Video_XProtoNet, self).__init__(**kwargs)
        super(PPNet, self).__init__()  # not calling init of PPNet and directly going to its parent!

        self.feat_channels = feat_channels  # Block -3: 384
        self.feat_shape = feat_shape  # (T=8, H=14, W=14)
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.prototype_class_identity = self.get_prototype_class_identity()
        self.proto_layer_rf_info = proto_layer_rf_info

        #self.proj = nn.Linear(4096, feat_channels)
        self.spatial_size = 14
        self.embed_dim = feat_channels

        # NO CNN backbone - feats already [N,C,T,H,W]

        # Feature projector (adapt 384→D=256)
        self.add_on_layers = nn.Sequential(
            nn.Conv3d(feat_channels, prototype_shape[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(prototype_shape[1], prototype_shape[1], kernel_size=1),
        )

        # Occurrence module
        self.occurrence_module = nn.Sequential(
            nn.Conv3d(feat_channels, prototype_shape[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(prototype_shape[1], prototype_shape[1] // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(prototype_shape[1] // 2, prototype_shape[0], kernel_size=1, bias=False),
        )

        self.om_softmax = nn.Softmax(dim=-1)
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

        # Learnable prototypes
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        # To be used for pruning
        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)  # do not use bias

        if init_weights:
            self._initialize_weights(self.add_on_layers)
            self._initialize_weights(self.occurrence_module)
            self.set_last_layer_incorrect_connection(incorrect_strength=0)

    def preprocess_embeddings(self, x):
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
    
    def forward(self, x):
        # Feature Extractor Layer
        # x: [B, 16, 196, 4096]  -> project to 512
        x = self.preprocess_embeddings(x)
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, D, T, H, W)
        occurrence_map = self.get_occurence_map_absolute_val(x)  # shape (N, P, 1, T, H, W)
        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3).sum(dim=3)  # shape (N, P, D)

        # Prototype Layer
        similarity = self.cosine_similarity(
            features_extracted, self.prototype_vectors.squeeze().unsqueeze(0)
        )  # shape (N, P)
        similarity = (similarity + 1) / 2.0  # normalizing to [0,1] for positive reasoning

        # classification layer
        logits = self.last_layer(similarity)

        return logits, similarity, occurrence_map

    def compute_occurence_map(self, x, preprocess=True):
        
        # Feature Extractor Layer
        if preprocess:
            x = self.preprocess_embeddings(x)
        occurrence_map = self.get_occurence_map_absolute_val(x)  # shape (N, P, 1, T, H, W)
        return occurrence_map

    def get_occurence_map_absolute_val(self, x):
        occurrence_map = self.occurrence_module(x)  # shape (N, P, T, H, W)
        occurrence_map = torch.abs(occurrence_map).unsqueeze(2)  # shape (N, P, 1, T, H, W)
        return occurrence_map
    
    def get_occurence_map_sigmoid_norm(self, x):
        occurrence_map = self.occurrence_module(x)  # shape (N, P, H, W)
        occurrence_map = F.sigmoid(occurrence_map).unsqueeze(2)  # shape (N, L, 1, H, W)
        return occurrence_map

    def get_occurence_map_min_max_norm(self, x):
        occurrence_map = self.occurrence_module(x)  # shape (N, P, H, W)

        # Option 3: min-max norm: subtract min, divide by max! to be in range [0-1]
        occurrence_map = occurrence_map - occurrence_map.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        epsilon = 1e-8
        occurrence_map = occurrence_map / (epsilon + occurrence_map.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0])
        occurrence_map = occurrence_map / (epsilon + occurrence_map.sum(dim=(2, 3), keepdim=True))
        occurrence_map = occurrence_map.unsqueeze(2)  # shape (N, L, 1, H, W)

        return occurrence_map


    def get_occurence_map(self, x):
        # Option 1: Absolute value
        # occurrence_map = self.get_occurence_map_absolute_val(x)  # shape (N, P, 1, H, W)
        # Option 2: Softmax!
        # occurrence_map = self.get_occurence_map_softmaxed(x)  # shape (N, P, 1, H, W)
        # Option 3: min-max norm: subtract min, divide by max! to be in range [0-1]
        occurrence_map = self.get_occurence_map_min_max_norm(x)  # shape (N, P, 1, H, W)
        # Option 4: Sigmoid like Xprotonet
        # occurrence_map = self.get_occurence_map_sigmoid_norm(x)  # shape (N, P, 1, H, W)
        return occurrence_map

    def push_forward(self, x):
        """
        this method is needed for the pushing operation
        """
        # Feature Extractor Layer
        feature_map = self.add_on_layers(x).unsqueeze(1)  # shape (N, 1, D, T, H, W)
        occurrence_map = self.get_occurence_map_absolute_val(x)  # shape (N, P, 1, T, H, W)
        features_extracted = (occurrence_map * feature_map).sum(dim=3).sum(dim=3).sum(dim=3)  # shape (N, P, D)

        # Prototype Layer
        similarity = self.cosine_similarity(
            features_extracted, self.prototype_vectors.squeeze().unsqueeze(0)
        )  # shape (N, P)
        similarity = (similarity + 1) / 2.0  # normalizing to [0,1] for positive reasoning

        # classification layer
        logits = self.last_layer(similarity)

        return features_extracted, 1 - similarity, occurrence_map, logits

    def __repr__(self):
        # XProtoNet(self, backbone, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            "PPNet(\n"
            "\tcnn_backbone: {},\n"
            "\timg_size: {},\n"
            "\tprototype_shape: {},\n"
            "\tproto_layer_rf_info: {},\n"
            "\tnum_classes: {},\n"
            ")"
        )

        return rep.format(
            self.cnn_backbone,
            self.img_size,
            self.prototype_shape,
            self.proto_layer_rf_info,
            self.num_classes,
        )


def construct_Video_XProtoNet_EchoPrime(
    feat_channels=384,
    feat_shape=(8, 14, 14),
    img_size=224,
    prototype_shape=(20, 256, 1, 1, 1),
    num_classes=4,
):
    # layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    # proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
    #                                                      layer_filter_sizes=layer_filter_sizes,
    #                                                      layer_strides=layer_strides,
    #                                                      layer_paddings=layer_paddings,
    #                                                      prototype_kernel_size=prototype_shape[2])
    return Video_XProtoNet_EchoPrime(
        feat_channels=feat_channels,
        feat_shape=feat_shape,
        img_size=img_size,
        prototype_shape=prototype_shape,
        proto_layer_rf_info=None,
        num_classes=num_classes,
        init_weights=True,
    )
