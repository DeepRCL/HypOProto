"""
Agent for video-based XprotoNet network, which is also used for our model ProtoASNet
trained end-to-end, inherits the image-based agent.
"""
import os
import numpy as np
import pandas as pd
import time
import wandb
import logging

import torch
import torch.nn as nn
from torch.backends import cudnn
from torchsummary import summary

from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split 

from src.loss.loss import (
    MAE,
    HyperbolicAngularSeparationLoss
)
from src.utils.lorentz import elementwise_dist
from src.agents.XProtoNet_e2e import XProtoNet_e2e
from src.data.dataloader import class_labels
from src.utils.utils import makedir
from src.utils.vis_prot_embd_space import plot_distance_histogram, plot_radius_vs_root_distance, plot_hyperboloid_projection, plot_radius_vs_root_distance_with_videos, plot_combined_hyperboloid_projection

cudnn.benchmark = True  # IF input size is same all the time, it's faster this way


class Hyper_Video_XProtoNet_e2e(XProtoNet_e2e):
    def __init__(self, config):
        super().__init__(config)
        self.MAELoss = MAE(**config['train']['criterion']["MAE"])
        self.HyperPASLoss = HyperbolicAngularSeparationLoss(**config['train']['criterion']["HyperPAS"])

                # DFR setup
        self.use_dfr = config.get('dfr', {}).get('use_dfr', False)
        if self.use_dfr:
            self.dfr_num_epochs = config['dfr']['num_epochs']
            self.dfr_lr = config['dfr']['lr']
            self.dfr_balanced_fraction = config['dfr'].get('balanced_fraction', 0.25)  # 10% balanced
            self.dfr_head = None
            print("DFR enabled - will retrain head on balanced val subset")

    def setup_dfr_head(self, feat_dim, num_classes):
        """Initialize DFR linear head with normalization"""
        self.dfr_head = nn.Sequential(
            nn.LayerNorm(feat_dim),  # Normalize features
            nn.Linear(feat_dim, num_classes)
        ).to(self.device)
        return torch.optim.Adam(self.dfr_head.parameters(), lr=self.dfr_lr), nn.CrossEntropyLoss()

    def extract_features_for_dfr(self, dataloader, mode="val"):
        """Extract frozen features (similarities) for DFR"""
        self.model.eval()
        self.model.requires_grad_(False)  # Freeze all
        
        feats, labels = [], []
        with torch.no_grad():
            for data_sample in tqdm(dataloader, desc=f"Extracting {mode} feats"):
                input = data_sample["video"].to(self.device)
                target = data_sample["label"].squeeze(-1).long().cpu()
                
                _, similarities, _, _ = self.model(input)  # Use similarities as feats (P-dim)
                feats.append(similarities.detach().cpu())
                labels.append(target)
        
        return torch.cat(feats), torch.cat(labels)

    def prepare_balanced_dfr_data(self, feats, labels, stratify_col=None):
        """Create balanced subset by class/view/E/e' bins"""
        # Simple class-balanced split (extend with views/ee if available)
        train_idx, val_idx = train_test_split(
            range(len(feats)), 
            test_size=1-self.dfr_balanced_fraction, 
            stratify=labels,  # Balance by class
            random_state=42
        )
        return feats[train_idx], labels[train_idx]

    def dfr_retrain(self):
        """Full DFR retraining on balanced val subset"""
        if not self.use_dfr or self.dfr_head is not None:
            return
            
        # Extract feats from val
        feats, labels = self.extract_features_for_dfr(self.data_loaders['val'])
        phi_bal, y_bal = self.prepare_balanced_dfr_data(feats, labels)
        
        logging.info(f"DFR: Using {len(phi_bal)} balanced samples (from {len(feats)} total)")
        
        # Setup & train head
        opt, criterion = self.setup_dfr_head(phi_bal.shape[1], self.model.num_classes)
        phi_bal_t = phi_bal.to(self.device)
        y_bal_t = y_bal.to(self.device)
        
        for epoch in range(self.dfr_num_epochs):
            logits = self.dfr_head(phi_bal_t)
            loss = criterion(logits, y_bal_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 10 == 0:
                logging.info(f"DFR epoch {epoch}: loss={loss.item():.4f}")
        
        self.model.requires_grad_(True)  # Unfreeze for normal training
        logging.info("DFR head trained & ready")

    def forward_with_dfr(self, similarities):
        """Use DFR head if active, else original logits"""
        if self.use_dfr and self.dfr_head is not None:
            return self.dfr_head(similarities)  # DFR logits
        # Original: model.last_layer(similarities)
        return self.model.last_layer(similarities)

    def apply_dfr_from_checkpoint(self, checkpoint_path, val_mode='val'):
        """
        Load pretrained model, extract val features, train DFR head.
        
        Usage:
        agent = Hyper_Video_XProtoNet_e2e(config)
        agent.load_model(checkpoint_path)  # Your existing load
        agent.apply_dfr_from_checkpoint(checkpoint_path)
        agent.save_model('model_with_dfr.pth')  # Save with DFR head
        """
        logging.info(f"Loading pretrained model from {checkpoint_path} for DFR")
        
        # Load your pretrained checkpoint (assumes existing load_model method)
        self.load_checkpoint(checkpoint_path)
        
        # Extract features & prepare balanced data
        feats, labels = self.extract_features_for_dfr(self.data_loaders[val_mode])
        phi_bal, y_bal = self.prepare_balanced_dfr_data(feats, labels)
        
        logging.info(f"DFR: Balanced subset {len(phi_bal)}/{len(feats)} samples")
        
        # Train DFR head
        opt, criterion = self.setup_dfr_head(phi_bal.shape[1], self.model.num_classes)
        phi_bal_t = phi_bal.to(self.device)
        y_bal_t = y_bal.to(self.device)
        
        self.model.eval()
        for epoch in tqdm(range(self.dfr_num_epochs), desc="DFR Retrain"):
            logits = self.dfr_head(phi_bal_t)
            loss = criterion(logits, y_bal_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 10 == 0:
                logging.info(f"DFR e{epoch}: loss={loss.item():.4f}")
        
        # Test DFR on full val
        self.model.eval()
        full_logits = self.dfr_head(feats.to(self.device))
        full_acc = balanced_accuracy_score(labels.cpu(), full_logits.argmax(1).cpu())
        logging.info(f"DFR val acc: {full_acc:.3f}")
        
        logging.info("✅ DFR head ready! Use forward_with_dfr() in inference")
    
    def save_model(self, model_dir, model_name):
        state = self.get_state()
        torch.save(
            state,
            f=os.path.join(model_dir, (model_name)),
        )

    def run_epoch(self, epoch, optimizer=None, mode="train"):
        logging.info(f"Epoch: {epoch} starting {mode}")
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        if "_push" in mode:
            # if val_push, use val for dataloder
            dataloader_mode = mode.split("_")[0]
        else:
            dataloader_mode = mode
        data_loader = self.data_loaders[dataloader_mode]
        epoch_steps = len(data_loader)

        label_names = class_labels
        logit_names = label_names + ["abstain"] if self.config["abstain_class"] else label_names

        n_batches = 0
        total_loss = np.zeros(9)

        y_pred_class_all = torch.FloatTensor()
        y_pred_all = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        epoch_pred_log_df = pd.DataFrame()

        start = time.time()

        # Diversity Metric
        count_array = np.zeros(self.model.prototype_shape[0])
        simscore_cumsum = torch.zeros(self.model.prototype_shape[0])

        # Reset sparsity metric
        getattr(self, f"{mode}_sparsity_80").reset()

        # Add this: Accumulate test ee for prototypes (only for test_vis)
        if mode == "test":
            # NEW: Collect test ee + proto assignments for test_vis scatter plot
            test_video_root_dists = []
            test_ee_list = []
            test_pred_ee_list = []
            video_ee_closest = [[] for _ in range(self.model.prototype_shape[0])]
            video_distances_closest = [[] for _ in range(self.model.prototype_shape[0])]

        with torch.set_grad_enabled(mode == "train"):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            accu_batch = 0
            for i in iterator:
                batch_log_dict = {}
                step = epoch * epoch_steps + i
                data_sample = next(data_iter)
                input = data_sample["video"].to(self.device)
                target = data_sample["label"].to(self.device)
                ee = data_sample["average_e_e_ratio"].to(self.device)

                logit, similarities, occurrence_map, pred_ee = self.model(input)

                # ADD THIS LINE (key fix!)
                if self.use_dfr and self.dfr_head is not None and mode != "train":
                    logit = self.forward_with_dfr(similarities)

                ############ Compute Loss ###############
                # CrossEntropy loss for Multiclass data
                target = target.squeeze(-1).long()
                ce_loss = self.CeLoss.compute(logits=logit, target=target)
                # cluster cost
                cluster_cost = self.Cluster.compute(similarities, target)
                # separation cost
                separation_cost = self.Separation.compute(similarities, target)
                # to encourage diversity on learned prototypes
                orthogonality_loss = self.Orthogonality.compute(self.model.get_prototype_vectors())
                # occurrence map L2 regularization
                occurrence_map_lnorm = self.Lnorm_occurrence.compute(occurrence_map, dim=(-3, -2, -1))
                # occurrence map transformation regularization
                occurrence_map_trans = self.Trans_occurrence.compute(input, occurrence_map, self.model)
                # FC layer L1 regularization
                fc_lnorm = self.Lnorm_fc.compute(self.model.last_layer.weight)
                # MAE loss
                valid_mask = ~torch.isnan(ee)
                if valid_mask.any():   # only compute if at least one valid sample exists
                    ee_valid = ee[valid_mask]
                    pred_ee_valid = pred_ee[valid_mask]
                    mae_ee = self.MAELoss.compute(pred_ee_valid, ee_valid)
                else:
                    # no supervision in this batch
                    mae_ee = torch.tensor(0.0, device=ee.device)
                # hyperbolic angular separation loss
                hyperpas_loss = self.HyperPASLoss.compute(self.model)

                loss = (
                    ce_loss
                    + cluster_cost
                    + separation_cost
                    + orthogonality_loss
                    + occurrence_map_lnorm
                    + occurrence_map_trans
                    + fc_lnorm
                    + mae_ee
                    + hyperpas_loss
                )

                ####### evaluation statistics ##########
                if self.config["abstain_class"]:
                    # take only logits from the non-abstention class
                    y_pred_prob = logit[:, : self.model.num_classes - 1].softmax(dim=1).cpu()
                else:
                    y_pred_prob = logit.softmax(dim=1).cpu()
                y_pred_max_prob, y_pred_class = y_pred_prob.max(dim=1)
                y_pred_class_all = torch.concat([y_pred_class_all, y_pred_class])
                y_pred_all = torch.concat([y_pred_all, y_pred_prob.detach()])
                y_true = target.detach().cpu()
                y_true_all = torch.concat([y_true_all, y_true])

                # f1 score
                f1_batch = f1_score(
                    y_true.numpy(),
                    y_pred_class.numpy(),
                    average=None,
                    labels=range(len(label_names)),
                    zero_division=0,
                )
                # confusion matrix
                cm = confusion_matrix(y_true, y_pred_class, labels=range(len(label_names)))
                # Accuracy
                accu_batch = balanced_accuracy_score(y_true.numpy(), y_pred_class.numpy())

                if mode == "train":
                    loss.backward()
                    if (i + 1) % self.train_config["accumulation_steps"] == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    self.current_iteration += 1

                total_loss += np.asarray(
                    [
                        ce_loss.item(),
                        cluster_cost.item(),
                        separation_cost.item(),
                        orthogonality_loss.item(),  # prototypical layer
                        occurrence_map_lnorm.item(),
                        occurrence_map_trans.item(),  # ROI layer
                        fc_lnorm.item(),  # FC layer
                        mae_ee.item(),
                        hyperpas_loss.item()
                    ]
                )
                n_batches += 1

                sparsity_batch = getattr(self, f"{mode}_sparsity_80")(similarities).item()

                # Determine the top 5 most similar prototypes to data
                # sort similarities in descending order
                sorted_similarities, sorted_indices = torch.sort(similarities[:, :30].detach().cpu(), descending=True)
                # Add the type 5 most similar prototypes to the count array
                np.add.at(count_array[:30], sorted_indices[:, :5], 1)

                if self.config["abstain_class"]:
                    # sort similarities in descending order
                    sorted_similarities, sorted_indices = torch.sort(
                        similarities[:, 30:].detach().cpu(), descending=True
                    )
                    # Add the type 5 most similar prototypes to the count array
                    np.add.at(count_array[30:], sorted_indices[:, :2], 1)

                simscore_cumsum += similarities.sum(dim=0).detach().cpu()

                # NEW: Accumulate test ee stats (inside loop, after valid_mask)
                if mode == "test" and valid_mask.any():
                    # Compute root distance for each VALID video's feature (not prototype!)
                    video_features, similarities = self.model.get_hyper_video_features(input)  # (B, D) - adapt to your feature extractor
                    # For VALID samples only
                    valid_sim = similarities[valid_mask]      # (B_valid, 20)
                    valid_feats = video_features[valid_mask]  # (B_valid, 20, 256)
                    
                    # Get top-1 prototype index per video
                    top_proto_idx = valid_sim.argmax(dim=1)   # (B_valid,)
                    
                    # Index features: video_features[b, top_proto_idx[b], :] → (B_valid, 256)
                    top_video_feats = valid_feats[range(len(valid_feats)), top_proto_idx]  # Fancy indexing!
                    batch_video_root_dists = elementwise_dist(
                        torch.zeros((1, top_video_feats.shape[1]), device=self.device), 
                        top_video_feats, 
                        self.model.curv.exp()
                    ).cpu().numpy()
                    test_video_root_dists.extend(batch_video_root_dists)
                    test_ee_list.extend(ee[valid_mask].cpu().numpy())
                    test_pred_ee_list.extend(pred_ee_valid.cpu().numpy())

                     # NEW: Top-10 closest videos PER prototype
                    valid_ee_np = ee[valid_mask].cpu().numpy()
                    
                    for b in range(len(valid_ee_np)):  # Per valid video
                        proto_id = top_proto_idx[b].item()
                        batch_ee = valid_ee_np[b]
                        if not np.isnan(batch_ee):
                            # Add to top-10 for this proto (truncate to 10)
                            video_ee_closest[proto_id].append(batch_ee)
                            # Compute distance for this video-proto pair
                            video_feat = top_video_feats[b:b+1]  # (1, 256)
                            vid_dist = elementwise_dist(
                                torch.zeros((1, video_feat.shape[1]), device=self.device), 
                                video_feat, 
                                self.model.curv.exp()
                            ).cpu().numpy()[0]
                            video_distances_closest[proto_id].append(vid_dist)
                            # FIXED: Convert to lists after truncation (not tuples!)
                            combined = sorted(zip(video_ee_closest[proto_id], video_distances_closest[proto_id]), 
                                            key=lambda x: x[1])[:10]
                            video_ee_closest[proto_id] = [x[0] for x in combined]
                            video_distances_closest[proto_id] = [x[1] for x in combined]

                # ########################## Logging batch information on console ###############################
                # cm_flattened = [list(cm[j].flatten()) for j in range(cm.shape[0])]
                iterator.set_description(
                    f"Epoch: {epoch} | {mode} | "
                    f"total Loss: {loss.item():.4f} | "
                    f"CE loss {ce_loss.item():.2f} | "
                    f"Cls {cluster_cost.item():.2f} | "
                    f"Sep {separation_cost.item():.2f} | "
                    f"Ortho {orthogonality_loss.item():.2f} | "
                    f"om_l2 {occurrence_map_lnorm.item():.4f} | "
                    f"om_trns {occurrence_map_trans.item():.2f} | "
                    f"fc_l1 {fc_lnorm.item():.4f} | "
                    f"Acc: {accu_batch:.2%} | f1: {f1_batch.mean():.2f} |"
                    f"Sparsity: {sparsity_batch:.1f}",
                    refresh=True,
                )

                # ########################## Logging batch information on Wandb ###############################
                if self.config["wandb_mode"] != "disabled":
                    batch_log_dict.update(
                        {
                            # mode is 'val', 'val_push', or 'train
                            f"batch_{mode}/step": step,
                            # ######################## Loss Values #######################
                            f"batch_{mode}/loss_all": loss.item(),
                            # f'batch_{mode}/loss_Fl': focal_loss.item(),
                            f"batch_{mode}/loss_CE": ce_loss,
                            f"batch_{mode}/loss_Clst": cluster_cost.item(),
                            f"batch_{mode}/loss_Sep": separation_cost.item(),
                            f"batch_{mode}/loss_Ortho": orthogonality_loss.item(),
                            f"batch_{mode}/loss_RoiNorm": occurrence_map_lnorm.item(),
                            f"batch_{mode}/loss_RoiTrans": occurrence_map_trans.item(),
                            f"batch_{mode}/loss_fcL1Norm": fc_lnorm.item(),
                            f"batch_{mode}/loss_MAEee": mae_ee.item(),
                            f"batch_{mode}/loss_hyperPAS": hyperpas_loss.item(),
                            # ######################## Eval metrics #######################
                            f"batch_{mode}/f1_mean": f1_batch.mean(),
                            f"batch_{mode}/accuracy": accu_batch,
                            f"batch_{mode}/sparsity": sparsity_batch,
                        }
                    )
                    batch_log_dict.update(
                        {f"batch_{mode}/f1_{as_label}": value for as_label, value in zip(label_names, f1_batch)}
                    )
                    # logging all information
                    wandb.log(batch_log_dict)

                # save model preds in CSV
                if mode == "val_push" or mode == "test":
                    # ##### creating the prediction log table for saving the performance for each case
                    epoch_pred_log_df = pd.concat(
                        [
                            epoch_pred_log_df,
                            self.create_pred_log_df(
                                data_sample,
                                logit.detach().cpu(),
                                logit_names=logit_names,
                            ),
                        ],
                        axis=0,
                    )

        end = time.time()

        ######################################################################################
        # ###################################### Calculating Metrics #########################
        ######################################################################################
        y_pred_class_all = y_pred_class_all.numpy()
        y_pred_all = y_pred_all.numpy()
        y_true_all = y_true_all.numpy()

        accu = balanced_accuracy_score(y_true_all, y_pred_class_all)
        f1 = f1_score(
            y_true_all,
            y_pred_class_all,
            average=None,
            labels=range(len(label_names)),
            zero_division=0,
        )
        f1_mean = f1.mean()

        # AUC = roc_auc_score(y_true_all, y_pred_all, average='weighted', multi_class='ovr',
        #                     labels=range(len(label_names)))
        try:
            AUC = roc_auc_score(
                y_true_all,
                y_pred_class_all,
                average="weighted",
                multi_class="ovr",
                labels=range(len(label_names)),
            )
        except ValueError:
            logging.exception("AUC calculation failed, setting it to 0")
            AUC = 0

        total_loss /= n_batches

        cm = confusion_matrix(y_true_all, y_pred_class_all, labels=range(len(label_names)))
        print(cm)

        ################################
        # get distribution of distances between prototypes and root (root is 0)
        ################################
        with torch.set_grad_enabled(False):
            # root feature is zeros with shape (1, self.model.prototype_shape[1])
            root_feature = torch.zeros((1, self.model.prototype_shape[1])).to(self.device)
            prototype_vectors = self.model.get_prototype_vectors()
            prototype_ee = self.model.prototype_ee.detach().cpu().numpy()
            _curv = self.model.curv.exp()
            root_distances = elementwise_dist(root_feature, prototype_vectors, _curv)

        root_distances = root_distances.cpu().numpy()

        if (mode == "train") or ("_push" in mode):
            # plot histograms of distances of local and broad prototypes to the root, on the same plot,
            fig_dist_to_root = plot_distance_histogram(root_distances,
                                                       "Distance to Origin", epoch, self.config["save_dir"],
                                                       f"{mode}-distance_to_root")
            
            fig_dist_vs_ee = plot_radius_vs_root_distance(root_distances, prototype_ee,
                                                       "Distance to Origin", epoch, self.config["save_dir"],
                                                       f"{mode}-ee-vs-distance")
        elif mode == "test":
            # Plot all video samples (ee vs their own root distance)
            # fig_dist_vs_ee = plot_hyperboloid_projection(
            #     np.array(test_pred_ee_list),
            #     np.array(test_ee_list),
            #     np.array(test_video_root_dists), 
            #     "Distance to Origin", 
            #     epoch, 
            #     self.config["save_dir"],
            #     f"{mode}-video-rootdist-vs-ee-hyper-all-0.5"
            # )

            fig_dist_vs_ee = plot_combined_hyperboloid_projection(
                np.array(test_pred_ee_list), 
                np.array(test_ee_list), 
                np.array(test_video_root_dists),
                root_distances, 
                prototype_ee,
                "Hyperboloid Projection: Videos + Prototypes", 
                epoch, self.config["save_dir"],
                f"{mode}-video-rootdist-vs-ee-hyper-proto")


            # # NEW: Prototypes + closest videos
            # plot_radius_vs_root_distance_with_videos(
            #     root_distances, prototype_ee,
            #     video_ee_closest, video_distances_closest,
            #     "Prototypes + 10 Closest Videos", epoch, self.config["save_dir"],
            #     f"{mode}-radius_vs_root_with_videos"
            # )

    

        # Diversity Metric Calculations
        # count how many prototypes were activated in at least 1% of the samples
        div_threshold = 0.05
        diversity = np.sum(count_array[:30] > div_threshold * len(y_true_all))
        diversity_log = f"diversity: {diversity}"
        if self.config["abstain_class"]:
            diversity_abstain = np.sum(count_array[30:] > div_threshold * len(y_true_all))
            diversity_log += f" | diversity_abstain: {diversity_abstain}"
        sorted_simscore_cumsum, sorted_indices = torch.sort(simscore_cumsum, descending=True)
        logging.info(f"sorted_simscore_cumsum is {sorted_simscore_cumsum}")
        # list(zip(range(40), (count_array > 0.3 * len(y_true_all)),count_array))
        # counts, bin_edges = np.histogram(count_array)
        # import termplotlib as tpl
        # # fig = tpl.figure()
        # # fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
        # # fig.show()
        # fig = tpl.figure()
        # x = np.arange(0, len(count_array))
        # fig.plot(x, count_array)
        # fig.show()

        sparsity_epoch = getattr(self, f"{mode}_sparsity_80").compute().item()

        #################################################################################
        # #################################### Consol Logs ##############################
        #################################################################################
        if mode == "test":
            logging.info(f"predicted labels for {mode} dataset are :\n {y_pred_class_all}")

        logging.info(
            f"Epoch:{epoch}_{mode} | Time:{end - start:.0f} | Total_Loss:{total_loss.sum() :.3f} | "
            f"[ce, clst, sep, ortho, om_l2, om_trns, fc_l1, mae, hyper_pas]={[f'{total_loss[j]:.3f}' for j in range(total_loss.shape[0])]} \n"
            f"Acc: {accu:.2%} | f1: {[f'{f1[j]:.2%}' for j in range(f1.shape[0])]} | f1_avg: {f1_mean:.4f} | AUC: {AUC} \n"
            f"Sparsity: {sparsity_epoch}  |  {diversity_log}"
        )
        logging.info(f"\tConfusion matrix: \n {cm}")
        logging.info(classification_report(y_true_all, y_pred_class_all, zero_division=0, target_names=label_names))

        #################################################################################
        ################################### CSV Log #####################################
        #################################################################################
        if mode == "val_push" or mode == "test":
            path_to_csv = os.path.join(self.config["save_dir"], f"csv_{mode}")
            makedir(path_to_csv)
            # epoch_pred_log_df.reset_index(drop=True).to_csv(os.path.join(path_to_csv, f'e{epoch:02d}_Auc{AUC.mean():.0%}.csv'))
            epoch_pred_log_df.reset_index(drop=True).to_csv(
                os.path.join(path_to_csv, f"e{epoch:02d}_f1_{f1_mean:.0%}.csv")
            )

        # ########################## Logging epoch information on Wandb ###############################
        if self.config["wandb_mode"] != "disabled":
            epoch_log_dict = {
                # mode is 'val', 'val_push', or 'train
                f"epoch": epoch,
                # ######################## Loss Values #######################
                f"epoch/{mode}/loss_all": total_loss.sum(),
                # ######################## Eval metrics #######################
                f"epoch/{mode}/f1_mean": f1_mean,
                f"epoch/{mode}/accuracy": accu,
                f"epoch/{mode}/AUC_mean": AUC,
                f"epoch/{mode}/diversity": diversity,
                f"epoch/{mode}/sparsity": sparsity_epoch,
            }
            if self.config["abstain_class"]:
                epoch_log_dict.update({f"epoch/{mode}/diversity_abstain": diversity_abstain})
            self.log_lr(epoch_log_dict)
            # log f1 scores separately
            epoch_log_dict.update({f"epoch/{mode}/f1_{as_label}": value for as_label, value in zip(label_names, f1)})
            # log AUC scores separately
            # epoch_log_dict.update({
            #     f'epoch/{mode}/AUC_{as_label}': value for as_label, value in zip(label_names, AUC)
            # })
            # log losses separately
            # loss_names = ["loss_Fl", "loss_Clst", "loss_Sep", "loss_Ortho", "loss_RoiNorm", "loss_RoiTrans", "loss_fcL1Norm"]
            loss_names = [
                "loss_CE",
                "loss_Clst",
                "loss_Sep",
                "loss_Ortho",
                "loss_RoiNorm",
                "loss_RoiTrans",
                "loss_fcL1Norm",
                "loss_MAEee",
                "loss_HyperPAS",
            ]
            epoch_log_dict.update(
                {f"epoch/{mode}/{loss_name}": value for loss_name, value in zip(loss_names, total_loss)}
            )
            # logging all information
            wandb.log(epoch_log_dict)

        return accu, f1_mean, AUC

    def print_model_summary(self):
        img_size = self.data_config["img_size"]
        frames = self.data_config["num_frames"]
        # summary(self.model, torch.rand((self.train_config['batch_size'], 3, img_size, img_size)))
        #summary(self.model, (3, frames, img_size, img_size), device="cpu")
        summary(self.model, (16, 196, 512), device="cpu")
        # print(self.model)
