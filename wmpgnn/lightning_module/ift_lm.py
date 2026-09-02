import pytorch_lightning as L

from collections import defaultdict
import copy

import torch.nn as nn

from wmpgnn.lightning_module.lightning_helper import *
from wmpgnn.util.pruners import edge_pruning, true_node_pruning
from wmpgnn.reconstruction.reconstruction import EventReconstruction
from wmpgnn.performance.reco_accuracy import obtain_reco_accuracy
from wmpgnn.performance.plotter import *
from wmpgnn.performance.tagging_power import analyze_tagging_power, process_ft
from wmpgnn.calibration.calibration import *


class IFTLightningModule(L.LightningModule):
    def __init__(self, model, dfei_model, optimizer_class, optimizer_params, configs, pos_weights):
        super().__init__()
        if "model" in configs["settings"]:
            self.version = configs["settings"]["model"]
        else:
            self.version = None
            self.save_hyperparameters({
                **configs,
                "pos_weights": make_loggable(pos_weights)
            })

        self.signal = "_".join(configs["evaluate"]["sample"])
        if configs["evaluate"]["over_write"] != "":
            self.signal += "__" + configs["evaluate"]["over_write"]

        self.configs = configs["inference"]
        self.model = model
        # DFEI model pass within IFT
        self.dfei_model = [dfei_model]
        if self.dfei_model[0] is not None:
            for param in self.dfei_model[0].parameters():
                param.requires_grad = False
        self.dfei_use_pid = configs['DFEI']["use_pid"] if "DFEI" in configs else False
        self.ift_use_pid = configs["IFT"]["use_pid"]

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        # Loss functions + associated inference class for plotting
        if self.configs["frag"]:
            self.frag_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["frag"])
        if self.configs["FT"]:
            self.ft_criterion = nn.CrossEntropyLoss(weight=pos_weights["FT"])

        # init log directories and event reconstruction class
        self.log_dir = configs["log_dir"]
        (self.trn_log, self.val_log), self.tst_log = init_logs(configs), init_logs(configs, mode="test")
        self.evt_reco = EventReconstruction(configs)
        self.tst_mode = 'MC'

    def forward(self, batch):
        # Obtaining LCA information
        if self.dfei_model[0] is not None:  # lca from dfei model, overwrites from pv asso
            dfei_input = copy.deepcopy(batch)
            if self.dfei_use_pid:
                dfei_input["tracks"].x = torch.cat([dfei_input["tracks"].x, dfei_input["tracks"].pid], dim=1)
            self.dfei_model[0] = self.dfei_model[0].to(self.device)
            dfei_outputs = self.dfei_model[0](dfei_input)
            lca = dfei_outputs[("tracks", "to", "tracks")].edges
        elif "lca" in batch[("tracks", "to", "tracks")]:  # using the information from the pv asso model
            lca = batch[("tracks", "to", "tracks")].lca
        else:  # using truth information
            labels = batch[("tracks", "tracks")].y.to(torch.long)
            lca = torch.nn.functional.one_hot(labels, num_classes=4).float()
        lca_score = torch.argmax(lca, dim=1).unsqueeze(1)
        # Attach DFEI information to graph
        batch[("tracks", "tracks")].edges = torch.cat([batch[("tracks", "tracks")].edges, lca_score], dim=1)
        if 'pred_y' in batch["tracks"]:
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pred_y.unsqueeze(dim=1)], dim=1)

        if self.ift_use_pid:
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)
        output = self.model(batch)
        output[("tracks", "to", "tracks")].lca = lca
        return output

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_params)

    def shared_step(self, batch, batch_idx, log):
        optimizers = self.optimizers()
        loss = init_loss(self.device)

        # Forward pass
        outputs_ft = self.forward(batch)

        y_ft = batch['tracks'].ft
        selbool = y_ft != 1  # this is actually a good question if one should use ft truth or the predicted from DFEI
        ift_loss = self.ft_criterion(outputs_ft["tracks"].x[selbool], y_ft[selbool])
        loss["ft_nodes"] += ift_loss

        log = loss_logging(log, loss, self.configs, mode="IFT")
        return ift_loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.trn_log)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.val_log)
        return loss

    def test_step(self, batch, batch_idx):
        org_x = batch["tracks"].x.clone()
        org_pid = batch["tracks"].pid.clone()

        # Forward pass
        outputs_ft = self.forward(batch)

        outputs_ft['tracks'].org_x = org_x
        outputs_ft['tracks'].org_pid = org_pid

        # Flavour tag decision
        ft_des = torch.softmax(outputs_ft["tracks"].x, dim=1)
        # Node decisions
        if self.dfei_model[0] is not None:
            outputs_ft["node_weights"] = self.dfei_model[0]._blocks[-1].node_weights["tracks"].squeeze()
        elif "pred_y" in outputs_ft["tracks"]:
            outputs_ft["node_weights"] = outputs_ft["tracks"].pred_y
        else:
            outputs_ft["node_weights"] = (outputs_ft["tracks"].ft != 1).int()
        # Edge decisions
        if self.dfei_model[0] is not None: # DFEI in IFT
            outputs_ft["edge_weights"] = self.dfei_model[0]._blocks[-1].edge_weights[
                ('tracks', 'to', 'tracks')].squeeze()
        elif "pred_y" in outputs_ft[("tracks", "tracks")]: # DFEI pre IFT
            outputs_ft["edge_weights"] = outputs_ft[("tracks", "tracks")].pred_y
        else: # truth information
            outputs_ft["edge_weights"] = (outputs_ft[("tracks", "tracks")].y != 0).int()

        """if "frag" in outputs_ft["tracks"]:
            frag_selbool = outputs_ft["tracks"].frag != -1  # this does not need to exist
            frag_in_evt = outputs_ft["tracks"].frag[frag_selbool]
            frag_pid = outputs_ft["part_ids"][frag_selbool]
            outputs_ft["frag_y"] = frag_in_evt
            outputs_ft["frag_pid"] = frag_pid"""

        self.evt_reco.reconstruct_heavyhadrons(outputs_ft, ft_des=ft_des)

        return {}

    def on_test_epoch_start(self):
        self.evt_reco.mode = self.tst_mode

    def on_train_epoch_end(self):
        avg_losses = epoch_end_loggable(self.trn_log)
        for key, val in avg_losses.items():
            self.log(f"train_{key}", val, prog_bar=(key == "ft_loss"), on_epoch=True, on_step=False)
        self.trn_log = defaultdict(list)

    def on_validation_epoch_end(self):
        avg_losses = epoch_end_loggable(self.val_log)
        for key, val in avg_losses.items():
            self.log(f"val_{key}", val, prog_bar=(key == "ft_loss"), on_epoch=True, on_step=False)
        self.val_log = defaultdict(list)

    def on_test_epoch_end(self):
        if self.version is None:
            self.version = self.logger.version
        sig_df = self.evt_reco.collect_results()
        outdir = f'{self.log_dir}/IFT/version_{self.version}'
        if self.tst_mode == 'MC':
            sig_df.to_csv(f'{outdir}/signal_reco_df_{self.signal}.csv', index=False)
        else:
            sig_df.to_csv(f'{outdir}/signal_reco_data_df_{self.signal}.csv', index=False)

        # If looking at MC add the additional performance scripts
        if self.tst_mode == 'MC':
            obtain_reco_accuracy(sig_df, self.version, self.signal, self.log_dir, model="IFT")
            if self.configs["FT"]:
                process_ft(self.tst_log, sig_df, self.version, self.signal, log_dir=self.log_dir)
                analyze_tagging_power(sig_df, self.version, self.signal, log_dir=self.log_dir)
                # Only consider event which are whitened
                if "is_whiten" in sig_df.keys():
                    whiten = self.signal + "__is_whiten"
                    whiten_df = sig_df[sig_df["is_whiten"] == 1]
                    obtain_reco_accuracy(whiten_df, self.version, whiten, self.log_dir, model="IFT")
                    process_ft(self.tst_log, whiten_df, self.version, whiten, log_dir=self.log_dir)
                    analyze_tagging_power(whiten_df, self.version, whiten, log_dir=self.log_dir)

                    # create calib root file
                    create_calib_root(whiten_df, self.version, whiten, log_dir=self.log_dir)
