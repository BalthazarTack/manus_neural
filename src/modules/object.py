import torch
from tqdm import tqdm
import hydra
from easydict import EasyDict as edict
from src.utils.extra import *
from src.utils.gaussian_utils import (
    render_gaussians,
    get_points_outside_mask,
)
from src.modules.base import BaseTrainingModule


class TrainingModule(BaseTrainingModule):
    def __init__(self, opts, mode):
        super().__init__(opts, mode)
        self.opts = opts
        self.mode = mode
        self.automatic_optimization = True

        if self.mode != "test":
            dataset = self.train_data
            points, points_colors = dataset.sample_gaussians(
                self.opts.model.opts.sample_size, sample_mesh=True
            )
        else:
            points, points_colors = None, None
        self.model = hydra.utils.instantiate(
            self.opts.model,
            points=points,
            points_colors=points_colors,
            _recursive_=False,
        )
        self.model.training_setup(self.opts.model.opts) if self.mode != "test" else None

    def forward(self, batch):
        posed_cov = self.model.get_covariance(full=True)
        # The object has no pose transform, so the canonical scale/rotation are
        # the correct render inputs. Avoid decomposing covariance back into
        # scale/rotation here; that round-trip is numerically unstable and can
        # poison training with NaNs.
        posed_scale = self.model.get_scaling
        posed_rotation = self.model.get_rotation
        pred = {
            "posed_xyz": self.model.get_xyz,
            "posed_cov": posed_cov,
            "posed_scale": posed_scale,
            "posed_rotation": posed_rotation,
            "cano_xyz": self.model.get_xyz,  ## Cano and Posed are same for object
            "cano_features": self.model.get_features,
            "cano_opacity": self.model.get_opacity,
            "cano_rotation": self.model.get_rotation,
            "cano_scale": self.model.get_scaling,
            "cano_frequency": self.model.get_frequencies,
            "cano_amplitude": self.model.get_amplitudes,
            "cano_phase": self.model.get_phases,
            "cano_offset": self.model.get_offsets,
            "frequencies_color": self.model.get_frequencies_color,
            "phases_color": self.model.get_phases_color,
            "amplitudes_color": self.model.get_amplitudes_color,
            "offsets_color": self.model.get_offsets_color,
            "tf": None,
        }
        return edict(pred)

    def render(self, batch):
        pred = self(batch)
        rendered = render_gaussians(
            pred.posed_xyz,
            pred.posed_cov,
            pred.cano_xyz,
            pred.cano_features,
            pred.cano_opacity,
            pred.posed_scale,
            pred.posed_rotation,
            pred.cano_frequency,
            pred.cano_amplitude,
            pred.cano_phase,
            pred.cano_offset,
            pred.frequencies_color,
            pred.phases_color,
            pred.amplitudes_color,
            pred.offsets_color,
            batch["camera"],
            batch["bg_color"],
            sh_degree=3,
            tf=pred.tf,
        )

        rendered.update(pred)

        if self.training:
            rendered["camera"] = batch["camera"]
            rendered["mask"] = batch["mask"]

        self.rendered = rendered
        return rendered

    def on_after_backward(self):
        if self.global_step < self.model.opts.remove_seg_end:
            camera = self.rendered["camera"]
            points = self.rendered["posed_xyz"].detach()
            mask = self.rendered["mask"]
            pts_mask = get_points_outside_mask(camera, points, mask)[..., 0]
            self.pts_mask += pts_mask
            prune_mask = self.pts_mask.clone() if self.pts_mask.sum() > 0 else None
        else:
            prune_mask = None

        res = self.density_update(self.rendered, self.trainer.optimizers[0], prune_mask)
        if res:
            self.pts_mask = torch.zeros(
                self.model.get_xyz.shape[0], dtype=torch.bool, device=self.device
            )

    def on_before_optimizer_step(self, opts):
        opts = self.update_learning_rate(opts)

    def on_train_epoch_start(self):
        self.pts_mask = torch.zeros(
            self.model.get_xyz.shape[0], dtype=torch.bool, device=self.device
        )

    def training_step(self, batch, batch_idx):
        rendered = self.render(batch)
        final_loss = self.loss_func(
            batch,
            rendered,
            self.opts.losses,
            self.opts.loss_weight,
            self.opts.trainer.log_losses,
        )

        self.log("loss", final_loss, sync_dist=True)
        return {"loss": final_loss}
