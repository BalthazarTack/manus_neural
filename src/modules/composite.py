import torch
from easydict import EasyDict as edict
from src.modules.base import BaseTrainingModule
from src.utils.transforms import project_points
from src.utils.gaussian_utils import render_gaussians, calculate_colors_from_sh, get_cmap, strip_symmetric, get_nocs_colors, get_nocs_grid, covariance_to_scale_rotation
from src.utils.train_utils import load_models, load_modules, freeze_model
from src.utils.loss_utils import psnr
from src.utils.vis_util import get_colors_from_cmap
from src.utils.extra import *


class TrainingModule(BaseTrainingModule):
    def __init__(self, opts, mode):
        super().__init__(opts, mode)
        self.opts = opts
        self.mode = mode

        hand_model_ckpt = find_best_checkpoint(os.path.join('../../../../', self.opts.hand_ckpt_dir))
        print("Hand model checkpoint:", hand_model_ckpt)
        object_model_ckpt = find_best_checkpoint(os.path.join('../../../../', self.opts.object_ckpt_dir))

        self.o_module = load_modules(self.opts.object_module, self.opts, "test", self.opts.object_model,
                                     checkpoint=object_model_ckpt)

        self.h_module = load_modules(self.opts.hand_module, self.opts, "test", self.opts.hand_model,
                                     checkpoint=hand_model_ckpt)

        print("Object points:", self.o_module.model.get_xyz.shape[0])
        print("Hand points:", self.h_module.model.get_xyz.shape[0])

        if self.opts.optimize_hand:
            self.h_module.model.training_setup()
        else:
            self.h_module.model = freeze_model(self.h_module.model)

        if self.opts.optimize_object:
            self.o_module.model.training_setup()
        else:
            self.o_module.model = freeze_model(self.o_module.model)
        self.model = self.h_module.model

        self.render_contact_type = self.opts.test_dataset.opts.contact_render_type
        skin_wts = self.h_module.model.get_skin_weights()
        self.skin_colors = to_tensor(visualize_skin_weights(skin_wts)[..., :3], device=self.device)

        if self.render_contact_type == 'nocs':
            bones_rest = self.test_data[0]['bones_rest']
            n_grid, n_colors, n_center, n_scale = get_nocs_grid(bones_rest, 64)
            self.nocs_grid = get_nocs_colors(self.h_module.model.get_xyz, n_colors, n_center, n_scale)

    def configure_optimizers(self):
            return self.h_module.model.optimizer
    
    def render_skeleton(self, bones_posed, camera, bg_color):
        heads = bones_posed.heads
        tails = bones_posed.tails

        def _scalar(value):
            if isinstance(value, torch.Tensor):
                return int(value.flatten()[0].item())
            return int(value)

        height = _scalar(camera.height)
        width = _scalar(camera.width)

        camera_k = camera.K[0] if isinstance(camera.K, torch.Tensor) and camera.K.dim() == 3 else camera.K
        camera_extr = (
            camera.extr[0]
            if isinstance(camera.extr, torch.Tensor) and camera.extr.dim() == 3
            else camera.extr
        )

        image = np.zeros((height, width, 3), dtype=np.uint8)

        projected_heads = project_points(heads[None], camera_k, camera_extr[:3, :4])[0]
        projected_tails = project_points(tails[None], camera_k, camera_extr[:3, :4])[0]

        projected_heads = to_numpy(projected_heads)
        projected_tails = to_numpy(projected_tails)

        radius = max(2, min(height, width) // 180)
        thickness = max(1, radius // 2)
        color = (0, 255, 0)

        for head, tail in zip(projected_heads, projected_tails):
            head_xy = tuple(head[:2].astype(np.int32))
            tail_xy = tuple(tail[:2].astype(np.int32))
            cv2.line(image, head_xy, tail_xy, color, thickness, lineType=cv2.LINE_AA)
            cv2.circle(image, head_xy, radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(image, tail_xy, radius, color, -1, lineType=cv2.LINE_AA)

        return to_tensor(image / 255.0, device=heads.device)

    

    def forward(self, batch):
        h_out = self.h_module(batch)
        o_out = self.o_module(batch)
        posed_xyz = torch.concat([h_out.posed_xyz, o_out.posed_xyz], dim=0)
        # posed_xyz = torch.concat([o_out.posed_xyz], dim=0)
        o_cov = strip_symmetric(o_out.posed_cov) if o_out.posed_cov.dim() == 3 else o_out.posed_cov
        cov = torch.concat([h_out.posed_cov, o_cov], dim=0)
        cano_xyz = torch.concat([h_out.cano_xyz, o_out.cano_xyz], dim=0)
        cano_features = torch.concat([h_out.cano_features, o_out.cano_features], dim=0)
        cano_opacity = torch.concat([h_out.cano_opacity, o_out.cano_opacity], dim=0)
        h_scale = getattr(h_out, "posed_scale", h_out.cano_scale)
        h_rotation = getattr(h_out, "posed_rotation", h_out.cano_rotation)
        o_scale = getattr(o_out, "posed_scale", o_out.cano_scale)
        o_rotation = getattr(o_out, "posed_rotation", o_out.cano_rotation)
        posed_scale = torch.concat([h_scale, o_scale], dim=0)
        posed_rotation = torch.concat([h_rotation, o_rotation], dim=0)
        cano_rotation = torch.concat([h_out.cano_rotation, o_out.cano_rotation], dim=0)
        cano_scale = torch.concat([h_out.cano_scale, o_out.cano_scale], dim=0)
        cano_frequency = torch.concat([h_out.cano_frequency, o_out.cano_frequency], dim=0)
        cano_amplitude = torch.concat([h_out.cano_amplitude, o_out.cano_amplitude], dim=0)
        cano_phase = torch.concat([h_out.cano_phase, o_out.cano_phase], dim=0)
        cano_offset = torch.concat([h_out.cano_offset, o_out.cano_offset], dim=0)
        o_out_tf = attach(torch.eye(4)[None].repeat((o_out.cano_xyz.shape[0], 1, 1)), cov.device)
        tf = torch.concat([h_out.tf, o_out_tf], dim=0)


        # log_prob = get_gmm(o_out.posed_xyz, h_out.posed_xyz, build_symmetric(h_out.posed_cov), chunk=1024)
        # prob = torch.exp(log_prob)
        # prob = prob / prob.max()
        # colors = get_colors_from_cmap(to_numpy(log_prob), cmap_name='viridis')
        # dump_points(o_out.posed_xyz, 'points.ply', colors*255)
        # breakpoint()


        pred = {
            "posed_xyz": posed_xyz,
            "posed_cov": cov,
            "cano_xyz": cano_xyz,
            "cano_features": cano_features,
            "cano_opacity": cano_opacity,
            "cano_rotation": cano_rotation,
            "cano_scale": cano_scale,
            "posed_rotation": posed_rotation,
            "posed_scale": posed_scale,
            "cano_frequency": cano_frequency,
            "cano_amplitude": cano_amplitude,
            "cano_phase": cano_phase,
            "cano_offset": cano_offset,
            "tf": tf,
            "h_out": h_out,
            "o_out": o_out
        }
        return edict(pred)

    def training_step(self, batch, batch_idx):
        pass

    def render(self, batch):
        pred = self(batch)

        if self.render_contact_type == 'gt_eval':
            h_dist, h_cmap = self.render_contacts(pred, batch, batch['cano_camera'], 'hand_only')
            self.h_ac.append(h_dist)
            local_h_ac = torch.stack(self.h_ac).sum(axis=0)
            _, acc_h_cmap = self.render_contacts(pred, batch, batch['cano_camera'], 'accumulated', acc_dist=local_h_ac)
            render = torch.cat([h_cmap, acc_h_cmap], dim=1)

        elif self.render_contact_type == 'acc_gt_eval':
            alpha = 0
            cmap_type = 'gray'
            _, acc_h_cmap = self.render_contacts(pred, batch, batch['camera'], 'acc_gt_eval', cmap_type, alpha,
                                                 self.acc_contacts)

            _, skin_wts = self.render_contacts(pred, batch, batch['camera'], 'skin_wts', cmap_type, alpha,
                                                 None)

            render = torch.cat([skin_wts, acc_h_cmap], dim=1)

        elif self.render_contact_type == 'results':

            rendered = render_gaussians( pred.posed_xyz,
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
                                        batch["camera"],
                                        batch["bg_color"],
                                        sh_degree=self.model.opts.sh_degree,
                                        tf=pred.tf
                                        )

            rgb_img = rendered['render']

            skeleton_img = self.render_skeleton(batch["bones_posed"], batch["camera"], torch.zeros_like(batch["bg_color"]))

            sklt_img = torch.clamp(rgb_img + skeleton_img, 0, 1)

            _, o_cmap = self.render_contacts(pred, batch, batch['camera'], 'object_only')
            h_dist, h_cmap = self.render_contacts(pred, batch, batch['cano_camera'], 'hand_only')
            self.h_ac.append(h_dist)
            local_h_ac = torch.stack(self.h_ac).sum(axis=0)
            _, acc_h_cmap = self.render_contacts(pred, batch, batch['cano_camera'], 'accumulated', acc_dist=local_h_ac)

            # render = torch.cat([rgb_img, skeleton_img, h_cmap, o_cmap, acc_h_cmap], dim=1)
            render = torch.cat([sklt_img, h_cmap, o_cmap, acc_h_cmap], dim=1)


        elif self.render_contact_type == 'nocs':
            rendered = render_gaussians( pred.posed_xyz,
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
                                        batch["camera"],
                                        batch["bg_color"],
                                        sh_degree=self.model.opts.sh_degree,
                                        tf=pred.tf
                                        )

            rgb_img = rendered['render']
            _, o_cmap = self.render_contacts(pred, batch, batch['camera'], 'nocs_object_only')
            _, h_cmap = self.render_contacts(pred, batch, batch['cano_camera'], 'nocs_hand_only')
            render = torch.cat([rgb_img, h_cmap, o_cmap], dim=1)

        pred['render'] = render

        if self.training:
            num_h = pred.h_out.posed_xyz.shape[0]

            density_update_dict = {
                "viewspace_points": rendered['viewspace_points'][:num_h],
                "visibility_filter": rendered['visibility_filter'][:num_h],
                "radii": rendered['radii'][:num_h]
            }

            pred['density_update_dict'] = density_update_dict

        return edict(pred)

    def render_contacts(self, pred, batch, camera, render_type='hand', cmap_type='magma', alpha=0.3, acc_dist=None):
        bg_color = batch['bg_color']
        pt1 = pred.o_out
        pt2 = pred.h_out
        source = pred
        posed_cov = None
        posed_xyz = None

        if render_type == 'object_only':
            source = pt1
            dist, indices, cmap = get_cmap(pt1.posed_xyz, pt2.posed_xyz, cmap_type=cmap_type)
            posed_xyz = pt1.posed_xyz
            posed_cov = self.o_module.model.get_covariance()
            rgb_colors = calculate_colors_from_sh(pt1.cano_xyz, pt1.cano_features, pt1.cano_xyz, camera, 3, pt1.tf)
            colors_precomp = rgb_colors * alpha + (1 - alpha) * cmap

        elif render_type == 'hand_only':
            source = pt2
            dist, indices, cmap = get_cmap(pt2.posed_xyz, pt1.posed_xyz, cmap_type=cmap_type)
            posed_xyz = pt2.cano_xyz
            posed_cov = self.h_module.model.get_covariance()
            rgb_colors = calculate_colors_from_sh(pt2.cano_xyz, pt2.cano_features, pt2.cano_xyz, camera, 3, pt2.tf)
            colors_precomp = rgb_colors * alpha + (1 - alpha) * cmap

        elif render_type == 'nocs_hand_only':
            source = pt2
            dist, indices, cmap = get_cmap(pt2.posed_xyz, pt1.posed_xyz, cmap_type=cmap_type)
            posed_xyz = pt2.cano_xyz
            posed_cov = self.h_module.model.get_covariance()
            mask = dist > 0
            rgb_colors = self.nocs_grid
            cmap = torch.zeros_like(rgb_colors)
            cmap[mask] = rgb_colors[mask]
            colors_precomp =cmap

        elif render_type == 'nocs_object_only':
            source = pt1
            dist, indices, cmap = get_cmap(pt1.posed_xyz, pt2.posed_xyz, cmap_type=cmap_type)
            posed_xyz = pt1.posed_xyz
            posed_cov = self.o_module.model.get_covariance()

            mask = dist > 0
            indices = indices.int()
            rgb_colors = self.nocs_grid[indices]
            cmap = torch.zeros_like(rgb_colors)
            cmap[mask] = rgb_colors[mask]
            colors_precomp =cmap

        elif render_type == 'accumulated':
            source = pt2
            dist = acc_dist
            cmap = get_colors_from_cmap(to_numpy(acc_dist), cmap_name=cmap_type)[..., :3]
            cmap = attach(to_tensor(cmap), pt2.posed_xyz.device)
            posed_xyz = pt2.cano_xyz
            posed_cov = self.h_module.model.get_covariance()
            rgb_colors = calculate_colors_from_sh(pt2.cano_xyz, pt2.cano_features, pt2.cano_xyz, camera, 3, pt2.tf)
            colors_precomp = rgb_colors * alpha + (1 - alpha) * cmap

        elif render_type == 'acc_gt_eval':
            source = pt2
            dist = acc_dist
            cmap = get_colors_from_cmap(to_numpy(acc_dist), cmap_name=cmap_type)[..., :3]
            cmap = attach(to_tensor(cmap), pt2.posed_xyz.device)
            posed_xyz = pt2.posed_xyz
            posed_cov = pt2.posed_cov
            colors_precomp = cmap

        elif render_type == 'skin_wts':
            source = pt2
            dist = None
            posed_xyz = pt2.posed_xyz
            posed_cov = pt2.posed_cov
            colors_precomp = self.skin_colors.to(posed_xyz.device)

        scale,rotation=covariance_to_scale_rotation(posed_cov)

        posed_xyz = pred.posed_xyz if posed_xyz is None else posed_xyz
        posed_cov = pred.posed_cov if posed_cov is None else posed_cov

        rendered = render_gaussians(posed_xyz, posed_cov, source.cano_xyz,
                        source.cano_features, source.cano_opacity, scale, rotation,
                        source.cano_frequency, source.cano_amplitude, source.cano_phase, source.cano_offset,
                                    camera, bg_color, colors_precomp, sh_degree=3, tf=source.tf)['render']
        return dist, rendered

    def on_test_epoch_start(self):
        name = "eval_results"
        self.test_results_dir = os.path.join(self.result_dir, name, "ours")
        os.makedirs(self.test_results_dir, exist_ok=True)
        self.test_images = []
        self.info = []
        self.h_ac = []

        if self.render_contact_type == 'acc_gt_eval':
            acc_contact_path = os.path.join(self.test_results_dir, 'acc_contacts.npy')
            self.acc_contacts = to_tensor(np.load(acc_contact_path))

    def test_step(self, batch, batch_idx):
        pred = self.render(batch)
        img = to_numpy(torch.clamp(pred['render'], 0, 1))
        img = (img * 255).astype(np.uint8)
        self.test_images.append(img)
        self.info.append(batch['info'])
        colors = visualize_skin_weights(self.model.get_skin_weights())
        if colors.shape[0] != pred['posed_xyz'].shape[0]:
            final_colors = np.zeros((pred['posed_xyz'].shape[0], 4))
            final_colors[..., -1] = 1
            final_colors[:colors.shape[0]] = colors
        else:
            final_colors = colors
        self.dump_gaussians(pred, batch_idx, colors = final_colors)

        if self.render_contact_type == 'acc_gt_eval':
            self.dump_gaussians(pred, batch_idx, None)

    def on_test_epoch_end(self):
        dir_name = self.render_contact_type

        if self.render_contact_type == 'acc_gt_eval':
            eval_dir = os.path.join(self.test_results_dir, dir_name)
            os.makedirs(eval_dir, exist_ok=True)

            for idx, img in enumerate(self.test_images):
                cam_name = self.info[idx][-1]
                img_name = self.info[idx][1]
                # cam_dir = os.path.join(eval_dir, cam_name)
                # os.makedirs(cam_dir, exist_ok=True)
                img_path = os.path.join(eval_dir, f'{cam_name}.png')
                dump_image(img, img_path)
            return

        elif self.render_contact_type == 'gt_eval':
            local_h_ac = torch.stack(self.h_ac).sum(axis=0).detach().cpu().numpy()
            acc_contact_path = os.path.join(self.test_results_dir, 'acc_contacts.npy')
            np.save(acc_contact_path, local_h_ac)
            cprint(f"Saved accumulated contacts to {acc_contact_path}", 'green')

        dump_video(self.test_images, os.path.join(self.test_results_dir, f'{dir_name}.mp4'))

    def on_load_checkpoint(self, checkpoint):
        if self.opts.model.opts.skin_weights_init_type == "mano_init_voxel":
            self.model.grid_scale = checkpoint['extra_params']['grid_scale']
            self.model.grid_center = checkpoint['extra_params']['grid_center']
            self.model.grid_points = checkpoint['extra_params']['grid_points']
            self.model.grid_weights = checkpoint['extra_params']['grid_weights']
        elif self.opts.model.opts.skin_weights_init_type == "mano_init_points":
            self.model._skin_weights = checkpoint['extra_params']['mano_weights']

    def on_save_checkpoint(self, checkpoint):
        if "extra_params" not in checkpoint:
            checkpoint['extra_params'] = {}

        if self.opts.model.opts.skin_weights_init_type == "mano_init_voxel":
            checkpoint['extra_params']['grid_scale'] = self.model.grid_scale
            checkpoint['extra_params']['grid_center'] = self.model.grid_center
            checkpoint['extra_params']['grid_points'] = self.model.grid_points
            checkpoint['extra_params']['grid_weights'] = self.model.grid_weights
        elif self.opts.model.opts.skin_weights_init_type == "mano_init_points":
            checkpoint['extra_params']['mano_weights'] = self.model.get_skin_weights()