"""Unguided sampling for ``EnVariationalDiffusion``: ancestral steps, DDIM, ``sample``, ``sample_chain``."""

from MolecularDiffusion.utils import assert_mean_zero_with_mask
from MolecularDiffusion.utils import check_quality
from MolecularDiffusion.utils import correct_edges
from MolecularDiffusion.utils import create_pyg_graph
from MolecularDiffusion.utils import find_close_points_torch_and_push_op2
from MolecularDiffusion.utils import remove_mean_with_mask
from MolecularDiffusion.utils import resolve_connectors
from MolecularDiffusion.utils import sample_center_gravity_zero_gaussian_with_mask
from MolecularDiffusion.utils import sample_gaussian_with_mask
from ase.data import covalent_radii
from torch.nn import functional as F
from tqdm import tqdm
import torch
from MolecularDiffusion.modules.models.edm._en_diffusion_math import (
    _resolve_jitter_scale,
    atomic_numbers,
    split_frame,
    split_z,
)


class EnDiffusionSamplingMixin:
    """Unguided sampling for ``EnVariationalDiffusion``: ancestral steps, DDIM, ``sample``, ``sample_chain``.

    Plain mixin: no parameters or state of its own, so the ``state_dict`` of
    ``EnVariationalDiffusion`` is unchanged. Methods use attributes set in its ``__init__``.
    """

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(
            mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise
        )

        x = xh[:, :, : self.n_dims]

        if len(self.extra_norm_values) > 0:
            h_int = z0[:, :, self.n_dims + self.in_node_nf - self.ndim_extra - 1] if self.include_charges else torch.zeros(0).to(z0.device)
            h_cat = torch.cat([
                z0[:, :, self.n_dims : self.n_dims + self.in_node_nf - self.ndim_extra - 1],
                z0[:, : , -self.ndim_extra:]   
                ],
                dim=2)
            
        else:
            _, h_cat, h_int = split_z(self, z0)

        if h_int.dim() < 3: 
            h_int = h_int.unsqueeze(-1)
        x, h_cat, h_int = self.unnormalize(
            x, h_cat, h_int, node_mask
        )
        
        if len(self.extra_norm_values) > 0:
            h_extra = h_cat[:, :, -len(self.extra_norm_values) :]
            h_cat = h_cat[:, :, : -len(self.extra_norm_values)]
        else:
            h_extra = torch.zeros(0).to(z0.device)

        h_cat = F.one_hot(
            torch.argmax(h_cat, dim=2),
            self.num_classes - len(self.extra_norm_values),
        ) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {"integer": h_int, "categorical": h_cat, "extra": h_extra}
        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def sample_p_zs_given_zt(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context,
        fix_noise=False,
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)
        )

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)
        
        if torch.all(eps_t == 0):
            raise ValueError("NaN in eps_t, stop sampling.")
        
        # Compute mu for p(zs | zt).
        assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        assert_mean_zero_with_mask(eps_t[:, :, : self.n_dims], node_mask)
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )
        return zs

    def sample_combined_position_feature_noise(
        self,
        n_samples,
        n_nodes,
        node_mask,
        std=1.0,
    ):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask,
            std=std,
        )
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf),
            device=node_mask.device,
            node_mask=node_mask,
            std=std,
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def sample_ddim_step(self, zt, s, t, node_mask, edge_mask, context, eta=0.0):
        """
        Deterministic DDIM step: z_s ← z_t - f(eps_theta), no stochasticity when eta=0.
        """
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
        
        # Predict noise
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)
        alpha_s = self.alpha(gamma_s, target_tensor=zt)
        alpha_t = self.alpha(gamma_t, target_tensor=zt)

        # Compute predicted z0
        z0_pred = (zt - eps_t * sigma_t) / alpha_t

        # this is sigma_t in the DDIM paper (NOT TO BE CONFUSED WITH sigma_t from noise scheduler)
        sigma_t_ddim = sigma_s/sigma_t*torch.sqrt(1-(alpha_t/alpha_s)**2)
        # DDIM deterministic step
        z_s = alpha_s * z0_pred + (torch.sqrt(sigma_s**2 - sigma_t_ddim **2)) * eps_t

        # Optional noise injection (eta > 0 reintroduces stochasticity)
        if eta > 0.0:
            eps = self.sample_combined_position_feature_noise(
                n_samples=zt.size(0), n_nodes=zt.size(1), node_mask=node_mask
            )
            z_s += sigma_t_ddim * eps * eta

        # Keep center of gravity zero
        z_s = torch.cat(
            [remove_mean_with_mask(z_s[:, :, : self.n_dims], node_mask), z_s[:, :, self.n_dims:]],
            dim=2,
        )
        return z_s

    @torch.no_grad()
    def sample_ddim(
        self,
        n_samples, 
        n_nodes, 
        node_mask,
        edge_mask,
        context, 
        fix_noise=False,
        n_steps=None,
        eta=0.0,
        n_frames=0, # since there are fewer steps, we save all frames
    ):
        """
        DDIM sampling: deterministic, optionally fewer steps.
        eta=0.0 corresponds to the original DDIM (no noise),
        eta > 0 adds noise (reverts toward DDPM-like).
        """
        n_steps = n_steps or self.T  # use full steps unless specified
        step_indices = torch.linspace(0, self.T - 1, n_steps, dtype=torch.long, device=node_mask.device)
        # Initialize with Gaussian noise
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )
        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        if n_frames > 0:
            
            z_size = (z.size(0), z.size(1), z.size(2)-len(self.extra_norm_values))
         
            chain = torch.zeros((n_frames,) + z_size, device=z.device)
            s_saves = torch.linspace(0, self.T, 
                                         steps=n_frames, device=z.device).long() 
            
        write_index = 0
        # Time steps for DDIM
        with tqdm(total=n_steps-1, dynamic_ncols=True, unit="steps") as pbar:
            for _, i in enumerate(reversed(range(1, len(step_indices)))):
                
                t = step_indices[i]
                s = step_indices[i-1]

                t_array = torch.full((n_samples, 1), t / self.T, device=z.device)
                s_array = torch.full((n_samples, 1), s / self.T, device=z.device)

                z = self.sample_ddim_step(z, s_array, t_array, node_mask, edge_mask, context, eta)
                if n_frames > 0:
                    if s in s_saves:
                        chain[write_index] = self.unnormalize_z(z, node_mask)
                        write_index += 1
                    
                pbar.update(1)
        # Final decode
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        # Center correction
        assert_mean_zero_with_mask(x, node_mask)
        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f"[DDIM] Warning cog drift: {max_cog:.3f}, projecting positions.")
            x = remove_mean_with_mask(x, node_mask)
        
        if n_frames > 0:
            # Save the chain of frames
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
            chain[-1] = xh   
            chain_flat = chain.view(n_frames, n_samples, z.size(1), z.size(2)-len(self.extra_norm_values)) 
        else:
            chain_flat = None
        return x, h, chain_flat

    @torch.no_grad()
    def sample(
        self,
        n_samples, 
        n_nodes,
        node_mask, 
        edge_mask, 
        context, 
        condition_tensor=None,
        condition_mode=None,
        inpaint_cfgs={},
        outpaint_cfgs={},
        use_noised_conditioning=False,
        fix_noise=False,
        n_frames=0,
        t_retry=180,
        n_retrys=0,
        silvr_cfgs=None,
    ):

        """
        Draw samples from the generative model.
        """
        self.use_noised_conditioning = use_noised_conditioning

        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )

        self.COV_R = torch.tensor(covalent_radii, dtype=torch.float32, device=z.device)

        # assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        if condition_mode:
            condition_alg, condition_component = condition_mode.split("_")
        else:
            condition_alg = None
            condition_component = None

        if condition_tensor is not None:
            condition_tensor = condition_tensor.to(device=z.device)
            if condition_tensor.size(0) == 1 and n_samples > 1:
                condition_tensor = condition_tensor.expand(n_samples, -1, -1).contiguous()
            elif condition_tensor.size(0) not in (1, n_samples):
                raise ValueError(
                    f"condition_tensor batch dim {condition_tensor.size(0)} "
                    f"!= n_samples {n_samples}"
                )

        self.clean_condition_tensor = condition_tensor

        # SILVR is excluded: this block centres the reference over its own
        # n_ref rows and throws the removed mean away, whereas SILVR must
        # centre over node_mask.sum() and KEEP the mean (it accumulates into
        # total_gravity_center_shift, which is what puts the sample back in
        # the binding site). SILVR does its own centring in its setup block.
        if condition_alg != "silvr" and condition_component in ("xh", "x"):
            n_node_cond = condition_tensor.size(1)

            node_mask_cond = torch.ones((n_samples, n_node_cond,1), device=z.device)
            if condition_alg != "outpaintft":
                condition_tensor[:,:, : self.n_dims] = remove_mean_with_mask(condition_tensor[:,:, : self.n_dims], node_mask_cond)


        if n_frames > 0:
            
            z_size = (z.size(0), z.size(1), z.size(2)-len(self.extra_norm_values))
         
            chain = torch.zeros((n_frames,) + z_size, device=z.device)
            if n_retrys > 0:
                n_frames_retry = int(n_frames*t_retry/self.T)
                try:
                    chain_retry = [torch.zeros((n_frames_retry,) + z_size, device=z.device) for _ in range(n_retrys)]
                except RuntimeError:
                    chain_retry = [torch.zeros((n_frames_retry+1,) + z_size, device=z.device) for _ in range(n_retrys)]
     
            s_saves = torch.linspace(0, self.T, 
                                        steps=n_frames, device=z.device).long() 
        
        t_int_start = self.T
        if condition_alg == "inpaint":

            denoising_strength = inpaint_cfgs.get("denoising_strength", None)
            if denoising_strength is None:
                raise ValueError("denoising_strength must be specified in to use inpainting")
            noise_initial_mask = inpaint_cfgs.get("noise_initial_mask", False)
            mask_node_index_raw = inpaint_cfgs.get("mask_node_index", [])
            if mask_node_index_raw is None:
                mask_node_index_raw = []
            mask_node_index = torch.tensor(mask_node_index_raw, device=z.device, dtype=torch.long)
            if mask_node_index.dim() == 1:
                mask_node_index = mask_node_index.unsqueeze(0)
            constraint_strength = inpaint_cfgs.get("constraint_strength", 0.8)
            scale_factor = inpaint_cfgs.get("scale_factor", 1.1)
            # Extended-inpainting init knobs (mirror outpaint).
            ext_init_method = str(inpaint_cfgs.get("init_method", "skeleton"))
            ext_skeleton_type = str(inpaint_cfgs.get("skeleton_type", "random_walk"))
            ext_bond_len = float(inpaint_cfgs.get("bond_len", 1.5))
            ext_forward_noise = str(inpaint_cfgs.get("forward_noise", "jitter"))
            ext_seed_dist = float(inpaint_cfgs.get("seed_dist", 2.0))
            ext_min_dist = float(inpaint_cfgs.get("min_dist", 1.0))
            ext_spread = float(inpaint_cfgs.get("spread", 1.0))
            ext_n_bq = int(inpaint_cfgs.get("n_bq_atom", 0))
            ext_jitter_scale = _resolve_jitter_scale(
                inpaint_cfgs, ext_init_method, ext_forward_noise
            )
            t_int_start =  self.T
            unmasked_node_indices = [i for i in range(n_node_cond) if i not in mask_node_index]
            n_node_cond = condition_tensor.size(1)
            d = torch.full((n_samples, 1), fill_value=denoising_strength, device=z.device)
            
            gamma_d = self.inflate_batch_array(self.gamma(d), condition_tensor)
        
            # Compute alpha_t and sigma_t from gamma.
            alpha_d = self.alpha(gamma_d, condition_tensor)
            sigma_d = self.sigma(gamma_d, condition_tensor)

            #-----------------------partial inpainting------------------------------------#
            if mask_node_index.size(1) > 0:
                # Reorder condition_tensor such that connector nodes are the first nodes
                connector_mask = torch.zeros(condition_tensor.size(1), dtype=torch.bool, device=z.device)
                connector_mask[mask_node_index] = True  
                non_connector_mask = ~connector_mask          
                reordered_condition_tensor = torch.cat([condition_tensor[:, non_connector_mask, :], 
                                                        condition_tensor[:, connector_mask, :]], dim=1)
                condition_tensor = reordered_condition_tensor

                x_ref = condition_tensor[:, :, : self.n_dims].squeeze(0)
                # NOTE: scaffold conditioning reads atomic numbers from the
                # latent's trailing charge column and so requires
                # include_charges=True. Left as-is deliberately: this path
                # is unreachable without a condition_tensor.
                h_int_ref = condition_tensor[:, :, -1].squeeze(0) * self.norm_values[2]

                if x_ref.dim() > 2:
                    x_ref = x_ref[0]
                if h_int_ref.dim() > 2:
                    h_int_ref = h_int_ref[0]
                        
                mol_graph = create_pyg_graph(x_ref, h_int_ref, r=3)
                mol_graph = correct_edges(mol_graph, scale_factor=1.2)

                connector_idxs = []
                connector_n_bonds = []
                mask_node_index_corr = torch.arange(condition_tensor.size(1) - mask_node_index.size(1), condition_tensor.size(1), device=z.device, dtype=torch.long)
                unmasked_node_indices = torch.arange(condition_tensor.size(1) - mask_node_index.size(1), device=z.device).tolist()
                for node in unmasked_node_indices:
                    adj_nodes = mol_graph.edge_index[1][mol_graph.edge_index[0] == node]
                    adj_nodes = adj_nodes.long().to(z.device)
                    if any(torch.isin(adj_nodes, mask_node_index_corr)):
                        connector_idxs.append(node)
                        connector_n_bonds.append(len(adj_nodes))  

                connector_idxs = list(set(connector_idxs))
                connector_indices = torch.tensor(connector_idxs, device=z.device, dtype=torch.long)
                connector_degrees = torch.tensor(connector_n_bonds, device=z.device, dtype=torch.long)      
                connector_dicts = {j.item(): [connector_degrees[i].item()] for i, j in enumerate(connector_indices)}   
                
                node_mask_m = torch.ones((n_samples, mask_node_index.size(1), 1), device=z.device)
                
                node_mask_um = torch.ones((n_samples, n_node_cond-mask_node_index.size(1)), device=z.device)  
                node_mask_um = node_mask_um.unsqueeze(-1)
    
                n_masked_nodes = mask_node_index.size(1)
                mask_node_bool_corr = torch.cat([
                    torch.zeros(n_node_cond - n_masked_nodes, device=z.device),
                    torch.ones(n_masked_nodes, device=z.device),
                ]).bool()
                
                xh_unmasked = condition_tensor[:,~mask_node_bool_corr, :] # not to be denoise

                # noise the masked nodes
                if noise_initial_mask:
                    eps_s_ref = self.sample_combined_position_feature_noise(
                        n_samples=n_samples, n_nodes=mask_node_index.size(1), node_mask=node_mask_m)
                    z = alpha_d * condition_tensor[:, mask_node_bool_corr, :] + sigma_d * eps_s_ref # to be denoised
                    
                    z = torch.cat([xh_unmasked, z], dim=1)
                    z = torch.cat(  
                        [
                            remove_mean_with_mask(z[:, :, : self.n_dims], node_mask_cond),
                            z[:, :, self.n_dims :],
                        ],
                        dim=2,
                    )      
       
                    ci_keys = torch.tensor(
                        list(connector_dicts.keys()), device=z.device, dtype=torch.long
                    )
                    for b in range(n_samples):
                        condition_pos_b = xh_unmasked[b, :, : self.n_dims]
                        zs_new_pos_b = z[b, mask_node_bool_corr, : self.n_dims]
                        _, zl_b = find_close_points_torch_and_push_op2(
                            condition_pos_b,
                            zs_new_pos_b,
                            connector_indices=ci_keys,
                            d_threshold_f=1.8,
                            w_b=2,
                            all_frozen=False,
                            z_ref=None,
                            z_tgt=None,
                            scale_factor=scale_factor,
                        )
                        z[b, mask_node_bool_corr, : self.n_dims] = zl_b

                
                
                # keep the masked nodes clean
                else:
                    z = torch.cat([xh_unmasked, condition_tensor[:, mask_node_bool_corr, :]], dim=1)

            #-----------------------all inpainting------------------------------------#
            else:
                if condition_component == "xh":
                    eps_s_ref = self.sample_combined_position_feature_noise(
                    n_samples=condition_tensor.size(0), n_nodes=condition_tensor.size(1), node_mask=node_mask[:, :condition_tensor.size(1), :])
                    if noise_initial_mask:
                        z = alpha_d * condition_tensor + sigma_d * eps_s_ref
                    else:
                        z = condition_tensor
                    z = torch.cat(
                        [
                            remove_mean_with_mask(z[:, :, : self.n_dims], node_mask[:, :condition_tensor.size(1), :]),
                            z[:, :, self.n_dims :],
                        ],
                        dim=2,
                    )
                    xh_unmasked = None
                elif condition_component == "x":
                    eps_s_ref = sample_center_gravity_zero_gaussian_with_mask(
                        size=(condition_tensor.size(0), condition_tensor.size(1), self.n_dims),
                        device=node_mask.device,
                        node_mask=node_mask[:, :condition_tensor.size(1), :],
                    )
                    if noise_initial_mask:
                        z = torch.cat([
                            alpha_d * condition_tensor[:, :, : self.n_dims] + sigma_d * eps_s_ref,
                            condition_tensor[:, :, self.n_dims:]
                        ], dim=2)
                    else:
                        z = condition_tensor
                    z = torch.cat(
                        [
                            remove_mean_with_mask(z[:, :, : self.n_dims], node_mask[:, :condition_tensor.size(1), :]),
                            z[:, :, self.n_dims :],
                        ],
                        dim=2,
                    )
                elif condition_component == "h":
                    eps_s_ref = sample_gaussian_with_mask(
                        size=(condition_tensor.size(0), condition_tensor.size(1), self.in_node_nf),
                        device=node_mask.device,
                        node_mask=node_mask[:, :condition_tensor.size(1), :],
                    )
                    if noise_initial_mask:
                        z = torch.cat([
                            condition_tensor[:, :, : self.n_dims],
                            alpha_d * condition_tensor[:, :, self.n_dims:] + sigma_d * eps_s_ref
                        ], dim=2)
                    else:
                        z = condition_tensor
                    z = torch.cat(
                        [
                            remove_mean_with_mask(z[:, :, : self.n_dims], node_mask[:, :condition_tensor.size(1), :]),
                            z[:, :, self.n_dims :],
                        ],
                        dim=2,)
                mask_node_bool_corr = torch.ones(condition_tensor.size(1), dtype=torch.bool, device=z.device)
                connector_dicts = {}
            #-----------------------extended inpainting-----------------------------------
            n_node_extend = n_nodes - n_node_cond
  
            if n_node_extend > 0:
                if mask_node_index.size(1) > 0:
                    n_connector = mask_node_index.size(1)
                    connector_indices = torch.arange(0 , n_connector, device=z.device) + condition_tensor[:, ~mask_node_bool_corr, :].shape[1]
                else:
                    n_connector = condition_pos.size(0)
                    connector_indices = torch.arange(0 , n_connector, device=z.device)
                
                new_nodes = self._init_inpaint_extend_nodes(
                    z, connector_indices, n_node_extend,
                    init_method=ext_init_method, skeleton_type=ext_skeleton_type,
                    bond_len=ext_bond_len, seed_dist=ext_seed_dist,
                    min_dist=ext_min_dist, spread=ext_spread, n_bq_atom=ext_n_bq,
                    forward_noise=ext_forward_noise, jitter_scale=ext_jitter_scale,
                    alpha_d=alpha_d, sigma_d=sigma_d,
                )
                z = torch.cat([z, new_nodes], dim=1)

                if mask_node_index.size(1) > 0:
                    mask_node_bool_corr = torch.cat([mask_node_bool_corr,
                                                     torch.ones(n_node_extend, device=z.device)]).bool()
                                       
                                       
            self.condition_tensor = z[:, ~mask_node_bool_corr, :] 
        
        elif condition_alg in ["outpaint", "outpaintft"]:

            t_start = outpaint_cfgs.get("t_start", 1.0)
            t_int_start = int(t_start * self.T)
            constraint_strength = outpaint_cfgs.get("constraint_strength", 0.8)
            scale_factor = outpaint_cfgs.get("scale_factor", 1.1)
            t_critical = outpaint_cfgs.get("t_critical", 0.05)
            
            connector_dicts = resolve_connectors(outpaint_cfgs)
            connector_indices = list(connector_dicts.keys())
            connector_indices = torch.tensor(connector_indices, device=z.device, dtype=torch.long)
            min_dist = outpaint_cfgs.get("min_dist", 1.0)
            seed_dist = outpaint_cfgs.get("seed_dist", 2.0)
            spread = outpaint_cfgs.get("spread", 1.0)
            n_bq_atom = outpaint_cfgs.get("n_bq_atom", 0)
            init_method = str(outpaint_cfgs.get("init_method", "skeleton"))
            skeleton_type = str(outpaint_cfgs.get("skeleton_type", "random_walk"))
            bond_len = float(outpaint_cfgs.get("bond_len", 1.5))
            forward_noise = str(outpaint_cfgs.get("forward_noise", "jitter"))
            jitter_scale = _resolve_jitter_scale(
                outpaint_cfgs, init_method, forward_noise
            )

            natom_ref = condition_tensor.size(1)
            natom_extra = n_nodes - natom_ref

            # Validate before building: every placement builder indexes the
            # scaffold with these, so a bad set otherwise dies as an opaque
            # ZeroDivisionError/IndexError inside the builder.
            if connector_indices.numel() == 0 and not (init_method == "seed" and n_bq_atom > 0):
                raise ValueError(
                    "connectors ({atom_index: [n_bonds]}) must be "
                    "specified to use outpainting"
                )
            if any(idx > natom_ref - 1 for idx in connector_indices):
                raise ValueError("connectors: atom index out of bounds for the reference structure")

            new_nodes = self._build_outpaint_extras(
                condition_tensor, connector_indices, natom_extra,
                init_method, skeleton_type, seed_dist, min_dist, spread,
                n_bq_atom, bond_len,
            )
            z = torch.cat([condition_tensor, new_nodes], dim=1)

            z[:, :natom_ref, :] = condition_tensor
            n_nodes = z.size(1)
            node_mask = torch.ones((n_samples, n_nodes), device=z.device)
            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=z.device).unsqueeze(0)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(z.size(0) * n_nodes * n_nodes, 1).to(self.device)  
            node_mask = node_mask.unsqueeze(-1)
            z_size = (z.size(0), z.size(1), z.size(2)-len(self.extra_norm_values))
            
            if n_frames > 0:
                chain = torch.zeros((n_frames,) + z_size, device=z.device)
                if n_retrys > 0:
                    n_frames_retry = int(n_frames*t_retry/self.T)
                    try:
                        chain_retry = [torch.zeros((n_frames_retry,) + z_size, device=z.device) for _ in range(n_retrys)]
                    except RuntimeError:
                        chain_retry = [torch.zeros((n_frames_retry+1,) + z_size, device=z.device) for _ in range(n_retrys)]
                    
            condition_tensor = z[:, :natom_ref, :]
            mask_node_bool_corr = torch.cat([torch.zeros(natom_ref, device=z.device),
                                                torch.ones(n_nodes - natom_ref, device=z.device)]).bool()
            self.condition_tensor = z[:, ~mask_node_bool_corr, :]

            # Training-consistent init: forward-noise the clean template to
            # t_start instead of using the raw seed blob directly.
            if init_method != "seed":
                z = self._outpaint_forward_noise(
                    z, condition_tensor, natom_ref, node_mask, t_int_start,
                    reference_freeze_mode="all",
                    forward_noise=forward_noise, jitter_scale=jitter_scale,
                )
                condition_tensor = z[:, :natom_ref, :]
                self.condition_tensor = z[:, ~mask_node_bool_corr, :]

        elif condition_alg == "silvr":
            # SILVR (Runcie & Mey 2023): every atom stays mobile; each reverse
            # step nudges the reference rows of the latent towards a freshly
            # noised copy of the reference. Setup mirrors the published
            # sampler (silvr_fork_en_diffusion.py L774-890).
            if self.ndim_extra > 0:
                raise ValueError(
                    "silvr does not support models with extra node features "
                    f"(ndim_extra={self.ndim_extra}): z is laid out as "
                    "[x, h_cat, h_int, h_extra] while the reference tensor is "
                    "[x, h_cat, h_extra, h_int], so the channels would be "
                    "silently scrambled."
                )
            if condition_tensor is None:
                raise ValueError(
                    "silvr requires a reference structure; set "
                    "condition_configs.reference_structure_path."
                )

            silvr_cfgs = silvr_cfgs or {}
            n_ref = condition_tensor.size(1)
            if n_ref > n_nodes:
                raise ValueError(
                    f"silvr reference has {n_ref} atoms but the target "
                    f"molecule size is {n_nodes}; the reference must fit."
                )
            shift_centre = bool(silvr_cfgs.get("shift_centre", True))

            # Per-atom rates (silvr_rates) or one uniform rate (silvr_rate).
            silvr_rates = silvr_cfgs.get("silvr_rates", None)
            if silvr_rates:
                if len(silvr_rates) != n_ref:
                    raise ValueError(
                        f"silvr_rates has {len(silvr_rates)} entries but the "
                        f"reference has {n_ref} atoms."
                    )
                rate_row = torch.tensor(
                    list(silvr_rates), dtype=z.dtype, device=z.device
                )
            else:
                rate_row = torch.full(
                    (n_ref,),
                    float(silvr_cfgs.get("silvr_rate", 0.01)),
                    dtype=z.dtype,
                    device=z.device,
                )
            if bool(((rate_row < 0.0) | (rate_row > 1.0)).any()):
                raise ValueError("silvr rate(s) must lie in [0, 1].")

            # (1, n_nodes, 1): rate on the reference rows, 0 on dummy/pad rows.
            silvr_rate = torch.zeros(
                (1, n_nodes, 1), dtype=z.dtype, device=z.device
            )
            silvr_rate[0, :n_ref, 0] = rate_row
            ref_node_mask = torch.zeros(
                (1, n_nodes, 1), dtype=z.dtype, device=z.device
            )
            ref_node_mask[:, :n_ref, :] = 1.0

            # Reference zero-padded to the full node count. The source
            # hardcodes 181 (GEOM-drugs max_n_nodes) here.
            #
            # DELIBERATE DEVIATION -- DO NOT "FIX" BACK: condition_tensor comes
            # from preprocess_ref_structure(), i.e. already divided by
            # model.norm_values, so it lives in the same latent space as z. The
            # published sampler mixes RAW one-hot into a latent normalized by
            # [1, 4, 10], leaving its feature channels 4x hot. Coordinates are
            # unaffected (norm_values[0] == 1), but feature guidance here is
            # therefore ~4x weaker than the paper's at the same silvr_rate.
            xh_ref = torch.zeros_like(z)
            xh_ref[:, :n_ref, :] = condition_tensor[:, :n_ref, :]

            # Initial mean, over node_mask.sum() (NOT n_ref) as in the source,
            # retained so it can be added back after the loop.
            n_live = node_mask.sum(1, keepdim=True)
            mean = xh_ref[:, :, : self.n_dims].sum(1, keepdim=True) / n_live
            mean = F.pad(mean, (0, z.size(2) - self.n_dims))
            total_gravity_center_shift = mean
            xh_ref = xh_ref - mean * node_mask

            # SILVR freezes nothing, so there is no scaffold block to carry;
            # explicitly clear the stateful attribute other modes leave behind
            # (otherwise the mask_node_bool_corr deref below fires).
            self.condition_tensor = None

        if context is not None:
            context = (
                context[0,0].repeat(1, z.size(1), 1).to(self.device)
            )*node_mask
        
        
        z = torch.cat(  
            [
                remove_mean_with_mask(z[:, :, : self.n_dims], node_mask),
                z[:, :, self.n_dims :],
            ],
            dim=2,
        )       
        if self.condition_tensor is not None:
            self.condition_tensor = z[:, ~mask_node_bool_corr, :]
    

        to_inspect_chain = False        
        if n_retrys > 0:
            snap_shot = {}
        write_index = 0
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        with tqdm(total=self.T, dynamic_ncols=True, unit="steps") as pbar:
            for i, s in enumerate(reversed(range(0, self.T))):
                s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
                t_array = s_array + 1
                s_array = s_array / self.T
                t_array = t_array / self.T

                if s > t_int_start:
                    pbar.update(1)
                    if n_frames > 0:
                        if s in s_saves:
                            chain[write_index] = self.unnormalize_z(z, node_mask)
                            write_index += 1
                            if n_retrys > 0:
                                info_dict = {
                                    "z": z,
                                    "condition_tensor": self.condition_tensor,
                                    "mask_node_bool_corr": mask_node_bool_corr
                                }
                                snap_shot[s] = info_dict
                    continue


                try:
                    if condition_alg == "ssgd":
                        z = self.sample_p_zs_given_zt_ssgd(
                        s_array,
                        t_array,
                        z, 
                        node_mask, 
                        edge_mask, 
                        context,
                        condition_tensor,
                        condition_component,
                        fix_noise=fix_noise,
                    )
                    elif condition_alg == "inpaint":
                            
                        if s/self.T < denoising_strength: 
                            z = self.sample_p_zs_given_zt_ip(
                                s_array,
                                t_array,
                                z,
                                node_mask,
                                edge_mask,
                                context,
                                mask_node_bool_corr,
                                connector_dicts,
                                constraint_strength=constraint_strength,
                                scale_factor=scale_factor,
                                fix_noise=fix_noise
                                )
                        else:
                            z = z
                    elif condition_alg == "outpaint":

                        z = self.sample_p_zs_given_zt_op(
                            s_array, t_array, z,
                            node_mask, edge_mask,
                            context,
                            mask_node_bool_corr,
                            connector_dicts=connector_dicts,
                            constraint_strength=constraint_strength,
                            scale_factor=scale_factor,
                            fix_noise=fix_noise,
                            )
                    elif condition_alg == "outpaintft": 
                        z = self.sample_p_zs_given_zt_op_ft(
                            s_array,
                            t_array, 
                            z, 
                            condition_tensor, 
                            node_mask, edge_mask,
                            context,
                            t_critical=t_critical,
                            fix_noise=fix_noise
                            )
                    elif condition_alg == "silvr":
                        z, xh_ref, total_gravity_center_shift = (
                            self.sample_p_zs_given_zt_silvr(
                                s_array,
                                t_array,
                                z,
                                node_mask,
                                edge_mask,
                                context,
                                xh_ref,
                                silvr_rate,
                                ref_node_mask,
                                total_gravity_center_shift,
                                condition_component,
                                fix_noise=fix_noise,
                            )
                        )
                    else:
                        z = self.sample_p_zs_given_zt(
                        s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise
                        )
                except ValueError as e:
                    if s < self.T/100: # arbitary threshold
                        to_inspect_chain = True
                    else:
                        t_retry = int(self.T*0.9)
                        if n_retrys > 0:
                            n_frames_retry = int(n_frames*t_retry/self.T)
                            chain_retry = [torch.zeros((n_frames_retry,) + z_size, device=z.device) for _ in range(n_retrys)]
                        # raise e  or just fuck this
                    break
                if n_frames > 0:
                    if s in s_saves:
                        chain[write_index] = self.unnormalize_z(z, node_mask)
                        write_index += 1
                        if n_retrys > 0:
                            info_dict = {
                                "z": z,
                                "condition_tensor": self.condition_tensor,
                                "mask_node_bool_corr": mask_node_bool_corr
                            }
                            snap_shot[s] = info_dict
                    
                pbar.update(1)
    
        N_check = 5
        N_COMPONENT_MAX = 10
        N_DEGREE_MAX = 8

        z = z.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        if condition_alg == "silvr" and shift_centre:
            # Undo the accumulated per-step re-centring once, putting the
            # sample back in the reference's frame (fork L1081-1089). Feature
            # channels are zeroed first: the source's "mean" spans all
            # channels, but only coordinates may be translated.
            total_gravity_center_shift = total_gravity_center_shift.type(z.dtype)
            total_gravity_center_shift[:, :, self.n_dims :] = 0
            z = z + total_gravity_center_shift * node_mask

        try:
            # Finally sample p(x, h | z_0).
            x, h = self.sample_p_xh_given_z0(
                    z, node_mask, edge_mask, context, fix_noise=fix_noise
                )
        except ValueError as e:
            to_inspect_chain = True       
        # 2 ---------------------------------------------------------------------
        # Address overshoot cases, there maybe a good xh already before 0    
        if to_inspect_chain and n_frames > 0:
            n_chain = chain.size(0)
            x_chain = chain[:, :, :, 0:self.n_dims]
            
            # Compute correct indices based on extra features
            if self.ndim_extra > 0:
                nextra = self.ndim_extra
                n_cat_core = self.num_classes - nextra
                h_int_idx = self.n_dims + n_cat_core
                one_hot_chain = chain[:, :, :, self.n_dims:h_int_idx]
                charges_chain = chain[:, :, :, h_int_idx:h_int_idx+1].squeeze(-1)
                extra_chain = chain[:, :, :, h_int_idx+1:]
            else:
                _, one_hot_chain, h_int_chain = split_frame(self, chain)
                charges_chain = atomic_numbers(self,
                    h_int_chain, one_hot_chain
                ).squeeze(-1)
                extra_chain = None
            
            one_hot_chain = F.one_hot(
                torch.argmax(one_hot_chain, dim=-1),
                num_classes=self.num_classes - self.ndim_extra,
            )
            charges_chain = torch.round(charges_chain).long()
            charges_chain = torch.abs(charges_chain)*self.norm_values[2]

            good_chain = False
            for i in range(N_check):
                try:
                    is_connected, num_components, n_degrees = check_quality(x_chain[-1-i], charges_chain[-1-i])
                except Exception as e:
                    # assume errors with check_quality is associated with bad mols
                    is_connected = False
                    num_components = 1000000
                    n_degrees = [0]*x_chain.size(1)      
                    
                good_chain = (num_components < N_COMPONENT_MAX) and all([n_degree < N_DEGREE_MAX for n_degree in n_degrees])
                if good_chain:
                    h = {}
                    x  = x_chain[-1-i]
                    h["integer"] = charges_chain[-1-i].unsqueeze(-1)
                    h["categorical"] = one_hot_chain[-1-i]
                    if extra_chain is not None:
                        h["extra"] = extra_chain[-1-i]
                    else:
                        h["extra"] = torch.zeros(0).to(z.device)
                    break
            # cannot find a good xh, last resort
            if not(good_chain):
                z = chain[-1-i]
                x, h = self.sample_p_xh_given_z0(
                    z, node_mask, edge_mask, context, fix_noise=fix_noise
                )
        
        # assert_mean_zero_with_mask(x, node_mask)

        # SILVR samples are deliberately off-origin (that is what shift_centre
        # buys: coordinates in the reference/binding-site frame), so this
        # projection must not run for it — it would undo the shift entirely.
        # The published sampler has the equivalent block commented out.
        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2 and condition_alg != "silvr":
            print(
                f"Warning cog drift with error {max_cog:.3f}. Projecting "
                f"the positions down."
            )
            x = remove_mean_with_mask(x, node_mask)

        # 3 ---------------------------------------------------------------------
        try:
            is_connected, num_components, n_degrees = check_quality(
                    x.squeeze(0),
                    atomic_numbers(self,
                        h["integer"], h["categorical"]
                    ).squeeze(0),
                )
        except:
            # assume errors with check_quality is associated with bad mols
            is_connected = False
            num_components = 1000000
            n_degrees = [0]*x.size(1)
    
        # 3.5 ---------------------------------------------------------------------
        # In the event EGNN fails at p(xh | z0), we retry the chain
        bad_xh = (num_components > N_COMPONENT_MAX) or any([n_degree > N_DEGREE_MAX for n_degree in n_degrees])
        if bad_xh and n_retrys > 0 and n_frames > 0:
            x = chain[-2, :, :, 0:self.n_dims]
            if self.ndim_extra > 0:
                nextra = self.ndim_extra
                n_cat_core = self.num_classes - nextra
                h_int_idx = self.n_dims + n_cat_core
                h["categorical"] = chain[-2, :, :, self.n_dims:h_int_idx]
                h["integer"] = chain[-2, :, :, h_int_idx:h_int_idx+1].long()
                h["extra"] = chain[-2, :, :, h_int_idx+1:]
            else:
                _, h_cat_m2, h_int_m2 = split_frame(self, chain[-2])
                h["integer"] = h_int_m2.long()
                h["categorical"] = h_cat_m2
            try:
                is_connected, num_components, n_degrees = check_quality(
                    x.squeeze(0),
                    atomic_numbers(self,
                        h["integer"], h["categorical"]
                    ).squeeze(0),
                )
            except:
                # assume errors with check_quality is associated with bad mols
                is_connected = False
                num_components = 1000000
                n_degrees = [0]*x.size(1)
                
        if condition_alg in ["outpaint", "inpaint"]:
            match_n_degree_connector = True
            for key, value in connector_dicts.items():
                if not(n_degrees[key] in value):
                    match_n_degree_connector = False
                    break
            to_retry = not(is_connected and match_n_degree_connector)
        else:
            to_retry = not(is_connected)

        if n_frames > 0:
            # Chain was allocated without extra features (z_size excludes them),
            # so we should NOT include h["extra"] when storing to chain
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
            chain[-1] = xh     
        else:
            chain_flat = None
            chain = None
        
        
        # 4 ---------------------------------------------------------------------
        if to_retry and n_retrys > 0 and n_frames > 0:
            n_retry = 0
            while (n_retry < n_retrys) and to_retry:
                closest_t = min(snap_shot.keys(), key=lambda x: abs(x - t_retry))
                z = snap_shot[closest_t]["z"]
                self.condition_tensor = snap_shot[closest_t]["condition_tensor"]
                mask_node_bool_corr = snap_shot[closest_t]["mask_node_bool_corr"]

                write_index = 0
                with tqdm(total=t_retry, dynamic_ncols=True, unit="steps") as pbar:
                    for i, s in enumerate(reversed(range(0, t_retry))):
                        s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
                        t_array = s_array + 1
                        s_array = s_array / self.T
                        t_array = t_array / self.T                
                        try:
                            if condition_alg == "inpaint":
                                    
                                if s/self.T < denoising_strength: 
                                    z = self.sample_p_zs_given_zt_ip(
                                        s_array,
                                        t_array,
                                        z,
                                        node_mask,
                                        edge_mask,
                                        context,
                                        mask_node_bool_corr,
                                        connector_dicts,
                                        constraint_strength=constraint_strength,
                                        scale_factor=scale_factor,
                                        fix_noise=fix_noise
                                        )
                            elif condition_alg == "outpaint":

                                z = self.sample_p_zs_given_zt_op(
                                    s_array, t_array, z,
                                    node_mask, edge_mask,
                                    context,
                                    mask_node_bool_corr,
                                    connector_dicts=connector_dicts,
                                    constraint_strength=constraint_strength,
                                    scale_factor=scale_factor,
                                    fix_noise=fix_noise,
                                    )
                            elif condition_alg == "outpaintft": 
                                z = self.sample_p_zs_given_zt_op_ft(
                                    s_array,
                                    t_array, 
                                    z, 
                                    condition_tensor, 
                                    node_mask, edge_mask,
                                    context,
                                    t_critical=t_critical,
                                    fix_noise=fix_noise
                                    )
                            else:
                                z = self.sample_p_zs_given_zt(
                                s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise
                                )
                        except ValueError as e:
                            if s < self.T/100: # arbitary threshold
                                to_inspect_chain = True
                            else:
                                t_retry = int(self.T*0.9)
                                if n_retrys > 0:
                                    n_frames_retry = int(n_frames*t_retry/self.T)
                                    chain_retry = [torch.zeros((n_frames_retry,) + z_size, device=z.device) for _ in range(n_retrys)]
                                # raise e  or just fuck this
                            break
                                    
                        if n_frames > 0 and n_retrys > 0:
                            if s in s_saves:
                                # print(f"write_index: {write_index}")
                                
                                if write_index < n_frames_retry:
                                    chain_retry[n_retry][write_index] = self.unnormalize_z(z, node_mask)
                                    write_index += 1
                        pbar.update(1)               
                    # Finally sample p(x, h | z_0).

                if to_inspect_chain:
                    n_chain = chain_retry[n_retry].size(0)
                    x_chain = chain_retry[n_retry][:, :, :, 0:self.n_dims]
                    x_chain = x_chain.squeeze(1)
                    
                    # Compute correct indices based on extra features
                    if self.ndim_extra > 0:
                        nextra = self.ndim_extra
                        n_cat_core = self.num_classes - nextra
                        h_int_idx = self.n_dims + n_cat_core
                        one_hot_chain = chain_retry[n_retry][:, :, :, self.n_dims:h_int_idx]
                        charges_chain = chain_retry[n_retry][:, :, :, h_int_idx:h_int_idx+1].squeeze(-1)
                        extra_chain_retry = chain_retry[n_retry][:, :, :, h_int_idx+1:]
                    else:
                        _, one_hot_chain, h_int_chain = split_frame(self,
                            chain_retry[n_retry]
                        )
                        charges_chain = atomic_numbers(self,
                            h_int_chain, one_hot_chain
                        ).squeeze(-1)
                        extra_chain_retry = None
                    
                    one_hot_chain = F.one_hot(
                        torch.argmax(one_hot_chain, dim=-1),
                        num_classes=self.num_classes - self.ndim_extra,
                    )
                    charges_chain = torch.round(charges_chain).long()

                    good_chain = False

                
                    for i in range(N_check):
                        try:
                            is_connected, num_components, n_degrees = check_quality(x_chain[-1-i], charges_chain[-1-i])
                        except Exception as e:
                            # assume errors with check_quality is associated with bad mols
                            is_connected = False
                            num_components = 1000000
                            n_degrees = [0]*x_chain.size(1)      
                        good_chain = (num_components < N_COMPONENT_MAX) and all([n_degree < N_DEGREE_MAX for n_degree in n_degrees])

                        if good_chain:
                            h_retry = {}
                            x_retry  = x_chain[-1-i]
                            h_retry["integer"] = charges_chain[-1-i].unsqueeze(-1)
                            h_retry["categorical"] = one_hot_chain[-1-i]
                            if extra_chain_retry is not None:
                                h_retry["extra"] = extra_chain_retry[-1-i]
                            else:
                                h_retry["extra"] = torch.zeros(0).to(z.device)
                            break
                    # cannot find a good xh, last resort
                    if not(good_chain):
                        z = chain[-1-i]
                        x_retry, h_retry = self.sample_p_xh_given_z0(
                            z, node_mask, edge_mask, context, fix_noise=fix_noise
                        )
                            
                else:
                    # Finally sample p(x, h | z_0).
                    x_retry, h_retry = self.sample_p_xh_given_z0(
                        z, node_mask, edge_mask, context, fix_noise=fix_noise
                    )
                    
                max_cog = torch.sum(x_retry, dim=1, keepdim=True).abs().max().item()
                if max_cog > 5e-2:
                    print(
                        f"Warning cog drift with error {max_cog:.3f}. Projecting "
                        f"the positions down."
                    )
                    x_retry = remove_mean_with_mask(x_retry, node_mask)
                    
                assert_mean_zero_with_mask(x_retry, node_mask)
                try:
                    is_connected, num_components, n_degrees = check_quality(
                        x_retry.squeeze(0),
                        atomic_numbers(self,
                            h_retry["integer"], h_retry["categorical"]
                        ).squeeze(0),
                    )
                except:
                    # assume errors with check_quality is associated with bad mols
                    is_connected = False
                    num_components = 1000000
                    n_degrees = [0]*x_retry.size(1)
                    print("Error in check_quality")

   
                bad_xh = (num_components > N_COMPONENT_MAX) or any([n_degree > N_DEGREE_MAX for n_degree in n_degrees])
                # In the event EGNN fails at p(xh | z0), we retry the chain
                if bad_xh and n_frames > 0:
                    x_retry = chain_retry[n_retry][-2, :, :, 0:self.n_dims]
                    if self.ndim_extra > 0:
                        nextra = self.ndim_extra
                        n_cat_core = self.num_classes - nextra
                        h_int_idx = self.n_dims + n_cat_core
                        h_retry["categorical"] = chain_retry[n_retry][-2, :, :, self.n_dims:h_int_idx]
                        h_retry["integer"] = chain_retry[n_retry][-2, :, :, h_int_idx:h_int_idx+1].long()
                        h_retry["extra"] = chain_retry[n_retry][-2, :, :, h_int_idx+1:]
                    else:
                        _, h_cat_m2, h_int_m2 = split_frame(self,
                            chain_retry[n_retry][-2]
                        )
                        h_retry["integer"] = h_int_m2.long()
                        h_retry["categorical"] = h_cat_m2
                    try:
                        is_connected, num_components, n_degrees = check_quality(
                        x_retry.squeeze(0),
                        atomic_numbers(self,
                            h_retry["integer"], h_retry["categorical"]
                        ).squeeze(0),
                    )
                    except:
                        # assume errors with check_quality is associated with bad mols
                        is_connected = False
                        num_components = 1000000
                        n_degrees = [0]*x.size(1)
                
                if condition_alg in ["outpaint", "inpaint"]:
                    match_n_degree_connector = True
                    for key, value in connector_dicts.items():
                        if not(n_degrees[key] in value):
                            match_n_degree_connector = False
                            break
                    to_retry = not(is_connected and match_n_degree_connector)
                else:
                    to_retry = not(is_connected)

                max_cog = torch.sum(x_retry, dim=1, keepdim=True).abs().max().item()
                if max_cog > 5e-2:
                    print(
                        f"Warning cog drift with error {max_cog:.3f}. Projecting "
                        f"the positions down."
                    )
                    x_retry = remove_mean_with_mask(x_retry, node_mask)
                    
                assert_mean_zero_with_mask(x_retry, node_mask)
  
                    
                x = torch.cat([x, x_retry], dim=0)
                h["categorical"] = torch.cat([h["categorical"], h_retry["categorical"]], dim=0) 
                h["integer"] = torch.cat([h["integer"], h_retry["integer"]], dim=0)
                if self.ndim_extra > 0 and "extra" in h and "extra" in h_retry:
                    h["extra"] = torch.cat([h["extra"], h_retry["extra"]], dim=0)

                n_retry+=1
                

            if n_frames > 0:
                # Chain_retry was also allocated without extra features,
                # so we should NOT include h_retry["extra"] when storing
                xh_retry = torch.cat([x_retry, h_retry["categorical"], h_retry["integer"]], dim=2)
                chain_retry[n_retry-1][-1] = xh_retry
                # Return the original denoising trajectory. Retry samples are
                # accumulated in x/h, but chain_flat must still be assigned on
                # this control-flow path.
                chain_flat = chain.view(
                    n_frames,
                    n_samples,
                    z.size(1),
                    z.size(2) - len(self.extra_norm_values),
                )
            else:
                chain_flat = None
                chain = None
        else:
            if n_frames:
                chain_flat = chain.view(n_frames, n_samples , z.size(1), z.size(2)-len(self.extra_norm_values)) 
  
        return x, h, chain_flat

    @torch.no_grad()
    def sample_chain(
        self,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        context,
        keep_frames=None,
    ):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context
            )

            assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        assert_mean_zero_with_mask(x[:, :, : self.n_dims], node_mask)

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat
