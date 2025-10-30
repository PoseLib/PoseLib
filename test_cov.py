import numpy as np
import torch
import torch.nn.functional as F
from typing import Literal
from einops import einsum
import poselib
 
def to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
 
 
def from_homogeneous(x: torch.Tensor) -> torch.Tensor:
    return x[..., :-1] / x[..., -1:]
 
 
def get_normalized_grid(
    B: int,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=overload_device or device)
            for n in (B, H, W)
        ],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n
 
 
def get_pixel_grid(
    B: int,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[torch.arange(n, device=overload_device or device) + 0.5 for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n
 
 
def to_normalized(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    return torch.stack((2 * x[..., 0] / W, 2 * x[..., 1] / H), dim=-1) - 1
 
 
def to_pixel(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    return torch.stack(((x[..., 0] + 1) / 2 * W, (x[..., 1] + 1) / 2 * H), dim=-1)
 
 
def bhwc_interpolate(
    x: torch.Tensor,
    size: tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool | None = None,
) -> torch.Tensor:
    return F.interpolate(
        x.permute(0, 3, 1, 2), size=size, mode=mode, align_corners=align_corners
    ).permute(0, 2, 3, 1)
 
 
def bhwc_grid_sample(
    x: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool | None = None,
) -> torch.Tensor:
    return F.grid_sample(
        x.permute(0, 3, 1, 2), grid, mode=mode, align_corners=align_corners
    ).permute(0, 2, 3, 1)
 
 
def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))
 
 
def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))
 
 
def compute_pose_error(R_gt, t_gt, R, t):
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R
 
 
def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)  # type: ignore
    return aucs
 
def cov_mat_from_cov_params(c: torch.Tensor) -> torch.Tensor:
    return prec_mat_from_prec_params(c)
 
def prec_mat_from_prec_params(p: torch.Tensor) -> torch.Tensor:
    P = p.new_zeros(p.shape[0], p.shape[1], p.shape[2], 2, 2)
    P[..., 0, 0] = p[..., 0]
    P[..., 1, 0] = p[..., 1]
    P[..., 0, 1] = p[..., 1]
    P[..., 1, 1] = p[..., 2]
    return P
 
def to_double_angle_rep(v: torch.Tensor) -> torch.Tensor:
    angle = torch.atan2(v[..., 1], v[..., 0])
    double_angle_rep = torch.stack((torch.cos(2*angle), torch.sin(2*angle)), dim=-1)
    return double_angle_rep
 
 
def prec_mat_to_flow(P: torch.Tensor, vis_max: float, mode: Literal["smallest", "largest"] = "largest") -> torch.Tensor:
    vals, vecs = torch.linalg.eigh(P)
    if mode == "smallest":
        # vis_val = vals[..., :1]
        vis_vec = vecs[..., 0]
    elif mode == "largest":
        # vis_val = vals[..., -1:]
        vis_vec = vecs[..., -1]
    else:
        raise ValueError(f"Invalid mode: {mode}")
    # select upper half-plane
    vis_vec = torch.where(vis_vec[..., 1:2] >= 0, vis_vec, -vis_vec)
    double_angle_rep = to_double_angle_rep(vis_vec)
    scale = (vals[..., 0] * vals[..., 1]).pow(0.25).clamp(0, vis_max)[..., None]
    # big_scale =  vis_val.clamp(min=1e-6).sqrt().clamp(0, vis_max)
    flow = scale * double_angle_rep
    return flow
 
def prec_params_to_flow(p: torch.Tensor, vis_max: float, mode: Literal["smallest", "largest"] = "largest") -> torch.Tensor:
    P = prec_mat_from_prec_params(p)
    return prec_mat_to_flow(P, vis_max, mode)
 

def sampson_error(pts1, pts2, F):
    """
    pts1: (N, 2)
    pts2: (N, 2)
    F: (3, 3)
    returns: (N,) Sampson error for each point correspondence
    """
    N = pts1.shape[0]
    pts1_h = np.hstack((pts1, np.ones((N, 1)))) # (N, 3)
    pts2_h = np.hstack((pts2, np.ones((N, 1)))) # (N, 3)

    Fx1 = F @ pts1_h.T # (3, N)
    Ftx2 = F.T @ pts2_h.T # (3, N)

    denom = Fx1[0,:]**2 + Fx1[1,:]**2 + Ftx2[0,:]**2 + Ftx2[1,:]**2
    numer = np.sum(pts2_h * (F @ pts1_h.T).T, axis=1)**2

    return np.sqrt(numer / denom)

# if your covs are (m,2,2) -> make a list[ (2,2) arrays ]
def cov_stack_to_list(cov_stack):
    cov_stack = np.asarray(cov_stack, dtype=np.float64)
    assert cov_stack.ndim == 3 and cov_stack.shape[1:] == (2,2), "covariances must be (m,2,2)"
    # ensure contiguous (pybind can be picky)
    return [np.ascontiguousarray(cov_stack[i]) for i in range(cov_stack.shape[0])]


def main():
    data = np.load('preds_with_precision.npy', allow_pickle=True).item()
    print(data.keys()) # dict_keys(['warp_pred', 'ov_and_prec_pred', 'ov_and_prec_pred_bkwd', 'warp_bkwd', 'K_A', 'K_B', 'T_AtoB', 'H', 'W'])

    warp_AB = data['warp_pred'] # (B, 768, 1024, 2)
    warp_BA = data['warp_bkwd'] # (B, 768, 1024, 2)
    ov_and_prec_AB = data['ov_and_prec_pred'] # (B, 768, 1024, 4)
    ov_and_prec_BA = data['ov_and_prec_pred_bkwd'] # (B, 768, 1024, 4)
    K_A = data['K_A'] # (B, 3, 3)
    K_B = data['K_B'] # (B, 3, 3)
    T_AtoB = data['T_AtoB'] # (B, 4, 4)
    R_gt = T_AtoB[:, :3, :3]
    t_gt = T_AtoB[:, :3, 3]
    H, W = data['H'], data['W']

    num_points = 1000
    tries_per_image = 100
    threshold = 16.0 # pixels
    threshold_cov = 5.0 # Whitened distance

    for im_k in range(warp_AB.shape[0]):
        print(f'-------------------------------- Image %d --------------------------' % im_k)

        errs_init = []
        errs_ref_pt = []
        errs_ref_cov = []
        errs_rsc_cov = []

        for tries in range(tries_per_image):
            points_A = np.vstack((np.random.randint(0, W, size=num_points),
                                np.random.randint(0, H, size=num_points))).T.astype(np.float32)
            
            
            
            points_B = bhwc_grid_sample(torch.from_numpy(warp_AB[im_k:im_k+1]).float(), 
                                            to_normalized(torch.from_numpy(points_A).unsqueeze(0).float(), H, W).unsqueeze(0),
                                            align_corners=True).squeeze(0)
            points_B = to_pixel(points_B, H, W).squeeze(0).numpy()
            
            ov_and_prec1 = bhwc_grid_sample(torch.from_numpy(ov_and_prec_AB[im_k:im_k+1]).float(), 
                                            to_normalized(torch.from_numpy(points_A).unsqueeze(0).float(), H, W).unsqueeze(0),
                                            align_corners=True).squeeze(0)
            
            ov_and_prec2 = bhwc_grid_sample(torch.from_numpy(ov_and_prec_BA[im_k:im_k+1]).float(), 
                                            to_normalized(torch.from_numpy(points_B).unsqueeze(0).float(), H, W).unsqueeze(0),
                                            align_corners=True).squeeze(0)
            prec1 = prec_mat_from_prec_params(ov_and_prec1[:, :, 1:4].unsqueeze(0)).squeeze(0).squeeze(0)
            prec2 = prec_mat_from_prec_params(ov_and_prec2[:, :, 1:4].unsqueeze(0)).squeeze(0).squeeze(0)

            good = (ov_and_prec1[:, :, 0] > 0.5) & (ov_and_prec2[:, :, 0] > 0.5)
            good = good.squeeze(0)

            points_A = points_A[good.numpy()]
            points_B = points_B[good.numpy()]
            prec1 = prec1[good.numpy()]
            prec2 = prec2[good.numpy()]
            
            cov1 = torch.linalg.inv(prec1).float().numpy()
            cov2 = torch.linalg.inv(prec2).float().numpy()

            T_AtoB = data['T_AtoB'] # (B, 4, 4)
            R_gt = T_AtoB[:, :3, :3]
            t_gt = T_AtoB[:, :3, 3]

            K1 = K_A[im_k]
            K2 = K_B[im_k]

            cam1 = {
                'model': 'PINHOLE',
                'width': W,
                'height': H,
                'params': [K1[0,0], K1[1,1], K1[0,2], K1[1,2]],  # fx, fy, cx, cy
            }
            cam2 = {
                'model': 'PINHOLE',
                'width': W,
                'height': H,
                'params': [K2[0,0], K2[1,1], K2[0,2], K2[1,2]],  # fx, fy, cx, cy
            }

            opt = {'max_epipolar_error': threshold}
            opt_cov = {'max_epipolar_error': threshold_cov}
            opt_ref = {'loss_type': 'TRIVIAL'}
            pose_init, info_init = poselib.estimate_relative_pose(points_A, points_B, cam1, cam2, opt, opt_ref)
            inl = info_init['inliers']

            if np.sum(inl) < 10:
                continue

            # Refine with points only, should be exactly the same as initial since we used all inliers and RANSAC already does LM refinement
            pose_ref_pt, info_ref_pt = poselib.refine_relative_pose(points_A[inl], points_B[inl], pose_init, cam1, cam2, opt_ref)

            # Refine with points + covariances
            opt_ref['verbose'] = False
            pose_ref_cov, info_ref_cov = poselib.refine_relative_pose_cov(points_A[inl], points_B[inl], 
                                                                cov1[inl],
                                                                cov2[inl],
                                                                pose_init, cam1, cam2, opt_ref)


            pose_rsc_cov, info_rsc_cov = poselib.estimate_relative_pose_cov(points_A, points_B, 
                                                                cov1, cov2,
                                                                cam1, cam2, opt_cov, opt_ref)


            error_pose_init = compute_pose_error(R_gt[im_k], t_gt[im_k], pose_init.R, pose_init.t)
            error_pose_ref_pt = compute_pose_error(R_gt[im_k], t_gt[im_k], pose_ref_pt.R, pose_ref_pt.t)
            error_pose_ref_cov = compute_pose_error(R_gt[im_k], t_gt[im_k], pose_ref_cov.R, pose_ref_cov.t)
            error_pose_rsc_cov = compute_pose_error(R_gt[im_k], t_gt[im_k], pose_rsc_cov.R, pose_rsc_cov.t)

            errs_init += [error_pose_init]
            errs_ref_pt += [error_pose_ref_pt]
            errs_ref_cov += [error_pose_ref_cov]
            errs_rsc_cov += [error_pose_rsc_cov]

            #print(info_init['inlier_ratio'],info_rsc_cov['inlier_ratio'])

            #import ipdb
            #ipdb.set_trace()
            

        errs_init = np.array(errs_init)
        errs_ref_pt = np.array(errs_ref_pt)
        errs_ref_cov = np.array(errs_ref_cov)
        errs_rsc_cov = np.array(errs_rsc_cov)

        if len(errs_init) == 0:
            print("No valid trials for this image, skipping...")
            continue
        print('Initial error from RANSAC:\t ', errs_init.mean(axis=0))
        print('Refinement with points only:\t ', errs_ref_pt.mean(axis=0))
        print('Refinement with points+cov:\t ', errs_ref_cov.mean(axis=0))
        print('RANSAC with points+cov:\t\t ', errs_rsc_cov.mean(axis=0))


    import ipdb
    ipdb.set_trace()

if __name__ == "__main__":
    main()