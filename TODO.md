# New features or missing implementations

- [ ] In `gen_relpose_5p1pt` add cheirality filter.
- [ ] Use per-camera `max_error` for `estimate_generalized_absolute_pose` (use `AbsolutePoseOptions::max_errors`). Also expose in pybind.
- [ ] Homography refinement with camera model.
- [ ] Implement Tangent Sampson error also for generalized relative pose
- [ ] Fix and test estimate_hybrid_pose (including non-P3P sampling)
- [ ] Refactor `estimate_relative_pose` to similarly to `estimate_absolute_pose` have a single entry point. Contact @vlarsson for more details.
- [ ] Refactor pybinds (split into multiple instead of a megafile)
- [ ] Clean up unit tests.
- [ ] Make sure `camera_models` model ids are consistent with COLMAP.
- [ ] Update README.md

# Bugs
- [ ] Fix RANSAC bug in `estimate_hybrid_pose`. See issues.

# Experimental
- [ ] Think about what the best way to handle line segments in non-pinhole camera models.
- [ ] Cheirality check for 2D-3D line correspondences. (Check `compute_msac_score` in utils.cc). Think about how this should be decided.
- [ ] Experiment with refinement with all residuals+truncated loss, vs. copying out the inliers and near inliers to only refine with these. Potientially this could be decided dynamically if it makes a difference.
- [ ] Implement and test `QRAccumulator` as an alternative for `NormalAccumulator` in `jacobian_accumulator.h`. Potentially more numerically stable for some problems.
- [ ] Experiment with normalization of world coordinate system in P3.5Pf/P4Pf/P5Pf.

