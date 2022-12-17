# PoseLib
This library provides a collection of minimal solvers for camera pose estimation. The focus is on calibrated absolute pose estimation problems from different types of correspondences (e.g. point-point, point-line, line-point, line-line).

The goals of this project are to provide
* Fast and robust implementation of the current state-of-the-art solvers.
* Consistent calling interface between different solvers.
* Minimize dependencies, both external (currently only [Eigen](http://eigen.tuxfamily.org/)) and internal. Each solver is (mostly) stand-alone, making it easy to extract only a specific solver to integrate into other frameworks.
* Robust estimators (based on LO-RANSAC) that just works out-of-the-box for most cases.

# Robust Estimation and Non-linear Refinement
We provide robust estimators for the most common problems
* Absolute pose from points (and lines)
* Essential / Fundamental matrix
* Homography
* Generalized relative pose

It is fairly straight-forward to implement robust estimators for other problems. See for example [absolute_pose.h](PoseLib/robust/estimators/absolute_pose.h). If you implement estimators for other problems, please consider submitting a pull-request.

In [robust.h](PoseLib/robust.h) we provide interfaces which normalizes the data, calls the RANSAC and runs a post-RANSAC non-linear refinement. It is also possible to directly call the individual components as well (see e.g. [ransac.h](PoseLib/robust/ransac.h), [bundle.h](PoseLib/robust/bundle.h), etc.). The RANSAC is straight-forward implementation of LO-RANSAC which generate hypothesis with minimal solvers and relies on non-linear refinement for refitting.

The robust estimator takes the following options
```c++
struct RansacOptions {
    size_t max_iterations = 100000;
    size_t min_iterations = 1000;
    double dyn_num_trials_mult = 3.0;
    double success_prob = 0.9999;
    double max_reproj_error = 12.0;  // used for 2D-3D matches
    double max_epipolar_error = 1.0; // used for 2D-2D matches
    unsigned long seed = 0;
    // If we should use PROSAC sampling. Assumes data is sorted
    bool progressive_sampling = false;
    size_t max_prosac_iterations = 100000;
};
```
and the non-linear refinement 
```c++
struct BundleOptions {
    size_t max_iterations = 100;
    enum LossType {
        TRIVIAL, TRUNCATED, HUBER, CAUCHY, TRUNCATED_LE_ZACH
    } loss_type = LossType::CAUCHY;
    double loss_scale = 1.0;
    double gradient_tol = 1e-8;
    double step_tol = 1e-8;
    double initial_lambda = 1e-3;
    double min_lambda = 1e-10;
    double max_lambda = 1e10;
    bool verbose = false;
};
```
Note that in [robust.h](PoseLib/robust.h) this is only used for the post-RANSAC refinement.

In [bundle.h](PoseLib/robust/bundle.h) we provide non-linear refinement for different problems. Mainly minimizing reprojection error and Sampson error as these performed best in our internal evaluations. These are used in the LO-RANSAC to perform non-linear refitting. Most estimators directly minimize the MSAC score (using `loss_type = TRUNCATED` and `loss_scale = threshold`) over all input correspondences. In practice we found that this works quite well and avoids recursive LO where inliers are added in steps.

## Camera models
PoseLib use [COLMAP](https://colmap.github.io/cameras.html)-compatible camera models. These are defined in [colmap_models.h](PoseLib/misc/colmap_models.h). Currently we only support
* SIMPLE_PINHOLE
* PINHOLE
* SIMPLE_RADIAL
* RADIAL
* OPENCV

but it is relatively straight-forward to add other models. If you do so please consider opening a pull-request. In contrast to COLMAP, we require analytical jacobians for the distortion mappings which make it a bit more work to port them.

The `Camera` struct currently contains `width`/`height` fields, however these are not used anywhere in the code-base and are provided simply to be consistent with COLMAP. The `Camera` class also provides the helper function `initialize_from_txt(str)` which initializes the camera from a line given by the `cameras.txt` file of a COLMAP reconstruction.

## Python bindings
See [pybind/README.md](pybind/README.md) for details on how to compile the python bindings. The python bindings expose all minimal solvers, e.g. `poselib.p3p(x,X)`, as well as all robust estimators from [robust.h](PoseLib/robust.h). 

Examples of how the robust estimators can be called are
```python
camera = {'model': 'SIMPLE_PINHOLE', 'width': 1200, 'height': 800, 'params': [960, 600, 400]}

pose, info = poselib.estimate_absolute_pose(p2d, p3d, camera, {'max_reproj_error': 16.0}, {})
```
or
```python
F, info = poselib.estimate_fundamental_matrix(p2d_1, p2d_2, {'max_epipolar_error': 0.75, 'progressive_sampling': True}, {})

```

The return value `info` is a dict containing information about the robust estimation (inliers, iterations, etc). The last two options are dicts which describe the `RansacOptions` and `BundleOptions`. Ommited values are set to their default (see above), except for the `loss_scale` used for the Cauchy loss which is set to half of the threshold used in RANSAC (which seems to be a good heuristic). Dicts with the default options can be obtained as `opt = poselib.RansacOptions()` or `poselib.BundleOptions()`.



Some of the available estimators are listed below, check [pyposelib.cpp](pybind/pyposelib.cpp) and [robust.h](PoseLib/robust.h) for more details. The table also shows which error threshold is used in the estimation (`RansacOptions.max_reproj_error` or `RansacOptions.max_epipolar_error`). All thresholds are given in pixels.

| Method | Arguments | (RansacOptions) Threshold |
| --- | --- | --- |
| <sub>`estimate_absolute_pose`</sub> | <sub> `(p2d, p3d, camera, ransac_opt,bundle_opt)`</sub> | <sub>`max_reproj_error` </sub> |
| <sub>`estimate_absolute_pose_pnpl`</sub> | <sub>`(p2d, p3d, l2d_1, l2d_2, l3d_1, l3d_2, camera, ransac_opt, bundle_opt)` </sub> | <sub>`max_reproj_error` (points), `max_epipolar_error` (lines) |
| <sub>`estimate_generalized_absolute_pose` | <sub>`(p2ds, p3ds, camera_ext, cameras, ransac_opt, bundle_opt)`</sub> | <sub>`max_reproj_error`</sub> |
| <sub>`estimate_relative_pose`</sub> | <sub>`(x1, x2, camera1, camera2, ransac_opt, bundle_opt)`</sub> | <sub>`max_epipolar_error` </sub>|
| <sub>`estimate_fundamental`</sub> | <sub>`(x1, x2, ransac_opt, bundle_opt)`</sub> | <sub>`max_epipolar_error`</sub> |
| <sub>`estimate_homography`</sub> | <sub>`(x1, x2, ransac_opt, bundle_opt)`</sub> | <sub>`max_reproj_error`</sub> |
| <sub>`estimate_generalized_relative_pose`</sub> | <sub>`(matches, camera1_ext, cameras1, camera2_ext, cameras2, ransac_opt, bundle_opt)`</sub> | <sub>`max_epipolar_error`</sub> |

### poselib.CameraPose
The python bindings expose a `poselib.CameraPose` class which is the return type for most methods. While the class internally represent the pose with `q` and `t`, it also exposes `R` (3x3) and `Rt` (3x4) which are read/write, i.e. you can do `pose.R = Rnew` and it will update the underlying quaternion `q`.

### Benchmarking the robust estimators
To sanity-check the robust estimators we benchmark against the LO-RANSAC implementation from [pycolmap](https://github.com/colmap/pycolmap).
    
<a href="https://user-images.githubusercontent.com/48490995/149815304-b3c1049a-ee64-4c14-be60-d4930535a3e7.png"><img src="https://user-images.githubusercontent.com/48490995/149815304-b3c1049a-ee64-4c14-be60-d4930535a3e7.png" width="75%"></a>
    
For all of the metrics higher is better (except for runtime).

# Minimal Solvers
## Naming convention
For the solver names we use a slightly non-standard notation where we denote the solver as

<pre>
p<b><i>X</i></b>p<b><i>Y</i></b>pl<b><i>Z</i></b>lp<b><i>W</i></b>ll
</pre>

where the number of correspondences required is given by
* <b><i>X</i></b>p - 2D point to 3D point,
* <b><i>Y</i></b>pl - 2D point to 3D line,
* <b><i>Z</i></b>lp - 2D line to 3D point,
* <b><i>W</i></b>ll - 2D line to 3D line.

The prefix with `u` is for upright solvers and  `g` for generalized camera solvers. Solvers that estimate focal length have the postfix with `f` and similarly `s` for solvers that estimate scale.

## Calling conventions
All solvers return their solutions as a vector of `CameraPose` structs, which defined as
```c++
struct CameraPose {
   Eigen::Vector4d q;
   Eigen::Vector3d t;
};
```
where the rotation is representation as a quaternion `q` and the convention is that `[R t]` maps from the world coordinate system into the camera coordinate system.


For <b>2D point to 3D point</b> correspondences, the image points are represented as unit-length bearings vectors. The returned camera poses `(R,t)` then satisfies (for some `lambda`)
```c++
  lambda * x[i] = R * X[i] + t
```
where `x[i]` is the 2D point and `X[i]` is the 3D point.
<b>Note</b> that only the P3P solver filters solutions with negative `lambda`.

Solvers that use point-to-point constraints take one vector with bearing vectors `x` and one vector with the corresponding 3D points `X`, e.g. for the P3P solver the function declaration is

```c++
int p3p(const std::vector<Eigen::Vector3d> &x,
        const std::vector<Eigen::Vector3d> &X,
        std::vector<CameraPose> *output);
```
Each solver returns the number of real solutions found.

For constraints with <b>2D lines</b>, the lines are represented in homogeneous coordinates. In the case of 2D line to 3D point constraints, the returned camera poses then satisfies
```c++
  l[i].transpose() * (R * X[i] + t) = 0
```
where `l[i]` is the line and  `X[i]` is the 3D point.

For constraints with <b>3D lines</b>, the lines are represented by a 3D point `X` and a bearing vector `V`. In the case of 2D point to 3D point constraints
```c++
  lambda * x[i] = R * (X[i] + mu * V[i]) + t
```
for some values of `lambda` and `mu`. Similarly, for line to line constraints we have
```c++
  l[i].transpose() * (R * (X[i] + mu * V[i]) + t) = 0
```
### Generalized Cameras
For generalized cameras we represent the image rays similarly to the 3D lines above, with an offset `p` and a bearing vector `x`. For example, in the case of point-to-point correspondences we have
```c++
p[i] + lambda * x[i] = R * X[i] + t
```
In the case of unknown scale we also estimate `alpha` such that
```c++
alpha * p[i] + lambda * x[i] = R * X[i] + t
```
For example, the generalized pose and scale solver (from four points) has the following signature
```c++
 int gp4ps(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
              const std::vector<Eigen::Vector3d> &X, std::vector<CameraPose> *output);
```

### Upright Solvers
For the upright solvers it assumed that the rotation is around the y-axis, i.e.
```c++
R = [a 0 -b; 0 1 0; b 0 a] 
```
To use these solvers it necessary to pre-rotate the input such that this is satisfied.

## Implemented solvers
The following solvers are currently implemented.

### Absolute Pose
| Solver | Point-Point | Point-Line | Line-Point | Line-Line | Upright | Generalized | Approx. runtime | Max. solutions | Comment |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| `p3p` | 3 | 0 | 0| 0|  |  | 250 ns | 4 | Persson and Nordberg, LambdaTwist (ECCV18) |
| `gp3p` | 3 | 0 | 0| 0|  | :heavy_check_mark:  | 1.6 us | 8 | Kukelova et al., E3Q3 (CVPR16) |
| `gp4ps` | 4 | 0 | 0| 0|  | :heavy_check_mark: | 1.8 us | 8 | Unknown scale.<br> Kukelova et al., E3Q3 (CVPR16)<br>Camposeco et al.(ECCV16) |
| `p4pf` | 4 | 0 | 0| 0|  |  | 2.3 us | 8 | Unknown focal length.<br> Kukelova et al., E3Q3 (CVPR16) |
| `p2p2pl` | 2 | 2 | 0| 0|  |  | 30 us | 16 | Josephson et al. (CVPR07) |
| `p6lp` | 0 | 0 | 6|  0| |  | 1.8 us | 8 | Kukelova et al., E3Q3 (CVPR16)  |
| `p5lp_radial` | 0 | 0 | 5|  0| |  | 1 us | 4 | Kukelova et al., (ICCV13)  |
| `p2p1ll` | 2 | 0 | 0 |  1| |  | 1.6 us | 8 | Kukelova et al., E3Q3 (CVPR16), Zhou et al. (ACCV18)  |
| `p1p2ll` | 1 | 0 | 0 |  2| |  | 1.7 us | 8 | Kukelova et al., E3Q3 (CVPR16), Zhou et al. (ACCV18)  |
| `p3ll` | 0 | 0 | 0 |  3| |  | 1.8 us | 8 | Kukelova et al., E3Q3 (CVPR16), Zhou et al. (ACCV18)  |
| `up2p` | 2 | 0 | 0| 0| :heavy_check_mark: |  | 65 ns | 2 | Kukelova et al. (ACCV10) |
| `ugp2p` | 2 | 0 | 0| 0| :heavy_check_mark: | :heavy_check_mark: | 65 ns | 2 | Adapted from Kukelova et al. (ACCV10)   |
| `ugp3ps` | 3 | 0 | 0| 0| :heavy_check_mark: | :heavy_check_mark: | 390 ns | 2 | Unknown scale. Adapted from Kukelova et al. (ACCV10)  |
| `up1p2pl` | 1 | 2 | 0| 0| :heavy_check_mark: |  | 370 ns | 4 |  |
| `up4pl` | 0 | 4 | 0| 0| :heavy_check_mark: |  | 1.4 us | 8 | Sweeney et al. (3DV14) |
| `ugp4pl` | 0 | 4 | 0| 0| :heavy_check_mark: | :heavy_check_mark: | 1.4 us | 8 | Sweeney et al. (3DV14) |


### Relative Pose
| Solver | Point-Point | Upright | Planar | Generalized | Approx. runtime | Max. solutions | Comment |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| `relpose_5pt` | 5 | | | | 5.5 us | 10 | Nister (PAMI 2004) |
| `relpose_8pt` | 8+ | | | | 2.2+ us | 1 |  |
| `relpose_upright_3pt` | 3 | :heavy_check_mark: | | | 210 ns | 4 | Sweeney et al. (3DV14)  | 
| `gen_relpose_upright_4pt` | 4 | :heavy_check_mark: | | :heavy_check_mark:  | 1.2 us | 6 | Sweeney et al. (3DV14)  | 
| `relpose_upright_planar_2pt` | 2 | :heavy_check_mark: | :heavy_check_mark: | | 120 ns | 2 | Choi and Kim (IVC 2018)  | 
| `relpose_upright_planar_3pt` | 3 | :heavy_check_mark: | :heavy_check_mark: | | 300 ns | 1 |  Choi and Kim (IVC 2018) | 
| `gen_relpose_5p1pt` | 5+1 |  | | :heavy_check_mark:  | 5.5 us | 10 | E + 1pt to fix scale  | 
| `gen_relpose_6pt` | 6 |  | | :heavy_check_mark:  | < 1ms | 64 | Larsson et al. (CVPR 2017)  | 



## How to compile?

Getting the code:

    > git clone https://github.com/vlarsson/PoseLib.git
    > cd PoseLib

Example of a local installation:

    > mkdir _build && cd _build
    > cmake -DCMAKE_INSTALL_PREFIX=../_install ..
    > cmake --build . --target install -j 8
      (equivalent to  'make install -j8' in linux)

Installed files:

    > tree ../_install
      .
      ├── bin
      │   └── benchmark
      ├── include
      │   └── PoseLib
      │       ├── solvers/gp3p.h
      │       ├──  ...
      │       ├── poselib.h          <==  Library header (includes all the rest)
      │       ├──  ...
      │       └── version.h
      └── lib
          ├── cmake
          │   └── PoseLib
          │       ├── PoseLibConfig.cmake
          │       ├── PoseLibConfigVersion.cmake
          │       ├── PoseLibTargets.cmake
          │       └── PoseLibTargets-release.cmake
          └── libPoseLib.a

Uninstall library:

    > make uninstall


## Benchmark

Conditional compilation of `benchmark` binary is controlled by `WITH_BENCHMARK` option. Default if OFF (without benchmark).

Add `-DWITH_BENCHMARK=ON` to cmake to activate.

    > cmake -DWITH_BENCHMARK=ON ..



## Use library (as dependency) in an external project.

    cmake_minimum_required(VERSION 3.13)
    project(Foo)

    find_package(PoseLib REQUIRED)

    add_executable(foo foo.cpp)
    target_link_libraries(foo PRIVATE PoseLib::PoseLib)


## Citing
If you are using the library for (scientific) publications, please cite the following source:
```
@misc{PoseLib,
  title = {{PoseLib - Minimal Solvers for Camera Pose Estimation}},
  author = {Viktor Larsson},
  URL = {https://github.com/vlarsson/PoseLib},
  year = {2020}
}
```
Please cite also the original publications of the different methods (see table above).

## Changelog

2.0 - Jan. 2022
* Added robust estimators (LO-RANSAC) and non-linear refinement
* Refactored CameraPose to use quaternion instead 3x3 matrix. Removed alpha.
* Implemented TR-IRLS  method from Le and Zach (3DV 2021)
* Restructured pybind11 interface
* Added support for PROSAC sampling
* Many minor fixes and improvements....

1.0 - Jan. 2020
* Initial release

## License
PoseLib is licensed under the BSD 3-Clause license. Please see [License](https://github.com/vlarsson/PoseLib/blob/master/LICENSE) for details.

## Acknowledgements
The RANSAC implementation is heavily inspired by [RansacLib](github.com/tsattler/RansacLib) from Torsten Sattler. 
