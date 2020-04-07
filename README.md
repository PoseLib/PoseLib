# PoseLib
This library provides a collection of minimal solvers for camera pose estimation. The focus is on calibrated absolute pose estimation problems from different types of correspondences (e.g. point-point, point-line, line-point, line-line).

The goals of this project are
* Fast and robust implementation of the current state-of-the-art solvers.
* Consistent calling interface between different solvers.
* Minimize dependencies, both external (currently only [Eigen](http://eigen.tuxfamily.org/) and [RE3Q3](https://github.com/vlarsson/re3q3)) and internal. Each solver is (mostly) stand-alone, making it easy to extract only a specific solver to integrate into other frameworks.


## Naming convention
For the solver names we use a slightly non-standard notation where we denote the solver as

<pre>
p<b><i>X</i></b>p<b><i>Y</i></b>pl<b><i>Z</i></b>lp<b><i>W</i></b>ll
</pre>

where the number of correspondences required is given by
* <b><i>X</i></b>p - 2D point to 3D point,
* <b><i>Y</i></b>pl - 2D point to 3D line,
* <b><i>Z</i></b>lp - 2D line to 3D point,
* <b><i>W</i></b>ll - 2D line to 2D line.

The prefix with `u` is for upright solvers and  `g` for generalized camera solvers. Solvers that estimate focal length have the postfix with `f` and similarly `s` for solvers that estimate scale.

## Calling conventions
All solvers return their solutions as a vector of `CameraPose` structs, which defined as
```
struct CameraPose {
   Eigen::Matrix3d R;
   Eigen::Vector3d t;
   double alpha = 1.0; // either focal length or scale
};
```
where `[R t]` maps from the world coordinate system into the camera coordinate system.


For <b>2D point to 3D point</b> correspondences, the image points are represented as unit-length bearings vectors. The returned camera poses `(R,t)` then satisfies (for some `lambda`)
```
  lambda * x[i] = R * X[i] + t
```
where `x[i]` is the 2D point and `X[i]` is the 3D point.
<b>Note</b> that only the P3P solver filters solutions with negative `lambda`.

Solvers that use point-to-point constraints take one vector with bearing vectors `x` and one vector with the corresponding 3D points `X`, e.g. for the P3P solver the function declaration is

```
int p3p(const std::vector<Eigen::Vector3d> &x,
        const std::vector<Eigen::Vector3d> &X,
        std::vector<CameraPose> *output);
```
Each solver returns the number of real solutions found.

For constraints with <b>2D lines</b>, the lines are represented in homogeneous coordinates. In the case of 2D line to 3D point constraints, the returned camera poses then satisfies
```
  l[i].transpose() * (R * X[i] + t) = 0
```
where `l[i]` is the line and  `X[i]` is the 3D point.

For constraints with <b>3D lines</b>, the lines are represented by a 3D point `X` and a bearing vector `V`. In the case of 2D point to 3D point constraints
```
  lambda * x[i] = R * (X[i] + mu * V[i]) + t
```
for some values of `lambda` and `mu`. Similarly, for line to line constraints we have
```
  l[i].transpose() * (R * (X[i] + mu * V[i]) + t) = 0
```
### Generalized Cameras
For generalized cameras we represent the image rays similarly to the 3D lines above, with an offset `p` and a bearing vector `x`. For example, in the case of point-to-point correspondences we have
```
p[i] + lambda * x[i] = R * X[i] + t
```
In the case of unknown scale we also estimate `alpha` such that
```
alpha * p[i] + lambda * x[i] = R * X[i] + t
```
For example, the generalized pose and scale solver (from four points) has the following signature
```
 int gp4ps(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
              const std::vector<Eigen::Vector3d> &X, std::vector<CameraPose> *output);
```

## Implemented solvers
The following solvers are currently implemented.

| Solver | Point-Point | Point-Line | Line-Point | Line-Line | Upright | Generalized | Approx. runtime | Max. solutions | Comment |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| `p3p` | 3 | 0 | 0| 0|  |  | 250 ns | 4 | Persson and Nordberg, LambdaTwist (ECCV18) |
| `gp3p` | 3 | 0 | 0| 0|  | :heavy_check_mark:  | 1.6 us | 8 | Kukelova et al., E3Q3 (CVPR16) |
| `gp4ps` | 4 | 0 | 0| 0|  | :heavy_check_mark: | 1.8 us | 8 | Unknown scale.<br> Kukelova et al., E3Q3 (CVPR16) |
| `p4pf` | 4 | 0 | 0| 0|  |  | 2.3 us | 8 | Unknown focal length.<br> Kukelova et al., E3Q3 (CVPR16) |
| `p2p2pl` | 2 | 2 | 0| 0|  |  | 30 us | 16 | Josephson et al. (CVPR07) |
| `p6lp` | 0 | 0 | 6|  0| |  | 1.8 us | 8 | Kukelova et al., E3Q3 (CVPR16)  |
| `p5lp_radial` | 0 | 0 | 5|  0| |  | 1 us | 4 | Kukelova et al., (ICCV13)  |
| `p2p1ll` | 2 | 0 | 0 |  1| |  | 1.6 us | 8 | Kukelova et al., E3Q3 (CVPR16), Zhou et al. (ACCV18)  |
| `p1p2ll` | 1 | 0 | 0 |  2| |  | 1.7 us | 8 | Kukelova et al., E3Q3 (CVPR16), Zhou et al. (ACCV18)  |
| `p3ll` | 0 | 0 | 0 |  3| |  | 1.8 us | 8 | Kukelova et al., E3Q3 (CVPR16), Zhou et al. (ACCV18)  |
| `up2p` | 2 | 0 | 0| 0| :heavy_check_mark: |  | 65 ns | 2 | Kukelova et al. (ACCV10) |
| `ugp2p` | 2 | 0 | 0| 0| :heavy_check_mark: | :heavy_check_mark: | 65 ns | 2 |  |
| `ugp3ps` | 3 | 0 | 0| 0| :heavy_check_mark: | :heavy_check_mark: | 390 ns | 2 | Unknown scale. |
| `up1p2pl` | 1 | 2 | 0| 0| :heavy_check_mark: |  | 370 ns | 4 |  |
| `up4pl` | 0 | 4 | 0| 0| :heavy_check_mark: |  | 1.4 us | 8 | Sweeney et al. (3DV14) |

## How to compile?

Getting the code:

    > git clone --recursive https://github.com/vlarsson/PoseLib.git
    > cd PoseLib

Example of a local installation:

    > mkdir build
    > cd build
    > cmake .. -DCMAKE_INSTALL_PREFIX=../installed
    > make install -j16

Installed files:

    > tree ../installed
      .
      ├── bin
      │   └── benchmark
      ├── include
      │   └── PoseLib
      │       ├── gp3p.h
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

### How to use the library (as dependency) in an external project?

    cmake_minimum_required(VERSION 3.0)
    project(Foo)

    find_package(PoseLib REQUIRED)
    include_directories(${POSELIB_INCLUDE_DIRS})

    add_executable(foo foo.cpp)
    target_link_libraries(foo ${POSELIB_LIBRARIES})


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


## License
PoseLib is licensed under the BSD 3-Clause license. Please see [License](https://github.com/vlarsson/PoseLib/blob/master/LICENSE) for details.


## TODO:
1. Add new solvers: ugp4l (Sweeney), 2d line-3d point solvers
2. Change upright solvers so that gravity is y-aligned (instead of z-aligned)
3. Non-minimal solvers (maybe Nakano?)
4. Weird bug with gp4ps and up2p which fail when compiled without -march=native on Ubuntu. Problem is that Matrix4d.Inverse() in Eigen does not play nice with -ffast-math (unless -march=native is active for some reason).
