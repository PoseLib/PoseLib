# PoseLib
Minimal solvers for calibrated pose estimation

## Naming convention
For the solver names we use a slightly non-standard notation where we denote the solver as

<pre>
p<b><i>X</i></b>p<b><i>Y</i></b>pl<b><i>Z</i></b>lp<b><i>W</i></b>ll
</pre>

where the number of correspondences required is given by
* <b><i>X</i></b> - 2D point to 3D point,
* <b><i>Y</i></b> - 2D point to 3D line,
* <b><i>Z</i></b> - 2D line to 3D point,
* <b><i>W</i></b> - 2D line to 2D line.

The prefix with `u` is for upright solvers and  `g` for generalized camera solvers. Solvers that estimate focal length have the postfix with `f` and similarly `s` for solvers that estimate scale.

## Implemented solvers
The following solvers are currently implemented.

| Solver | Point-Point | Point-Line | Line-Point | Line-Line | Upright | Generalized | Approx. runtime | Max. solutions | Comment |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| `p3p` | 3 | 0 | 0| 0|  |  | 250 ns | 4 | Persson and Nordberg, LambdaTwist (ECCV18) |
| `gp3p` | 3 | 0 | 0| 0|  | :heavy_check_mark:  | 4.9 us | 8 | Kukelova et al., E3Q3 (CVPR16) |
| `gp4ps` | 4 | 0 | 0| 0|  | :heavy_check_mark: | 4.9 us | 8 | Unknown scale.<br> Kukelova et al., E3Q3 (CVPR16) |
| `p4pf` | 4 | 0 | 0| 0|  |  | 6 us | 8 | Unknown focal length.<br> Kukelova et al., E3Q3 (CVPR16) |
| `p2p2pl` | 2 | 2 | 0| 0|  |  | 30 us | 16 | Josephson et al. (CVPR07) |
| `p6lp` | 0 | 0 | 6|  0| |  | 4.9 us | 8 | Kukelova et al., E3Q3 (CVPR16)  |
| `p2p1ll` | 2 | 0 | 0 |  1| |  | 4.9 us | 8 | Kukelova et al., E3Q3 (CVPR16)  |
| `p1p2ll` | 1 | 0 | 0 |  2| |  | 4.9 us | 8 | Kukelova et al., E3Q3 (CVPR16)  |
| `p3ll` | 0 | 0 | 0 |  3| |  | 4.9 us | 8 | Kukelova et al., E3Q3 (CVPR16)  |
| `up2p` | 2 | 0 | 0| 0| :heavy_check_mark: |  | 65 ns | 2 | Kukelova et al. (ACCV10) |
| `ugp2p` | 2 | 0 | 0| 0| :heavy_check_mark: | :heavy_check_mark: | 65 ns | 2 |  |
| `ugp3ps` | 3 | 0 | 0| 0| :heavy_check_mark: | :heavy_check_mark: | 390 ns | 2 | Unknown scale. |
| `up1p2pl` | 1 | 2 | 0| 0| :heavy_check_mark: |  | 370 ns | 4 |  |
| `up4pl` | 0 | 4 | 0| 0| :heavy_check_mark: |  | 7.4 us | 8 | Sweeney et al. (3DV14) |


**TODO:**
1. Add new solvers: ugp4l (Sweeney), 2d line-3d point solvers, radial p5p (Kukelova)
2. Add root bracketing instead of companion matrix (E3Q3)
3. Change upright solvers so that gravity is y-aligned (instead of z-aligned)
4. Non-minimal solvers (maybe Nakano?)
5. Weird bug with gp4ps and up2p which fail when compiled without -march=native on Ubuntu.
6. Make sure solvers either consistently call output->clear(), or they dont. Currently it is a mixed bag.
