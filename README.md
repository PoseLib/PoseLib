# PoseLib
Minimal solvers for calibrated pose estimation

The following solvers are currently implemented.

| Solver | Point-Point | Point-Line | Upright | Generalized | Approx. runtime | Max. solutions | Comment |
| --- | --- | --- | :---: | :---: | :---: | :---: | --- |
| `p3p` | 3 | 0 |  |  | 250 ns | 4 | Persson and Nordberg, LambdaTwist (ECCV 2018) |
| `gp3p` | 3 | 0 |  | :heavy_check_mark:  | 4.9 us | 8 | Kukelova et al., E3Q3 (CVPR 2016) |
| `gp4ps` | 4 | 0 |  | :heavy_check_mark: | 4.9 us | 8 | Unknown scale. Kukelova et al., E3Q3 (CVPR 2016) |
| `p4pf` | 4 | 0 |  |  | 6 us | 8 | Unknown focal length. Kukelova et al., E3Q3 (CVPR 2016) |
| `p2p2l` | 2 | 2 |  |  | 30 us | 16 | Josephson et al. (CVPR 2007) |
| `up2p` | 2 | 0 | :heavy_check_mark: |  | 65 ns | 2 |  |
| `ugp2p` | 2 | 0 | :heavy_check_mark: | :heavy_check_mark: | 65 ns | 2 |  |
| `up1p2l` | 1 | 2 | :heavy_check_mark: |  | 370 ns | 4 |  |
| `up4l` | 0 | 4 | :heavy_check_mark: |  | 7.4 us | 8 | Sweeney et al. (3DV 2014) |


**TODO:**
1. Add new solvers: ugp4l (Sweeney), 2d line-3d point solvers, radial p5p (Kukelova)
2. Add root bracketing instead of companion matrix (E3Q3)
3. Change upright solvers so that gravity is y-aligned (instead of z-aligned)
4. Non-minimal solvers (maybe Nakano?)
5. Weird bug with gp4ps and up2p which fail when compiled without -march=native on Ubuntu.
