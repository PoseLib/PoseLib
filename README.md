# PoseLib
Minimal solvers for calibrated pose estimation

The following solvers are currently implemented.

| Solver | Point-Point | Point-Line | Upright | Generalized | Approx. runtime | Max. solutions | Comment |
| --- | --- | --- | :---: | :---: | :---: | :---: | --- |
| `p3p` | 3 | 0 |  |  | 290 ns | 4 | Persson and Nordberg, LambdaTwist (ECCV 2018) |
| `gp3p` | 3 | 0 |  | :heavy_check_mark:  | 5.4 us | 8 | Kukelova et al., E3Q3 (CVPR 2016) |
| `gp4ps` | 4 | 0 |  | :heavy_check_mark:  | 5.4 us | 8 | Kukelova et al., E3Q3 (CVPR 2016) |
| `p2p2l` | 2 | 2 |  |  | 33 us | 16 | Josephson et al. CVPR 2007 |
| `up2p` | 2 | 0 | :heavy_check_mark: |  | 73 ns | 2 |  |
| `up1p2l` | 1 | 2 | :heavy_check_mark: |  | 570 ns | 4 |  |
| `up4l` | 0 | 4 | :heavy_check_mark: |  | 8 us | 8 | Sweeney et al. (3DV 2014) |


