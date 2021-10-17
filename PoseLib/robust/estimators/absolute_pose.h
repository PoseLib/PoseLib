#ifndef POSELIB_ROBUST_ESTIMATORS_ABSOLUTE_POSE_H
#define POSELIB_ROBUST_ESTIMATORS_ABSOLUTE_POSE_H

#include <PoseLib/robust/types.h>
#include <PoseLib/robust/utils.h>
#include <PoseLib/types.h>

namespace pose_lib {

class AbsolutePoseEstimator {
  public:
    AbsolutePoseEstimator(const RansacOptions &ransac_opt,
                          const std::vector<Eigen::Vector2d> &points2D,
                          const std::vector<Eigen::Vector3d> &points3D)
        : num_data(points2D.size()), opt(ransac_opt), x(points2D), X(points3D) {
        rng = opt.seed;
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 3;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Eigen::Vector2d> &x;
    const std::vector<Eigen::Vector3d> &X;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> xs, Xs;
    std::vector<size_t> sample;
};

class GeneralizedAbsolutePoseEstimator {
  public:
    GeneralizedAbsolutePoseEstimator(const RansacOptions &ransac_opt,
                                     const std::vector<std::vector<Eigen::Vector2d>> &points2D,
                                     const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                     const std::vector<CameraPose> &camera_ext)
        : num_cams(points2D.size()), opt(ransac_opt),
          x(points2D), X(points3D), rig_poses(camera_ext) {
        rng = opt.seed;
        ps.resize(sample_sz);
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        sample.resize(sample_sz);
        camera_centers.resize(num_cams);
        for (size_t k = 0; k < num_cams; ++k) {
            camera_centers[k] = -camera_ext[k].R.transpose() * camera_ext[k].t;
        }

        num_data = 0;
        num_pts_camera.resize(num_cams);
        for (size_t k = 0; k < num_cams; ++k) {
            num_pts_camera[k] = points2D[k].size();
            num_data += num_pts_camera[k];
        }
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 3;
    size_t num_data;
    const size_t num_cams;

  private:
    const RansacOptions &opt;
    const std::vector<std::vector<Eigen::Vector2d>> &x;
    const std::vector<std::vector<Eigen::Vector3d>> &X;
    const std::vector<CameraPose> &rig_poses;
    std::vector<Eigen::Vector3d> camera_centers;
    std::vector<size_t> num_pts_camera; // number of points in each camera

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> ps, xs, Xs;
    std::vector<std::pair<size_t, size_t>> sample;
};

class AbsolutePosePointLineEstimator {
  public:
    AbsolutePosePointLineEstimator(const RansacOptions &ransac_opt,
                                   const std::vector<Point2D> &x,
                                   const std::vector<Point3D> &X,
                                   const std::vector<Line2D> &l,
                                   const std::vector<Line3D> &L)
        : num_data(x.size() + l.size()), opt(ransac_opt), points2D(x), points3D(X), lines2D(l), lines3D(L) {
        rng = opt.seed;
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        ls.resize(sample_sz);
        Cs.resize(sample_sz);
        Vs.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 3;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &points2D;
    const std::vector<Point3D> &points3D;
    const std::vector<Line2D> &lines2D;
    const std::vector<Line3D> &lines3D;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> xs, Xs, ls, Cs, Vs;
    std::vector<size_t> sample;
};

class Radial1DAbsolutePoseEstimator {
  public:
    Radial1DAbsolutePoseEstimator(const RansacOptions &ransac_opt,
                                  const std::vector<Eigen::Vector2d> &points2D,
                                  const std::vector<Eigen::Vector3d> &points3D)
        : num_data(points2D.size()), opt(ransac_opt), x(points2D), X(points3D) {
        rng = opt.seed;
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Eigen::Vector2d> &x;
    const std::vector<Eigen::Vector3d> &X;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector2d> xs;
    std::vector<Eigen::Vector3d> Xs;
    std::vector<size_t> sample;
};

} // namespace pose_lib

#endif
