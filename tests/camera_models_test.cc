#include "test.h"
#include <PoseLib/misc/camera_models.h>
#include "example_cameras.h"
#include "test.h"

using namespace poselib;

bool test_id_from_string() {

    REQUIRE_EQ(Camera::id_from_string("SIMPLE_PINHOLE"), 0);
    REQUIRE_EQ(Camera::id_from_string("PINHOLE"), 1);
    REQUIRE_EQ(Camera::id_from_string("SIMPLE_RADIAL"), 2);
    REQUIRE_EQ(Camera::id_from_string("RADIAL"), 3);
    REQUIRE_EQ(Camera::id_from_string("OPENCV"), 4);
    REQUIRE_EQ(Camera::id_from_string("OPENCV_FISHEYE"), 5);
    REQUIRE_EQ(Camera::id_from_string("FULL_OPENCV"), 6);


    return true;
}

bool test_id_string_converstion() {

    for(size_t id = 0; id < 5; ++id) {
        std::string name = Camera::name_from_id(id);
        REQUIRE_EQ(Camera::id_from_string(name), id);
    }
    return true;
}


bool test_model_name() {
    Camera camera("SIMPLE_RADIAL", {1.0, 0.0, 0.0, 0.0}, 1.0, 1.0);
    REQUIRE_EQ(camera.model_id, 2);
    REQUIRE_EQ(camera.model_name(), "SIMPLE_RADIAL");
    REQUIRE_EQ(camera.model_id, Camera::id_from_string("SIMPLE_RADIAL"));

    return true;
}


bool test_from_txt() {


    Camera camera1, camera2;

    int camera1_id = camera1.initialize_from_txt("0 SIMPLE_RADIAL 1936 1296 2425.85 932.383 628.265 -0.0397695");    
    REQUIRE_EQ(camera1_id, 0); 
    REQUIRE_EQ(camera1.model_name(), "SIMPLE_RADIAL"); 
    REQUIRE_EQ(camera1.width, 1936);
    REQUIRE_EQ(camera1.height, 1296);
    REQUIRE_EQ(camera1.params.size(), 4);
    REQUIRE_EQ(camera1.params[0], 2425.85);
    REQUIRE_EQ(camera1.params[1],  932.383);
    REQUIRE_EQ(camera1.params[2], 628.265);
    REQUIRE_EQ(camera1.params[3], -0.0397695);
    
    int camera2_id = camera2.initialize_from_txt("1 PINHOLE 6214 4138 3425.62 3426.29 3118.41 2069.07");    
    REQUIRE_EQ(camera2_id, 1); 
    REQUIRE_EQ(camera2.model_name(), "PINHOLE"); 
    REQUIRE_EQ(camera2.width, 6214);
    REQUIRE_EQ(camera2.height, 4138);
    REQUIRE_EQ(camera2.params.size(), 4);
    REQUIRE_EQ(camera2.params[0], 3425.62);
    REQUIRE_EQ(camera2.params[1], 3426.29);
    REQUIRE_EQ(camera2.params[2], 3118.41);
    REQUIRE_EQ(camera2.params[3], 2069.07);
    

    return true;
}

bool test_to_txt() {
    std::string txt0 = "SIMPLE_RADIAL 987 123 2000 1234 4567 -0.1234";
    std::string txt1 = "999 SIMPLE_RADIAL 987 123 2000 1234 4567 -0.1234";
    
    Camera camera("SIMPLE_RADIAL", {2000, 1234, 4567, -0.1234}, 987, 123);

    REQUIRE_EQ(camera.to_cameras_txt(), txt0);
    REQUIRE_EQ(camera.to_cameras_txt(999), txt1);

    // Check with example cameras
    for(size_t i = 0; i < example_cameras.size(); ++i) {
        const std::string example_txt = example_cameras[i];
        camera.initialize_from_txt(example_txt);

        const std::string example_txt2 = camera.to_cameras_txt();
        Camera camera2;
        camera2.initialize_from_txt(example_txt2);
        REQUIRE_EQ(camera.width, camera2.width);
        REQUIRE_EQ(camera.height, camera2.height);
        REQUIRE_EQ(camera.params.size(), camera2.params.size());
        for(size_t k = 0; k < camera.params.size(); ++k) {
            REQUIRE_EQ(camera.params[k], camera2.params[k]);
        }
    }
    
    return true;
}

bool check_project_unproject(Camera camera, const Eigen::Vector2d &xp) {
    Eigen::Vector2d xp2;
    Eigen::Vector3d x;
    camera.unproject(xp, &x);
    camera.project(x, &xp2);
    //std::cout << "xp = " << xp << " x = " << x << " xp2 = " << xp2 << "\n";
    REQUIRE_SMALL_M((xp - xp2).norm(), 1e-6, camera.model_name() + " (xp=" + to_string(xp.transpose()) + ", x=" + to_string(x.transpose()) + ", xp2=" + to_string(xp2.transpose()) + ")");
    return true;
}

bool test_project_unproject() {

    for(const std::string &camera_txt : example_cameras) {
        Camera camera;
        camera.initialize_from_txt(camera_txt);
        std::cout << camera.model_name() << "\n";

        for(size_t i = 20; i <= 80; ++i) {
            for(size_t j = 20; j <= 80; ++j) {
                Eigen::Vector2d xp(i/100.0, j/100.0);
                xp(0) *= camera.width;
                xp(1) *= camera.height;
                if(!check_project_unproject(camera, xp))
                    return false;
             }
        }
        
        // check close to principal point
        for(size_t i = -1; i <= 1; ++i) {
            for(size_t j = -1; j <= 1; ++j) {
                Eigen::Vector2d xp = camera.principal_point();
                xp(0) += i * 1e-6;
                xp(1) += j * 1e-6;            
                if(!check_project_unproject(camera, xp))
                    return false;
            }
        }
    }
    return true;
}


void compute_jacobian_central_diff(Camera camera, Eigen::Vector3d x, Eigen::Matrix<double,2,3> &jac, Eigen::Matrix<double, 2, Eigen::Dynamic> &jac_p) {
    const double h = 1e-8;
    Eigen::Vector3d x1p(x(0) + h, x(1), x(2));
    Eigen::Vector3d x2p(x(0), x(1) + h, x(2));
    Eigen::Vector3d x3p(x(0), x(1), x(2) + h);
    
    Eigen::Vector3d x1m(x(0) - h, x(1), x(2));
    Eigen::Vector3d x2m(x(0), x(1) - h, x(2));
    Eigen::Vector3d x3m(x(0), x(1), x(2) - h);

    Eigen::Vector2d yp, ym;

    camera.project(x1p, &yp);
    camera.project(x1m, &ym);
    jac.col(0) = (yp - ym) / (2*h);

    camera.project(x2p, &yp);
    camera.project(x2m, &ym);
    jac.col(1) = (yp - ym) / (2*h);

    camera.project(x3p, &yp);
    camera.project(x3m, &ym);
    jac.col(2) = (yp - ym) / (2*h);

    const double h2 = 1e-6;
    jac_p.resize(2, camera.params.size());
    std::vector<double> p0 = camera.params;    
    for(size_t k = 0; k < camera.params.size(); ++k) {
        camera.params = p0;
        camera.params[k] += h2;
        camera.project(x, &yp);

        camera.params = p0;
        camera.params[k] -= h2;
        camera.project(x, &ym);
        jac_p.col(k) = (yp - ym) / (2*h2);
    }
}

void compute_unproj_jacobian_central_diff(Camera camera, Eigen::Vector2d xp, Eigen::Matrix<double, 3, 2> &jac,
                                          Eigen::Matrix<double, 3, Eigen::Dynamic> &jac_p) {
    const double h = 1e-8;
    Eigen::Vector2d x1p(xp(0) + h, xp(1));
    Eigen::Vector2d x2p(xp(0), xp(1) + h);

    Eigen::Vector2d x1m(xp(0) - h, xp(1));
    Eigen::Vector2d x2m(xp(0), xp(1) - h);

    Eigen::Vector3d yp, ym;

    camera.unproject(x1p, &yp);
    camera.unproject(x1m, &ym);
    jac.col(0) = (yp - ym) / (2 * h);

    camera.unproject(x2p, &yp);
    camera.unproject(x2m, &ym);
    jac.col(1) = (yp - ym) / (2 * h);

    const double h2 = 1e-6;
    jac_p.resize(3, camera.params.size());
    std::vector<double> p0 = camera.params;
    for (size_t k = 0; k < camera.params.size(); ++k) {
        camera.params = p0;
        camera.params[k] += h2;
        camera.unproject(xp, &yp);

        camera.params = p0;
        camera.params[k] -= h2;
        camera.unproject(xp, &ym);
        jac_p.col(k) = (yp - ym) / (2 * h2);
    }
}

double compute_max_colwise_error(Eigen::MatrixXd A, Eigen::MatrixXd B) {
    double err = 0;
    for(size_t k = 0; k < A.cols(); ++k) {
        err = std::max(err, (A.col(k)-B.col(k)).norm() / std::max(1.0, B.col(k).norm()));
    }
    return err;
}

bool check_jacobian(Camera camera, const Eigen::Vector2d &xp) {
    // Unproject
    Eigen::Vector3d x;
    camera.unproject(xp, &x);

    Eigen::Matrix<double,2,3> jac_finite, jac;
    Eigen::Matrix<double,2,Eigen::Dynamic> jac_p_finite, jac_p;
    compute_jacobian_central_diff(camera, x, jac_finite, jac_p_finite);    
    Eigen::Vector2d xp2;
    jac.setZero();
    camera.project_with_jac(x, &xp2, &jac, &jac_p);
    //std::cout << "jac = \n" << jac << "\n jac_finite = \n" << jac_finite << "\n";

    REQUIRE_SMALL_M((xp - xp2).norm(), 1e-6, camera.model_name());
    double jac_err = (jac - jac_finite).norm() / jac_finite.norm();
    REQUIRE_SMALL_M(jac_err, 1e-6, camera.model_name() + ", x=" + std::to_string(xp(0)) + "," + std::to_string(xp(1)) + "\n jac_p=" + to_string(jac) + "\njac_p_finite=" + to_string(jac_finite)+ "\ndiff=\n" + to_string(jac - jac_finite));    
    

    //std::cout << "jac_p = \n" << jac_p << "\n jac_p_finite = \n" << jac_p_finite << "\n";

    
    double jac_p_err = compute_max_colwise_error(jac_p, jac_p_finite);
    REQUIRE_SMALL_M(jac_p_err, 1e-3, camera.model_name() + ", x=" + std::to_string(xp(0)) + "," + std::to_string(xp(1)) + "\n jac_p=\n" + to_string(jac_p) + "\njac_p_finite=\n" + to_string(jac_p_finite) + "\ndiff=\n" + to_string(jac_p - jac_p_finite));    

    return true;
}


bool check_unproj_jacobian(Camera camera, const Eigen::Vector2d &xp) {
    // Unproject
    Eigen::Vector3d x;
    camera.unproject(xp, &x);

    Eigen::Matrix<double, 3, 2> jac_finite, jac;
    Eigen::Matrix<double, 3, Eigen::Dynamic> jac_p_finite, jac_p;
    compute_unproj_jacobian_central_diff(camera, xp, jac_finite, jac_p_finite);
    Eigen::Vector3d x2;
    jac.setZero();
    jac_p.setZero();
    camera.unproject_with_jac(xp, &x2, &jac, &jac_p);

    REQUIRE_SMALL_M((x - x2).norm(), 1e-6, camera.model_name());
    double jac_unproj_err = (jac - jac_finite).norm() / jac_finite.norm();
    REQUIRE_SMALL_M(jac_unproj_err, 1e-4,
                    camera.model_name() + ", x=" + std::to_string(xp(0)) + "," + std::to_string(xp(1)) +
                        "\n jac_unproj=" + to_string(jac) + "\njac_unproj_finite=" + to_string(jac_finite) +
                        "\ndiff=\n" + to_string(jac - jac_finite));

    double jac_unproj_p_err = compute_max_colwise_error(jac_p, jac_p_finite);
    REQUIRE_SMALL_M(jac_unproj_p_err, 1e-3,
                    camera.model_name() + ", x=" + std::to_string(xp(0)) + "," + std::to_string(xp(1)) +
                        "\n jac_unproj_p=\n" + to_string(jac_p) + "\njac_unproj_p_finite=\n" + to_string(jac_p_finite) +
                        "\ndiff=\n" + to_string(jac_p - jac_p_finite));

    return true;
}


bool test_jacobian() {

    for(const std::string &camera_txt : example_cameras) {
        Camera camera;
        camera.initialize_from_txt(camera_txt);
        std::cout << "CAMERA = " << camera.model_name() << "\n";
        for(size_t i = 20; i <= 80; ++i) {
            for(size_t j = 20; j <= 80; ++j) {
                Eigen::Matrix<double,2,3> jac;
                Eigen::Vector2d xp(i/100.0, j/100.0);
                xp(0) *= camera.width;
                xp(1) *= camera.height;
                if(!check_jacobian(camera, xp)) {
                    return false;
                }
                if (!check_unproj_jacobian(camera, xp)) {
                    return false;
                }
            }
        }
        // check close to principal point
        for(size_t i = -1; i <= 1; ++i) {
            for(size_t j = -1; j <= 1; ++j) {
                Eigen::Vector2d xp = camera.principal_point();
                xp(0) += i * 1e-6;
                xp(1) += j * 1e-6;            
                if(!check_jacobian(camera, xp)) {
                    return false;
                }
            }
        }
    }
    return true;
}


bool test_jacobian_1D_radial() {
    const std::string camera_txt = "0 1D_RADIAL 1920 1080 1920 1080";
    Camera camera;
    camera.initialize_from_txt(camera_txt);
    for(size_t iter = 0; iter < 10; ++iter) {
        Eigen::Vector2d xp, xp2;
        Eigen::Vector3d x; 
        Eigen::Matrix<double,2,3> jac;
        xp.setRandom();
        xp = 0.5 * (xp + Eigen::Vector2d(1.0, 1.0));
        xp(0) *= 0.8 * camera.width;
        xp(1) *= 0.8 * camera.height;
        xp(0) += 0.2 * camera.width;
        xp(1) += 0.2 * camera.height;

        // Unproject
        camera.unproject(xp, &x);
        x.normalize();

        Eigen::Matrix<double,2,3> jac_finite;
        Eigen::Matrix<double,2,Eigen::Dynamic> jac_p_finite;
        compute_jacobian_central_diff(camera, x, jac_finite, jac_p_finite);
        
        jac.setZero();
        camera.project_with_jac(x, &xp2, &jac);
        //std::cout << "jac = \n" << jac << "\n jac_finite = \n" << jac_finite << "\n";

        double jac_err = (jac - jac_finite).norm() / jac_finite.norm();
        //std::cout << "err = " << err <<"\n";
        REQUIRE(jac_err < 1e-6);            
        //std::cout << "point res = " << (xp - xp2).norm() << "\n";
    }
    return true;
}


std::vector<Test> register_camera_models_test() {
    return {
        TEST(test_id_from_string),
        TEST(test_id_string_converstion),
        TEST(test_model_name),
        TEST(test_from_txt),
        TEST(test_to_txt),
        TEST(test_project_unproject),
        TEST(test_jacobian),
        TEST(test_jacobian_1D_radial)
    };
}