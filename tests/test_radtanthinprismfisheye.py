import numpy as np
import pytest
import poselib
import pycolmap
from projectaria_tools.projects import ase
from projectaria_tools.core.calibration import CameraCalibration, CameraModelType


@pytest.fixture
def cameras_ase():
    ase_camera = ase.get_ase_rgb_calibration()

    if False:  # Test with random parameters
        params = ase_camera.get_projection_params()
        params[3:] = np.random.rand(12) * 0.01
        ase_camera = CameraCalibration(
            ase_camera.get_label(),
            CameraModelType.FISHEYE624,
            params,
            ase_camera.get_transform_device_camera(),
            *ase_camera.get_image_size(),
            ase_camera.get_valid_radius(),
            ase_camera.get_max_solid_angle(),
            ase_camera.get_serial_number(),
            ase_camera.get_time_offset_sec_device_camera(),
            ase_camera.get_readout_time_sec())

    params = ase_camera.get_projection_params()
    focal_length = params[0]
    params = np.insert(params, 0, focal_length)
    width, height = ase_camera.get_image_size()

    pycolmap_camera = pycolmap.Camera(
        model="RAD_TAN_THIN_PRISM_FISHEYE", 
        width=width, height=height, 
        params=params.tolist()
    )
    poselib_camera = poselib.Camera(
        "RAD_TAN_THIN_PRISM_FISHEYE", 
        params.tolist(), 
        width, height
    )
    return ase_camera, pycolmap_camera, poselib_camera


def test_project(cameras_ase):
    ase_camera, pycolmap_camera, poselib_camera = cameras_ase
    for _ in range(100):
        X = np.random.rand(3) - np.array([0.5, 0.5, 0])
        x1 = ase_camera.project(X)
        x2 = pycolmap_camera.img_from_cam(X)
        x3 = poselib_camera.project(X)

        if x1 is not None:
            assert np.allclose(x1, x3, atol=1e-6), f"ASE vs PoseLib: {x1} vs {x3}"
        assert np.allclose(x2, x3, atol=1e-6), f"PyCOLMAP vs PoseLib: {x2} vs {x3}"


def test_project_with_jac(cameras_ase):
    ase_camera, _, poselib_camera = cameras_ase

    for _ in range(100):
        X = np.random.rand(3) - np.array([0.5, 0.5, 0])

        J1 = np.empty((2, 3), dtype=np.float64, order="F")
        Jp1 = np.empty((2, ase_camera.num_parameters()), dtype=np.float64, order="F")
        x1 = ase_camera.project(X, J1, Jp1)

        x2, J2, Jp2 = poselib_camera.project_with_jac_params(X)
        Jp2 = np.stack([np.delete(Jp2[0], 1), np.delete(Jp2[1], 0)], axis=0)

        if x1 is not None:
            assert np.allclose(x1, x2, atol=1e-6), f"ASE vs PoseLib: {x1} vs {x2}"
            assert np.allclose(J1, J2, atol=1e-6), f"ASE vs PoseLib Jacobian: {J1} vs {J2}"
            assert np.allclose(Jp1, Jp2, atol=1e-6), f"ASE vs PoseLib Param Jacobian: {Jp1} vs {Jp2}"
