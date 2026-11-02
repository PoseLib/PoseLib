import subprocess
import sys
import textwrap

import numpy as np
import poselib
import pytest


@pytest.mark.parametrize("function_name", ["estimate_homography", "refine_homography"])
@pytest.mark.parametrize(
    "num_points1, num_points2", [(9, 4), (4, 9), (0, 4), (4, 0), (1, 3), (3, 1)]
)
def test_homography_rejects_mismatched_point_counts(
    function_name, num_points1, num_points2
):
    # Isolate native crashes so a regression cannot terminate the pytest process.
    code = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import numpy as np
        import poselib
        import pytest

        function_name, num_points1, num_points2, expected_core = sys.argv[1:]
        assert Path(poselib._core.__file__).resolve() == Path(expected_core).resolve()
        rng = np.random.default_rng(0)
        points1 = rng.uniform(-1, 1, (int(num_points1), 2))
        points2 = rng.uniform(-1, 1, (int(num_points2), 2))
        with pytest.raises(ValueError) as exc_info:
            if function_name == "estimate_homography":
                poselib.estimate_homography(points1, points2)
            else:
                poselib.refine_homography(points1, points2, np.eye(3))
        assert str(exc_info.value) == (
            "points2D_1 and points2D_2 must have the same number of points."
        )
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            function_name,
            str(num_points1),
            str(num_points2),
            poselib._core.__file__,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{function_name}({num_points1}, {num_points2}) exited with "
        f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("function_name", ["estimate_homography", "refine_homography"])
def test_homography_recovers_known_transform(function_name):
    points1 = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [0.2, 0.3],
            [-0.4, 0.7],
            [0.8, -0.2],
            [-0.7, -0.3],
            [0.3, -0.8],
        ]
    )
    expected_H = np.array([[1.1, 0.1, 0.3], [-0.05, 0.9, -0.2], [0.01, -0.02, 1.0]])
    points1_h = np.column_stack((points1, np.ones(len(points1))))
    points2_h = points1_h @ expected_H.T
    points2 = points2_h[:, :2] / points2_h[:, 2:]

    if function_name == "estimate_homography":
        H, _ = poselib.estimate_homography(
            points1,
            points2,
            {"ransac": {"seed": 0, "min_iterations": 64, "max_iterations": 64}},
        )
    else:
        initial_H = expected_H.copy()
        initial_H[0, 2] += 0.05
        H, _ = poselib.refine_homography(points1, points2, initial_H)

    np.testing.assert_allclose(H / H[2, 2], expected_H, rtol=1e-6, atol=1e-6)
    projected = points1_h @ H.T
    np.testing.assert_allclose(
        projected[:, :2] / projected[:, 2:], points2, rtol=1e-6, atol=1e-6
    )
