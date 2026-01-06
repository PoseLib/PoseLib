import sys
import poselib

def test_poselib():
    print(f'Python version: {sys.version}')
    print(f'PoseLib version: {poselib.__version__}')
    if sys.version_info < (3, 14):
        from posebench import run_benchmark
        print(f'Running posebench...')
        result = run_benchmark(
            subsample=10,
            subset=True,
        )
        print(result)
        print('Posebench done.')
    else:
        print('Skipping tests for Python >= 3.14 due to deps not ready yet (h5py)')