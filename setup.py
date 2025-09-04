import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DPython_EXECUTABLE=' + sys.executable,
                      '-DPYTHON_PACKAGE=ON',
                      '-DBUILD_SHARED_LIBS=OFF',]
        if os.environ.get('CMAKE_INSTALL_PREFIX') is not None:
            cmake_args += [f"-DCMAKE_INSTALL_PREFIX={os.environ.get('CMAKE_INSTALL_PREFIX')}"]
        else:
            # Set the install prefix to the extension directory so install() commands work
            cmake_args += [f"-DCMAKE_INSTALL_PREFIX={extdir}"]
            
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            if os.environ.get('CMAKE_TOOLCHAIN_FILE') is not None:
                cmake_toolchain_file = os.environ.get('CMAKE_TOOLCHAIN_FILE')
                cmake_args += [f'-DCMAKE_TOOLCHAIN_FILE={cmake_toolchain_file}']
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                if os.environ.get('CMAKE_TOOLCHAIN_FILE') is not None:
                    cmake_args += ['-DVCPKG_TARGET_TRIPLET=x64-windows']
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        
        # Run cmake --install but only install the python component to avoid library headers/libs
        install_args = ['--config', cfg] if platform.system() == "Windows" else []
        install_args += ['--component', 'python']
        subprocess.check_call(['cmake', '--install', '.'] + install_args, cwd=self.build_temp)



# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="poselib",
    version="2.0.5",
    author="Viktor Larsson and contributors",
    author_email="viktor.larsson@math.lth.se",
    description="",
    long_description="",
    ext_modules=[CMakeExtension("poselib._core")],
    packages=["poselib"],
    package_dir={"poselib": "pyposelib"},
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
