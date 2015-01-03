import os

import numpy
import jinja2
from paver.easy import task, needs, path, sh, cmdopts
from paver.setuputils import setup, install_distutils_tasks, find_package_data
from distutils.extension import Extension
from optparse import make_option
from Cython.Build import cythonize

import version


DEVICE_VECTOR_TYPES = (('int8_t', 'np.int8'),
                       ('uint8_t', 'np.uint8'),
                       ('int16_t', 'np.int16'),
                       ('uint16_t', 'np.uint16'),
                       ('int32_t', 'np.int32'),
                       ('uint32_t', 'np.uint32'),
                       ('int64_t', 'np.int64'),
                       ('uint64_t', 'np.uint64'),
                       ('float', 'np.float32'),
                       ('double', 'np.float64'))

COMMON_DEVICE_VECTOR_TYPES = [DEVICE_VECTOR_TYPES[i] for i in (4, 5, 8)]
INTEGRAL_DEVICE_VECTOR_TYPES = DEVICE_VECTOR_TYPES[:-2]
FLOAT_DEVICE_VECTOR_TYPES = DEVICE_VECTOR_TYPES[-2:]


pyx_files = ['cythrust/si_prefix.pyx']


ext_modules = [Extension(f[:-4].replace('/', '.'), [f],
                         extra_compile_args=['-O3'],
                         include_dirs=['cythrust'])
               for f in pyx_files]

ext_modules = cythonize(ext_modules)


setup(name='cythrust',
      version=version.getVersion(),
      description='Cython bindings for the Thrust parallel library.',
      keywords='cython thrust cuda gpu numpy',
      author='Christian Fobel',
      url='https://github.com/cfobel/cythrust',
      license='GPL',
      packages=['cythrust'],
      package_data=find_package_data('cythrust', package='cythrust',
                                     only_in_packages=False),
      ext_modules=ext_modules,
      install_requires=['pandas', 'Cybuild', 'jinja2', 'theano-helpers'])


@task
def build_ext():
    pass


@task
@needs('build_ext', 'generate_setup', 'minilib', 'setuptools.command.sdist')
def sdist():
    """Overrides sdist to make sure that our setup.py is generated."""
    pass
