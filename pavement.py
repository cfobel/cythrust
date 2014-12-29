import os

import numpy
import jinja2
from paver.easy import task, needs, path, sh, cmdopts
from paver.setuputils import setup, install_distutils_tasks, find_package_data
from distutils.extension import Extension
from optparse import make_option

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


pyx_files = (['cythrust/device_vector/%s/device_vector.pyx' % (dtype[3:])
              for ctype, dtype in DEVICE_VECTOR_TYPES] +
             ['cythrust/device_vector/copy.pyx',
              'cythrust/device_vector/count.pyx',
              'cythrust/device_vector/extrema.pyx',
              'cythrust/device_vector/partition.pyx',
              'cythrust/device_vector/sort.pyx',
              'cythrust/device_vector/sum.pyx',
              'cythrust/si_prefix.pyx',
              'cythrust/functional.pyx', 'cythrust/sparse.pyx',
              'cythrust/reduce.pyx', 'cythrust/describe.pyx',
              'cythrust/tests/test_fusion.pyx'])



if os.environ.get('CYTHON_BUILD') is None:
    pyx_files = [f.replace('.pyx', '.cpp') for f in pyx_files]

ext_modules = [Extension(f[:-4].replace('/', '.'), [f],
                         extra_compile_args=['-O3', '-msse3', '-std=c++0x',
                                             '-fopenmp'],
                         #extra_link_args=['-lgomp'],
                         include_dirs=[path('~/local/include').expand(),
                                       '/usr/local/cuda-6.5/include',
                                       'cythrust', numpy.get_include()],
                         define_macros=[('THRUST_DEVICE_SYSTEM',
                                         'THRUST_DEVICE_SYSTEM_CPP')])
                                         #'THRUST_DEVICE_SYSTEM_OMP')])
               for f in pyx_files]

if os.environ.get('CYTHON_BUILD') is not None:
    from Cython.Build import cythonize

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
      install_requires=['pandas', 'Cybuild', 'jinja2'])


@task
def generate_device_vector_source():
    '''
    Generate Cython code from template for `DeviceVector` extension types.
    '''
    package_root = path(__file__).abspath().parent
    device_root = package_root.joinpath('cythrust', 'device_vector')
    _generate_device_vector_source(device_root)


@task
def generate_device_vector_source_cuda():
    '''
    Generate Cython code from template for `DeviceVector` extension types.
    '''
    package_root = path(__file__).abspath().parent
    device_root = package_root.joinpath('cythrust', 'cuda', 'device_vector')
    _generate_device_vector_source(device_root)


def _generate_device_vector_source(device_root):
    for ctype, dtype in DEVICE_VECTOR_TYPES:
        dtype_module = device_root.joinpath(dtype[3:])
        dtype_module.makedirs_p()

        for f in ('device_vector.pxd', 'device_vector.pyx', '__init__.py'):
            output_path = dtype_module.joinpath(f)
            template_path = device_root.joinpath(f + 't')

            if not output_path.exists() or (output_path.mtime <
                                            template_path.mtime):
                with output_path.open('wb') as output:
                    template = jinja2.Template(template_path.bytes())
                    output.write(template.render({'C_DTYPE': ctype,
                                                  'NP_DTYPE': dtype}))
                    print 'wrote:', output_path


    for f in ('__init__.py', '__init__.pxd', 'copy.pyx', 'count.pyx',
              'extrema.pyx', 'partition.pyx', 'sort.pyx', 'sum.pyx'):
        output_path = device_root.joinpath(f)
        template_path = device_root.joinpath(f + 't')

        if not output_path.exists() or (output_path.mtime <
                                        template_path.mtime):
            with output_path.open('wb') as output:
                template = jinja2.Template(template_path.bytes())
                output.write(template.render({'DEVICE_VECTOR_TYPES':
                                              DEVICE_VECTOR_TYPES,
                                              'COMMON_DEVICE_VECTOR_TYPES':
                                              COMMON_DEVICE_VECTOR_TYPES,
                                              'INTEGRAL_DEVICE_VECTOR_TYPES':
                                              INTEGRAL_DEVICE_VECTOR_TYPES,
                                              'FLOAT_DEVICE_VECTOR_TYPES':
                                              FLOAT_DEVICE_VECTOR_TYPES}))
                print 'wrote:', output_path


@task
@needs('generate_device_vector_source_cuda')
def build_device_vector_cu():
    cwd = os.getcwd()
    package_root = path(__file__).abspath().parent
    device_root = package_root.joinpath('cythrust', 'cuda', 'device_vector')

    try:
        for ctype, dtype in DEVICE_VECTOR_TYPES:
            dtype_module = device_root.joinpath(dtype[3:])
            os.chdir(dtype_module)
            sh('cython device_vector.pyx --cplus -I../.. -o device_vector.cu')
        os.chdir(device_root.parent.joinpath('tests'))
        sh('cython test_fusion.pyx --cplus -I../.. -o test_fusion.cu')

        os.chdir(device_root)
        for f in ('copy', 'count', 'extrema', 'partition', 'sort', 'sum'):
            sh('cython %s.pyx --cplus -I../.. -o %s.cu' % (f, f))

        os.chdir(device_root.parent)
        for f in ('cycudart', ):
            sh('cython %s.pyx --cplus -I.. -o %s.cu' % (f, f))
    finally:
        os.chdir(cwd)


@task
@cmdopts([
    make_option('-a', '--arch', help='CUDA architecture to compile for '
                '_(e.g., `sm_20`, `sm_13`, etc.)_.')
])
@needs('build_device_vector_cu')
def build_device_vector_pyx(options):
    arch = options['build_device_vector_pyx'].get('arch', 'sm_20')
    cwd = os.getcwd()
    package_root = path(__file__).abspath().parent
    device_root = package_root.joinpath('cythrust', 'cuda', 'device_vector')

    NVCC_BUILD = ('nvcc -use_fast_math -shared -arch %s --compiler-options '
                  '"-fPIC -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv '
                  '-Wall ' '-Wstrict-prototypes '
                  '-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA" '
                  '-I{home}/local/include -I/usr/local/cuda-6.5/include '
                  '-I../.. -I/usr/include/python2.7 {namebase}.cu '
                  '-I%s '
                  '-o {namebase}.so' % (arch, numpy.get_include()))

    try:
        for ctype, dtype in DEVICE_VECTOR_TYPES:
            dtype_module = device_root.joinpath(dtype[3:])
            os.chdir(dtype_module)
            sh(NVCC_BUILD.format(home=path('~').expand(),
                                 namebase='device_vector'))

        #os.chdir(device_root.parent.joinpath('tests'))
        #sh(NVCC_BUILD.format(home=path('~').expand(),
                              #namebase='test_fusion'))

        os.chdir(device_root)
        for f in ('copy', 'count', 'extrema', 'partition', 'sort', 'sum'):
            sh(NVCC_BUILD.format(home=path('~').expand(), namebase=f))
    finally:
        os.chdir(cwd)


@task
@needs('generate_device_vector_source', 'generate_setup', 'minilib',
       'setuptools.command.build_ext')
def build_ext():
    """Overrides sdist to make sure that our setup.py is generated."""
    pass


@task
@needs('build_ext', 'generate_setup', 'minilib', 'setuptools.command.sdist')
def sdist():
    """Overrides sdist to make sure that our setup.py is generated."""
    pass
