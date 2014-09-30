import os

import jinja2
from paver.easy import task, needs, path, sh
from paver.setuputils import setup, install_distutils_tasks, find_package_data
from distutils.extension import Extension

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


pyx_files = (['cythrust/device_vector/%s/device_vector.pyx' % (dtype[3:])
              for ctype, dtype in DEVICE_VECTOR_TYPES] +
             ['cythrust/si_prefix.pyx', 'cythrust/functional.pyx',
              'cythrust/sparse.pyx', 'cythrust/reduce.pyx',
              'cythrust/describe.pyx', 'cythrust/tests/test_fusion.pyx'])



if os.environ.get('CYTHON_BUILD') is None:
    pyx_files = [f.replace('.pyx', '.cpp') for f in pyx_files]

ext_modules = [Extension(f[:-4].replace('/', '.'), [f],
                         extra_compile_args=['-O3', '-msse3', '-std=c++0x'],
                         include_dirs=[path('~/local/include').expand(),
                                       '/usr/local/cuda-6.5/include',
                                       'cythrust'],
                         define_macros=[('THRUST_DEVICE_SYSTEM',
                                         'THRUST_DEVICE_SYSTEM_CPP')])
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
      ext_modules=ext_modules)


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

        with dtype_module.joinpath('device_vector.pxd').open('wb') as output:
            template_path = device_root.joinpath('device_vector.pxdt')
            template = jinja2.Template(template_path.bytes())
            output.write(template.render({'C_DTYPE': ctype,
                                          'NP_DTYPE': dtype}))

        with dtype_module.joinpath('device_vector.pyx').open('wb') as output:
            template_path = device_root.joinpath('device_vector.pyxt')
            template = jinja2.Template(template_path.bytes())
            output.write(template.render({'C_DTYPE': ctype,
                                          'NP_DTYPE': dtype}))

        with dtype_module.joinpath('__init__.py').open('wb') as output:
            template_path = device_root.joinpath('dtype.__init__.pyt')
            template = jinja2.Template(template_path.bytes())
            output.write(template.render({'C_DTYPE': ctype,
                                          'NP_DTYPE': dtype}))

    with device_root.joinpath('__init__.py').open('wb') as output:
        template_path = device_root.joinpath('__init__.pyt')
        template = jinja2.Template(template_path.bytes())
        output.write(template.render({'DEVICE_VECTOR_TYPES':
                                      DEVICE_VECTOR_TYPES}))

    with device_root.joinpath('__init__.pxd').open('wb') as output:
        template_path = device_root.joinpath('__init__.pxdt')
        template = jinja2.Template(template_path.bytes())
        output.write(template.render({'DEVICE_VECTOR_TYPES':
                                      DEVICE_VECTOR_TYPES}))


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
    finally:
        os.chdir(cwd)


@task
@needs('build_device_vector_cu')
def build_device_vector_pyx():
    cwd = os.getcwd()
    package_root = path(__file__).abspath().parent
    device_root = package_root.joinpath('cythrust', 'cuda', 'device_vector')

    try:
        for ctype, dtype in DEVICE_VECTOR_TYPES:
            dtype_module = device_root.joinpath(dtype[3:])
            os.chdir(dtype_module)
            sh('nvcc -use_fast_math -shared -arch sm_20 --compiler-options '
               '"-fPIC -pthread '
               '-fno-strict-aliasing -DNDEBUG -g -fwrapv -Wall '
               '-Wstrict-prototypes -fPIC '
               '-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA" '
               '-I%s/local/include -I/usr/local/cuda-6.5/include '
               '-Icythrust -I/usr/include/python2.7 device_vector.cu '
               '-o device_vector.so')
        os.chdir(device_root.parent.joinpath('tests'))
        sh('nvcc -use_fast_math -shared -arch sm_20 --compiler-options '
           '"-fPIC -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -Wall '
           '-Wstrict-prototypes '
           '-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA" '
           '-I%s/local/include -I/usr/local/cuda-6.5/include '
           '-I../.. -I/usr/include/python2.7 test_fusion.cu '
           '-o test_fusion.so')
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
