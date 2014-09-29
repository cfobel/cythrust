import os
from paver.easy import task, needs, path, sh
from paver.setuputils import setup, install_distutils_tasks

import version


setup(name='cythrust',
      version=version.getVersion(),
      description='Cython bindings for the Thrust parallel library.',
      keywords='cython thrust cuda gpu numpy',
      author='Christian Fobel',
      url='https://github.com/cfobel/cythrust',
      license='GPL',
      packages=['cythrust'],
      package_data={'cythrust': ['*.p??']})


#DEVICE_VECTOR_TYPES = (('int32_t', 'np.int32'), )

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


@task
def generate_device_vector_source():
    '''
    Generate Cython code from template for `DeviceVector` extension types.
    '''
    import jinja2

    package_root = path(__file__).abspath().parent

    for ctype, dtype in DEVICE_VECTOR_TYPES:
        dtype_module = package_root.joinpath('cythrust', 'device_vector',
                                             dtype[3:])
        dtype_module.makedirs_p()

        with dtype_module.joinpath('device_vector.pxd').open('wb') as output:
            template_path = package_root.joinpath('cythrust', 'device_vector',
                                                  'device_vector.pxdt')
            template = jinja2.Template(template_path.bytes())
            output.write(template.render({'C_DTYPE': ctype,
                                          'NP_DTYPE': dtype}))

        with dtype_module.joinpath('device_vector.pyx').open('wb') as output:
            template_path = package_root.joinpath('cythrust', 'device_vector',
                                                  'device_vector.pyxt')
            template = jinja2.Template(template_path.bytes())
            output.write(template.render({'C_DTYPE': ctype,
                                          'NP_DTYPE': dtype}))

        with dtype_module.joinpath('__init__.py').open('wb') as output:
            template_path = package_root.joinpath('cythrust', 'device_vector',
                                                  'dtype.__init__.pyt')
            template = jinja2.Template(template_path.bytes())
            output.write(template.render({'C_DTYPE': ctype,
                                          'NP_DTYPE': dtype}))

    device_vector_path = package_root.joinpath('cythrust', 'device_vector')

    with device_vector_path.joinpath('__init__.py').open('wb') as output:
        template_path = device_vector_path.joinpath('__init__.pyt')
        template = jinja2.Template(template_path.bytes())
        output.write(template.render({'DEVICE_VECTOR_TYPES':
                                      DEVICE_VECTOR_TYPES}))

    with device_vector_path.joinpath('__init__.pxd').open('wb') as output:
        template_path = device_vector_path.joinpath('__init__.pxdt')
        template = jinja2.Template(template_path.bytes())
        output.write(template.render({'DEVICE_VECTOR_TYPES':
                                      DEVICE_VECTOR_TYPES}))


@task
@needs('generate_device_vector_source')
def build_device_vector_cpp():
    cwd = os.getcwd()
    package_root = path(__file__).abspath().parent

    try:
        for ctype, dtype in DEVICE_VECTOR_TYPES:
            dtype_module = package_root.joinpath('cythrust', 'device_vector',
                                                 dtype[3:])
            os.chdir(dtype_module)
            sh('cython device_vector.pyx --cplus -I../..')
    finally:
        os.chdir(cwd)


@task
@needs('build_device_vector_cpp')
def build_device_vector_pyx():
    cwd = os.getcwd()
    package_root = path(__file__).abspath().parent

    try:
        for ctype, dtype in DEVICE_VECTOR_TYPES:
            dtype_module = package_root.joinpath('cythrust', 'device_vector',
                                                 dtype[3:])
            os.chdir(dtype_module)
            sh('g++ -O3 -shared -pthread -fPIC -fwrapv -fno-strict-aliasing '
               '-I/usr/include/python2.7 -o device_vector.so device_vector.cpp'
               ' -I/usr/local/cuda-6.5/include -I../.. '
               '-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP')
    finally:
        os.chdir(cwd)


@task
@needs('build_device_vector_pyx', 'generate_setup', 'minilib',
       'setuptools.command.sdist')
def sdist():
    """Overrides sdist to make sure that our setup.py is generated."""
    pass
