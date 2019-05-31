from setuptools import setup

def readme():
    with open('README.rst') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'gpumap',
    'version': '0.1.0',
    'description' : 'UMAP with GPUs',
    'long_description' : readme(),
    'classifiers' : [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7',
    ],
    'keywords' : 'dimension reduction t-sne manifold umap',
    'url' : 'http://github.com/p3732/gpumap',
    'maintainer' : 'Peter Eisenmann',
    'maintainer_email' : 'p3732@gmx.de',
    'license' : 'BSD',
    'packages' : ['gpumap'],
    'install_requires': ['numpy >= 1.13',
                         'scikit-learn >= 0.16',
                         'scipy >= 0.19',
                         'numba >= 0.37',
                         'faiss >= 1.5.3'],
    'ext_modules' : [],
    'cmdclass' : {},
    'test_suite' : 'nose.collector',
    'tests_require' : ['nose'],
    'data_files' : ()
    }

setup(**configuration)
