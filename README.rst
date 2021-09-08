see README.2nd, quick fix to facilitate installation with CUDA 10.2 (Sept 2021)

======
GPUMAP
======

GPU Parallelized Uniform Manifold Approximation and Projection (GPUMAP) is the
GPU-ported version of the UMAP dimension reduction technique that can be used
for visualisation similarly to t-SNE, but also for general non-linear dimension
reduction.

At the moment only CUDA capable GPUs are supported. Due to a dependency on
FAISS, only Linux (and potentially MacOS) platforms are supported at the moment.

For further information on UMAP see the `the original implementation
https://github.com/lmcinnes/umap/`.

-----------------
How to use GPUMAP
-----------------

The gpumap package inherits from sklearn classes, and thus drops in neatly
next to other sklearn transformers with an identical calling API.

.. code:: python

    import gpumap
    from sklearn.datasets import load_digits

    digits = load_digits()

    embedding = gpumap.GPUMAP().fit_transform(digits.data)

There are a number of parameters that can be set for the GPUMAP class; the
major ones are as follows:

 -  ``n_neighbors``: This determines the number of neighboring points used in
    local approximations of manifold structure. Larger values will result in
    more global structure being preserved at the loss of detailed local
    structure. In general this parameter should often be in the range 5 to
    50, with a choice of 10 to 15 being a sensible default.

 -  ``min_dist``: This controls how tightly the embedding is allowed compress
    points together. Larger values ensure embedded points are more evenly
    distributed, while smaller values allow the algorithm to optimise more
    accurately with regard to local structure. Sensible values are in the
    range 0.001 to 0.5, with 0.1 being a reasonable default.

The metric parameter is supported to keep the interface aligned with UMAP,
however, setting it to anything but 'euclidean' will fall back to the sequential
version. Processing sparse matrices is not supported either, and will similarly
cause a fallback to the sequential version for parts of the algorithm.

------------------------
Performance and Examples
------------------------

See `https://github.com/p3732/gpumap`.

Testing was done with an RTX2070 and Intel Xeon CPUs.

----------
Installing
----------

GPUMAP has the same dependecies of UMAP, namely ``scikit-learn``, ``numpy``,
``scipy`` and ``numba``. GPUMAP adds a requirement for ``faiss`` to perform
nearest-neighbor search on GPUs.

**Requirements:**

* scikit-learn
* (numpy)
* (scipy)
* numba
* faiss-gpu

**Install Options**

GPUMAP needs to be installed from source, depencies via pip, preferably in a python 3.7 virtual environment,
which can be created on the following way, e.g., in /applications/gpumap:

.. code:: bash
 
    venvname=gpumap
    venvpath=/applications/
    mypython=python3.7
    
    cd $venvpath
    $mypython -m venv $venvname
    source $venvpath/$venvname/bin/activate


**Build**

Make sure cuda and the NVIDIA drivers are present on your system. Tested with Ubuntu 18.04
x86_46 only (so far). The following repository has been used (added to /etc/apt/sources.list):

.. code:: 

    deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /
    
.. code:: bash

    sudo apt update
    sudo apt install cuda cuda-10-0 cuda-10-2

Building from source is easy, clone the repository or get the code onto your
computer by other means, install the 

.. code:: bash

    pip install --upgrade pip
    pip install cupy-cuda102==9.0.0rc1
    pip install faiss-gpu==1.7.1.post2
    pip install numpy==1.20.2
    pip install scikit-learn==0.24.1
    pip install scipy==1.6.2

    python setup.py install

Note that the dependecies need to be installed beforehand. Then test, is the installation
is working with the script. You will probably get deprecation warnings.

.. code:: bash

    python test_gpumap.py

-------
License
-------

The gpumap package is based on the umap package and thus is also 3-clause BSD
licensed.

------------
Contributing
------------

Contributions, in particular solutions to make GPUMAP compatible with current CUDA version, are highly welcome.
