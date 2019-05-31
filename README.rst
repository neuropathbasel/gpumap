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

GPUMAP, like UMAP, is very efficient at embedding large high dimensional
datasets. In particular it scales well with both input dimension and embedding
dimension. Performance depends strongly depends on the used GPU. For a problem
such as the 784-dimensional MNIST digits dataset with 70000 data samples, GPUMAP
can complete the embedding in around 30 seconds with an (outdated) NVIDIA GTX
745 graphics card. More recent hardware will scale accordingly. Despite this
runtime efficiency UMAP still produces high quality embeddings.

The obligatory MNIST digits dataset, embedded in 29 seconds using a 3.6 GHz
Intel Core i7 processor and an NVIDIA GTX 745 GPU (n_neighbors=10,
min_dist=0.001):

.. image:: images/gpumap_example_mnist1.png
    :alt: GPUMAP embedding of MNIST digits

The MNIST digits dataset is fairly straightforward however. A better test is
the more recent "Fashion MNIST" dataset of images of fashion items (again
70000 data sample in 784 dimensions). GPUMAP
produced this embedding in 2 minutes exactly (n_neighbors=5, min_dist=0.1):

.. image:: images/gpumap_example_fashion_mnist1.png
    :alt: GPUMAP embedding of "Fashion MNIST"

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
* faiss

**Install Options**

GPUMAP can be installed via Conda, PyPi or from source:

**Option 1: Conda**

Set up a new conda environment, if needed.

.. code:: bash

    conda create -n env

    conda activate env

    conda install python

Install dependecies: Numba and FAISS

.. code:: bash

    conda install numba
    conda install scikit-learn

    conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10
    # For older CUDA versions:
    # conda install faiss-gpu cudatoolkit=8.0 -c pytorch # For CUDA8
    # conda install faiss-gpu cudatoolkit=9.0 -c pytorch # For CUDA9

    conda install -c conda-forge gpumap

**Option 2: PyPi**

GPUMAP is also available as a PyPi package.

.. code:: bash

    pip install scikit-learn numba faiss gpumap

Note that the prebuilt FAISS library is not officially supported by upstream.

**Option 3: Build**

Building from source is easy, clone the repository or get the code onto your
computer by other means and run the installer with:

.. code:: bash

    python setup.py install

Note that the dependecies need to be installed beforehand. These are
the `FAISS https://github.com/facebookresearch/faiss/blob/master/INSTALL.md`
library and `Numba http://numba.pydata.org/numba-doc/latest/user/installing.html`.

-------
License
-------

The gpumap package is based on the umap package and thus is also 3-clause BSD
licensed.

------------
Contributing
------------

Contributions are always welcome! Fork away!

