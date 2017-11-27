# Beanland Atlas

Pioneering work by Dr Richard Beanland on high dynamic range electron diffraction seabed patterns has led to the development of a new ultra-high resolution structure determination technique. This technique offers orders of magnitude higher resolution than traditional x-ray diffraction techniques and is able to resolve the bonding contributions of individual atomic orbitals.

This repository is being built to make this vision a reality. It will house code that

* Automates the construction of Beanland's seabed atlas
* Performs aberration and other corrections
* Identifies quantum mechanical symmetries
* Quantifies material structure, including the individual contributions of atomic orbitals to material bonds

Beanland's procedure is being automated in a statistically robust way that requires minimal user intervention, allowing his procedure to be used as a "black box" routine by other scientists. 

The procedure is extremely complex and currently relies on supercomputational support. Parallelisable portions of the code are therefore being optimised to run on GPUs or over multiple CPUs. It is hoped that in the future the full procedure will be implementable in less than 30s on most desktop computors. 

The code is being built on the following libraries:

* ArrayFire 3
* FFTW3
* OpenCL 2.0
* OpenCV 3.3
* OpenMP

A full list of dependencies can be found in the main project header.

The Beanland Atlas project is in a pre-release state. Critical portions of the code are still in development and it has not been fully optimised. Nevertheless, the reposited code may be freely used; without restriction, as described by the MIT License.
