# Beanland Atlas

Pioneering work by Dr Richard Beanland on high dynamic range electron diffraction seabed patterns has led to the development of a new ultra-high resolution structure determination technique. This technique offers orders of magnitude higher resolution than traditional x-ray diffraction techniques and is able to resolve the bonding contributions of individual atomic orbitals.

This repository will house code that:

* Automates the construction of Beanland's seabed atlases
* Performs aberration and other corrections
* Identifies and makes use of atlas diffraction pattern symmetries
* Quantifies material structure, including the individual contributions of atomic orbitals to material bonds

Beanland's procedure is being automated in a statistically robust way that requires minimal user intervention, allowing his procedure to be used as a "black box" routine by other scientists. The procedure is extremely intensive and currently relies on supercomputational support. Parallelisable portions of the procedure are therefore being optimised to run on GPUs or over multiple CPUs.

Libraries and acceleration packages that the code is built upon include:

* [ArrayFire 3](http://arrayfire.org/docs/index.htm)
* [FFTW3](http://www.fftw.org/)
* [OpenCL 2.2](https://www.khronos.org/opencl/)
* [OpenCV 3.3](https://opencv.org/opencv-3-3.html)
* [OpenMP](https://msdn.microsoft.com/en-us/library/tt15eb9t.aspx)

A full list of dependencies can be found in the main [project header](https://github.com/Jeffrey-Ede/Beanland-Atlas/blob/master/Beanland-Atlas/Beanland-Atlas/beanland_atlas.h).

The Beanland Atlas project is in a pre-release state: critical portions of the code are still in development and the reposited code has not been fully optimised. Nevertheless, the reposited code may be used without restriction, as described by the MIT license.
