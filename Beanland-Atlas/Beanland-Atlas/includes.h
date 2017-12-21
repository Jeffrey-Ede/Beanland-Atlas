#pragma once

//OpenCL
#include <CL/cl.hpp>

//Input and output
#include <fstream>
#include <iostream>

//Data containers
#include <vector>
#include <array>

#include "opencv2/highgui/highgui.hpp" //Loading images
#include <opencv2/imgproc/imgproc.hpp> //Convert RGB to greyscale
#include "opencv2/core/ocl.hpp"

//ArrayFire GPU accelerated algorithms
#include <arrayfire.h>
#include <af/array.h> //ArrayFire only acts on ArrayFire arrays
#include <opencv2/core/core.hpp>
#include <af/opencl.h> //Use ArrayFire with OpenCL

#include <cstdio>

//Blob detection
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

//Speed testing
#include <time.h>

//Dynamic, adaptive feature detection
#include <opencv2/features2d/features2d.hpp>

//Number of threads supported
#include <thread>

//CPU Parallelism
#include <omp.h>

//Accelerated CPU-based FFT
#include <complex.h>
#include <fftw3.h>

//Data type size limits
#include <limits>

//Gradient based symmetry detection
//#include <symmetry_func.h> //The algorithm is too slow

//Solve systems of equations
#include <Eigen/Dense>

#include "utility.hpp"
