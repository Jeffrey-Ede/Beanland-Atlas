#include <utility.h>

namespace ba
{
	/*Calculate Pearson normalised product moment correlation coefficient between 2 vectors of floats
	**Inputs:
	**vect1: std::vector<float>, One of the datasets to use in the calculation
	**vect2: std::vector<float>, The other dataset to use in the calculation
	**NUM_THREADS: const int, The number of threads to use for OpenMP CPU acceleration
	**Return:
	**float, Pearson normalised product moment correlation coefficient between the 2 datasets
	*/
	float pearson_corr(std::vector<float> vect1, std::vector<float> vect2, const int NUM_THREADS) {

		//Sums for Pearson product moment correlation coefficient
		float sum_xy = 0.0f;
		float sum_x = 0.0f;
		float sum_y = 0.0f;
		float sum_x2 = 0.0f;
		float sum_y2 = 0.0f;

		//Reflect points across mirror line and compare them to the nearest pixel
		#pragma omp parallel for num_threads(NUM_THREADS), reduction(sum:sum_xy), reduction(sum:sum_x), reduction(sum:sum_y), reduction(sum:sum_x2), reduction(sum:sum_y2)
		for (int i = 0; i < vect1.size(); i++) {

			//Contribute to Pearson correlation coefficient
			sum_xy += vect1[i]*vect2[i];
			sum_x += vect1[i];
			sum_y += vect2[i];
			sum_x2 += vect1[i]*vect1[i];
			sum_y2 += vect2[i]*vect2[i];
		}

		return (vect1.size()*sum_xy - sum_x*sum_y) / (std::sqrt(vect1.size()*sum_x2 - sum_x*sum_x) * 
			std::sqrt(vect1.size()*sum_y2 - sum_y*sum_y));
	}

	/*Utility function that builds a named kernel from source code. Will print errors if there are problems compiling it
	**Inputs:
	**kernel_sourceFile: const char*, File containing kernel source code
	**kernel_name: const char*, Name of kernel to be built
	**af_context: cl_context, Context to create kernel in
	**af_device_id: cl_device_id, Device to run kernel on
	**Returns:
	**cl_kernel, Build kernel ready for arguments to be passed to it
	*/
	cl_kernel create_kernel(const char* kernel_sourceFile, const char* kernel_name, cl_context af_context, cl_device_id af_device_id)
	{
		//Read the program source
		std::ifstream sourceFile(kernel_sourceFile);
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		const char * source = sourceCode.c_str();

		//Create program
		int status = CL_SUCCESS;
		cl_program program = clCreateProgramWithSource(af_context, 1, &source, NULL, &status);

		//Build the program
		if (clBuildProgram(program, 1, &af_device_id, "", NULL, NULL) != CL_SUCCESS) {

			//Print the source code
			std::cout << source << std::endl;

			//Print the buildlog if the kernel fails to compile
			char buffer[10240];
			clGetProgramBuildInfo(program, af_device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
			fprintf(stderr, "CL Compilation failed:\n%s", buffer);
			abort();
		}

		//Create kernel
		cl_kernel kernel = clCreateKernel(program, kernel_name, &status);

		//Free OpenCL resources
		clReleaseProgram(program);

		return kernel;
	}

	/*Calculate weighted 1st order autocorrelation using weighted Pearson normalised product moment correlation coefficient.
	**That is, the ratio of the weighted covariance to the product of the weighted variances of the data and the lagged data
	**Inputs:
	**data: std::vector<float>, One of the datasets to use in the calculation
	**Errors: std::vector<float>, Errors in dataset elements used in the calculation
	**Return:
	**float, Measure of the autocorrelation. 2-2*<return value> approximates the Durbin-Watson statistic for large datasets
	*/
	float weighted_pearson_autocorr(std::vector<float> data, std::vector<float> err, const int NUM_THREADS) 
	{
		//sums for pearson product moment correlation coefficient
		//x - data; e - error, 1 - lagged data, 2 - forward data

		//Start by calculating means using weighted sums
		float sum_x = 0.0f;
		float sum_x_err = 0.0f;

		int size_minus1 = data.size()-1;
		#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 1; i < size_minus1; i++) {

			#pragma omp atomic
			sum_x += data[i]/(err[i]*err[i]);
			#pragma omp atomic
			sum_x_err += 1.0f/(err[i]*err[i]);
		}

		//Calculate weighted means (Bessel correction)
		float x1_mean = (sum_x + data[0]/(err[0]*err[0])) / (sum_x_err + 1.0f/(err[0]*err[0]));
		float x2_mean = (sum_x + data[size_minus1]/(err[size_minus1]*err[size_minus1])) / (sum_x_err + 1.0f/(err[size_minus1]*err[size_minus1]));

		//Calculate Pearson normalised product moment correlation coefficient, disregarding errors of means
		float sum_xy = 0.0f;
		float sum_xy_err = 0.0f;
		float sum_x2 = 0.0f;
		float sum_x2_err1 = 0.0f;
		float sum_x2_err2 = 0.0f;

		int i;
		#pragma omp parallel for num_threads(NUM_THREADS)
		for (i = 1; i < size_minus1; i++) 
		{
			#pragma omp atomic
			sum_xy += (data[i]-x2_mean)*(data[i-1]-x1_mean) / (std::abs(data[i]-x2_mean)*err[i-1] + std::abs(data[i-1]-x1_mean)*err[i]);
			#pragma omp atomic
			float weight = 1.0f /(std::abs(data[i]-x2_mean)*err[i-1] + std::abs(data[i-1]-x1_mean)*err[i]);
			sum_xy_err += weight*(1-weight);

			#pragma omp atomic
			sum_x2 += 1.0f / (err[i]*err[i]);
			#pragma omp atomic
			sum_x2_err1 += 1.0f / (std::abs(data[i]-x1_mean)*std::abs(data[i]-x1_mean)*err[i]*err[i]);
			#pragma omp atomic
			sum_x2_err2 += 1.0f / (std::abs(data[i]-x2_mean)*std::abs(data[i]-x2_mean)*err[i]*err[i]);
		}

		sum_xy += (data[i]-x2_mean)*(data[i-1]-x1_mean) / (std::abs(data[i]-x2_mean)*err[i-1] + std::abs(data[i-1]-x1_mean)*err[i]);
		float weight = 1.0f /(std::abs(data[i]-x2_mean)*err[i-1] + std::abs(data[i-1]-x1_mean)*err[i]);
		sum_xy_err += weight*(1-weight);

		//Weighted sums of squares
		float x1 = (sum_x2 + 1.0f/(err[0]*err[0])) / 
			(sum_x2_err1 + 1.0f / (std::abs(data[0]-x1_mean)*std::abs(data[0]-x1_mean)*err[0]*err[0]));
		float x2 = (sum_x2 + 1.0f/(err[size_minus1]*err[size_minus1])) / 
			(sum_x2_err2 + 1.0f / (std::abs(data[size_minus1]-x2_mean)*std::abs(data[size_minus1]-x2_mean)*err[size_minus1]*err[size_minus1]));

		return ( sum_xy/(data.size()*sum_xy_err) ) / ( std::sqrt( x1 ) * std::sqrt( x2 ) );
	}

	/*Calculates the factorial of a small integer
	**Input:
	**n: long unsigned int, Number to find factorial of
	**Return:
	**long unsigned int, Reciprocal of input
	*/
	long unsigned int factorial(long unsigned int n)
	{
		return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
	}

	/*Calculates the power of 2 greater than or equal to the supplied number
	**Inputs:
	**n: int, Number to find the first positive power of 2 greater than or equal to
	**ceil: int, This parameter should not be inputted. It is used to recursively find the power of 2 greater than or equal to the supplied number
	**Return:
	**int, Power of 2 greater than or equal to the input
	*/
	int ceil_power_2(int n, int ceil)
	{
		return n <= ceil ? ceil : ceil_power_2(n, 2*ceil);
	}

	/*Calculate Pearson normalised product moment correlation coefficient between 2 OpenCV mats for some offset between them
	**Beware: function needs debugging
	**Inputs:
	**img1: cv::Mat &, One of the mats
	**img2: cv::Mat &, The other mat
	**j: const int, Offset of the second mat's columns from the first's
	**i: const int, Offset of the second mat's rows from the first's
	**Return:
	**float, Pearson normalised product moment correlation coefficient between the 2 mats
	*/
	float pearson_corr(cv::Mat &img1, cv::Mat &img2, const int j, const int i) 
	{
		float pear;

		//Check if this coefficient is greater than the maximum
		if (i >= 0 && j >= 0)
		{
			int row_span = std::min(img1.rows-i, img2.rows);
			int col_span = std::min(img1.cols-j, img2.cols);
			pear = pearson_corr(img1(cv::Rect(j, i, col_span, row_span)), 
				img2(cv::Rect(0, 0, col_span, row_span)));
		}
		else
		{
			if (i >= 0 && j < 0)
			{
				int row_span = std::min(img1.rows-i, img2.rows);
				int col_span = std::min(img1.cols+j, img2.cols);
				pear = pearson_corr(img1(cv::Rect(0, i, col_span, row_span)), 
					img2(cv::Rect(img2.cols-col_span, 0, col_span, row_span)));
			}
			else
			{
				if (i < 0 && j >= 0)
				{
					int row_span = std::min(img1.rows+i, img2.rows);
					int col_span = std::min(img1.cols-j, img2.cols);
					pear = pearson_corr(img1(cv::Rect(j, 0, col_span, row_span)), 
						img2(cv::Rect(0, img2.rows-row_span, col_span, row_span)));
				}
				else
				{
					int row_span = std::min(img1.rows+i, img2.rows);
					int col_span = std::min(img1.cols+j, img2.cols);

					pear = pearson_corr(img1(cv::Rect(0, 0, col_span, row_span)), 
						img2(cv::Rect(img2.cols-col_span, img2.rows-row_span, col_span, row_span)));
				}
			}
		}

		return pear;
	}

	/*Create a copy of an image where the black pixels are replaced with the mean values of the image
	**img: cv::Mat &, Floating point mat to make a black pixel free copy of
	**Returns:
	**cv::Mat, Copy of input mat where black pixels have been replaced with the mean matrix value
	*/
	cv::Mat black_to_mean(cv::Mat &img)
	{
		//Create the dbackened mat
		cv::Mat no_black = cv::Mat(img.size(), img.type());

		//Get the mean px value
		cv::Scalar mean = cv::mean(img);

		float *r, *s;
		//Iterate across mat rows...
        #pragma omp parallel for
		for (int m = 0; m < img.rows; m++) 
		{
			//...and iterate across mat columns
			r = img.ptr<float>(m);
			s = no_black.ptr<float>(m);
			for (int n = 0; n < img.cols; n++) 
			{
				s[n] = r[n] ? r[n] : mean.val[0];
			}
		}

		return no_black;
	}

	/*Gaussian blur an image based on its size
	**img: cv::Mat &, Floating point mam to blur
	**frac: float, Gaussian kernel size as a fraction of the image's smallest dimension's size
	**Returns,
	**cv::Mat, Blurred copy of the input mat
	*/
	cv::Mat blur_by_size(cv::Mat &img, float blur_frac)
	{
		cv::Mat blurred;

		if (blur_frac != 0.0f)
		{
			//Make sure the kernel size is at least one
			int k_size = (int)(blur_frac*std::min(img.rows, img.cols)) < 1 ? 1 : (int)(blur_frac*std::min(img.rows, img.cols));

			//Blur the image
			cv::filter2D(img, blurred, img.depth(), cv::getGaussianKernel(k_size, -1, CV_32F));

			return blurred;
		}
		else
		{
			return img;
		}
	}

	/*Calculate the autocorrelation of an OpenCV mat
	**img: cv::Mat &, Image to calculate the autocorrelation of
	**Returns,
	**cv::Mat, Autocorrelation of the image
	*/
	cv::Mat autocorrelation(cv::Mat &img)
	{
		cv::Mat autocorr;
		cv::filter2D(img, autocorr, CV_32FC1, img, cv::Point(-1,-1), 0.0, cv::BORDER_CONSTANT);

		return autocorr;
	}

	/*Calculate Pearson's normalised product moment correlation coefficient between 2 floating point same-size OpenCV mats
	**img1: cv::Mat &, One of the mats
	**img2: cv::Mat &, The other mat
	**Returns,
	**float, Pearson normalised product moment correlation coefficient between the 2 mats
	*/
	float pearson_corr(cv::Mat &img1, cv::Mat &img2)
	{
		//Sums for Pearson product moment correlation coefficient
		float sum_xy = 0.0f;
		float sum_x = 0.0f;
		float sum_y = 0.0f;
		float sum_x2 = 0.0f;
		float sum_y2 = 0.0f;

		//Reflect points across mirror line and compare them to the nearest pixel
		float *p, *q;
        #pragma omp parallel for reduction(sum:sum_xy), reduction(sum:sum_x), reduction(sum:sum_y), reduction(sum:sum_x2), reduction(sum:sum_y2)
		for (int i = 0; i < img1.rows; i++)
		{
			p = img1.ptr<float>(i);
			q = img2.ptr<float>(i);
			for (int j = 0; j < img1.cols; j++)
			{
				//If the pixels are both not black - safegaurd against rows or columns with some black in the middle
				if (p[j] && q[j])
				{
					//Contribute to Pearson correlation coefficient
					sum_xy += p[i]*q[i];
					sum_x += p[i];
					sum_y += q[i];
					sum_x2 += p[i]*p[i];
					sum_y2 += q[i]*q[i];
				}
			}
		}

		return (img1.rows*img2.cols*sum_xy - sum_x*sum_y) / (std::sqrt(img1.rows*img2.cols*sum_x2 - sum_x*sum_x) * 
			std::sqrt(img1.rows*img2.cols*sum_y2 - sum_y*sum_y));
	}

	/*Calculate the average feature size in an image by summing the components of its 2D Fourier transform in quadrature to produce a 
	**1D frequency spectrum to find the weighted centroid of
	**Inputs:
	**img: cv::Mat &, Image to get the average feature size in
	**Return:
	**float, Estimated average feature size in px
	*/
	float get_avg_feature_size(cv::Mat &img)
	{
		//Expand the input image to optimal size
		cv::Mat padded;
		int m = cv::getOptimalDFTSize( img.rows );
		int n = cv::getOptimalDFTSize( img.cols );
		cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0)); //On the border add zero values

		cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32FC1)};
		cv::Mat complexI;
		cv::merge(planes, 2, complexI); //Add to the expanded another plane with zeros

		cv::dft(complexI, complexI); //This way the result may fit in the source matrix

	    //Compute the FFT magnitude
		cv::split(complexI, planes); //planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
		cv::magnitude(planes[0], planes[1], planes[0]); //planes[0] = magnitude
		cv::Mat mag = planes[0];

		//Convert the 2D FFt magnitudes into a 1D frequency spectrum
		cv::Mat spectrum1D = cv::Mat(1, std::max(img.rows, img.cols), CV_32FC1, cv::Scalar(0.0)); //1D spectrum
		cv::Mat num_contrib = cv::Mat(1, std::max(img.rows, img.cols), CV_16UC1, cv::Scalar(0)); //Number of elements contributing to 1D spectrum components

		/* Top left quadrant */
		float longest_diag = std::sqrt((mag.cols / 2 + mag.cols % 2 - 1)*(mag.cols / 2 + mag.cols % 2 - 1) + 
			(mag.rows / 2 + mag.rows % 2 - 1)*(mag.rows / 2 + mag.rows % 2 - 1));
		float *p;
		for (int i = 0; i < mag.rows / 2 + mag.rows % 2; i++)
		{
			p = mag.ptr<float>(i);
			for (int j = 0; j < mag.cols / 2 + mag.cols % 2; j++)
			{
				int bin_num = std::min((int)(std::sqrt(i*i + j*j) * spectrum1D.cols / longest_diag), 
					spectrum1D.cols-1); //Take min in case of rounding errors

				//Add contributions to 1D Fourier spectrum bins
				spectrum1D.at<float>(0, bin_num) += p[j];
				num_contrib.at<uchar>(0, bin_num)++;
			}
		}

		/* Top right quadrant */
		longest_diag = std::sqrt((mag.cols / 2 - mag.cols % 2 - 1)*(mag.cols / 2 - mag.cols % 2 - 1) + 
			(mag.rows / 2 - mag.rows % 2 - 1)*(mag.rows / 2 - mag.rows % 2 - 1));
		for (int i = mag.rows / 2 + mag.rows % 2; i < mag.rows; i++)
		{
			p = mag.ptr<float>(i);
			for (int j = mag.cols / 2 + mag.cols % 2; j < mag.cols; j++)
			{
				int bin_num = std::min((int)(std::sqrt((i - mag.rows - 1)*(i - mag.rows - 1) + (j - mag.cols - 1)*(j - mag.cols - j)) *
					spectrum1D.cols / longest_diag), spectrum1D.cols-1); //Take min in case of rounding errors

				//Add contributions to 1D Fourier spectrum bins
				spectrum1D.at<float>(0, bin_num) += p[j];
				num_contrib.at<uchar>(0, bin_num)++;
			}
		}

		//Calculate the centroid of the 1D spectrum
		p = spectrum1D.ptr<float>(0);
		uchar *q; q = num_contrib.ptr<uchar>(0);
		float inv_feature_size = 0.0f;
		float sum = p[0], sum_err = 1.0f / q[0];
		for (int i = 1; i < spectrum1D.cols; i++)
		{
			sum += p[i];
			sum_err += q[i] ? 1.0f / q[i] : 0;
			inv_feature_size += q[i] ? p[i] * i / q[i] : 0;
		}

		return std::sqrt(m*m + n*n) * inv_feature_size / (sum * sum_err);
	}
}

