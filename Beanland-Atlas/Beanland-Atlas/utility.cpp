#include <beanland_atlas.h>

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
	float wighted_pearson_autocorr(std::vector<float> data, std::vector<float> err, const int NUM_THREADS) 
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
	**Inputs:
	**img1: cv::Mat &, One of the mats
	**img1: cv::Mat &, One of the mats
	**col_offset: const int, Offset of the second mat's columnss from the first's
	**row_offset: const int, Offset of the second mat's rows from the first's
	**Return:
	**float, Pearson normalised product moment correlation coefficient between the 2 mats
	*/
	float pearson_corr(cv::Mat img1, cv::Mat img2, const int row_offset, const int col_offset) 
	{
		//Sums for Pearson product moment correlation coefficient
		float sum_xy = 0.0f;
		float sum_x = 0.0f;
		float sum_y = 0.0f;
		float sum_x2 = 0.0f;
		float sum_y2 = 0.0f;

		//Special case the 4 different combinations of positive and negative offsets
		if (row_offset >= 0 && col_offset >= 0)
		{
			//Iterate across mat rows...
			float *r;
			float *s;
            #pragma omp parallel reduction(sum:sum_xy), reduction(sum:sum_x), reduction(sum:sum_y), reduction(sum:sum_x2), reduction(sum:sum_y2)
			for (int m = row_offset; m < img1.rows; m++) 
			{
				//...and iterate across mat columns
				r = img1.ptr<float>(m);
				s = img2.ptr<float>(m-row_offset);
				for (int n = col_offset; n < img1.cols; n++) 
				{
					//If the pixels are both not black - safegaurd against rows or columns with some black in the middle
					if (r[n] && s[n-col_offset])
					{
						//Contribute to Pearson correlation coefficient
						sum_xy += r[n]*s[n-col_offset];
						sum_x += r[n];
						sum_y += s[n-col_offset];
						sum_x2 += r[n]*r[n];
						sum_y2 += s[n-col_offset]*s[n-col_offset];
					}
				}
			}
		}
		else 
		{
			if (row_offset < 0 && col_offset < 0)
			{
				//Iterate across mat rows...
				float *r;
				float *s;
                #pragma omp parallel reduction(sum:sum_xy), reduction(sum:sum_x), reduction(sum:sum_y), reduction(sum:sum_x2), reduction(sum:sum_y2)
				for (int m = -row_offset; m < img1.rows; m++) 
				{
					//...and iterate across mat columns
					r = img1.ptr<float>(m+row_offset);
					s = img2.ptr<float>(m);
					for (int n = -col_offset; n < img1.cols; n++) 
					{
						//If the pixels are both not black - safegaurd against rows or columns with some black in the middle
						if (r[n+col_offset] && s[n])
						{
							//Contribute to Pearson correlation coefficient
							sum_xy += r[n+col_offset]*s[n];
							sum_x += r[n+col_offset];
							sum_y += s[n];
							sum_x2 += r[n+col_offset]*r[n+col_offset];
							sum_y2 += s[n]*s[n];
						}
					}
				}
			}
			else
			{
				if (row_offset >= 0 && col_offset < 0)
				{
					//Iterate across mat rows...
					float *r;
					float *s;
                    #pragma omp parallel reduction(sum:sum_xy), reduction(sum:sum_x), reduction(sum:sum_y), reduction(sum:sum_x2), reduction(sum:sum_y2)
					for (int m = row_offset; m < img1.rows; m++) 
					{
						//...and iterate across mat columns
						r = img1.ptr<float>(m);
						s = img2.ptr<float>(m-row_offset);
						for (int n = -col_offset; n < img1.cols; n++) 
						{
							//If the pixels are both not black - safegaurd against rows or columns with some black in the middle
							if (r[n+col_offset] && s[n])
							{
								//Contribute to Pearson correlation coefficient
								sum_xy += r[n+col_offset]*s[n];
								sum_x += r[n+col_offset];
								sum_y += s[n];
								sum_x2 += r[n+col_offset]*r[n+col_offset];
								sum_y2 += s[n]*s[n];
							}
						}
					}
				}
				//The remaining case is row_offset < 0 && col_offset >= 0
				else
				{
					//Iterate across mat rows...
					float *r;
					float *s;
                    #pragma omp parallel reduction(sum:sum_xy), reduction(sum:sum_x), reduction(sum:sum_y), reduction(sum:sum_x2), reduction(sum:sum_y2)
					for (int m = -row_offset; m < img1.rows; m++) 
					{
						//...and iterate across mat columns
						r = img1.ptr<float>(m+row_offset);
						s = img2.ptr<float>(m);
						for (int n = col_offset; n < img1.cols; n++) 
						{
							//If the pixels are both not black - safegaurd against rows or columns with some black in the middle
							if (r[n] && s[n-col_offset])
							{
								//Contribute to Pearson correlation coefficient
								sum_xy += r[n]*s[n-col_offset];
								sum_x += r[n];
								sum_y += s[n-col_offset];
								sum_x2 += r[n]*r[n];
								sum_y2 += s[n-col_offset]*s[n-col_offset];
							}
						}
					}
				}
			}
		}

		return (img1.rows*img1.cols*sum_xy - sum_x*sum_y) / (std::sqrt(img1.rows*img1.cols*sum_x2 - sum_x*sum_x) * 
			std::sqrt(img1.rows*img1.cols*sum_y2 - sum_y*sum_y));
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
		int k_size = (int)(blur_frac*std::min(img.rows, img.cols)) < 1 ? 1 : (int)(blur_frac*std::min(img.rows, img.cols));

		cv::filter2D(img, blurred, img.depth(), cv::getGaussianKernel(k_size, -1, CV_32F));

		return blurred;
	}

	/*Create phase correlation specturm
	**src1: cv::Mat &, One of the images
	**src2: cv::Mat &, The second image
	**Returns,
	**cv::Mat, phase correlation spectrum
	*/
	cv::Mat phase_corr_spectrum(cv::Mat &src1, cv::Mat &src2)
	{
		cv::Mat window;
		CV_Assert( src1.type() == src2.type());
		CV_Assert( src1.type() == CV_32FC1 || src1.type() == CV_64FC1 );
		CV_Assert( src1.size == src2.size);

		if(!window.empty())
		{
			CV_Assert( src1.type() == window.type());
			CV_Assert( src1.size == window.size);
		}

		int M = cv::getOptimalDFTSize(src1.rows);
		int N = cv::getOptimalDFTSize(src1.cols);

		cv::Mat padded1, padded2, paddedWin;

		if(M != src1.rows || N != src1.cols)
		{
			cv::copyMakeBorder(src1, padded1, 0, M - src1.rows, 0, N - src1.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
			cv::copyMakeBorder(src2, padded2, 0, M - src2.rows, 0, N - src2.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

			if(!window.empty())
			{
				cv::copyMakeBorder(window, paddedWin, 0, M - window.rows, 0, N - window.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
			}
		}
		else
		{
			padded1 = src1;
			padded2 = src2;
			paddedWin = window;
		}

		cv::Mat FFT1, FFT2, P, Pm, C;

		// perform window multiplication if available
		if(!paddedWin.empty())
		{
			// apply window to both images before proceeding...
			cv::multiply(paddedWin, padded1, padded1);
			cv::multiply(paddedWin, padded2, padded2);
		}

		//Execute phase correlation equation
		cv::dft(padded1, FFT1, cv::DFT_REAL_OUTPUT);
		cv::dft(padded2, FFT2, cv::DFT_REAL_OUTPUT);

		//Compute FF* / |FF*|
		cv::mulSpectrums(FFT1, FFT2, P, 0, true);
		cv::mulSpectrums(P, 1 / (cv::abs(P)+1) , C, 0, false);

		//cv::divide(P, cv::abs(P), C, 0, false); 

		cv::Mat D;
		C.convertTo(D, CV_32FC1);

		cv::idft(D, D); // gives us the nice peak shift location...

		return D;
	}

	/*Sum of squared differences between 2 images. The second images is correlated against the first in the Fourier domain
	**src1: cv::Mat &, One of the images
	**src2: cv::Mat &, The second image
	**frac: float, Proportion of the first image's dimensions to pad it by
	**Returns,
	**cv::Mat, Sum of the squared differences
	*/
	cv::Mat ssd(cv::Mat &src1, cv::Mat &src2, float frac)
	{
		//Pad the first matrix to prepare it for the second matrix being correlated accross it
		cv::Mat pad1;
		int pad_rows = (int)(frac*src1.rows);
		int pad_cols = (int)(frac*src1.cols);
		cv::copyMakeBorder(src1, pad1, pad_rows, pad_rows, pad_cols, pad_cols, cv::BORDER_CONSTANT, 0.0);

		//Calculate the sum of squared differences in the Fourier domain
		cv::Mat ssd;
		cv::matchTemplate(pad1, src2, ssd, cv::TM_SQDIFF);

		//Calculate the cumulative rowwise and columnwise sums of squares. These can be used to calculate additive corrections
		//to the calculate sum of squared differences where the templates did not overlap
		float *r, *s;

		/* Left */
		//Iterate across mat rows...
		cv::Mat left = cv::Mat(src2.rows, pad_cols, CV_32FC1);
		for (int i = 0; i < left.rows; i++)
		{
			//Initialise pointers
			r = left.ptr<float>(i);
			s = src2.ptr<float>(i);

			//Calculate the square of the first column element
			r[left.cols-1] = s[0]*s[0];

			//Iterate across the remaining mat columns, accumulating the sum of squares
			for (int j = left.cols-2, k = 1; j >= 0; j--, k++)
			{
				r[j] = s[k]*s[k] + r[j+1];
			}
		}

		/* Right */
		//Iterate across mat rows...
		cv::Mat right = cv::Mat(src2.rows, pad_cols, CV_32FC1);
		for (int i = 0; i < right.rows; i++)
		{
			//Initialise pointers
			r = right.ptr<float>(i);
			s = src2.ptr<float>(i);

			//Calculate the square of the first column element
			r[0] = s[src2.cols-1]*s[src2.cols-1];

			//Iterate across the remaining mat columns, accumulating the sum of squares
			for (int j = 1, k = src2.cols-2; j < right.cols; j++, k--)
			{
				r[j] = s[k]*s[k] + r[j-1];
			}
		}

		/* Top */
		//Iterate across mat cols
		cv::Mat top = cv::Mat(pad_rows, src2.cols, CV_32FC1);
		for (int i = 0; i < top.cols; i++)
		{

			//Calculate the square of the first row element
			top.at<float>(top.rows-1, i) = src2.at<float>(0, i)*src2.at<float>(0, i);

			//Iterate across the remaining mat rows, accumulating the sum of squares
			for (int j = top.rows-2, k = 1; j >= 0; j--, k++)
			{
				top.at<float>(j, i) = src2.at<float>(k, i)*src2.at<float>(k, i) + top.at<float>(j+1, i);
			}
		}

		/* Bottom */
		//Iterate across mat cols...
		cv::Mat bot = cv::Mat(pad_rows, src2.cols, CV_32FC1);
		for (int i = 0; i < bot.cols; i++)
		{
			//Calculate the square of the first row element
			bot.at<float>(0, i) = src2.at<float>(src2.rows-1, i)*src2.at<float>(src2.rows-1, i);

			//Iterate across the remaining mat rows, accumulating the sum of squares
			for (int j = 1, k = src2.rows-2; j < bot.rows; j++, k--)
			{
				bot.at<float>(j, i) = src2.at<float>(k, i)*src2.at<float>(k, i) + bot.at<float>(j-1, i);
			}
		}


		//Calulate the linear map needed to map the ssd values from the Fourier analysis to the correct values
		cv::Mat offset = cv::Mat(ssd.rows, ssd.cols, CV_32FC1, cv::Scalar(0.0));
		cv::Mat scale = cv::Mat(ssd.rows, ssd.cols, CV_32FC1, cv::Scalar(0.0));

		//Accumulate the top and bot accumulator rows
		std::vector<float> vert_acc(2*pad_rows+1, 0);
		for (int i = 0; i < top.rows; i++)
		{
			//Iterate across the accumulator columns
			r = top.ptr<float>(i);
			for (int j = 0; j < top.cols; j++)
			{
				vert_acc[i] += r[j];
			}
		}
		for (int i = 0, k = top.rows+1; i < bot.rows; i++, k++)
		{
			//Iterate across the accumulator columns
			r = bot.ptr<float>(i);
			for (int j = 0; j < bot.cols; j++)
			{
				vert_acc[k] += r[j];
			}
		}

		//Accumulate the left accumulator columns
		for (int i = 0; i < left.cols; i++)
		{
			float acc = 0.0f;
			for (int j = 0; j < top.rows; j++)
			{
				offset.at<float>(j, i) += acc + vert_acc[j];
				acc += left.at<float>(j, i);
			}

			//Continue across the left rows
			for (int j = 0, k = bot.rows; j < bot.rows; j++, k++)
			{
				offset.at<float>(k, i) += acc + vert_acc[k];
				acc -= left.at<float>(j, i);
			}
		}

		//Middle columns
		for (int i = 0; i < top.rows + bot.rows + 1; i++)
		{
			offset.at<float>(i, left.cols) = vert_acc[i];
		}

		//Accumulate the right accumulator columns
		for (int i = 0, l = right.cols+1; i < right.cols; i++, l++)
		{
			float acc = 0.0f;
			for (int j = 0; j < top.rows; j++)
			{
				offset.at<float>(j, l) += acc + vert_acc[j];
				acc += right.at<float>(j, i);
			}

			//Continue across the left rows
			for (int j = 0, k = bot.rows; j < bot.rows; j++, k++)
			{
				offset.at<float>(k, l) += acc + vert_acc[k];
				acc -= right.at<float>(j, i);
			}
		}

		//Iterate across mat rows...
		for (int m = 0; m < scale.rows; m++) 
		{
			//...and iterate across mat columns
			s = scale.ptr<float>(m);
			for (int n = 0; n < scale.cols; n++) 
			{
				s[n] = (src2.rows-std::abs(m-top.rows))*(src2.cols-std::abs(n-left.cols));
			}
		}

		//Linearly map to the correct results
		return (ssd - offset) / scale;
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
}

