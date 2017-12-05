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
		#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 0; i < vect1.size(); i++) {

			//Contribute to Pearson correlation coefficient
			#pragma omp atomic
			sum_xy += vect1[i]*vect2[i];
			#pragma omp atomic
			sum_x += vect1[i];
			#pragma omp atomic
			sum_y += vect2[i];
			#pragma omp atomic
			sum_x2 += vect1[i]*vect1[i];
			#pragma omp atomic
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

		//Iterate across mat rows...
		float *r;
		float *s;
        #pragma omp parallel reduction(sum:sum_xy), reduction(sum:sum_x), reduction(sum:sum_y), reduction(sum:sum_x2), reduction(sum:sum_y2)
		for (int m = 0; m < img1.rows-row_offset; m++) 
		{
			//...and iterate across each mat columns
			r = img1.ptr<float>(m);
			s = img2.ptr<float>(m+row_offset);
			for (int n = 0; n < img1.cols-col_offset; n++) 
			{
				//If the pixels are both not black
				if (r[n] && s[n+col_offset])
				{
					//Contribute to Pearson correlation coefficient
					sum_xy += r[n]*s[n+col_offset];
					sum_x += r[n];
					sum_y += s[n+col_offset];
					sum_x2 += r[n]*r[n];
					sum_y2 += s[n+col_offset]*s[n+col_offset];
				}
			}
		}

		return (img1.rows*img1.cols*sum_xy - sum_x*sum_y) / (std::sqrt(img1.rows*img1.cols*sum_x2 - sum_x*sum_x) * 
			std::sqrt(img1.rows*img1.cols*sum_y2 - sum_y*sum_y));
	}
}