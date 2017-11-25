#include <beanland_atlas.h>

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

		//Print the buildlog if the kernel fails to compile
		char buffer[10240];
		clGetProgramBuildInfo(program, af_device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		fprintf(stderr, "CL Compilation failed:\n%s", buffer);
		abort();
	}

	//Create kernel
	return clCreateKernel(program, kernel_name, &status);
}

/*Calculate weighted 1st order autocorrelation using weighted Pearson normalised product moment correlation coefficient.
**That is, the ratio of the weighted covariance to the product of the weighted variances of the data and the lagged data
**Inputs:
**data: std::vector<float>, One of the datasets to use in the calculation
**Errors: std::vector<float>, Errors in dataset elements used in the calculation
**Return:
**float, Measure of the autocorrelation. 2-2*<return value> approximates the Durbin-Watson statistic for large datasets
*/
float wighted_pearson_autocorr(std::vector<float> data, std::vector<float> err) 
{
	//sums for pearson product moment correlation coefficient
	//x - data; e - error, 1 - lagged data, 2 - forward data

	//Start by calculating means using weighted sums
	float sum_x = 0.0f;
	float sum_x_err = 0.0f;

	int size_minus1 = data.size()-1;
	for (int i = 1; i < size_minus1; i++) {

		sum_x += data[i]/(err[i]*err[i]);
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
	for (i = 1; i < size_minus1; i++) {

		sum_xy += (data[i]-x2_mean)*(data[i-1]-x1_mean) / (std::abs(data[i]-x2_mean)*err[i-1] + std::abs(data[i-1]-x1_mean)*err[i]);
		float weight = 1.0f /(std::abs(data[i]-x2_mean)*err[i-1] + std::abs(data[i-1]-x1_mean)*err[i]);
		sum_xy_err += weight*(1-weight);

		sum_x2 += 1.0f / (err[i]*err[i]);
		sum_x2_err1 += 1.0f / (std::abs(data[i]-x1_mean)*std::abs(data[i]-x1_mean)*err[i]*err[i]);
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

///*Refine position of maxima to sub-index accuracy by finding the indices on either side where the second derivative of the spectrum
//**changes sign and then weighting the average of the indices in this range by their weighted amplitudes
//**Inputs:
//**max_loc: int, Index of maxima to refine
//**data: std::vector<float> &, Data containing a maxima to refine the position of
//**data: std::vector<float> &, Error spectrum of the data
//**Returns:
//**float, Position of maxima refined to sub-index accuracy
//*/
//float refined_max_loc(int max_loc, std::vector<float> &data, std::vector<float> &err, const int NUM_THREADS)
//{
//	//The weighted amplitudes are generated by weighting the height of the spectra by their errors. If the maxima is near an edge of the spectra and the point containing the second derivative is cut off, derivaties of the series and its error spectrum are used to extrapolate the spectrum and its errors to it.
//
//	//Sweep in both directions from the maximum to find the positions where the 2nd derivative changes sign using central finite differences
//    #pragma omp parallel sections num_threads(NUM_THREADS >= 2 ? 2 : 1)
//	{
//		int i = max_loc, j = max_loc;
//
//		//Find location where 2nd derivate changes sign at higher index
//        #pragma omp section
//		{
//			for ( ; i < data.size()-2; i++)
//			{
//				//Forwards finite differences
//				if (data[i+2] / err[i+2] - 2.0f * data[i+1] / err[i+1] + data[i] / err[i] > 0)
//				{
//					break;
//				}
//			}
//		}
//
//		//Find location where 2nd derivate changes sign at lower index
//        #pragma omp section
//		{
//			for ( ; j > 1; j--)
//			{
//				//Backwards finite differences
//				if (data[j-2] / err[j-2] - 2.0f * data[j-1] / err[j-1] + data[j] / err[j] < 0)
//				{
//					break;
//				}
//			}
//		}
//
//		//Both derivates in spectrum
//		if(j > 1 && i ){
//
//			//Find weighted centroid
//			float sum = j / err[j], sum_err = 1.0f / err[j];
//            #pragma omp parallel for num_threads(NUM_THREADS)
//			for (int k = j+1; k <= i; k++)
//			{
//				sum += k / err[k];
//				sum_err += 1.0f / err[k];
//			}
//
//			return sum / sum_err;
//		}
//		else
//		{
//			//Only low index 2nd derivative change is not in spectrum
//			if (j <= 1 && i <= data.size() - 2) 
//			{
//				//Take centroid of data within range, cutting off the high index data once the 2nd derivative reaches minus
//				//the same value
//				float limit = data[0] / err[0] - 2.0f * data[1] / err[1] + data[2] / err[2];
//
//				//Decrement i until the limit is passed
//				for (i--; data[i + 2] / err[i + 2] - 2.0f * data[i + 1] / err[i + 1] + data[i] / err[i] <= limit; i--){}
//
//				//0th index special case
//				float sum = 0.0f, sum_err = 1.0f / err[0];
//
//				//Find weighted centroid
//                #pragma omp parallel for num_threads(NUM_THREADS)
//				for (int k = 1; k <= i; k++)
//				{
//					sum += k / err[k];
//					sum_err += 1.0f / err[k];
//				}
//
//				return sum / sum_err;
//			}
//			else 
//			{
//				//Only high index 2nd derivative change is not in spectrum
//				if (i > data.size()-2  && j > 1) 
//				{
//					//Take centroid of data within range, cutting off the high index data once the 2nd derivative reaches minus
//					//the same value
//					float limit = data[data.size()-3] / err[data.size()-3] - 2.0f * data[data.size()-2] / err[data.size()-2] + 
//						data[data.size()-1] / err[data.size()-1];
//
//					//Increment j until the limit is passed
//					for (j++; data[j-2] / err[j-2] - 2.0f * data[j-1] / err[j-1] + data[j] / err[j] <= limit; i++){}
//
//					//Find weighted centroid
//					float sum = j / err[j], sum_err = 1.0f / err[j];
//                    #pragma omp parallel for num_threads(NUM_THREADS)
//					for (int k = j+1; k <= i; k++)
//					{
//						sum += k / err[k];
//						sum_err += 1.0f / err[k];
//					}
//
//					return sum / sum_err;
//				}
//				//Both low and high index 2nd derivative change are not in spectrum
//				else
//				{
//					//Find the limit with the smallest rate of change in gradiation
//					float llimit = data[0] / err[0] - 2.0f * data[1] / err[1] + data[2] / err[2];
//					float ulimit = data[data.size()-3] / err[data.size()-3] - 2.0f * data[data.size()-2] / err[data.size()-2] + 
//						data[data.size()-1] / err[data.size()-1];
//
//					//Rate of change in gradiation higher at high indices
//					if (std::abs(ulimit) > std::abs(llimit))
//					{
//						//Decrement i until the limit is passed
//						for (i--; data[i + 2] / err[i + 2] - 2.0f * data[i + 1] / err[i + 1] + data[i] / err[i] <= llimit; i--){}
//
//						//0th index special case
//						float sum = 0.0f, sum_err = 1.0f / err[0];
//
//						//Find weighted centroid
//                        #pragma omp parallel for num_threads(NUM_THREADS)
//						for (int k = 1; k <= i; k++)
//						{
//							sum += k / err[k];
//							sum_err += 1.0f / err[k];
//						}
//
//						return sum / sum_err;
//					}
//					else
//					{
//						//Rate of change in gradiation is the same at both ends of the spectrum
//						if (std::abs(ulimit) == std::abs(llimit))
//						{
//							//Find weighted centroid
//							float sum = 0.0f, sum_err = 1.0f / err[0];
//                            #pragma omp parallel for num_threads(NUM_THREADS)
//							for (int k = 1; k < data.size(); k++)
//							{
//								sum += k / err[k];
//								sum_err += 1.0f / err[k];
//							}
//
//							return sum / sum_err;
//						}
//						//Rate of change in gradiation higher at low indices
//						else
//						{
//							//Increment j until the limit is passed
//							for (j++; data[j-2] / err[j-2] - 2.0f * data[j-1] / err[j-1] + data[j] / err[j] <= ulimit; i++){}
//
//							//Find weighted centroid
//							float sum = j / err[j], sum_err = 1.0f / err[j];
//                            #pragma omp parallel for num_threads(NUM_THREADS)
//							for (int k = j+1; k <= i; k++)
//							{
//								sum += k / err[k];
//								sum_err += 1.0f / err[k];
//							}
//
//							return sum / sum_err;
//						}
//					}
//				}
//			}
//		}
//	}
//}