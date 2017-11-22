#include <beanland_atlas.h>

/*Calculate Pearson normalised product moment correlation coefficient between 2 vectors of floats
**Inputs:
**vect1: std::vector<float>, One of the datasets to use in the calculation
**vect2: std::vector<float>, The other dataset to use in the calculation
**Return:
**float, Pearson normalised product moment correlation coefficient between the 2 datasets
*/
float pearson_corr(std::vector<float> vect1, std::vector<float> vect2) {

	//Sums for Pearson product moment correlation coefficient
	float sum_xy = 0.0f;
	float sum_x = 0.0f;
	float sum_y = 0.0f;
	float sum_x2 = 0.0f;
	float sum_y2 = 0.0f;

	//Reflect points across mirror line and compare them to the nearest pixel
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

	std::cout << x1_mean << ", " << x2_mean << std::endl;

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