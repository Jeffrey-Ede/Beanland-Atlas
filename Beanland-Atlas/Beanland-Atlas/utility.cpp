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

/*Calculate weighted 1st order autocorrelation using weighted Pearson normalised product moment correlation coefficient
**Inputs:
**data: std::vector<float>, One of the datasets to use in the calculation
**Errors: std::vector<float>, Errors in dataset elements used in the calculation
**Return:
**float, Measure of the autocorrelation. 2-2*<return value> approximates the Durbin-Watson statistic
*/
float wighted_pearson_autocorr(std::vector<float> data, std::vector<float> err) {

	//Sums for Pearson product moment correlation coefficient
	//x - data; e - error
	float sum_1_over_e = 0.0f;
	float sum_1_over_e2 = 0.0f;
	float sum_1_over_xe = 0.0f;
	float sum_x_over_e = 0.0f;
	float sum_x_over_e2 = 0.0f;

	float sum_xy = 0.0f;
	float sum_xy_err = 0.0f;

	//Previous values needed for summation
	float x_prev, xe_prev, x2e2_prev, x_over_e_prev, x_over_e2_prev;
	x_prev = data[0];
	x2e2_prev = xe_prev*xe_prev;

	//Reflect points across mirror line and compare them to the nearest pixel
	int size_minus1 = data.size()-1;
	for (int i = 1; i < size_minus1; i++) {

		float one_over_e = 1.0f/err[i];
		float one_over_e2 = one_over_e*one_over_e;

		float xe = data[i]*err[i];
		float x2e2 = xe*xe;

		sum_1_over_e += one_over_e;
		sum_1_over_e2 += one_over_e2;
		sum_1_over_xe += 1.0f/xe;
		sum_x_over_e += data[i]*one_over_e;
		sum_x_over_e2 +=  data[i]*one_over_e2;

		float inv_sum = 1.0f/(x2e2+x2e2_prev);
		sum_xy += data[i]*x_prev*inv_sum;
		sum_xy_err += inv_sum;

		x_prev = data[i];
		x2e2_prev = x2e2;
	}

	//First element
	float one_over_e_first = 1.0f/err[0];
	float one_over_e2_first = one_over_e_first*one_over_e_first;
	float xe_first = data[0]*err[0];
	float x2e2_first = xe_first*xe_first;

	float sum_1_over_e_first = sum_1_over_e + one_over_e_first;
	float sum_1_over_e2_first = sum_1_over_e2 + one_over_e2_first;
	float sum_1_over_xe_first = sum_1_over_xe + 1.0f/xe_first;
	float sum_x_over_e_first = sum_x_over_e + data[0]*one_over_e_first;
	float sum_x_over_e2_first = sum_x_over_e2 + data[0]*one_over_e2_first;

	//Last element
	float one_over_e_last = 1.0f/err[size_minus1];
	float one_over_e2_last = one_over_e_last*one_over_e_last;
	float xe_last = data[size_minus1]*err[size_minus1];
	float x2e2_last = xe_last*xe_last;

	float sum_1_over_e_last = sum_1_over_e + one_over_e_last;
	float sum_1_over_e2_last = sum_1_over_e2 + one_over_e2_last;
	float sum_1_over_xe_last = sum_1_over_xe + 1.0f/xe_last;
	float sum_x_over_e_last = sum_x_over_e + data[size_minus1]*one_over_e_last;
	float sum_x_over_e2_last = sum_x_over_e2 + data[size_minus1]*one_over_e2_last;

	//Add last element to sum
	float inv_sum = 1.0f/(x2e2_last+x2e2_prev);
	sum_xy += data[size_minus1]*x_prev*inv_sum;
	sum_xy_err += inv_sum;

	//Break calculation of Pearson normalised product moment correlation coefficient into parts for readability
	float p1, p2, p3, p4, p5, p6;
	p1 = size_minus1 * sum_xy / sum_xy_err;
	p2 = size_minus1 * sum_x_over_e2_first * sum_x_over_e2_last / ( sum_1_over_e2_first * sum_1_over_e2_last );
	p3 = size_minus1 * sum_x_over_e_first / sum_1_over_xe_first;
	p4 = sum_x_over_e2_first / sum_1_over_e2_first;
	p5 = size_minus1 * sum_x_over_e_last / sum_1_over_xe_last;
	p6 = sum_x_over_e2_last / sum_1_over_e2_last;

	return (p1-p2)/std::sqrt((p3-p4*p4)*(p5-p6*p6));
}