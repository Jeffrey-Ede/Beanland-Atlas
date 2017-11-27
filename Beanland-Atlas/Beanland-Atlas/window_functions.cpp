#include <beanland_atlas.h>

/*Calculates values of Hann window function so that they are ready for repeated application
**Inputs:
**mat_rows: int, Number of rows in window
**mat_cols: int, Number of columns in windo
**NUM_THREADS: const int, Number of threads to use for OpenMP CPU acceleration
**Returns:
**cv::Mat, Values of Hann window function at the pixels making up the 
*/
cv::Mat create_hann_window(int mat_rows, int mat_cols, const int NUM_THREADS)
{
	//Create OpenCV matrix to store look up table values
	cv::Mat lut;
	lut.create(mat_rows, mat_cols, CV_32FC1);

	//Apply Hanning window along rows
	float *p;
    #pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < mat_rows; i++) 
	{
		//Apply Hanning window along columns
		p = lut.ptr<float>(i);
		for (int j = 0; j < mat_cols; j++) 
		{
			p[j] = std::sin(j*PI/(mat_cols-1))*std::sin(j*PI/(mat_cols-1)) * 
				std::sin(i*PI/(mat_rows-1)) * std::sin(i*PI/(mat_rows-1));
		}
	}

	return lut;
}

/*Calculates values of Hann window function so that they are ready for repeated application
**Inputs:
**mat: cv::Mat &, Image to apply window to
**win: cv::Mat &, Window to apply
**NUM_THREADS: const int, Number of threads to use for OpenMP CPU acceleration
**Returns:
**cv::Mat, Hanning windowed image
*/
cv::Mat apply_win_func(cv::Mat &mat, cv::Mat &win, const int NUM_THREADS)
{
	//Create OpenCV matrix to store data
	cv::Mat windowed_img;
	windowed_img.create(mat.rows, mat.cols, CV_32FC1);

	//Apply Hanning window along rows
	float *p, *q, *r;
    #pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < mat.rows; i++) 
	{
		//Apply Hanning window along columns
		p = mat.ptr<float>(i);
		q = win.ptr<float>(i);
		r = windowed_img.ptr<float>(i);
		for (int j = 0; j < mat.cols; j++) 
		{
			r[j] = p[j] * q[j];
		}
	}

	return windowed_img;
}