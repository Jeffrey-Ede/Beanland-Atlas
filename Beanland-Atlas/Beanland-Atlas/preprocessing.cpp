#include <beanland_atlas.h>

namespace ba
{
	/*Preprocess each of the images by applying a median filter and resizing them
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual images to preprocess
	**med_filt_size: int, Size of median filter
	*/
	void preprocess(std::vector<cv::Mat> &mats, int med_filt_size)
	{
		//ArrayFire can only performs Fourier analysis on arrays that are a power of 2 in size so pad the input arrays to this size
		int cols = ceil_power_2(mats[0].cols);
		int rows = ceil_power_2(mats[0].rows);

		//Preprocess every image in the image stack
        //#pragma omp parallel for
		for (int i = 0; i < mats.size(); i++)
		{
			//Apply median filter
			cv::Mat med_filtrate;
			cv::medianBlur(mats[i], med_filtrate, med_filt_size);

			//Resize the median filtrate
			cv::Mat resized_med;
			cv::resize(med_filtrate, resized_med, cv::Size(cols, rows), 0, 0, cv::INTER_LANCZOS4); //Resize the array so that it is a power of 2 in size

			//Convert the image type to 32 bit floating point
			cv::Mat image32F;
			resized_med.convertTo(image32F, CV_32FC1);

			mats[i] = image32F;
		}
	}
}