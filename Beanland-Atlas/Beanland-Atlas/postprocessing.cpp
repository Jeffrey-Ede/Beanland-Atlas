#include <postprocessing.h>

namespace ba
{
	/*Mask noting the positions of the black pixels that are not padding an image
	**img: cv::Mat &, Floating point OpenCV mat to find the black non-bounding pixels of
	**Returns: 
	**cv::Mat, Mask indicating the positions of non-padding black pixels so that they can be infilled
	*/
	cv::Mat infilling_mask(cv::Mat &img)
	{
		//Mark the values of the padding pixels and the outermost non-zero image pixels with with zeros
		cv::Mat not_padding = cv::Mat::ones(img.size(), CV_8UC1);

		//Work inwards from the top and bottom of every column to find the length of the non-black region
		for (int i = 0; i < img.cols; i++)
		{
			//From the top of the image
			for (int j_top = 0; j_top < img.rows; j_top++)
			{
				//Set the padding and outermost non-zero pixel mask values to 0
				not_padding.at<float>(j_top, i) = 0;

				//Check if this is the first non-zero pixel from the image edge
				if (img.at<float>(j_top, i))
				{
					break;
				}
			}

			//From the bottom of the image
			for (int j_bot = img.rows-1; j_bot >= 0; j_bot--)
			{
				//Set the padding and outermost non-zero pixel mask values to 0
				not_padding.at<float>(j_bot, i) = 0;

				//Check if this is the first non-zero pixel from the image edge
				if (img.at<float>(j_bot, i))
				{
					break;
				}
			}
		}

		//Work inwards from the top and bottom of every row to find the length of the non-black region
		for (int i = 0; i < img.rows; i++)
		{
			//From the top of the image
			for (int j_top = 0; j_top < img.cols; j_top++)
			{
				//Set the padding and outermost non-zero pixel mask values to 0
				not_padding.at<float>(i, j_top) = 0;

				//Check if this is the first non-zero pixel from the image edge
				if (img.at<float>(i, j_top))
				{
					break;
				}
			}

			//From the bottom of the image
			for (int j_bot = img.cols-1; j_bot >= 0; j_bot--)
			{
				//Set the padding and outermost non-zero pixel mask values to 0
				not_padding.at<float>(i, j_bot) = 0;

				//Check if this is the first non-zero pixel from the image edge
				if (img.at<float>(i, j_bot))
				{
					break;
				}
			}
		}

		//Return a mask where the non-padding zero values are marked with 1s
		return not_padding & (img == 0);
	}

	/*Navier-Stokes interpolate the value of each pixel in an image from the surrounding pixels using the spatial scales
	**on the image
	**img: cv::Mat &, Image to find the Navier-stokes inflows to the pixels of
	**dst: cv::Mat &, Output Navier-Stokes reconstruction
	*/
	void navier_stokes_reconstruction(cv::Mat &img, cv::Mat &dst)
	{
		dst = cv::Mat(img.size(), CV_32FC1);

		//Get the average feature size in the image so that this can be used as a scale to apply the Navier-Stokes
		//equations over
		int half = (int)get_avg_feature_size(img) / 2;

		//Apply the reconstruction to each pixel using its neighbours
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				//Extent of rectangle to infill from about the point
				int left = j > half ? half : j;
				int right = j > img.cols-1 - half ? img.cols-j-1 : half;
				int top = i > half ? half : i;
				int bot = i > img.rows-1 - half ? img.rows-i-1 : half;
				
				//Mark the point to be infilled
				cv::Mat infill_mask = cv::Mat(top+bot+1, left+right+1, CV_8UC1, cv::Scalar(0));
				infill_mask.at<byte>(top, left) = 1;

				cv::Rect rect = cv::Rect(cv::Point(i-top, j-left), cv::Point(i+bot, j+right));
				cv::Mat crop;
				img(rect).copyTo(crop);

				//Navier-Stokes inpaint the pixel
				cv::inpaint(crop, infill_mask, crop, half, cv::INPAINT_NS);

				//Set the value of the pixel in the output image equal to the inpainted value
				dst.at<float>(i, j) = crop.at<float>(top, left);
			}
		}
	}
}