#include <beanland_atlas.h>

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
}