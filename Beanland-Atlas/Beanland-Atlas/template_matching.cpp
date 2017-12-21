#include <beanland_atlas.h>

namespace ba
{
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

		//Get the sum of the squares of the elements in the second image
		float sum_sqr_2 = 0.0f;
		//Sum across the first source's rows...
		for (int i = 0; i < src2.rows; i++)
		{
			//...and the first source's columns
			r = src2.ptr<float>(i);
			for (int j = 0; j < src2.cols; j++)
			{
				sum_sqr_2 += r[j]*r[j];
			}
		}

		cv::Mat x = (ssd - offset) / scale;

		cv::Point y;
		double min, max;
		cv::minMaxLoc(x, &min, &max, &y, NULL);
		std::cout << min << ", " << max << ", " << y << std::endl;
		std::cout << offset.rows << ", " << offset.cols << std::endl;
		std::cout << x.at<float>(y.y, y.x) << ", " << offset.at<float>(y.y, y.x) << ", " << offset.at<float>(y.y, y.x) << std::endl;
		std::cout << sum_sqr_2 << std::endl;
		std::getchar();

		display_CV(offset);
		display_CV(scale);
		cv::Mat x = (ssd - offset) / scale;
		display_CV(x);

		//Linearly map to the correct results
		return (ssd - offset) / scale;
	}

	/*Pearson normalised product moment correlation coefficient between 2 images. The second images is correlated against the 
	**first in the Fourier domain
	**src1: cv::Mat &, One of the images
	**src2: cv::Mat &, The second image
	**frac: float, Proportion of the first image's dimensions to pad it by
	**Returns,
	**cv::Mat, Pearson normalised product moment correlation coefficients
	*/
	cv::Mat fourier_pearson_corr(cv::Mat &src1, cv::Mat &src2, float frac)
	{
		//Pad the first matrix to prepare it for the second matrix being correlated accross it
		cv::Mat pad1;
		int pad_rows = (int)(frac*src1.rows);
		int pad_cols = (int)(frac*src1.cols);
		cv::copyMakeBorder(src1, pad1, pad_rows, pad_rows, pad_cols, pad_cols, cv::BORDER_CONSTANT, 0.0);

		//Calculate the Pearson product moment correlation coefficients in the Fourier domain
		cv::Mat pear;
		cv::matchTemplate(pad1, src2, pear, CV_TM_CCOEFF);

		//Calculate the cumulative rowwise and columnwise sums of squares. These can be used to calculate additive corrections
		//to the calculate sum of squared differences where the templates did not overlap
		float *r, *s;

		/* Left of 1st image */
		//Iterate across mat rows...
		cv::Mat left = cv::Mat(src1.rows, pad_cols, CV_32FC1);
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

		/* Right of 1st image */
		//Iterate across mat rows...
		cv::Mat right = cv::Mat(src1.rows, pad_cols, CV_32FC1);
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

		/* Top of 1st image */
		//Iterate across mat cols
		cv::Mat top = cv::Mat(pad_rows, src1.cols, CV_32FC1);
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

		/* Bottom of 1st image */
		//Iterate across mat cols...
		cv::Mat bot = cv::Mat(pad_rows, src1.cols, CV_32FC1);
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

		/* Left of 2nd image */
		//Iterate across mat rows...
		cv::Mat left2 = cv::Mat(src2.rows, pad_cols, CV_32FC1);
		for (int i = 0; i < left.rows; i++)
		{
			//Initialise pointers
			r = left2.ptr<float>(i);
			s = src2.ptr<float>(i);

			//Calculate the square of the first column element
			r[left.cols-1] = s[0]*s[0];

			//Iterate across the remaining mat columns, accumulating the sum of squares
			for (int j = left.cols-2, k = 1; j >= 0; j--, k++)
			{
				r[j] = s[k]*s[k] + r[j+1];
			}
		}

		/* Right of 2nd image */
		//Iterate across mat rows...
		cv::Mat right2 = cv::Mat(src2.rows, pad_cols, CV_32FC1);
		for (int i = 0; i < right.rows; i++)
		{
			//Initialise pointers
			r = right2.ptr<float>(i);
			s = src2.ptr<float>(i);

			//Calculate the square of the first column element
			r[0] = s[src2.cols-1]*s[src2.cols-1];

			//Iterate across the remaining mat columns, accumulating the sum of squares
			for (int j = 1, k = src2.cols-2; j < right.cols; j++, k--)
			{
				r[j] = s[k]*s[k] + r[j-1];
			}
		}

		/* Top of 2nd image */
		//Iterate across mat cols
		cv::Mat top2 = cv::Mat(pad_rows, src2.cols, CV_32FC1);
		for (int i = 0; i < top.cols; i++)
		{

			//Calculate the square of the first row element
			top2.at<float>(top.rows-1, i) = src2.at<float>(0, i)*src2.at<float>(0, i);

			//Iterate across the remaining mat rows, accumulating the sum of squares
			for (int j = top.rows-2, k = 1; j >= 0; j--, k++)
			{
				top2.at<float>(j, i) = src2.at<float>(k, i)*src2.at<float>(k, i) + top2.at<float>(j+1, i);
			}
		}

		/* Bottom of 2nd image */
		//Iterate across mat cols...
		cv::Mat bot2 = cv::Mat(pad_rows, src2.cols, CV_32FC1);
		for (int i = 0; i < bot.cols; i++)
		{
			//Calculate the square of the first row element
			bot2.at<float>(0, i) = src2.at<float>(src2.rows-1, i)*src2.at<float>(src2.rows-1, i);

			//Iterate across the remaining mat rows, accumulating the sum of squares
			for (int j = 1, k = src2.rows-2; j < bot.rows; j++, k--)
			{
				bot2.at<float>(j, i) = src2.at<float>(k, i)*src2.at<float>(k, i) + bot2.at<float>(j-1, i);
			}
		}

		//Calulate the linear map needed to map the correlation coefficients to their normalised values
		cv::Mat scale = cv::Mat(pear.rows, pear.cols, CV_32FC1, cv::Scalar(0.0));
		cv::Mat scale2 = cv::Mat(pear.rows, pear.cols, CV_32FC1, cv::Scalar(0.0));

		//Accumulate the top and bot accumulator rows of the 1st source
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

		//Accumulate the top and bot accumulator rows of the second source
		std::vector<float> vert_acc_2(2*pad_rows+1, 0);
		for (int i = 0; i < top2.rows; i++)
		{
			//Iterate across the accumulator columns
			r = top2.ptr<float>(i);
			for (int j = 0; j < top2.cols; j++)
			{
				vert_acc_2[i] += r[j];
			}
		}
		for (int i = 0, k = top2.rows+1; i < bot2.rows; i++, k++)
		{
			//Iterate across the accumulator columns
			r = bot2.ptr<float>(i);
			for (int j = 0; j < bot2.cols; j++)
			{
				vert_acc_2[k] += r[j];
			}
		}

		//Accumulate the right accumulator columns of the first source
		for (int i = 1; i < right.cols+1; i++)
		{
			float acc = 0.0f;
			for (int j = 0; j < top.rows; j++)
			{
				scale.at<float>(j, i-1) += acc + vert_acc[j];
				acc += right.at<float>(j, i-1);
			}

			//Continue across the left rows
			for (int j = 0, k = bot.rows; j < bot.rows; j++, k++)
			{
				scale.at<float>(k, i-1) += acc + vert_acc[k];
				acc -= right.at<float>(j, i-1);
			}
		}

		//Accumulate the left accumulator columns of the first source
		for (int i = 1, l = left.cols+1; i < left.cols+1; i++, l++)
		{
			float acc = 0.0f;
			for (int j = 0; j < top.rows; j++)
			{
				scale.at<float>(j, l-1) += acc + vert_acc[j];
				acc += left.at<float>(j, i-1);
			}

			//Continue across the left rows
			for (int j = 0, k = bot.rows; j < bot.rows; j++, k++)
			{
				scale.at<float>(k, l-1) += acc + vert_acc[k];
				acc -= left.at<float>(j, i-1);
			}
		}

		//Accumulate the left accumulator columns of the second source
		for (int i = 0; i < left2.cols; i++)
		{
			float acc2 = 0.0f;
			for (int j = 0; j < top2.rows; j++)
			{
				scale2.at<float>(j, i) += acc2 + vert_acc_2[j];
				acc2 += left2.at<float>(j, i);
			}

			//Continue across the left rows
			for (int j = 0, k = bot2.rows; j < bot2.rows; j++, k++)
			{
				scale2.at<float>(k, i) += acc2 + vert_acc_2[k];
				acc2 -= left2.at<float>(j, i);
			}
		}

		//Middle column of the second source
		for (int i = 0; i < top2.rows + bot2.rows + 1; i++)
		{
			scale2.at<float>(i, left2.cols) = vert_acc_2[i];
		}

		//Accumulate the right accumulator columns of the second source
		for (int i = 0, l = right2.cols+1; i < right2.cols; i++, l++)
		{
			float acc2 = 0.0f;
			for (int j = 0; j < top2.rows; j++)
			{
				scale2.at<float>(j, l) += acc2 + vert_acc_2[j];
				acc2 += right2.at<float>(j, i);
			}

			//Continue across the left rows
			for (int j = 0, k = bot2.rows; j < bot2.rows; j++, k++)
			{
				scale2.at<float>(k, l) += acc2 + vert_acc_2[k];
				acc2 -= right2.at<float>(j, i);
			}
		}

		//Get the sum of the squares of the elements in the first image
		float sum_sqr = 0.0f;
		//Sum across the first source's rows...
		for (int i = 0; i < src1.rows; i++)
		{
			//...and the first source's columns
			r = src1.ptr<float>(i);
			for (int j = 0; j < src1.cols; j++)
			{
				sum_sqr += r[j]*r[j];
			}
		}

		//Get the sum of the squares of the elements in the second image
		float sum_sqr_2 = 0.0f;
		//Sum across the first source's rows...
		for (int i = 0; i < src2.rows; i++)
		{
			//...and the first source's columns
			r = src2.ptr<float>(i);
			for (int j = 0; j < src2.cols; j++)
			{
				sum_sqr_2 += r[j]*r[j];
			}
		}

		double min2, max2;
		cv::minMaxLoc(scale, &min2, &max2, NULL, NULL);
		std::cout << min2 << ", " << max2 << std::endl;

		cv::minMaxLoc(scale2, &min2, &max2, NULL, NULL);
		std::cout << min2 << ", " << max2 << std::endl;

		//Calculate the elements of the linear map needed to correct the Pearson correlation values
		for (int i = 0; i < scale.rows; i++)
		{
			r = scale.ptr<float>(i);
			s = scale2.ptr<float>(i);
			for (int j = 0; j < scale.cols; j++)
			{
				s[j] = std::sqrt(sum_sqr - r[j]) * std::sqrt(sum_sqr_2 - s[j]) + 1; // +1 to prevent divide by 0 errors
			}
		}

		cv::Point y;
		double min, max;
		cv::minMaxLoc(pear / scale, &min, &max, &y, NULL);
		std::cout << min << ", " << max << ", " << y << std::endl;
		std::cout << scale.rows << ", " << scale.cols << std::endl;
		std::cout << pear.at<float>(y.y, y.x) << ", " << scale.at<float>(y.y, y.x) << ", " << scale2.at<float>(y.y, y.x) << std::endl;
		std::cout << sum_sqr << ", " << sum_sqr_2 << std::endl;
		std::getchar();

		//Linearly map to the correct results
		return pear / scale;
	}
}