#include <template_matching.h>

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
		cv::Mat scale = cv::Mat(ssd.rows, ssd.cols, CV_32FC1);

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
			for (int k = src1.rows-1; k >= top.rows; k--)
			{
				acc += left.at<float>(k, i);
			}

			for (int j = 0, k = top.rows-1; j < top.rows; j++, k--)
			{
				offset.at<float>(j, i) += acc + vert_acc[j];
				acc += left.at<float>(k, i);
			}

			//Continue across the left rows
			for (int j = 0, k = bot.rows, m = src1.rows-1; j < bot.rows; j++, k++, m--)
			{
				offset.at<float>(k, i) += acc + vert_acc[k];
				acc -= left.at<float>(m, i);
			}
		}

		//Middle columns
		for (int i = 0; i < top.rows + bot.rows + 1; i++)
		{
			offset.at<float>(i, left.cols) = vert_acc[i];
		}

		//Accumulate the right accumulator columns
		for (int i = 0, l = left.cols+1; i < right.cols; i++, l++)
		{
			float acc = 0.0f;
			for (int k = src1.rows-1; k >= top.rows; k--)
			{
				acc += right.at<float>(k, i);
			}

			for (int j = 0, k = top.rows-1; j < top.rows; j++, k--)
			{
				offset.at<float>(j, l) += acc + vert_acc[j];
				acc += right.at<float>(k, i);
			}

			//Continue across the left rows
			for (int j = 0, k = bot.rows, m = src1.rows-1; j < bot.rows; j++, k++, m--)
			{
				offset.at<float>(k, l) += acc + vert_acc[k];
				acc -= right.at<float>(m, i);
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

	/*Pearson product moment correlation coefficient between 2 images. The second images is correlated against the 
	**first in the Fourier domain
	**src1: cv::Mat &, One of the images
	**src2: cv::Mat &, The second image
	**frac: float, Proportion of the first image's dimensions to pad it by. Must be smaller than 1.0f so that there is at 
	**least some overlap
	**Returns,
	**cv::Mat, Pearson normalised product moment correlation coefficients. The result is not normalised.
	*/
	cv::Mat fourier_pearson_corr(cv::Mat &src1, cv::Mat &src2, float frac)
	{
		//Pad the first matrix to prepare it for the second matrix being correlated accross it
		cv::Mat pad1;
		int pad_rows = (int)(frac*src1.rows);
		int pad_cols = (int)(frac*src1.cols);
		float src1_mean = cv::mean(src1).val[0];
		cv::copyMakeBorder(src1, pad1, pad_rows, pad_rows, pad_cols, pad_cols, cv::BORDER_CONSTANT, cv::Scalar(src1_mean));

		//Calculate the Pearson product moment correlation coefficients in the Fourier domain
		cv::Mat pear;
		cv::matchTemplate(pad1, src2, pear, CV_TM_CCOEFF);

		//Subtract the mean of the images so that it can be subtracted from them during the correction
		cv::Mat sub1 = src1 - src1_mean;
		cv::Mat sub2 = src2 - cv::mean(src2).val[0];

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
			s = sub1.ptr<float>(i);

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
			s = sub1.ptr<float>(i);

			//Calculate the square of the first column element
			r[0] = s[src1.cols-1]*s[src1.cols-1];

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
			top.at<float>(top.rows-1, i) = sub1.at<float>(0, i)*sub1.at<float>(0, i);

			//Iterate across the remaining mat rows, accumulating the sum of squares
			for (int j = top.rows-2, k = 1; j >= 0; j--, k++)
			{
				top.at<float>(j, i) = sub1.at<float>(k, i)*sub1.at<float>(k, i) + top.at<float>(j+1, i);
			}
		}

		/* Bottom of 1st image */
		//Iterate across mat cols...
		cv::Mat bot = cv::Mat(pad_rows, src1.cols, CV_32FC1);
		for (int i = 0; i < bot.cols; i++)
		{
			//Calculate the square of the first row element
			bot.at<float>(0, i) = sub1.at<float>(src2.rows-1, i)*sub1.at<float>(src2.rows-1, i);

			//Iterate across the remaining mat rows, accumulating the sum of squares
			for (int j = 1, k = src2.rows-2; j < bot.rows; j++, k--)
			{
				bot.at<float>(j, i) = sub1.at<float>(k, i)*sub1.at<float>(k, i) + bot.at<float>(j-1, i);
			}
		}

		/* Left of 2nd image */
		//Iterate across mat rows...
		cv::Mat left2 = cv::Mat(src2.rows, pad_cols, CV_32FC1);
		for (int i = 0; i < left.rows; i++)
		{
			//Initialise pointers
			r = left2.ptr<float>(i);
			s = sub2.ptr<float>(i);

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
			s = sub2.ptr<float>(i);

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
			top2.at<float>(top.rows-1, i) = sub2.at<float>(0, i)*sub2.at<float>(0, i);

			//Iterate across the remaining mat rows, accumulating the sum of squares
			for (int j = top.rows-2, k = 1; j >= 0; j--, k++)
			{
				top2.at<float>(j, i) = sub2.at<float>(k, i)*sub2.at<float>(k, i) + top2.at<float>(j+1, i);
			}
		}

		/* Bottom of 2nd image */
		//Iterate across mat cols...
		cv::Mat bot2 = cv::Mat(pad_rows, src2.cols, CV_32FC1);
		for (int i = 0; i < bot.cols; i++)
		{
			//Calculate the square of the first row element
			bot2.at<float>(0, i) = sub2.at<float>(src2.rows-1, i)*sub2.at<float>(src2.rows-1, i);

			//Iterate across the remaining mat rows, accumulating the sum of squares
			for (int j = 1, k = src2.rows-2; j < bot.rows; j++, k--)
			{
				bot2.at<float>(j, i) = sub2.at<float>(k, i)*sub2.at<float>(k, i) + bot2.at<float>(j-1, i);
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

		//Accumulate the left accumulator columns of the first source
		for (int i = 0; i < left.cols; i++)
		{
			float acc = 0.0f;
			for (int k = src1.rows-1; k >= top.rows; k--)
			{
				acc += left.at<float>(k, i);
			}

			for (int j = 0, k = top.rows-1; j < top.rows; j++, k--)
			{
				scale.at<float>(j, i) += acc + vert_acc[j];
				acc += left.at<float>(k, i);
			}

			//Continue across the left rows
			for (int j = 0, k = bot.rows, m = src1.rows-1; j < bot.rows; j++, k++, m--)
			{
				scale.at<float>(k, i) += acc + vert_acc[k];
				acc -= left.at<float>(m, i);
			}
		}

		//Middle column of the second source
		for (int i = 0; i < top.rows + bot.rows + 1; i++)
		{
			scale.at<float>(i, left.cols) = vert_acc[i];
		}

		//Accumulate the right accumulator columns of the first source
		for (int i = 0, l = left.cols+1; i < left.cols; i++, l++)
		{
			float acc = 0.0f;
			for (int k = src1.rows-1; k >= top.rows; k--)
			{
				acc += right.at<float>(k, i);
			}

			for (int j = 0, k = top.rows-1; j < top.rows; j++, k--)
			{
				scale.at<float>(j, l) += acc + vert_acc[j];
				acc += right.at<float>(k, i);
			}

			//Continue across the left rows
			for (int j = 0, k = bot.rows, m = src1.rows-1; j < bot.rows; j++, k++, m--)
			{
				scale.at<float>(k, l) += acc + vert_acc[k];
				acc -= right.at<float>(m, i);
			}
		}

		//Accumulate the left accumulator columns of the second source
		for (int i = 0; i < left2.cols; i++)
		{
			float acc2 = 0.0f;
			for (int k = src2.rows-1; k >= top2.rows; k--)
			{
				acc2 += left2.at<float>(k, i);
			}

			for (int j = 0, k = top2.rows-1; j < top2.rows; j++, k--)
			{
				scale2.at<float>(j, i) += acc2 + vert_acc_2[j];
				acc2 += left2.at<float>(k, i);
			}

			//Continue across the left rows
			for (int j = 0, k = bot2.rows, m = src2.rows-1; j < bot2.rows; j++, k++, m--)
			{
				scale2.at<float>(k, i) += acc2 + vert_acc_2[k];
				acc2 -= left2.at<float>(m, i);
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
			for (int k = src2.rows-1; k >= top2.rows; k--)
			{
				acc2 += right2.at<float>(k, i);
			}

			for (int j = 0, k = top2.rows-1; j < top2.rows; j++, k--)
			{
				scale2.at<float>(j, l) += acc2 + vert_acc_2[j];
				acc2 += right2.at<float>(k, i);
			}

			//Continue across the left rows
			for (int j = 0, k = bot2.rows, m = src2.rows-1; j < bot2.rows; j++, k++, m--)
			{
				scale2.at<float>(k, l) += acc2 + vert_acc_2[k];
				acc2 -= right2.at<float>(m, i);
			}
		}

		//Get the sum of the squares of the elements in the first image
		float sum_sqr = 0.0f;
		//Sum across the first source's rows...
		for (int i = 0; i < src1.rows; i++)
		{
			//...and the first source's columns
			r = sub1.ptr<float>(i);
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
			r = sub2.ptr<float>(i);
			for (int j = 0; j < src2.cols; j++)
			{
				sum_sqr_2 += r[j]*r[j];
			}
		}

		//Calculate the elements of the linear map needed to correct the Pearson correlation values
		for (int i = 0, k = scale.rows-1; i < scale.rows; i++, k--)
		{
			r = scale.ptr<float>(k);
			s = scale2.ptr<float>(i);
			for (int j = 0, l = scale.cols-1; j < scale.cols; j++, l--)
			{
				s[j] = std::sqrt(sum_sqr > r[l] ? sum_sqr - r[l] : 0) * 
					std::sqrt(sum_sqr_2 > s[j] ? sum_sqr_2 - s[j] : 0); //Ternary operations safeguard against rounding errors
			}
		}

		//Linearly map to the correct results
		return pear / scale2;
	}

	/*Create phase correlation specturm
	**src1: cv::Mat &, One of the images
	**src2: cv::Mat &, The second image
	**Returns,
	**cv::Mat, Phase correlation spectrum
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

		//Get the phase correlation spectrum...
		cv::idft(D, D);

		//...and shift its energy into the center
		fftShift(D);

		return D;
	}
}