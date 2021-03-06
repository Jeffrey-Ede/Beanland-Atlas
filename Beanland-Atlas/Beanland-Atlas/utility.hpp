#pragma once

namespace ba
{
	/*OpenCV function that shifts the energy into the center of an OpenCV mat containing an FFT
	**Input:
	**_out: cv::InputOutputArray, OpenCV mat containing an FFt to shift the center of energy of
	*/
	static void fftShift(cv::InputOutputArray _out)
	{
		using namespace cv;

		Mat out = _out.getMat();

		if(out.rows == 1 && out.cols == 1)
		{
			// trivially shifted.
			return;
		}

		std::vector<Mat> planes;
		split(out, planes);

		int xMid = out.cols >> 1;
		int yMid = out.rows >> 1;

		bool is_1d = xMid == 0 || yMid == 0;

		if(is_1d)
		{
			int is_odd = (xMid > 0 && out.cols % 2 == 1) || (yMid > 0 && out.rows % 2 == 1);
			xMid = xMid + yMid;

			for(size_t i = 0; i < planes.size(); i++)
			{
				Mat tmp;
				Mat half0(planes[i], Rect(0, 0, xMid + is_odd, 1));
				Mat half1(planes[i], Rect(xMid + is_odd, 0, xMid, 1));

				half0.copyTo(tmp);
				half1.copyTo(planes[i](Rect(0, 0, xMid, 1)));
				tmp.copyTo(planes[i](Rect(xMid, 0, xMid + is_odd, 1)));
			}
		}
		else
		{
			int isXodd = out.cols % 2 == 1;
			int isYodd = out.rows % 2 == 1;
			for(size_t i = 0; i < planes.size(); i++)
			{
				// perform quadrant swaps...
				Mat q0(planes[i], Rect(0,    0,    xMid + isXodd, yMid + isYodd));
				Mat q1(planes[i], Rect(xMid + isXodd, 0,    xMid, yMid + isYodd));
				Mat q2(planes[i], Rect(0,    yMid + isYodd, xMid + isXodd, yMid));
				Mat q3(planes[i], Rect(xMid + isXodd, yMid + isYodd, xMid, yMid));

				if(!(isXodd || isYodd))
				{
					Mat tmp;
					q0.copyTo(tmp);
					q3.copyTo(q0);
					tmp.copyTo(q3);

					q1.copyTo(tmp);
					q2.copyTo(q1);
					tmp.copyTo(q2);
				}
				else
				{
					Mat tmp0, tmp1, tmp2 ,tmp3;
					q0.copyTo(tmp0);
					q1.copyTo(tmp1);
					q2.copyTo(tmp2);
					q3.copyTo(tmp3);

					tmp0.copyTo(planes[i](Rect(xMid, yMid, xMid + isXodd, yMid + isYodd)));
					tmp3.copyTo(planes[i](Rect(0, 0, xMid, yMid)));

					tmp1.copyTo(planes[i](Rect(0, yMid, xMid, yMid + isYodd)));
					tmp2.copyTo(planes[i](Rect(xMid, 0, xMid + isXodd, yMid)));
				}
			}
		}

		merge(planes, out);
	}

	/*Get value at a location in an image. If the location is outside the image, move the index so that it is inside
	**Inputs:
	**img: cv::Mat &, Image the point is restricted to be in
	**col: const int, Column index of the point
	**row: const int, Row index of the point
	**val: T &, Reference to a variable to store the value in. This must be of the same type as the matrix
	*/
	template <typename T> void lim_at(cv::Mat &img, const int col, const int row, T &val)
	{
		val = img.at<T>(row < img.rows ? (row >= 0 ? row : 0) : img.rows-1, 
			col < img.cols ? (col >= 0 ? col : 0) : img.cols-1);
	}

	/*Check if a container contains an object
	**Inputs:
	**container: std::vector<typename S> &, The container
	**thing: typename T &, The thing to check if the container contains
	*/
	template <typename S, typename T> bool contains(std::vector<S> &container, T &thing)
	{
		return std::find(container.begin(), container.end(), thing) != container.end();
	}

	/*Convert a 2D image to a 1D array by moving across its columns and rows, in that order, and recording the values of 
	**pixels corresponding to non-zero values on the mask
	**Inputs:
	**img: cv::Mat &, 32-bit Image to convert from 2D to 1D
	**img1D: std::vector<typename T>, Vector to store the 1D version of the image
	**mask: cv::Mat &, 8-bit mask whose non-zero pixels indicate pixels to record in the 1D vector
	*/
	template <typename T> void img_2D_to_1D(cv::Mat &img, std::vector<T> &img1D, cv::Mat &mask = cv::Mat())
	{
		if (mask.empty())
		{
			//Assign memory to hold the 1D image
			img1D = std::vector<T>(img.rows*img.cols);

			//Iterate over the image, recording the pixel
			float *p;
			byte *b;
			for (int i = 0, k = 0; i < mask.rows; i++)
			{
				b = mask.ptr<byte>(i);
				p = img.ptr<float>(i);
				for (int j = 0; j < mask.cols; j++, k++)
				{
					img1D[k] = (T)p[j];
				}
			}
		}
		else
		{
			//Get number of pixels to place in the 1D vector
			int nnz_px = cv::countNonZero(mask);

			//Assign memory to hold the 1D image
			img1D = std::vector<T>(nnz_px);

			//Iterate over the image, recording the pixel
			float *p;
			byte *b;
			for (int i = 0, k = 0; i < mask.rows; i++)
			{
				b = mask.ptr<byte>(i);
				p = img.ptr<float>(i);
				for (int j = 0; j < mask.cols; j++)
				{
					//Check if the pixel is indicated to be recorded by the mask
					if (b[j])
					{
						img1D[k] = (T)p[j];

						k++;
					}
				}
			}
		}
	}

	/*Calculate Pearson's normalised product moment correlation coefficient between 2 vectors
	**Inputs:
	**vect1: std::vector<typename S> &, One of the data vectors
	**vect2: std::vector<typename T> &, The other data vector
	**Returns:
	**double, Pearson normalised product moment correlation coefficient between the vectors
	*/
	template <typename T, typename S> double pearson_corr(std::vector<S> &vect1, std::vector<T> &vect2)
	{
		//Sums for Pearson product moment correlation coefficient
		double sum_xy = 0.0;
		double sum_x = 0.0;
		double sum_y = 0.0;
		double sum_x2 = 0.0;
		double sum_y2 = 0.0;

		//Compare the vectors index by index
        #pragma omp parallel for reduction(sum:sum_xy), reduction(sum:sum_x), reduction(sum:sum_y), reduction(sum:sum_x2), reduction(sum:sum_y2)
		for (int i = 0; i < vect1.size(); i++)
		{
			sum_xy += vect1[i]*vect2[i];
			sum_x += vect1[i];
			sum_y += vect2[i];
			sum_x2 += vect1[i]*vect1[i];
			sum_y2 += vect2[i]*vect2[i];
		}

		return (vect1.size()*sum_xy - sum_x*sum_y) / (std::sqrt(vect1.size()*sum_x2 - sum_x*sum_x) * 
			std::sqrt(vect1.size()*sum_y2 - sum_y*sum_y));
	}

	/*Convert an OpenCV mat to a 1D vector
	**Inputs: 
	**mat: cv::Mat &, OpenCV mat to convert
	**vect: std::vector<typename> &, Vector to store the matrix i
	**p: typename &, Variable of the same type as the matrix elements
	**transpose: const bool, If true, the vector is filled with transpositional data. Defaults to false
	*/
	template <typename T, typename U> void cvMat_to_vect(cv::Mat &mat, std::vector<T> &vect, U &p, const bool transpose = false)
	{
		vect = std::vector<T>(mat.rows*mat.cols);
		if (transpose)
		{
			for (int j = 0, k = 0; j < mat.cols; j++)
			{
				for (int i = 0; i < mat.rows; i++)
				{
					vect[k++] = mat.at<U>(i, j);
				}
			}
		}
		else
		{
			U *q;
			for (int i = 0, k = 0; i < mat.rows; i++)
			{
				q = mat.ptr<U>(i);
				for (int j = 0; j < mat.cols; j++)
				{
					vect[k++] = q[j];
				}
			}
		}
	}
}