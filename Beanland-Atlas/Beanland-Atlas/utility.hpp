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
			nnz_px = cv::countNonZero(mask);

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
}