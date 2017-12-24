#include <beanland_commensuration.h>

namespace ba
{
	/*Commensurate the individual images so that the Beanland atlas can be constructed
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual floating point images to extract spots from
	**spot_pos: std::vector<cv::Point>, Positions of located spots in aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images
	**col_max: int, Maximum column difference between spot positions
	**row_max: int, Maximum row difference between spot positions
	**radius: const int, Radius about the spot locations to extract pixels from
	**samp_to_detect_sphere: cv::Vec2f, Initial estimate of the sample-to-detector sphere radius of curvature and average direction, respectively
	**Returns:
	**
	*/
	std::vector<cv::Mat> beanland_commensurate(std::vector<cv::Mat> &mats, cv::Point &spot_pos, std::vector<std::vector<int>> &rel_pos,
		int col_max, int row_max, int radius, cv::Vec2f &samp_to_detect_sphere)
	{
		//Find the non-consecutively same position spots and record the indices of multiple spots with the same positions
		std::vector<std::vector<int>> grouped_idx = consec_same_pos_spots(rel_pos);

		//Create collection of the spot images at different positions, averaging the consecutively same position spots. Also, slightly blur the 
		//spots to reduce high frequency noise
		int diam = 2*radius+1;
		std::vector<cv::Mat> commensuration = bragg_profile_preproc(mats, grouped_idx, spot_pos, rel_pos, col_max, row_max, radius, 
			diam, BRAGG_PROF_PREPROC_GAUSS);

		//Create a mask to indicate which pixels in the square are circle pixels to reduce calculations
		cv::Mat circ_mask = cv::Mat(commensuration[0].size(), CV_8UC1, cv::Scalar(0));

		//Draw circle at the position of the spot on the mask
		cv::circle(circ_mask, cv::Point(radius, radius), radius+1, cv::Scalar(1), -1, 8, 0);

		//Separation parameter needs to be calculated properly later ************************FLAG***************************
		float max_sep = 1.0*radius/*bragg_get_max_sep()*/;

		//Commensurate the angles and intensities
		//perspective_warp(commensuration, rel_pos, circ_mask, grouped_idx, max_sep, ewald_rad); 

		//Calculate the smallest radii of the spots that can be used when constructing the Beanland atlas. We want to use the smallest possible
		//radii of each circle to minimise the effect of the dark field decoupled Bragg profile curvature
		//get_smallest_spot_radii() - This will be written later. I need to calculate what the perspective warped spots look like first

		//Create a 3D mat to store the data
		//
		//int dims[] = {diam, diam, grouped_idx.size()};
		//cv::Mat intensities = cv::Mat(3, dims, CV_32FC1);



		////...and extract the spot from each micrograph
		//for (int j = 0; j < mats.size(); j++)
		//{		
		//	if (spot_pos.y >= row_max-rel_pos[1][j] && spot_pos.y < row_max-rel_pos[1][j]+mats[j].rows &&
		//		spot_pos.x >= col_max-rel_pos[0][j] && spot_pos.x < col_max-rel_pos[0][j]+mats[j].cols)
		//	{



		//		///Mask to extract spot from micrograph
		//		cv::Mat circ_mask = cv::Mat::zeros(mats[j].size(), CV_8UC1);

		//		//Draw circle at the position of the spot on the mask
		//		cv::Point circleCenter(spot_pos[k].x-col_max+rel_pos[0][j], spot_pos[k].y-row_max+rel_pos[1][j]);
		//		cv::circle(circ_mask, circleCenter, radius, cv::Scalar(1), -1, 8, 0);

		//		//Copy the part of the micrograph containing the spot
		//		cv::Mat imagePart = cv::Mat::zeros(mats[j].size(), mats[j].type());
		//		mats[j].copyTo(imagePart, circ_mask);

		//		//Compend spot to map
		//		float *r;
		//		ushort *s;
		//		ushort *t;
		//		byte *u;
		//		for (int m = 0; m < indv_num_mappers[k].rows; m++) 
		//		{
		//			r = indv_maps[k].ptr<float>(m);
		//			s = indv_num_mappers[k].ptr<ushort>(m);
		//			t = imagePart.ptr<ushort>(m);
		//			u = circ_mask.ptr<byte>(m);
		//			for (int n = 0; n < indv_num_mappers[k].cols; n++) 
		//			{
		//				//Add contributing pixels to maps
		//				r[n] += t[n];
		//				s[n] += u[n];
		//			}
		//		}
		//	}
		//}

		std::vector<cv::Mat> something;
		return something;
	}

	/*Identify groups of consecutive spots that all have the same position
	**Inputs:
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the spots in the image stack
	**Returns:
	**std::vector<std::vector<int>>, Groups of spots with the same location
	*/
	std::vector<std::vector<int>> consec_same_pos_spots(std::vector<std::vector<int>> &rel_pos)
	{
		//Find consecutive spots with the same position
		std::vector<std::vector<int>> groups;
		int first_same = 0; //Start of same position train
		for (int i = 1; i < rel_pos[0].size(); i++)
		{
			//Check if relative position is different from the previous
			if (rel_pos[0][i] != rel_pos[0][i-1] || rel_pos[1][i] != rel_pos[1][i-1])
			{
				//Create the train of same position indices
				std::vector<int> same_train(i - first_same);
				for (int j = 0; j < same_train.size(); j++)
				{
					same_train[j] = first_same + j;
				}

				//Append the indices sequence of the same train
				groups.push_back(same_train);

				//Restart the same train
				first_same = i;
			}
		}

		//Create the last train of same position indices
		std::vector<int> same_train(rel_pos[0].size() - first_same);
		for (int j = 0; j < same_train.size(); j++)
		{
			same_train[j] = first_same + j;
		}

		//Append the indices sequence of the same train
		groups.push_back(same_train);

		//Return the groups of same position indices
		return groups;
	}

	/*Extract a Bragg peak from an image stack, averaging the spots that are in the same position in consecutive images
	**Inputs:
	**mats: std::vector<cv::Mat> &, Images to extract the spots from
	**grouped_idx: std::vector<std::vector<int>> &, Groups of consecutive image indices where the spots are all in the same position
	**spot_pos: cv::Point &, Position of the spot on the aligned images average px values diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the spots in the input images to the first image
	**col_max: const int &, Maximum column difference between spot positions
	**row_max: const int &, Maximum row difference between spot positions
	**radius: const int &, Radius of the spot
	**diam: const int &, Diameter of the spot
	**gauss_size: const int &, Size of the Gaussian blurring kernel applied during the last preprocessing step to remove unwanted noise
	**Returns:
	**std::vector<cv::Mat>, Preprocessed Bragg peaks, ready for dark field decoupled profile extraction
	*/
	std::vector<cv::Mat> bragg_profile_preproc(std::vector<cv::Mat> &mats, std::vector<std::vector<int>> &grouped_idx, cv::Point &spot_pos, 
		std::vector<std::vector<int>> &rel_pos, const int &col_max, const int &row_max, const int &radius, const int &diam, 
		const int &gauss_size)
	{
		//Assign memory to store the preprocessed Bragg peaks
		std::vector<cv::Mat> blur_not_consec;

		//For each group of spots...
		for (int i = 0; i < grouped_idx.size(); i++)
		{
			//Index of the first image in the group
			int j = grouped_idx[i][0];

			//Check if the spot is in the image. If the first in the group isn't, none of them are as they are all in the same position
			if (spot_pos.y >= row_max-rel_pos[1][j] && spot_pos.y < row_max-rel_pos[1][j]+mats[j].rows &&
				spot_pos.x >= col_max-rel_pos[0][j] && spot_pos.x < col_max-rel_pos[0][j]+mats[j].cols)
			{
				//Accumulate the mats in the group for averaging
				cv::Mat acc = cv::Mat(diam, diam, CV_32FC1, cv::Scalar(0.0));

				//...extract the group of spots, averaging them together if there are multiple in the group
				for (int k = 0; k < grouped_idx[i].size(); k++)
				{
					//Index of image in the image stack
					j = grouped_idx[i][k];

					//Accumulate the circle in the accumulator
					accumulate_circle(mats[j], spot_pos.x-col_max+rel_pos[0][j], spot_pos.y-row_max+rel_pos[1][j], radius,
						acc, radius, radius);
				}

				//Gaussian blur the accumulator once the accumulations have been divided by the number of contributing circes
				cv::Mat blur;
				cv::GaussianBlur(acc / grouped_idx.size(), blur, cv::Size(gauss_size, gauss_size), 0, 0);

				//Store the preprocessed Bragg peak
				blur_not_consec.push_back(blur);
			}
		}

		return blur_not_consec;
	}

	/*Extracts a circle of data from an OpenCV mat and accumulates it in another mat. It is assumed that the dimensions specified for
	**the accumulator will allow the full circle-sized extraction to be accumulated
	**Inputs:
	**mat: cv::Mat &, Reference to a floating point OpenCV mat to extract the data from
	**col: const int &, column of circle origin
	**row: const int &, row of circle origin
	**rad: const int &, radius of the circle to extract data from
	**acc: cv::Mat &, Floating point OpenCV mat to accumulate the data in
	**acc_col: const int &, Column of the accumulator mat to position the circle at
	**acc_row: const int &, Row of the accumulator mat to position the circle at
	*/
	void accumulate_circle(cv::Mat &mat, const int &col, const int &row, const int &rad, cv::Mat &acc, const int &acc_col,
		const int &acc_row)
	{
		//Get the minimum and maximum rows and columns to iterate between
		int min_col = std::max(0, col-rad);
		int max_col = std::min(mat.cols-1, col+rad);
		int min_row = std::max(0, row-rad);
		int max_row = std::min(mat.rows-1, row+rad);

		//Iterate accross the circle rows
		float *p, *q;
        #pragma omp parallel for
		for (int i = min_row, rel_row = -rad, k = acc_row-rad; i <= max_row; i++, rel_row++, k++)
		{
			//Create C-style pointers to interate across the circle with
			p = mat.ptr<float>(i);
			q = acc.ptr<float>(k);

			//Get columns to iterate between
			int c = (int)std::sqrt(rad*rad-rel_row*rel_row);
			int min = std::max(min_col, col-c);
			int max = std::min(max_col, col+c);

			//Iterate across columns
			for (int j = min, l = acc_col-c; j <= max; j++, l++)
			{
				q[l] += p[j];
			}
		}
	}

	/*Homographic perspective warp of an OpenCV mat. The returned mat is cropped down to a size specified by the user
	**Input:
	**img: cv::Mat &, Image to homagraphically warp
	**Returns:
	**cv::Mat, Homographically perspective warped image
	*/
	cv::Mat homographic_perspective_warp(cv::Mat &img)
	{
		//Move this to utility? It will be called by the main perspective warp finding function

		cv::Mat warped;
		return warped;
	}

	/*Commensurate the images using homographic perspective warps, homomorphic warps and intensity rescaling
	**Input:
	**commensuration: std::vector<cv::Mat>> &, Preprocessed Bragg peaks. Consecutive Bragg peaks in the same position have been averaged
	**and they have been Gaussian blurred to remove unwanted high frequency noise
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the spots
	**grouped_idx: std::vector<std::vector<int>> &, Spots where they are grouped if they are consecutively in the same position
	**circ_mask: cv::Mat &, Mask indicating the spot pixels
	**max_dst: const float &, The maximum distance between 2 instances of a spot for the overlap between them to be considered
	**ewald_rad: const float &, Estimated Ewald radius
	*/
	void perspective_warp(std::vector<cv::Mat> &commensuration, std::vector<std::vector<int>> &rel_pos, cv::Mat &circ_mask, 
		std::vector<std::vector<int>> &grouped_idx, const float &max_dist, const float &ewald_rad)
	{
		//Find all the overlaps that must be considered between the groups of consecutive spots that are in the same position
		std::vector<std::vector<std::vector<int>>> overlaps = get_spot_overlaps(rel_pos, grouped_idx, max_dist);

		//The positions of the spot centres in the preprocessed images are all the same: they are in the middle of the images
		cv::Point spot_centre = cv::Point(circ_mask.cols/2 + 1, circ_mask.rows/2 + 1);

		//For each of the significant overlaps, calculate the homographic warps and homomorphisms needed for perspective correction
		for (int i = 0; i < overlaps.size(); i++)
		{
			//For each of the spots the spot overlaps with
			for (int j = 0; j < overlaps[i].size(); j++)
			{
				//Calculate the difference between the overlapping regions of the significantly overlapping groups of spots
				cv::Mat diff_overlap, mask;
				get_diff_overlap(commensuration[i], commensuration[overlaps[i][j][0]], spot_centre,	spot_centre, 
					overlaps[i][j][1], overlaps[i][j][2], circ_mask, diff_overlap, mask);

				/*Use the difference between the overlapping regions to estimate the perspective warping parameters needed to commensurate
				  the spots*/

			}
		}
	}

	/*Calculate an initial estimate for the dark field decoupled Bragg profile using the preprocessed Bragg peaks. This function is redundant.
	**It remains in case I need to generate data from it for my thesis, etc. in the future
	**Input:
	**blur_not_consec: std::vector<cv::Mat>> &, Preprocessed Bragg peaks. Consecutive Bragg peaks in the same position have been averaged
	**and they have been Gaussian blurred to remove unwanted high frequency noise
	**circ_mask: cv::Mat &, Mask indicating the spot pixels
	**Returns:
	**cv::Mat, Dark field decouple Bragg profile of the accumulation
	*/
	cv::Mat get_acc_bragg_profile(std::vector<cv::Mat> &blur_not_consec, cv::Mat &circ_mask)
	{
		//Assign memory to store the dark field decoupled Bragg profile estimate
		cv::Mat profile = cv::Mat(blur_not_consec[0].size(), CV_32FC1, cv::Scalar(0.0));

		//Get the maximum pixel in every column
		byte *c;

		//Iterate across mask rows...
		for (int i = 0; i < circ_mask.rows; i++)
		{
			//...and columns
			c = circ_mask.ptr<byte>(i);
			for (int j = 0; j < circ_mask.cols; j++)
			{
				//Get maximum values for mask pixels
				if (c[j])
				{
					//Iterate through every preprocessed Bragg peak to find the maximum
					float max = 0;
					for (int k = 0; k < blur_not_consec.size(); k++)
					{
						//Check if the pixel value is higher than the maximum
						if (blur_not_consec[k].at<float>(i, j) > max)
						{
							max = blur_not_consec[k].at<float>(i, j);
						}
					}

					//Record the maximum on the Bragg profile estimate
					profile.at<float>(i, j) = max;
				}
			}
		}

		return profile;
	}
}