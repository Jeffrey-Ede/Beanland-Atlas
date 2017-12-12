#include <beanland_atlas.h>

namespace ba
{
	/*Combine the k spaces mapped out by spots in each of the images create maps of the whole k space navigated by that spot.
	**Individual maps are summed together. The total map is then divided by the number of spot k space maps contributing to 
	**each px in the total map. These maps are then combined into an atlas
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual images to extract spots from
	**spot_pos: std::vector<cv::Point>, Positions of located spots in aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images
	**radius: const int, Radius about the spot locations to extract pixels from
	**ns_radius: const int, Radius to Navier-Stokes infill when removing the diffuse background
	**inpainting_method: Method to inpaint the Bragg peak regions in the diffraction pattern. Defaults to the Navier-Stokes method
	**Returns:
	**std::vector<cv::Mat>, Regions of k space surveys by the spots
	*/
	std::vector<cv::Mat> create_spot_maps(std::vector<cv::Mat> &mats, std::vector<cv::Point> &spot_pos, std::vector<std::vector<int>> &rel_pos,
		const int radius, const int ns_radius, const int inpainting_method)
	{
		//Initialise vectors of OpenCV mats to hold individual paths and number of contributions to those paths
		std::vector<cv::Mat> indv_maps(spot_pos.size());
		std::vector<cv::Mat> indv_num_mappers(spot_pos.size());
	
		//Get the maximum relative rows and columns and the difference between the maximum and minimum rows and columns
		int col_max = 0, row_max = 0, col_min = INT_MAX, row_min = INT_MAX;
        #pragma omp parallel for reduction(max:col_max), reduction(max:row_max), reduction(min:col_min), reduction(min:row_min)
		for (int i = 0; i < rel_pos[0].size(); i++)
		{
			//Minimum relitive position column
			if (rel_pos[0][i] < col_min)
			{
				col_min = rel_pos[0][i];
			}
			else
			{
				//Maximum relative position column
				if (rel_pos[0][i] > col_max)
				{
					col_max = rel_pos[0][i];
				}
			}

			//Minimum relative position row
			if (rel_pos[1][i] < row_min)
			{
				row_min = rel_pos[1][i];
			}
			else
			{
				//Maximum relaive position row
				if (rel_pos[1][i] > row_max)
				{
					row_max = rel_pos[1][i];
				}
			}
		}
		
		//Differences between maxima and minima are how far the spots travelled
		int cols_diff = col_max - col_min;
		int rows_diff = row_max - row_min;

		//Fill the path mats with zeros
        #pragma omp parallel for
		for (int j = 0; j < spot_pos.size(); j++) 
		{
			indv_maps[j] = cv::Mat::zeros(mats[j].size(), CV_32FC1);
			indv_num_mappers[j] = cv::Mat::zeros(mats[j].size(), CV_16UC1);
		}

		//Perform background subtraction
		subtract_background(mats, spot_pos, rel_pos, inpainting_method, col_max, row_max, ns_radius);

		//For each spot...
        #pragma omp parallel for
		for (int k = 0; k < spot_pos.size(); k++) 
		{
			//...get it's dark field decoupled Bragg profile in each micrograph...
			std::vector<cv::Mat> bragg_profiles = get_bragg_profiles(mats, spot_pos[k], rel_pos, col_max, row_max, radius);

			//...and extract the spot from each micrograph
			for (int j = 0; j < mats.size(); j++)
			{	
				//Check if the spot is in the image
				if (spot_pos[k].y >= row_max-rel_pos[1][j] && spot_pos[k].y < row_max-rel_pos[1][j]+mats[j].rows &&
					spot_pos[k].x >= col_max-rel_pos[0][j] && spot_pos[k].x < col_max-rel_pos[0][j]+mats[j].cols)
				{
					///Mask to extract spot from micrograph
					cv::Mat circ_mask = cv::Mat::zeros(mats[j].size(), CV_8UC1);
		
					//Draw circle at the position of the spot on the mask
					cv::Point circleCenter(spot_pos[k].x-col_max+rel_pos[0][j], spot_pos[k].y-row_max+rel_pos[1][j]);
					cv::circle(circ_mask, circleCenter, radius+1, cv::Scalar(1), -1, 8, 0);
		
					//Copy the part of the micrograph containing the spot
					cv::Mat imagePart = cv::Mat::zeros(mats[j].size(), mats[j].type());
					mats[j].copyTo(imagePart, circ_mask);
		
					//Compend spot to map
					float *r, *t;
					ushort *s;
					byte *u;
					for (int m = 0; m < indv_num_mappers[k].rows; m++) 
					{
						r = indv_maps[k].ptr<float>(m);
						s = indv_num_mappers[k].ptr<ushort>(m);
						t = imagePart.ptr<float>(m);
						u = circ_mask.ptr<byte>(m);
						for (int n = 0; n < indv_num_mappers[k].cols; n++) 
						{
							//Add contributing pixels to maps
							r[n] += t[n];
							s[n] += u[n];
						}
					}
				}
			}
		}

		//Normalise maps using the number of mappers contributing to each pixel
        #pragma omp parallel for
		for (int k = 0; k < spot_pos.size(); k++) {
				
			//Divide non-zero accumulator matrix pixel values by number of overlapping contributing micrographs
			float *r;
			ushort *s;
			for (int m = 0; m < indv_num_mappers[k].rows; m++) 
			{
				r = indv_maps[k].ptr<float>(m);
				s = indv_num_mappers[k].ptr<ushort>(m);
				for (int n = 0; n < indv_num_mappers[k].cols; n++) 
				{
					//Divide pixels contributed to by the number of contributing pixels
					if (s[n]) {
						r[n] /= s[n];
					}
				}
			}
		}
		
		//Crop maps so that they only contain the paths mapped out by the spots
		std::vector<cv::Mat> surveys(spot_pos.size());
        #pragma omp parallel for
		for (int k = 0; k < spot_pos.size(); k++) {
				
			//Minimum row
			int x;
			int x_pad;
			if (spot_pos[k].x >= rows_diff+radius) {
				x = spot_pos[k].x - rows_diff - radius;
				x_pad = 0;
			}
			else {
				x = 0;
				x_pad = rows_diff + radius - spot_pos[k].x;
			}
		
			//Minimum column
			int y;
			int y_pad;
			if (spot_pos[k].y >= cols_diff+radius) {
				y = spot_pos[k].y - cols_diff - radius;
				y_pad = 0;
			}
			else {
				y = 0;
				y_pad = cols_diff + radius - spot_pos[k].y;
			}
				
			//Maximum row
			int x_2;
			if (spot_pos[k].x + radius <= mats[0].rows) {
				x_2 = spot_pos[k].x + radius;
			}
			else {
				x_2 = mats[0].rows;
			}
		
			//Maximum column
			int y_2;
			if (spot_pos[k].y + radius <= mats[0].cols) {
				y_2 = spot_pos[k].y + radius;
			}
			else {
				y_2 = mats[0].cols;
			}
		
			//Establish roi in map and cast
			cv::Rect roi_map = cv::Rect(x, y, x_2-x, y_2-y);
			cv::Rect roi_crop = cv::Rect(x_pad, y_pad, x_2-x, y_2-y);
		
			//Cropped map
			surveys[k] = cv::Mat(cols_diff+2*radius, rows_diff+2*radius, CV_32FC1, cv::Scalar(0.0));
			indv_maps[k](roi_map).copyTo(surveys[k](roi_crop));
		}

		//display_CV(create_raw_atlas(surveys, spot_pos, radius, cols_diff, rows_diff), 1e-3);

		return surveys;
	}

	/*Subtract the bacground from micrographs by masking the spots, infilling the masked image and then subtracting the infilled
	**image from the original
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual floating point images to extract spots from
	**spot_pos: std::vector<cv::Point>, Positions of located spots in aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images
	**inpainting_method: Method to inpaint the Bragg peak regions in the diffraction pattern
	**col_max: int, Maximum column difference between spot positions
	**row_max: int, Maximum row difference between spot positions
	**ns_radius: const int, Radius to Navier-Stokes infill when removing the diffuse background
	*/
	void subtract_background(std::vector<cv::Mat> &mats, std::vector<cv::Point> &spot_pos, std::vector<std::vector<int>> &rel_pos, 
		int inpainting_method, int col_max, int row_max, int ns_radius)
	{
		//If the user wants to use inpainting to remove the diffuse background
		if(inpainting_method != -1)
		{
			//Find position of spots in each micrograph;
			#pragma omp parallel for
			for (int j = 0; j < mats.size(); j++) 
			{
				///Mask to mark the positions of all the spots to be Navier-Stokes infilled on the micrograph
				cv::Mat ns_mask = cv::Mat::zeros(mats[j].size(), CV_8UC1);

				//Use Navier-Stokes infilling to remove the diffuse background from the micrographs
				//Start by creating a mask that marks out the regions to infill
				for (int k = 0; k < spot_pos.size(); k++) 
				{
					if (spot_pos[k].y >= row_max-rel_pos[1][j] && spot_pos[k].y < row_max-rel_pos[1][j]+mats[j].rows &&
						spot_pos[k].x >= col_max-rel_pos[0][j] && spot_pos[k].x < col_max-rel_pos[0][j]+mats[j].cols) 
					{
						//Draw circle at the position of the spot on the mask
						cv::Point circleCenter(spot_pos[k].x-col_max+rel_pos[0][j], spot_pos[k].y-row_max+rel_pos[1][j]);
						cv::circle(ns_mask, circleCenter, ns_radius, cv::Scalar(1), -1, 8, 0);
					}
				}

				//Create Navier-Stokes inpainted diffraction pattern
				cv::Mat inpainted = cv::Mat::zeros(mats[j].size(), CV_32FC1);
				cv::inpaint(mats[j], ns_mask, inpainted, 7, inpainting_method);

				//Subtract the background from the image
				float *r, *s;
				for (int m = 0; m < mats[j].rows; m++) 
				{
					r = mats[j].ptr<float>(m);
					s = inpainted.ptr<float>(m);
					for (int n = 0; n < mats[j].cols; n++) 
					{
						r[n] = r[n] > s[n] ? r[n] - s[n] : 0;
					}
				}
			}
		}
	}

	/*Extract a spot's dark field decoupled Bragg profile for each micrograph it is in
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual floating point images to extract spots from
	**mats: std::vector<cv::Mat> &, Individual floating point images to extract spots from
	**spot_pos: std::vector<cv::Point>, Positions of located spots in aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images
	**col_max: int, Maximum column difference between spot positions
	**row_max: int, Maximum row difference between spot positions
	**radius: const int, Radius about the spot locations to extract pixels from
	*/
	std::vector<cv::Mat> get_bragg_profiles(std::vector<cv::Mat> &mats, cv::Point &spot_pos, std::vector<std::vector<int>> &rel_pos,
		int col_max, int row_max, int radius)
	{
		//Find the non-consecutively same position spots and record the indices of multiple spots with the same positions
		std::vector<std::vector<int>> grouped_idx = consec_same_pos_spots(rel_pos);
		
		//Add calculation to extract sensible smaller radius to use
		int small_rad = 0.7*radius; //Approximate the smaller radius for now

		//Create collection of the spot images at different positions, averaging the consecutively same position spots. Also, blur the 
		//spots to remove high frequency noise
		int diam = 2*radius+1;
		std::vector<cv::Mat> blur_not_consec = bragg_profile_preproc(mats, grouped_idx, spot_pos, rel_pos, col_max, row_max, radius, 
			diam, BRAGG_PROF_PREPROC_GAUSS);

		//Create a mask to indicate which pixels in the square are circle pixels to reduce calculations
		cv::Mat circ_mask = cv::Mat(blur_not_consec[0].size(), CV_8UC1, cv::Scalar(0));

		//Draw circle at the position of the spot on the mask
		cv::circle(circ_mask, cv::Point(radius, radius), radius+1, cv::Scalar(1), -1, 8, 0);

		/*display_CV(circ_mask);
		display_CV(blur_not_consec[0]);*/

		float max_sep = 1.2*radius/*bragg_get_max_sep()*/;

		//Get an approximate dark field decoupled Bragg profile
		cv::Mat decoupled_bragg = get_angl_intens(blur_not_consec, circ_mask, max_sep); //Separation parameter needs to be calculated properly later

		std::cout << decoupled_bragg;

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

	/*Calculate an initial estimate for the dark field decoupled Bragg profile using the preprocessed Bragg peaks
	**Input:
	**blur_not_consec: std::vector<cv::Mat>> &, Preprocessed Bragg peaks. Consecutive Bragg peaks in the same position have been averaged
	**and they have been Gaussian blurred to remove unwanted high frequency noise
	**circ_mask: cv::Mat &, Mask indicating the spot pixels
	**max_dst: float, The maximum distance between 2 instances of a spot for the overlap between them to be considered
	**Returns:
	**cv::Mat, Dark field decouple Bragg profile of the accumulation
	*/
	cv::Mat get_angl_intens(std::vector<cv::Mat> &blur_not_consec, cv::Mat &circ_mask, float &max_dist)
	{
		cv::Mat r;
		return r;
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