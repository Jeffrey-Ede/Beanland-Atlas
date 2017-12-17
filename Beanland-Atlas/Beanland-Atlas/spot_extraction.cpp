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

		//Perform background subtraction using Navier-Stokes infilling or otherwise
		subtract_background(mats, spot_pos, rel_pos, inpainting_method, col_max, row_max, ns_radius);

		//Get the approximate radius and orientation of the sample-to-detector sphere
		//float ewald_rad = ewald_radius(spot_pos);

		//For each spot...
        #pragma omp parallel for
		for (int k = 0; k < spot_pos.size(); k++) 
		{
			//...get it's dark field decoupled Bragg profile in each micrograph...
			//std::vector<cv::Mat> bragg_profiles = beanland_commensurate(mats, spot_pos[k], rel_pos, col_max, row_max, radius, ewald_rad);

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
}