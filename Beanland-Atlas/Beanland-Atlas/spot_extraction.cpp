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
	**Returns:
	**cv::Mat, Atlas showing the k space surveyed by each of the spots
	*/
	cv::Mat create_spot_maps(std::vector<cv::Mat> &mats, std::vector<cv::Point> &spot_pos, std::vector<std::vector<int>> &rel_pos,
		const int radius, const int ns_radius)
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
		for (int j = 0; j < spot_pos.size(); j++) {
			indv_maps[j] = cv::Mat::zeros(mats[j].size(), CV_32FC1);
			indv_num_mappers[j] = cv::Mat::zeros(mats[j].size(), CV_16UC1);
		}

		//Find position of spots in each micrograph;
        //#pragma omp parallel for
		for (int j = 0; j < mats.size(); j++) {

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
			cv::Mat ns_inpainted = cv::Mat::zeros(mats[j].size(), CV_32FC1);
			cv::inpaint(mats[j], ns_mask, ns_inpainted, 3, cv::INPAINT_NS);

			cv::Mat ns_subtracted = mats[j] - ns_inpainted;

			//For each spot
			for (int k = 0; k < spot_pos.size(); k++)
			{		
				if (spot_pos[k].y >= row_max-rel_pos[1][j] && spot_pos[k].y < row_max-rel_pos[1][j]+mats[j].rows &&
					spot_pos[k].x >= col_max-rel_pos[0][j] && spot_pos[k].x < col_max-rel_pos[0][j]+mats[j].cols)
				{
					///Mask to extract spot from micrograph
					cv::Mat circ_mask = cv::Mat::zeros(mats[j].size(), CV_8UC1);
		
					//Draw circle at the position of the spot on the mask
					cv::Point circleCenter(spot_pos[k].x-col_max+rel_pos[0][j], spot_pos[k].y-row_max+rel_pos[1][j]);
					cv::circle(circ_mask, circleCenter, radius, cv::Scalar(1), -1, 8, 0);
		
					//Copy the part of the micrograph containing the spot
					cv::Mat imagePart = cv::Mat::zeros(mats[j].size(), mats[j].type());
					ns_subtracted.copyTo(imagePart, circ_mask);
		
					//Compend spot to map
					float *r;
					ushort *s;
					ushort *t;
					byte *u;
					for (int m = 0; m < indv_num_mappers[k].rows; m++) 
					{
						r = indv_maps[k].ptr<float>(m);
						s = indv_num_mappers[k].ptr<ushort>(m);
						t = imagePart.ptr<ushort>(m);
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

		return create_raw_atlas(surveys, spot_pos, radius, cols_diff, rows_diff);
	}

	/*Combines individual spots' surveys of k space into a single atlas. Surveys are positioned proportionally to their spot's position in 
	**the aligned average px values pattern
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by each spot
	**spot_pos: std::vector<cv::Point> &, Positions of spots in aligned average image
	**radius: int, Radius of the spots being used
	**cols_diff: int, Difference between the minimum and maximum spot columns
	**rows_diff: int, Difference between the minimum and maximum spot rows
	**Return:
	cv::Mat, Atlas of the k space surveyed by the diffraction spots
	*/
	cv::Mat create_raw_atlas(std::vector<cv::Mat> &surveys, std::vector<cv::Point> &spot_pos, int radius, int cols_diff, int rows_diff)
	{
		//Find spots with minimum value for the maximum value of their row or column separations
		float min_sep = INT_MAX;
	    #pragma omp parallel for reduction(min:min_sep)
		for (int m = 0; m < spot_pos.size(); m++) {
			for (int n = m+1; n < spot_pos.size(); n++) {
					
				//Check if the maximum row or column value is smaller than the minimum
				float new_min = std::max(std::abs(spot_pos[m].y - spot_pos[n].y), std::abs(spot_pos[m].x - spot_pos[n].x));
				if (new_min < min_sep) {
					min_sep = new_min;
				}
			}
		}

		//Get the maximum rows and columns of the spots in the aligned average image
		int col_max = 0, row_max = 0, col_min = INT_MAX, row_min = INT_MAX;
        #pragma omp parallel for reduction(max:col_max), reduction(max:row_max), reduction(min:col_min), reduction(min:row_min)
		for (int i = 0; i < spot_pos.size(); i++)
		{
			//Minimum spot position column
			if (spot_pos[i].x < col_min)
			{
				col_min = spot_pos[i].x;
			}
			else
			{
				//Maximum spot position column
				if (spot_pos[i].x > col_max)
				{
					col_max = spot_pos[i].x;
				}
			}

			//Minimum spot position row
			if (spot_pos[i].y < row_min)
			{
				row_min = spot_pos[i].y;
			}
			else
			{
				//Maximum spot position row
				if (spot_pos[i].y > row_max)
				{
					row_max = spot_pos[i].y;
				}
			}
		}

		//Factor to scale positions by so that individual surveys will be positions proportionally their spot's position in the aligned pattern
		float pos_scaler = (2*radius + std::max(rows_diff, cols_diff)) / min_sep;
		
		//Use maximum separation of spots to find size of final image
		int maps_cols = std::ceil(pos_scaler*(row_max - row_min)) + 2*radius + cols_diff;
		int maps_rows = std::ceil(pos_scaler*(col_max - col_min)) + 2*radius + rows_diff;
		
		//Create atlas
		cv::Mat raw_atlas = cv::Mat(maps_cols, maps_rows, CV_32FC1, cv::Scalar(0.0));
		
		//Position each survey proportionally to its spot position in the amalgamation
		for (int k = 0; k < spot_pos.size(); k++) {
		
			//Calculate where to position the survey
			int row = (int)(pos_scaler*(spot_pos[k].y - row_min));
			int col = (int)(pos_scaler*(spot_pos[k].x - col_min));
		
			//Compend the survey in the atlas
			cv::Rect roi = cv::Rect(col, row, 2*radius+rows_diff, 2*radius+cols_diff);		
			surveys[k].copyTo(raw_atlas(roi));
		}

		return raw_atlas;
	}
}