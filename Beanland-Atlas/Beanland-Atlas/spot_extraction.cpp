#include <beanland_atlas.h>

namespace ba
{
	/*Combine the k spaces mapped out by spots in each of the images create maps of the whole k space navigated by that spot.
	**Individual maps are summed together. The total map is then divided by the number of spot k space maps contributing to 
	**each px in the total map.
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual images to extract spots from
	**spot_pos: std::vector<cv::Point>, Positions of located spots in aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images
	**radius: const int, Radius about the spot locations to extract pixels from
	**Returns:
	**std::vector<cv::Mat>, k space mapped out by each spot
	*/
	std::vector<cv::Mat> create_spot_maps(std::vector<cv::Mat> &mats, std::vector<cv::Point> &spot_pos, std::vector<std::vector<int>> &rel_pos,
		const int radius)
	{
		//Initialise vectors of OpenCV mats to hold individual paths and number of contributions to those paths
		std::vector<cv::Mat> indv_maps(spot_pos.size());
		std::vector<cv::Mat> indv_num_mappers(spot_pos.size());
	
		//Get the maximum relative rows and columns and the difference between the maximum and minimum rows and columns
		int col_max = 0, row_max = 0, col_min = 0, row_min = 0;
        #pragma omp parallel for reduction(max:col_max), reduction(max:row_max), reduction(min:col_min), reduction(min:row_min)
		for (int i = 0; i < rel_pos[0].size(); i++)
		{
			//Minimum column
			if (rel_pos[0][i] < col_min)
			{
				col_min = rel_pos[0][i];
			}
			else
			{
				//Maximum column
				if (rel_pos[0][i] > col_max)
				{
					col_max = rel_pos[0][i];
				}
			}

			//Minimum row
			if (rel_pos[1][i] < row_min)
			{
				row_min = rel_pos[1][i];
			}
			else
			{
				//Maximum row
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
        #pragma omp parallel for
		for (int j = 0; j < mats.size(); j++) {
		
			//For each spot
			for (int k = 0; k < spot_pos.size(); k++) {

				/*std::cout << spot_pos[k].y << ", " << row_max << ", " << rel_pos[1][j] << ", " << mats[j].rows << std::endl;
				std::cout << spot_pos[k].x << ", " << col_max << ", " << rel_pos[0][j] << ", " << mats[j].cols << std::endl;*/
		
				if (spot_pos[k].y >= row_max-rel_pos[1][j] && spot_pos[k].y < row_max-rel_pos[1][j]+mats[j].rows &&
					spot_pos[k].x >= col_max-rel_pos[0][j] && spot_pos[k].x < col_max-rel_pos[0][j]+mats[j].cols) {
		
					///Mask to extract spot from micrograph
					cv::Mat circ_mask = cv::Mat::zeros(mats[j].size(), CV_8UC1);
		
					//Draw circle at the position of the spot on the mask
					cv::Point circleCenter(spot_pos[k].x-col_max+rel_pos[0][j], spot_pos[k].y-row_max+rel_pos[1][j]);
					cv::circle(circ_mask, circleCenter, radius, cv::Scalar(1), -1, 8, 0);
		
					//Copy the part of the micrograph containing the spot
					cv::Mat imagePart = cv::Mat::zeros(mats[j].size(), mats[j].type());
					mats[j].copyTo(imagePart, circ_mask);
		
					//Compend spot to map
					float *r;
					ushort *s;
					ushort *t;
					byte *u;
					for (int m = 0; m < indv_num_mappers[k].rows; m++) {
		
						r = indv_maps[k].ptr<float>(m);
						s = indv_num_mappers[k].ptr<ushort>(m);
						t = imagePart.ptr<ushort>(m);
						u = circ_mask.ptr<byte>(m);
						for (int n = 0; n < indv_num_mappers[k].cols; n++) {
		
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
			for (int m = 0; m < indv_num_mappers[k].rows; m++) {
		
				r = indv_maps[k].ptr<float>(m);
				s = indv_num_mappers[k].ptr<ushort>(m);
				for (int n = 0; n < indv_num_mappers[k].cols; n++) {
		
					//Divide pixels contributed to by the number of contributing pixels
					if (s[n]) {
						r[n] /= s[n];
					}
				}
			}
		}
		
		//Crop maps so that they only contain the paths mapped out by the spots
		std::vector<cv::Mat> cropped_maps(spot_pos.size());
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
			cropped_maps[k] = cv::Mat(cols_diff+2*radius, rows_diff+2*radius, CV_32FC1, cv::Scalar(0.0));
			indv_maps[k](roi_map).copyTo(cropped_maps[k](roi_crop));
		}

		for (int i = 0; i < cropped_maps.size(); i++)
		{
			double max;
			cv::minMaxLoc(cropped_maps[i], NULL, &max, NULL, NULL);

			std::cout << max << std::endl;

			display_CV(cropped_maps[i], 2e-3);
		}

		return cropped_maps;
	}
}