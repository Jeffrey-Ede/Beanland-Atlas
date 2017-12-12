#include <beanland_atlas.h>

namespace ba
{
	/*Display C++ API ArrayFire array
	**Inputs:
	**arr: af::array &, ArrayFire C++ API array to display
	**scale: float, Multiply the array elements by this value before displaying
	**plt_name: char *, Optional name for plot
	**dim0: int, Optional image side size
	**dim1: int, Optional image side size
	*/
	void display_AF(af::array &arr, float scale, char * plt_name, int dim0, int dim1)
	{
		af::Window window(dim0, dim1, plt_name);
		do
		{
			window.image(arr.as(f32)*scale);
		} while( !window.close() );
	}

	/*Display C API ArrayFire array
	**Inputs:
	**arr: af_array &, ArrayFire C API array to display
	**scale: float, Multiply the array elements by this value before displaying
	**plt_name: char *, Optional name for plot
	**dim0: int, Optional image side size
	**dim1: int, Optional image side size
	*/
	void display_AF(af_array &arr, float scale, char * plt_name, int dim0, int dim1)
	{
		af::Window window(dim0, dim1, plt_name);
		do
		{
			window.image(af::array(arr).as(f32)*scale);
		} while( !window.close() );
	}

	/*Display OpenCV mat
	**Inputs:
	**mat: cv::Mat &, OpenCV mat to display
	**scale: float, Multiply the mat elements by this value before displaying
	**norm: bool, If true, min-max normalise the mat before displaying it with values in the range 0-255
	**plt_name: char *, Optional name for plot
	*/
	void display_CV(cv::Mat &mat, float scale, bool norm, char * plt_name)
	{
		//Set up the window to be resizable while keeping its aspect ratio
		cv::namedWindow( plt_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO );

		//Normalise
		if (norm)
		{
			//Min-max normalise the mat
			cv::Mat norm;
			cv::normalize(mat, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

			//Show the OpenCV mat
			cv::imshow( plt_name, norm );
			cv::waitKey(0);
		}
		//Don't normalise (default)
		else
		{
			//Show the OpenCV mat
			cv::imshow( plt_name, mat*scale );
			cv::waitKey(0);
		}
	}

	/*Print the size of the first 2 dimensions of a C++ API ArrayFire array
	**Input:
	**arr: af::array &, Arrayfire C++ API array
	*/
	void print_AF_dims(af::array &arr)
	{
		printf("dims = [%lld %lld]\n", arr.dims(0), arr.dims(1));
	}

	/*Print the size of the first 2 dimensions of a C API ArrayFire array
	**Input:
	**arr: af_array &, Arrayfire C API array
	*/
	void print_AF_dims(af_array &arr)
	{
		printf("dims = [%lld %lld]\n", af::array(arr).dims(0), af::array(arr).dims(1));
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