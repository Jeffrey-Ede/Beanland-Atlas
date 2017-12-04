#include <beanland_atlas.h>

namespace ba
{
	/*Get the surveys made by spots equidistant from the central spot in the aligned images average px values diffraction pattern. These 
	**will be used to identify the atlas symmetry
	**Inputs:
	**positions: , Relative positions of the individual spots in the aligned images average px values diffraction pattern
	**threshold: float, Maximum proportion of the distance between the brightest spot and the spot least distant from it that a spot can be
	**and still be considered to be one of the equidistant spots
	**Returns:
	**struct equidst_surveys, Indices of the spots nearest to and equidistant from the brightest spot and their angles to a horizontal
	**line drawn horizontally through the brightest spot
	*/
	struct equidst_surveys equidistant_surveys(std::vector<cv::Point> &spot_pos, float threshold)
	{
		//Find the minimum distance from the central spot
		int min_dst = INT_MAX;
        #pragma omp parallel for reduction(min:min_dst)
		for (int i = 1; i < spot_pos.size(); i++)
		{
			//Get position relative to the central spot
			int dx = spot_pos[i].x - spot_pos[0].x;
			int dy = spot_pos[i].y - spot_pos[0].y;

			//Check if the distance of the spot is smaller than the minimum
			if (std::sqrt(dx*dx + dy*dy) < min_dst)
			{
				//Make distance the new minimum
				min_dst = std::sqrt(dx*dx + dy*dy);
			}
		}

		//Find the indices of the surveys equidistant from the central spot, within the threshold
		std::vector<int> equidst_surveys;
		std::vector<float> survey_angles;
		for (int i = 1; i < spot_pos.size(); i++)
		{
			//Get position relative to the central spot
			int dx = spot_pos[i].x - spot_pos[0].x;
			int dy = spot_pos[i].y - spot_pos[0].y;

			//Check if the distance of the spot is smaller than the threshold
			if (std::sqrt(dx*dx + dy*dy) < (1.0+threshold)*min_dst)
			{
				//Note the position's index
				equidst_surveys.push_back(i);

				//Get the angle to the horizontal drawn through the brightest spot
				float angle = std::acos(dx / std::sqrt(dx*dx + dy*dy));
				survey_angles.push_back(dy > 0 ? angle : angle+PI);
			}
		}

		struct equidst_surveys survey_param;
		survey_param.indices = equidst_surveys;
		survey_param.angles = survey_angles;

		return survey_param;
	}

	/*Identifies the symmetry of the Beanland Atlas using Fourier analysis and Pearson normalised product moment correlation
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**angles: std::vector<float> &, Angles of spots used to create surveys relative to a horizontal line drawn through the brightest spot
	**indices: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	**Returns:
	**int, Number between 0 and 30 representing the type of symmetry
	*/
	int identify_symmetry(std::vector<cv::Mat> &surveys, std::vector<float> &angles, std::vector<int> &indices)
	{
		//Order the indices into order order of increasing position angle from the horizontal
		std::vector<int> ordered_indices = order_indices_by_angle(angles, indices);

		//Check for outright mirror symmetry between the surveys
		float mir_between = mir_sym_between_surveys(surveys, angles, ordered_indices);

		//Check for rotational symmetry between the surveys
		float rot_between = rot_sym_between_surveys(surveys, ordered_indices);

		//Check for mirror symmetry inside individual surveys
		float mir_in = mir_sym_in_surveys(surveys, ordered_indices);

		//Check for rotational symmetry inside individual surveys
		float rot_in = rot_sym_in_surveys(surveys, ordered_indices);

		//Use the symmetry score to identify the symmetry type

		//Return integer for now. Return more meaningful data later
		return 7;
	}

	/*Order indices in order of increasing angle from the horizontal
	**Inputs:
	**angles: std::vector<float> &, Angles between the survey spots and a line drawn horizontally through the brightest spot
	**indices: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	**Returns:
	**std::vector<int>, Indices in order of increasing angle from the horizontal
	*/
	std::vector<int> order_indices_by_angle(std::vector<float> &angles, std::vector<int> &indices)
	{
		//Bubble sort the indices by increasing angle size
        #pragma omp parallel for
		for (int i = 0; indices.size(); i++) //Brute force: don't check if array is sorted after each iteration
		{
			//Swap pairs of indices in the wrong order
			for (int j = 1; j < indices.size(); j++)
			{
				if (angles[j] < angles[j-1])
				{
					int temp = indices[j];
					indices[j] = indices[j-1];
					indices[j-1] = indices[j];
				}
			}
		}
	}

	/*Calculate the mirror symmetry between adjacent surveys by phase correlating their reflections and then calculating their Pearson normalised product
	**moment correlation coefficient. Return the lowest score
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**indice: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	**Returns:
	**float, Minimum Pearson normalised product moment correlation coefficient for mirror symmetry between surveys
	*/
	float mir_sym_between_surveys(std::vector<cv::Mat> &surveys, std::vector<float> &angles, std::vector<int> &indices)
	{
		//Average highest correlation
		float pearson_corr = 0.0f;

		//Compare each matrix with...
        #pragma omp parallel for reduction(sum:pearson_corr)
		for (int i = 0; i < surveys.size(); i++)
		{
			//...the reflections of the next matrices
			for (int j = i + 1; j < surveys.size(); j++)
			{
				//Find the highest possible Pearson correlation between the image and its reflection
				pearson_corr += highest_mir_sym_between(surveys[indices[i]], surveys[indices[j]], 0.5*(angles[i] + angles[j]));
			}
		}

		//Use the Hockey-Stick formula to get the number of comparisons to divide the sum by to get the mean
		pearson_corr *= 3 /((surveys.size()+1) * surveys.size() * (surveys.size()-1));

		
		return sym;
	}

	/*Calculate the rotational symmetry between the surveys by phase correlating their rotations and then calculating their Pearson normalised product
	**moment correlation coefficient. Return the lowest score
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**indice: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	**Returns:
	**float, Minimum Pearson normalised product moment correlation coefficient for rotational symmetry between surveys
	*/
	float rot_sym_between_surveys(std::vector<cv::Mat> &surveys, std::vector<int> &indices)
	{
		float sym;
		return sym;
	}

	/*Calculate the mirror symmetry in the surveys by phase correlating their reflections and then calculating their Pearson normalised product
	**moment correlation coefficient. Return the lowest score
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**indice: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	**Returns:
	**float, Minimum Pearson normalised product moment correlation coefficient for mirror symmetry in the surveys
	*/
	float mir_sym_in_surveys(std::vector<cv::Mat> &surveys, std::vector<int> &indices)
	{
		float sym;
		return sym;
	}

	/*Calculate the rotational symmetry in the surveys by phase correlating their rotations and then calculating their Pearson normalised product
	**moment correlation coefficient. Return the lowest score
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**indice: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	**Returns:
	**float, Minimum Pearson normalised product moment correlation coefficient for rotational symmetry in surveys
	*/
	float rot_sym_in_surveys(std::vector<cv::Mat> &surveys, std::vector<int> &indices)
	{
		float sym;
		return sym;
	}

	/*Create mirror image to use when phase correlating it against it's partner. Image is mirrored in a line perpendicular to the one
	**joining it to its partner
	**Inputs:
	**mat: cv::Mat &, Image to reflect and extract part of the mirror from
	**group
	**Returns:
	**cv::Mat, Image reflected in mirror line
	*/
	cv::Mat reflect_mat(cv::Mat &mat)
	{
		cv::Mat reflection;
		return reflection;
	}

	/*Rotates an image keeping the image the same size, embedded in a larger black rectangle
	**Inputs:
	**src: cv::Mat &, Image to rotate
	**angle: float, Angle to rotate the image (anticlockwise)
	**Returns:
	**cv::Mat, Rotated image
	*/
	cv::Mat rotate_CV(cv::Mat src, float angle)
	{
		cv::Mat dst;
		cv::Point2f pt(src.cols/2., src.rows/2.);    
		cv::Mat r = getRotationMatrix2D(pt, angle, 1.0);
		cv::warpAffine(src, dst, r, cv::Size(src.cols*std::cos(angle) + src.rows*std::sin(angle), 
			src.cols*std::sin(angle) + src.rows*std::cos(angle)));
		return dst;
	}

	/*Highest possible Pearson normalise product moment correlation coefficient between a survey and the reflection of a second survey where
	**the surveys are at a known mean angle to the horizontal
	**Inputs:
	**img1: cv::Mat &, One of the surveys
	**img1: cv::Mat &, The other survey
	**angle: float, Mean angle between the 2 survey's spots and a horizontal line going through the brightest spot
	**Returns:
	**float, Highest Pearson product moment correlation coefficient between the the first image and the reflection of the second
	*/
	float highest_mir_sym_between(cv::Mat &img1, cv::Mat img2, float angle)
	{
		//Rotate the images clockwise by their mean angular difference from a horizontal line drawn through the brightest spot
		//This maximises their overlap when the second image is reflected by a horizontal line above it
		cv::Mat rot1 = rotate_CV(img1, 360 - RAD_TO_DEG*angle);

		//Rotate the second image. Then flip the second image's columns to mirror it
		cv::Mat mir2;
		cv::flip(rotate_CV(img2, 360 - RAD_TO_DEG*angle), mir2, 1);

		//Get the biggest portions of the images that are not black background
		cv::Rect rot1_roi = biggest_not_black(rot1);
		cv::Rect mir2_roi = biggest_not_black(mir2);

		//Match up the region of interest rowspans so they are both the smallest
		if (rot1_roi.width > mir2_roi.width)
		{
			rot1_roi.width -= rot1_roi.width - mir2_roi.width;
			rot1_roi.x += (rot1_roi.width - mir2_roi.width) / 2;
		}
		else
		{
			if (rot1_roi.width < mir2_roi.width)
			{
				mir2_roi.width -= mir2_roi.width - rot1_roi.width;
				mir2_roi.x += (mir2_roi.width - rot1_roi.width) / 2;
			}
		}

		//Match up the region of interest columnspans so they are are both the smallest
		if (rot1_roi.height > mir2_roi.height)
		{
			rot1_roi.height -= rot1_roi.height - mir2_roi.height;
			rot1_roi.y += (rot1_roi.height - mir2_roi.height) / 2;
		}
		else
		{
			if (rot1_roi.height < mir2_roi.height)
			{
				mir2_roi.height -= mir2_roi.height - rot1_roi.height;
				mir2_roi.y += (mir2_roi.height - rot1_roi.height) / 2;
			}
		}

		//Get the position of the maximum phase correlation between the rotated image and the reflection of its partner
		cv::Point2d max_phase_corr = cv::phaseCorrelate(rot1(rot1_roi), mir2(mir2_roi));

		//Calculate the phase correlation for this offset, including the offset of the regions of interest themselves

		//Get Find the Pearson normalised product moment coefficient for the alignment of maximum phase correlation
		return ;
	}

	/*Get the largest rectangular portion of an image inside a surrounding black background
	**Inputs:
	**img: cv::Mat &, Input floating point image to extract the largest possible non-black region of
	**Returns:
	**cv::Rect, Largest non-black region
	*/
	cv::Rect biggest_not_black(cv::Mat &img)
	{
		//Work inwards from the top and bottom of every column to find the length of the non-black region
		std::vector<int> thicknesses(img.cols);
        #pragma omp parallel for
		for (int i = 0; i < img.cols; i++)
		{
			//From the top of the image
			int j_top;
			for (j_top = 0; j_top < img.rows; j_top++)
			{
				if (img.at<float>(j_top, i))
				{
					break;
				}
			}

			//From the bottom of the image
			int j_bot;
			for (j_bot = img.rows-1; j_bot >= 0; j_bot--)
			{
				if (img.at<float>(j_bot, i))
				{
					break;
				}
			}

			thicknesses[i] = img.rows - (j_bot - j_top) + 1;
		}

		//Work inwards from the top and bottom of every row to find the length of the non-black region. Record the one whose
		//product with the row non-black length has the highest value
		int max_area = -1;
		int max_row, max_col, max_rows, max_cols;
		for (int i = 0; i < img.rows; i++)
		{
			//From the left of the image
			int j_left;
			for (j_left = 0; j_left < img.rows; j_left++)
			{
				if (img.at<float>(i, j_left))
				{
					break;
				}
			}

			//From the right of the image
			int j_right;
			for (j_right = img.cols-1; j_right >= 0; j_right--)
			{
				if (img.at<float>(i, j_right))
				{
					break;
				}
			}

			int thickness = img.cols - (j_right - j_left) + 1;

			//Check if this produces the highest area
			if(j_left < j_right) //Safeguard against the entire row being black
			{
				int min_thick_vert_idx = std::distance(thicknesses.begin(), std::min_element(thicknesses.begin() + j_left, thicknesses.end() - j_left));

				if (thickness * thicknesses[min_thick_vert_idx] > max_area)
				{
					max_area = thickness * thicknesses[min_thick_vert_idx];
					max_row = i;
					max_col = j_left;
					max_rows = thickness;
					max_cols = thicknesses[min_thick_vert_idx];
				}
			}
		}

		return cv::Rect(max_row, max_col, max_rows, max_cols);
	}
}