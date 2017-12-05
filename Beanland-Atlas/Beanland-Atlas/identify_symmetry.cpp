#include <beanland_atlas.h>

namespace ba
{
	/*Get the surveys made by spots equidistant from the central spot in the aligned images average px values diffraction pattern. These 
	**will be used to identify the atlas symmetry
	**Inputs:
	**positions: std::vector<cv::Point> &, Relative positions of the individual spots in the aligned images average px values diffraction 
	**pattern
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
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym identify_symmetry(std::vector<cv::Mat> &surveys, std::vector<float> &angles, std::vector<int> &indices, 
		std::vector<cv::Point> &spot_pos, bool cascade)
	{
		//Order the indices into order order of increasing position angle from the horizontal
		std::vector<int> ordered_indices = order_indices_by_angle(angles, indices);

		//Rotate the surveys so that they all have the same angle relative to a horizontal line drawn through the brightest spot
		std::vector<cv::Mat> rot_to_align = rotate_to_align(surveys, angles, ordered_indices);

		//Get the largest non-black regions in the rotated images
		std::vector<cv::Rect> big_not_black = biggest_not_blacks(rot_to_align);

		//Special case:
		if (indices.size() == 2)
		{
			return alas_sym_2(rot_to_align, big_not_black, spot_pos, cascade);
		}
		else
		{
			if (indices.size() == 3)
			{
				return alas_sym_3(rot_to_align, big_not_black, spot_pos, cascade);
			}
			else
			{
				if (indices.size() == 4)
				{
					return alas_sym_4(rot_to_align, big_not_black, spot_pos, cascade);
				}
				else
				{
					return alas_sym_6(rot_to_align, big_not_black, spot_pos, cascade);
				}
			}
		}

		//Mirror symmetry Pearson normalised product moment coefficient spectrums between surveys
		std::vector<float> mirror_between = get_mirror_between_sym(rot_to_align, big_not_black);

		//Rotational symmetry Pearson normalised product moment coefficient spectrum between surveys
		std::vector<float> rotational_between = get_rotational_between_sym(rot_to_align, big_not_black);

		//Rotational symmetry Pearson normalised product moment coefficient spectrum in surveys
		std::vector<float> rotational_in = get_rotational_in_sym(rot_to_align, big_not_black);
	}

	/*Calculate the symmetry of a 2 survey atlas
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**big_not_black: std::vector<cv::Rect> &, The highest-are non-black region in the survey
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be slightly faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym atlas_sym_2(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Rect> &big_not_black, std::vector<cv::Point> &spot_pos,
		bool cascade)
	{
		//Mirror symmetry Pearson normalised product moment coefficient spectrums between surveys
		std::vector<std::vector<float>> mirror_between = get_mirror_between_sym(rot_to_align, big_not_black);


	}

	/*Calculate the symmetry of a 3 survey atlas
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**big_not_black: std::vector<cv::Rect> &, The highest-are non-black region in the survey
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be slightly faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym atlas_sym_3(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Rect> &big_not_black, std::vector<cv::Point> &spot_pos,
		bool cascade)
	{

	}

	/*Calculate the symmetry of a 4 survey atlas
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**big_not_black: std::vector<cv::Rect> &, The highest-are non-black region in the survey
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be slightly faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym atlas_sym_4(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Rect> &big_not_black, std::vector<cv::Point> &spot_pos,
		bool cascade)
	{

	}

	/*Calculate the symmetry of a 6 survey atlas
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**big_not_black: std::vector<cv::Rect> &, The highest-are non-black region in the survey
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be slightly faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym atlas_sym_6(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Rect> &big_not_black, std::vector<cv::Point> &spot_pos,
		bool cascade)
	{

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
				int min_thick_vert_idx = std::distance(thicknesses.begin(),
					std::min_element(thicknesses.begin() + j_left, thicknesses.end() - j_left));

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

	/*Rotate the surveys so that they are all aligned at the same angle to a horizontal line drawn through the brightest spot
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**angles: std::vector<float> &, Angles between the survey spots and a line drawn horizontally through the brightest spot
	**indices: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	**Returns:
	**std::vector<cv::Mat>, Images rotated so that they are all aligned. 
	*/
	std::vector<cv::Mat> rotate_to_align(std::vector<cv::Mat> &surveys, std::vector<float> &angles, std::vector<int> &indices)
	{
		//Rotate each mat so that they all have the same orientation to a horizontal line drawn through the brightest spot in the aligned patter
		std::vector<cv::Mat> rot_to_align(indices.size());
        #pragma omp parallel for
		for (int i = 0; i < indices.size(); i++)
		{
			rot_to_align[i] = rotate_CV(surveys[indices[i]], 360 - RAD_TO_DEG*angles[i]);
		}

		return rot_to_align;
	}

	/*Find the largest-area regions that are non-black in each of the mat
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**Returns:
	**std::vector<cv::Rect>, Largest non-black region in the mat 
	*/
	std::vector<cv::Rect> biggest_not_blacks(std::vector<cv::Mat> &mats)
	{
		//Find the largest non-black region in each mat
		std::vector<cv::Rect> rois(mats.size());
        #pragma omp parallel for
		for (int i = 0; i < mats.size(); i++)
		{
			rois[i] = biggest_not_black(mats[i]);
		}

		return rois;
	}

	/*Decrease the size of the larger rectangular region of interest so that it is the same size as the smaller
	**Inputs:
	**roi1: cv::Rect, One of the regions of interest
	**roi2: cv::Rect, The other region of interest
	**Returns:
	**std::vector<cv::Rect>, Regions of interest that are the same size, in the same order as the input arguments
	*/
	std::vector<cv::Rect> same_size_rois(cv::Rect roi1, cv::Rect roi2)
	{
		std::vector<cv::Rect> same_size_rects(2);

		//Match up the region of interest rowspans so they are both the smallest
		if (roi1.width > roi2.width)
		{
			roi1.width -= roi1.width - roi2.width;
			roi1.x += (roi1.width - roi2.width) / 2;
		}
		else
		{
			if (roi1.width < roi2.width)
			{
				roi2.width -= roi2.width - roi1.width;
				roi2.x += (roi2.width - roi1.width) / 2;
			}
		}

		//Match up the region of interest columnspans so they are are both the smallest
		if (roi1.height > roi2.height)
		{
			roi1.height -= roi1.height - roi2.height;
			roi1.y += (roi1.height - roi2.height) / 2;
		}
		else
		{
			if (roi1.height < roi2.height)
			{
				roi2.height -= roi2.height - roi1.height;
				roi2.y += (roi2.height - roi1.height) / 2;
			}
		}

		same_size_rects[0] = roi1;
		same_size_rects[1] = roi2;

		return same_size_rects;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys and reflections of the other surveys in
	**a mirror line between the 2 surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**big_not_black: std::vector<cv::Rect> &, The highest-are non-black region in the survey
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_mirror_between_sym(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Rect> &big_not_black)
	{
		//Use the Hockey-Stick formula to get the number of comparisons
		int num_comp = (rot_to_align.size()+1) * rot_to_align.size() * (rot_to_align.size()-1) / 3;

		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(2);
		std::vector<float> mirror_between(num_comp); //Pearson normalised product moment correlation coefiicients
		std::vector<float> sym_centres(num_comp); //position of the mirror symmetry phase correlation maximum

		//Compare each matrix with...
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			//...the higher index matrices
			for (int j = i + 1; j < rot_to_align.size(); j++, k++)
			{
				//Flip the second mat in the comparison
				cv::Mat mir;
				cv::flip(rot_to_align[j], mir, 1);

				//Find the mirrored roi
				cv::Rect mir_roi = cv::Rect(rot_to_align[j].rows - big_not_black[j].x - big_not_black[j].width, 
					rot_to_align[j].cols - big_not_black[j].y - big_not_black[j].height, big_not_black[j].width, big_not_black[j].height);

				//Reduce one of the rois size, if necessary, so that they are the same size
				std::vector<cv::Rect> same_sized_roi = same_size_rois(big_not_black[i], mir_roi);

				//Get the position of the maximum phase correlation between the rotated image and the reflection of its partner
				cv::Point2d max_phase_corr = cv::phaseCorrelate(rot_to_align[i](same_sized_roi[0]), mir(same_sized_roi[1]));

				//Record the location of the maximum phase correlation
				sym_centres[k] = 1/*0.5*(rot_to_align[i].rows - (max_phase_corr.y + same_sized_roi[1].y-same_sized_roi[0].y))*/;

				//Calculate the phase correlation for this offset, including the offset of the regions of interest themselves
				mirror_between[k] = pearson_corr(rot_to_align[i], mir, max_phase_corr.x + 
					same_sized_roi[1].x-same_sized_roi[0].x, max_phase_corr.y + same_sized_roi[1].y-same_sized_roi[0].y);
			}
		}

		sym_param[0] = mirror_between;
		sym_param[1] = sym_centres;

		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys when rotated to the positions of the other
	**surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**big_not_black: std::vector<cv::Rect> &, The highest-are non-black region in the survey
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_rotational_between_sym(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Rect> &big_not_black)
	{
		//Use the Hockey-Stick formula to get the number of comparisons
		int num_comp = (rot_to_align.size()+1) * rot_to_align.size() * (rot_to_align.size()-1) / 3;

		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(2);
		std::vector<float> rotation_between(num_comp); //Pearson normalised product moment correlation coefiicients
		std::vector<float> sym_centres(num_comp); //position of the mirror symmetry phase correlation maximum

		//Compare each matrix with...
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			//...the higher index matrices
			for (int j = i; j < rot_to_align.size(); j++, k++)
			{
				//Reduce one of the rois size, if necessary, so that they are the same size
				std::vector<cv::Rect> same_sized_roi = same_size_rois(big_not_black[i], big_not_black[j]);

				//Get the position of the maximum phase correlation between the rotated image and its partner
				cv::Point2d max_phase_corr = cv::phaseCorrelate(rot_to_align[i](same_sized_roi[0]), rot_to_align[j](same_sized_roi[1]));

				//Record the location of the maximum phase correlation
				sym_centres[k] = 1/*0.5*(rot_to_align[i].rows - (max_phase_corr.y + same_sized_roi[1].y-same_sized_roi[0].y))*/;

				//Calculate the phase correlation for this offset, including the offset of the regions of interest themselves
				rotation_between[k] = pearson_corr(rot_to_align[i], rot_to_align[j], max_phase_corr.x + 
					same_sized_roi[1].x-same_sized_roi[0].x, max_phase_corr.y + same_sized_roi[1].y-same_sized_roi[0].y);
			}
		}

		sym_param[0] = rotation_between;
		sym_param[1] = sym_centres;

		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients for 180 deg rotational symmetry in the surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**big_not_black: std::vector<cv::Rect> &, The highest-are non-black region in the survey
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_rotational_in_sym(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Rect> &big_not_black)
	{
		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(2);
		std::vector<float> rotation_in(rot_to_align.size()); //Pearson normalised product moment correlation coefiicients
		std::vector<float> sym_centres(rot_to_align.size()); //position of the mirror symmetry phase correlation maximum

		//Look for rotational symmetry in each matrix
		for (int i = 0; i < rot_to_align.size(); i++)
		{
			//Rotate the matrix 180 degrees
			cv::Mat rot_180;
			cv::flip(rot_to_align[i], rot_180, -1);

			//Get the largest non-black region of interest in the rotated mat
			cv::Rect rot_180_roi = cv::Rect(rot_to_align[i].rows - big_not_black[i].x - big_not_black[i].width, 
				rot_to_align[i].cols - big_not_black[i].y - big_not_black[i].height, big_not_black[i].width, big_not_black[i].height);

			//Get the position of the maximum phase correlation between the rotated image and itself
			cv::Point2d max_phase_corr = cv::phaseCorrelate(rot_to_align[i](big_not_black[i]), rot_180(rot_180_roi));

			//Record the location of the maximum phase correlation
			sym_centres[i] = 1/*0.5*(rot_to_align[i].rows - (max_phase_corr.y + same_sized_roi[1].y-same_sized_roi[0].y))*/;

			//Calculate the phase correlation for this offset, including the offset of the regions of interest themselves
			rotation_in[i] = pearson_corr(rot_to_align[i], rot_180, max_phase_corr.x + 
				rot_180_roi.x-big_not_black[i].x, max_phase_corr.y + rot_180_roi.y-big_not_black[i].y);
		}

		sym_param[0] = rotation_in;
		sym_param[1] = sym_centres;

		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients for mirror rotational symmetry inside the surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**big_not_black: std::vector<cv::Rect> &, The highest-are non-black region in the survey
	**Returns:
	**std::vector<float>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_mir_rot_in_sym(std::vector<cv::Mat> &rot_to_align, float mir_row)
	{
		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(2);
		std::vector<float> rot_mir_in(rot_to_align.size()); //Pearson normalised product moment correlation coefiicients
		std::vector<float> sym_centres(rot_to_align.size()); //position of the mirror symmetry phase correlation maximum

		//Look for rotational symmetry in each matrix
		for (int i = 0; i < rot_to_align.size(); i++)
		{
			//Partition matrix into 2 parts, along the known mirror line
			cv::Rect roi_left = cv::Rect(0, 0, (int)mir_row, rot_to_align[i].cols);
			cv::Rect roi_right = cv::Rect(mir_row > (int)mir_row ? std::ceil(mir_row) : (int)mir_row+1, 0,
				rot_to_align[i].rows - (int)mir_row, rot_to_align[i].cols);

			//Rotate the rightmost region of interest 180 degrees
			cv::Mat rot_180;
			cv::flip(rot_to_align[i](roi_right), rot_180, -1);

			//Mirror the rotated region of interest
			cv::Mat mir_rot;
			cv::flip(rot_180, mir_rot, 1);

			//Calculate the largest non-black regions of interest for both regions
			cv::Rect roi_left_bnb = biggest_not_black(rot_to_align[i](roi_left));
			cv::Rect mir_rot_bnb = biggest_not_black(mir_rot);

			//Find the largest possible roi that fits in both images
			std::vector<cv::Rect> same_size_rects = same_size_rois(roi_left_bnb, mir_rot_bnb);

			//Get the position of the maximum phase correlation between the rotated image and itself
			cv::Point2d max_phase_corr = cv::phaseCorrelate(rot_to_align[i](same_size_rects[0]), mir_rot(same_size_rects[1]));

			//Record the location of the maximum phase correlation
			sym_centres[i] = 1/*0.5*(rot_to_align[i].rows - (max_phase_corr.y + same_sized_roi[1].y-same_sized_roi[0].y))*/;

			//Calculate the phase correlation for this offset, including the offset of the regions of interest themselves
			rot_mir_in[i] = pearson_corr(rot_to_align[i](roi_left_bnb), mir_rot(mir_rot_bnb), max_phase_corr.x + 
				same_size_rects[1].x-same_size_rects[0].x, max_phase_corr.y + same_size_rects[1].y-same_size_rects[0].y);
		}

		sym_param[0] = rot_mir_in;
		sym_param[1] = sym_centres;

		return sym_param;
	}
}