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
				survey_angles.push_back(dy > 0 ? angle : 2*PI-angle);
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
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**threshold: float, Maximum proportion of the distance between the brightest spot and the spot least distant from it that a spot can be
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym identify_symmetry(std::vector<cv::Mat> &surveys, std::vector<cv::Point> &spot_pos, float threshold, bool cascade)
	{
		//Get the indices of the surveys to compare to identify the atlas symmetry and the angles of the spots used to create surveys
		//relative to a horizontal line drawn through the brightest spot
		struct equidst_surveys equidst = equidistant_surveys(spot_pos, threshold);

		//Order the indices into order order of increasing position angle from the horizontal
		order_indices_by_angle(equidst.angles, equidst.indices);

		//Rotate the surveys so that they all have the same angle relative to a horizontal line drawn through the brightest spot
		std::vector<cv::Mat> rot_to_align = rotate_to_align(surveys, equidst.angles, equidst.indices);

		//Get the largest non-black regions in the rotated images
		//std::vector<cv::Rect> big_not_black = biggest_not_blacks(rot_to_align);

		//Special case based on the number of spots equidistant from the straight through beam
		if (equidst.indices.size() == 2)
		{
			return atlas_sym_2(rot_to_align, spot_pos, cascade);
		}
		else
		{
			if (equidst.indices.size() == 3)
			{
				return atlas_sym_3(rot_to_align, spot_pos, cascade);
			}
			else
			{
				if (equidst.indices.size() == 4)
				{
					return atlas_sym_4(rot_to_align, spot_pos, cascade);
				}
				else
				{
					return atlas_sym_6(rot_to_align, spot_pos, cascade);
				}
			}
		}
	}

	/*Calculate the symmetry of a 2 survey atlas
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be slightly faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym atlas_sym_2(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Point> &spot_pos, bool cascade)
	{
		atlas_sym a;

		return a;
	}

	/*Calculate the symmetry of a 3 survey atlas
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be slightly faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym atlas_sym_3(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Point> &spot_pos, bool cascade)
	{
		atlas_sym a;

		return a;
	}

	/*Calculate the symmetry of a 4 survey atlas
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be slightly faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym atlas_sym_4(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Point> &spot_pos, bool cascade)
	{
		//Mirror symmetry Pearson normalised product moment coefficient spectrums between surveys
		std::vector<float> mirror_between = get_mirror_between_sym(rot_to_align);

		for (int i = 0; i < mirror_between.size(); i++)
		{
			std::cout << mirror_between[i] << std::endl;
		}

		std::getchar();

		atlas_sym a;

		return a;
	}

	/*Calculate the symmetry of a 6 survey atlas
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be slightly faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym atlas_sym_6(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Point> &spot_pos, bool cascade)
	{
		atlas_sym a;

		return a;
	}

	/*Order indices in order of increasing angle from the horizontal
	**Inputs:
	**angles: std::vector<float> &, Angles between the survey spots and a line drawn horizontally through the brightest spot
	**indices_orig: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	*/
	void order_indices_by_angle(std::vector<float> &angles, std::vector<int> &indices)
	{
		//Structure to combine the angles and indices vectors so that they can be coupled for sorting
		struct angl_idx {
			float angle;
			int idx;
		};

		//Custom comparison of coupled elements based on angle
		struct by_angle { 
			bool operator()(angl_idx const &a, angl_idx const &b) { 
				return a.angle < b.angle;
			}
		};

		//Populate the coupled structure
		std::vector<angl_idx> angls_idxs(angles.size());
		for (int i = 0; i < angles.size(); i++)
		{
			angls_idxs[i].angle = angles[i];
			angls_idxs[i].idx = indices[i];
		}

		//Use the custom comparison to sort the coupled elements
		std::sort(angls_idxs.begin(), angls_idxs.end(), by_angle());

		//Replace the contents of the original vectors with the ordered vlues
		for (int i = 0; i < angles.size(); i++)
		{
			angles[i] = angls_idxs[i].angle ;
			indices[i] = angls_idxs[i].idx;
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
		//Center of rotation
		cv::Point2f pt(src.cols/2., src.rows/2.);

		//Rotation matrix
		cv::Mat rot = getRotationMatrix2D(pt, angle, 1.0);

		//Determine bounding rectangle
		cv::Rect bbox = cv::RotatedRect(pt, src.size(), angle).boundingRect();

		//Adjust the transformation matrix
		rot.at<double>(0,2) += bbox.width/2.0 - pt.x;
		rot.at<double>(1,2) += bbox.height/2.0 - pt.y;

		//Rotate the image
		cv::Mat dst;
		cv::warpAffine(src, dst, rot, bbox.size());

		return dst;
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
			rot_to_align[i] = rotate_CV(surveys[indices[i]], RAD_TO_DEG*angles[i]);
		}

		return rot_to_align;
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
	**Returns:
	**std::vector<float>, Pearson normalised product moment correlation coefficients and relative row and column shift of the second 
	**image, in that order
	*/
	std::vector<float> get_mirror_between_sym(std::vector<cv::Mat> &rot_to_align)
	{
		//Use the Hockey-Stick formula to get the number of comparisons
		int num_comp = (rot_to_align.size()+1) * rot_to_align.size() * (rot_to_align.size()-1) / 3;

		//Assign vectors to hold the symmetry parameters
		std::vector<float> sym_param(2);
		std::vector<float> pearson(num_comp); //Pearson normalised product moment correlation coefiicients
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

				std::vector<float> quant_sym = quantify_rel_shift(rot_to_align[i], mir);

				//Restrict range of larger region of interest dimensions so that they are the same size as the smaller dimensions
				//cv::Point2d max_phase_corr = cv::phaseCorrelate(rot_to_align[i](resized_rois[0]), mir(resized_rois[1]));



				////Create 2x3 warp matrix
				//cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);

				//// Specify the number of iterations.
				//int number_of_iterations = 5000;

				//// Specify the threshold of the increment
				//// in the correlation coefficient between two iterations
				//double termination_eps = 1e-10;

				//cv::Mat x1, x2;
				//rot_to_align[i](resized_rois[0]).convertTo(x1, CV_8UC1);
				//mir(resized_rois[1]).convertTo(x2, CV_8UC1);
				//cv::Mat trans = estimateRigidTransform(
				//	x1,
				//	x2,
				//	true
				//);


				//// Define termination criteria
				//cv::TermCriteria criteria (cv::TermCriteria::COUNT+cv::TermCriteria::EPS, number_of_iterations, termination_eps);

				//// Run the ECC algorithm. The results are stored in warp_matrix.
				//cv::findTransformECC(
				//	rot_to_align[i](resized_rois[0]),
				//	mir(resized_rois[1]),
				//	warp_matrix,
				//	cv::MOTION_TRANSLATION,
				//	criteria
				//);

				//std::cout << warp_matrix << std::endl;

				//// Storage for warped image.
				//Mat im2_aligned;

				//if (warp_mode != MOTION_HOMOGRAPHY)
				//	// Use warpAffine for Translation, Euclidean and Affine
				//	warpAffine(im2, im2_aligned, warp_matrix, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);
				//else
				//	// Use warpPerspective for Homography
				//	warpPerspective (im2, im2_aligned, warp_matrix, im1.size(),INTER_LINEAR + WARP_INVERSE_MAP);


				//std::cout << roi << " # " << roi_mir << std::endl;
				//std::cout << max_phase_corr.x << ", " << max_phase_corr.y << std::endl;
				//display_CV(blur_by_size(black_to_mean(rot_to_align[i](resized_rois[0])), 0.1), 1e-3);
				//display_CV(blur_by_size(black_to_mean(mir(resized_rois[1])), 0.1), 1e-3);

				////Account for offset difference of the regions of interest
				//max_phase_corr.x += roi_mir.x - roi.x;
				//max_phase_corr.y += roi_mir.y - roi.y;

				////Record the location of the maximum phase correlation
				//sym_centres[k] = max_phase_corr.y;

				////Calculate the phase correlation for this offset, including the offset of the regions of interest themselves
				//pearson[k] = pearson_corr(rot_to_align[i], mir, max_phase_corr.x, max_phase_corr.y);

				//std::cout << "pearson: " << pearson[k] << std::endl;
			}
		}

		/*sym_param[0] = pearson;
		sym_param[1] = sym_centres;
*/
		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys when rotated to the positions of the other
	**surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_rotational_between_sym(std::vector<cv::Mat> &rot_to_align)
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
				//Get the position of the maximum phase correlation between the rotated image and its partner
				cv::Point2d max_phase_corr = cv::phaseCorrelate(rot_to_align[i], rot_to_align[j]);

				//Record the location of the maximum phase correlation
				sym_centres[k] = 1/*0.5*(rot_to_align[i].rows - (max_phase_corr.y + same_sized_roi[1].y-same_sized_roi[0].y))*/;

				//Calculate the phase correlation for this offset, including the offset of the regions of interest themselves
				rotation_between[k] = pearson_corr(rot_to_align[i], rot_to_align[j], max_phase_corr.x, max_phase_corr.y);
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
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_rotational_in_sym(std::vector<cv::Mat> &rot_to_align)
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

			//Get the position of the maximum phase correlation between the rotated image and itself
			cv::Point2d max_phase_corr = cv::phaseCorrelate(rot_to_align[i], rot_180);

			//Record the location of the maximum phase correlation
			sym_centres[i] = 1/*0.5*(rot_to_align[i].rows - (max_phase_corr.y + same_sized_roi[1].y-same_sized_roi[0].y))*/;

			//Calculate the phase correlation for this offset, including the offset of the regions of interest themselves
			rotation_in[i] = pearson_corr(rot_to_align[i], rot_180, max_phase_corr.x, max_phase_corr.y);
		}

		sym_param[0] = rotation_in;
		sym_param[1] = sym_centres;

		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients for mirror rotational symmetry inside the surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<float>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_mir_rot_in_sym(std::vector<cv::Mat> &rot_to_align, float mir_row)
	{
		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(2);
		std::vector<float> rot_mir_in(rot_to_align.size()); //Pearson normalised product moment correlation coefiicients
		std::vector<float> sym_centres(rot_to_align.size()); //position of the mirror symmetry phase correlation maximum

		////Look for rotational symmetry in each matrix
		//for (int i = 0; i < rot_to_align.size(); i++)
		//{
		//	//Partition matrix into 2 parts, along the known mirror line
		//	cv::Rect roi_left = cv::Rect(0, 0, (int)mir_row, rot_to_align[i].cols);
		//	cv::Rect roi_right = cv::Rect(mir_row > (int)mir_row ? std::ceil(mir_row) : (int)mir_row+1, 0,
		//		rot_to_align[i].rows - (int)mir_row, rot_to_align[i].cols);

		//	//Rotate the rightmost region of interest 180 degrees
		//	cv::Mat rot_180;
		//	cv::flip(rot_to_align[i](roi_right), rot_180, -1);

		//	//Mirror the rotated region of interest
		//	cv::Mat mir_rot;
		//	cv::flip(rot_180, mir_rot, 1);

		//	//Calculate the largest non-black regions of interest for both regions
		//	cv::Rect roi_left_bnb = biggest_not_black(rot_to_align[i](roi_left));
		//	cv::Rect mir_rot_bnb = biggest_not_black(mir_rot);

		//	//Find the largest possible roi that fits in both images
		//	std::vector<cv::Rect> same_size_rects = same_size_rois(roi_left_bnb, mir_rot_bnb);

		//	//Get the position of the maximum phase correlation between the rotated image and itself
		//	cv::Point2d max_phase_corr = cv::phaseCorrelate(rot_to_align[i](same_size_rects[0]), mir_rot(same_size_rects[1]));

		//	//Record the location of the maximum phase correlation
		//	sym_centres[i] = 1/*0.5*(rot_to_align[i].rows - (max_phase_corr.y + same_sized_roi[1].y-same_sized_roi[0].y))*/;

		//	//Calculate the phase correlation for this offset, including the offset of the regions of interest themselves
		//	rot_mir_in[i] = pearson_corr(rot_to_align[i](roi_left_bnb), mir_rot(mir_rot_bnb), max_phase_corr.x + 
		//		same_size_rects[1].x-same_size_rects[0].x, max_phase_corr.y + same_size_rects[1].y-same_size_rects[0].y);
		//}

		sym_param[0] = rot_mir_in;
		sym_param[1] = sym_centres;

		return sym_param;
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
		std::vector<int> col_thick(img.cols);
		std::vector<int> col_top(img.cols);
		std::vector<int> col_bot(img.cols);
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

			col_thick[i] = j_bot > j_top ? j_bot - j_top + 1 : 0;
			col_top[i] = j_top;
			col_bot[i] = j_bot;
		}

		//Work inwards from the top and bottom of every row to find the length of the non-black region
		std::vector<int> row_thick(img.rows);
		std::vector<int> row_top(img.rows);
		std::vector<int> row_bot(img.rows);
		for (int i = 0; i < img.rows; i++)
		{
			//From the top of the image
			int j_top;
			for (j_top = 0; j_top < img.cols; j_top++)
			{
				if (img.at<float>(i, j_top))
				{
					break;
				}
			}

			//From the bottom of the image
			int j_bot;
			for (j_bot = img.cols-1; j_bot >= 0; j_bot--)
			{
				if (img.at<float>(i, j_bot))
				{
					break;
				}
			}

			row_thick[i] = j_bot > j_top ? j_bot - j_top + 1 : 0;
			row_top[i] = j_top;
			row_bot[i] = j_bot;
		}

		//Find the combination that has the highest area
		int max_area = -1;
		int max_row, max_col, max_rows, max_cols;
		for (int i = 0; i < img.rows; i++)
		{
			//For each non-black region of rows with non-zero thickness, find the column of minimum thickness
			if(row_thick[i])
			{
				int min_thick_vert_idx = std::distance(col_thick.begin(),
					std::min_element(col_thick.begin() + row_top[i], col_thick.begin() + row_bot[i]));

				if(col_thick[min_thick_vert_idx])
				{
					//Find the minimum row thickness for the rowspan of this row of minimum thickness
					int min_thick_horiz_idx = std::distance(row_thick.begin(),
						std::min_element(row_thick.begin() + col_top[min_thick_vert_idx], row_thick.begin() + col_bot[min_thick_vert_idx]));

					////Find the differences between the top and bot columns of the rows at the top and bottom of this column of
					////minimum thickness
					//int top_row_thick = row_thick[col_top[min_thick_vert_idx]];
					//int bot_row_thick = row_thick[col_bot[min_thick_vert_idx]];



					//Check if the area is higher than the maximum...
					int area = row_thick[min_thick_horiz_idx]*col_thick[min_thick_vert_idx];
					if (area > max_area)
					{
						//...and update the roi parameters if it is the largest
						max_area = area;
						max_row = row_top[min_thick_horiz_idx];
						max_col = col_top[min_thick_vert_idx];
						max_rows = row_thick[min_thick_horiz_idx];
						max_cols = col_thick[min_thick_vert_idx];
					}
				}
			}
		}

		return cv::Rect(max_row, max_col, max_rows, max_cols);
	}

	/*Get the shift of a second image relative to the first
	**Inputs:
	**img1: cv::Mat &, One of the images
	**img1: cv::Mat &, The second image
	**blur_frac: const float, Fraction od image to base Gaussian blurring kernel standard deviation on
	**Returns:
	**std::vector<float>, Pearson normalised product moment correlation coefficients and relative row and column shift of the second 
	**image, in that order
	*/
	std::vector<float> quantify_rel_shift(cv::Mat &img1, cv::Mat &img2, const float blur_frac)
	{
		//Trim padding from the survey region of interest
		cv::Rect roi1 = biggest_not_black(img1);

		//Trim the padding from the second survey region of interest
		cv::Rect roi2 = biggest_not_black(img2);

		//Resize the rois so that they share the smallest rows and smallers columns
		std::vector<cv::Rect> resized_rois = same_size_rois(roi1, roi2);

		//Gaussian blur the regions of interest
		cv::Mat blur1 = blur_by_size(img1(resized_rois[0]), blur_frac);
		cv::Mat blur2 = blur_by_size(img2(resized_rois[1]), blur_frac);

		display_CV(blur1, 1e-3);
		display_CV(blur2, 1e-3);

		//Speed up this process by only considering pixels of low ssd later...

		cv::Mat a = ssd(blur1, blur2);

		display_CV(a, 1e-5);

		//float frac = 0.3;

		//int i_pear, j_pear;
		//float max_pearson = -1.0;
		//for (int i = -frac*blur1.rows; i < frac*blur1.rows; i++)
		//{
		//	for (int j = -frac*blur1.cols; j < frac*blur1.cols; j++)
		//	{
		//		//Calculate the Pearson normalised produce moment correlation coefficient for this offset
		//		float pear_corr = pearson_corr(img1(roi1), img2(roi2), i, j);
		//		if (pear_corr > max_pearson)
		//		{
		//			max_pearson = pear_corr;
		//			i_pear = i;
		//			j_pear = j;
		//		}
		//	}
		//}

		//std::cout << i_pear << ", " << j_pear << ", " << max_pearson << std::endl;
		//display_CV(blur1, 1e-3);
		//display_CV(blur2, 1e-3);
	}
}