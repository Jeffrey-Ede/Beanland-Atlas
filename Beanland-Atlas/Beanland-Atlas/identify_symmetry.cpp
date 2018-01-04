#include <identify_symmetry.h>

namespace ba
{
	/*Identifies the symmetry of the Beanland Atlas using Fourier analysis and Pearson normalised product moment correlation
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**threshold: float, Maximum proportion of the distance between the brightest spot and the spot least distant from it that a spot can be
	**frac_for_sym: const float, Mean Pearson product moment correlation coefficients for each symmetry must be at least this fraction of 
	**the maximum Pearson coefficient for that symmetry to be considered present
	**Returns:
	**atlas_sym, Atlas symmetries
	*/
	atlas_sym identify_symmetry(std::vector<cv::Mat> &surveys, std::vector<cv::Point> &spot_pos, const float threshold,
		const float frac_for_sym)
	{
		//Get the indices of the surveys to compare to identify the atlas symmetry and the angles of the spots used to create surveys
		//relative to a horizontal line drawn through the brightest spot
		std::vector<int> indices; 
		std::vector<float> angles;
		equidistant_surveys(spot_pos, threshold, indices, angles);

		//Order the indices into order order of increasing position angle from the horizontal
		order_indices_by_angle(angles, indices);

		//Rotate the surveys so that they all have the same angle relative to a horizontal line drawn through the brightest spot
		std::vector<cv::Mat> rot_to_align = rotate_to_align(surveys, angles, indices);

		//Mirror symmetry Pearson normalised product moment coefficient spectrums between surveys
		std::vector<std::vector<float>> mirror_between = get_mirror_between_sym(rot_to_align);

		//Mirror symmetry Pearson normalised product moment coefficient spectrums in surveys
		std::vector<std::vector<float>> mirror_in0 = get_mirror_in_sym(rot_to_align);

		//Mirror radially outwards symmetry Pearson normalised product moment coefficient spectrums in surveys
		std::vector<std::vector<float>> mirror_in1 = get_mirror_in_rad_sym(rot_to_align);

		//Rotation symmetry Pearson normalised product moment coefficient spectrums between surveys
		std::vector<std::vector<float>> rot_between = get_rotational_between_sym(rot_to_align);

		//Caclulate the average mirror symmetry quantifications between spots
		float mean_mir_between = 0.0f;
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			for (int j = i+1; j < rot_to_align.size(); j++, k++)
			{
				mean_mir_between += mirror_between[k][0];
			}
		}
		mean_mir_between /= rot_to_align.size() * (rot_to_align.size()-1) / 2;

		//Caclulate the average mirror symmetry quantifications in spots for mirroring perpendicularly to the radially outwards direction
		float mean_mir_in0 = 0.0f;
		for (int i = 0; i < rot_to_align.size(); i++)
		{
			mean_mir_in0 += mirror_in0[i][0];
		}
		mean_mir_in0 /= rot_to_align.size();

		//Caclulate the average mirror symmetry quantifications in spots for mirroring radially outwards
		float mean_mir_in1 = 0.0f;
		for (int i = 0; i < rot_to_align.size(); i++)
		{
			mean_mir_in1 += mirror_in1[i][0];
		}
		mean_mir_in1 /= rot_to_align.size();

		//Calculate the average symmetry quantification for spots multiples of 2; not 1, apart
		float mean_mir_between_2 = 0.0f; //Only present for 4 or 6 equidistant spots
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			for (int j = i+2; j < rot_to_align.size(); j += 2, k += 2)
			{
				mean_mir_between_2 += mirror_between[k][0];
			}
		}
		mean_mir_between_2 /= rot_to_align.size() * (rot_to_align.size()-1) / 4;

		//Mean rotational symmetry between surveys quantification
		float mean_rot_between = 0.0f;
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			for (int j = i+1; j < rot_to_align.size(); j++, k++)
			{
				mean_rot_between += rot_between[k][0];
			}
		}
		mean_rot_between /= rot_to_align.size() * (rot_to_align.size()-1) / 2;

		//Decide which symmetries are present
		float max_pear = std::max(mean_mir_in0, std::max(mean_mir_between, std::max(mean_mir_between_2, mean_rot_between)));

		//Record the symmetries present. By index: 0 - mir_in, 1 - mir_between, 2 - mir_between_2, 3 - rot_between, 4 - rot_in, 
		//5 - mir_in_rad (mirrored outwards, in radial direction)
		std::vector<bool> symmetries(6);
		symmetries[0] = mean_mir_in0 > frac_for_sym * max_pear;
		symmetries[1] = mean_mir_between > frac_for_sym * max_pear;
		symmetries[2] = symmetries[1] || mean_mir_between_2 > frac_for_sym * max_pear;
		symmetries[3] = mean_rot_between > frac_for_sym * max_pear;

		//Give the other symmetries false placeholders
		symmetries[4] = false;
		symmetries[5] = false;

		//Get an initial estimate of the symmetry centres to guide the internal rotational symmetry quantification
		std::vector<std::vector<float>> rot_in;
		std::vector<cv::Point2f> est_sym_centers = get_sym_centers(mirror_between, rot_between, rot_in, symmetries,
			rot_to_align.size(), rot_to_align);

		for (int i = 0; i < est_sym_centers.size(); i++)
		{
			std::cout << est_sym_centers[i] << std::endl;
		}

		//Calculate the internal rotational symmetry Pearson normalised product moment coefficient spectrums in surveys
		rot_in = get_rotational_in_sym(rot_to_align, est_sym_centers);

		//Mean rotational symmetry in surveys quntification
		float mean_rot_in = 0.0f;
		for (int i = 0; i < rot_to_align.size(); i++)
		{
			mean_rot_in += rot_in[i][0];
		}
		mean_rot_in /= rot_to_align.size();

		symmetries[4] = mean_rot_in > frac_for_sym * max_pear;

		std::cout << mean_mir_in0 << ", " << mean_mir_between << ", " << mean_mir_between_2 << ", " << mean_rot_between << ", " << mean_rot_in << std::endl;

		std::cout << "***" << std::endl;
		print_vect(symmetries);

		//Solve an overconstrained system of simultaneous equations to get the symmetry centres
		std::vector<cv::Point2f> sym_centers = get_sym_centers(mirror_between, rot_between, rot_in, symmetries, 
			rot_to_align.size(), rot_to_align);

		atlas_sym a;
		a.x = 1;
		return a;
	}

	/*Get the surveys made by spots equidistant from the central spot in the aligned images average px values diffraction pattern. These 
	**will be used to identify the atlas symmetry
	**Inputs:
	**positions: std::vector<cv::Point> &, Relative positions of the individual spots in the aligned images average px values diffraction 
	**pattern
	**threshold: float, Maximum proportion of the distance between the brightest spot and the spot least distant from it that a spot can be
	**and still be considered to be one of the equidistant spots
	**indices: std::vector<int> &, Output indices of the spots nearest to and equidistant from the brightest spot 
	**angles: std::vector<int> &, Output angles of the spots to a horizontal line drawn horizontally through the brightest spot
	*/
	void equidistant_surveys(std::vector<cv::Point> &spot_pos, float threshold, std::vector<int> &indices, std::vector<float> &angles)
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
		for (int i = 1; i < spot_pos.size(); i++)
		{
			//Get position relative to the central spot
			int dx = spot_pos[i].x - spot_pos[0].x;
			int dy = spot_pos[i].y - spot_pos[0].y;

			//Check if the distance of the spot is smaller than the threshold
			if (std::sqrt(dx*dx + dy*dy) < (1.0+threshold)*min_dst)
			{
				//Note the position's index
				indices.push_back(i);

				//Get the angle to the horizontal drawn through the brightest spot
				float angle = std::acos(dx / std::sqrt(dx*dx + dy*dy));
				angles.push_back(dy > 0 ? angle : 2*PI-angle);
			}
		}
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

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys and reflections of the other surveys in
	**a mirror line between the 2 surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and relative row and column shift of the 
	**second image, in that order
	*/
	std::vector<std::vector<float>> get_mirror_between_sym(std::vector<cv::Mat> &rot_to_align)
	{
		//Use sum of squares formula to get the number of comparisons
		int num_comp = rot_to_align.size() * (rot_to_align.size()-1) / 2;

		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(num_comp);
		
		//Compare each matrix with...
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			//...the higher index matrices
			for (int j = i+1; j < rot_to_align.size(); j++, k++)
			{
				//Flip the second mat in the comparison
				cv::Mat mir;
				cv::flip(rot_to_align[j], mir, 0);

				//Quantify the symmetry and record the shift of highest symmetry
				sym_param[k] = quantify_rel_shift(rot_to_align[i], mir);
			}
		}

		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys and reflections of themselves perpendicular
	**to the radially outwards direction
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and relative row and column shift of the 
	**second image, in that order
	*/
	std::vector<std::vector<float>> get_mirror_in_sym(std::vector<cv::Mat> &rot_to_align)
	{
		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(rot_to_align.size());

		//Compare each matrix with...
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			//Flip the second mat in the comparison
			cv::Mat mir;
			cv::flip(rot_to_align[i], mir, 0);

			//Quantify the symmetry and record the shift of highest symmetry
			sym_param[i] = quantify_rel_shift(rot_to_align[i], mir, INTERAL_MIR0_SSD_FRAC, REL_SHIFT_WIS_INTERNAL_MIR0);
		}

		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys and reflections of themselves parallel
	**to the radially outwards direction
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and relative row and column shift of the 
	**second image, in that order
	*/
	std::vector<std::vector<float>> get_mirror_in_rad_sym(std::vector<cv::Mat> &rot_to_align)
	{
		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(rot_to_align.size());

		//Compare each matrix with...
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			//Flip the second mat in the comparison
			cv::Mat mir;
			cv::flip(rot_to_align[i], mir, 1);

			//Quantify the symmetry and record the shift of highest symmetry
			sym_param[i] = quantify_rel_shift(rot_to_align[i], mir, INTERAL_MIR1_SSD_FRAC, REL_SHIFT_WIS_INTERNAL_MIR1);
		}

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
		//Get the number of comparisons
		int num_comp = (rot_to_align.size()-1) * rot_to_align.size() / 2;

		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(num_comp);

		//Compare each matrix with...
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			//...the higher index matrices and itself
			for (int j = i+1; j < rot_to_align.size(); j++, k++)
			{
				//Quantify the symmetry and record the shift of highest symmetry
				sym_param[k] = quantify_rel_shift(rot_to_align[i], rot_to_align[j]);
			}
		}

		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients for 180 deg rotational symmetry in the surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**est_sym_centers: std::vector<cv::Point2f> &, Estimated symmetry center positions
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_rotational_in_sym(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Point2f> &est_sym_centers)
	{
		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(rot_to_align.size());

		//Look for rotational symmetry in each matrix
        #pragma omp parallel for
		for (int i = 0; i < rot_to_align.size(); i++)
		{
			//Rotate the matrix 180 degrees
			cv::Mat rot_180;
			cv::flip(rot_to_align[i], rot_180, -1);

			//Quantify the symmetry and record the shift of highest symmetry
			sym_param[i] = quantify_rel_shift(rot_to_align[i], rot_180, INTERAL_ROT_SSD_FRAC, REL_SHIFT_WIS_INTERNAL_ROT);
		}

		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients for mirror rotational symmetry inside an image. This function
	**is not currently being used as this symmetry does not need to be detected in surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**refl_lines: std::vector<int> &, Lines to perform reflections in
	**Returns:
	**std::vector<float>, Pearson normalised product moment correlation coefficient for internal mirror rotation symmetry
	*/
	std::vector<float> get_mir_rot_in_sym(std::vector<cv::Mat> &rot_to_align, std::vector<int> &refl_lines)
	{
		//Assign vectors to hold the symmetry parameters
		std::vector<float> sym_param(rot_to_align.size());

		//Look for rotational symmetry in each matrix
        #pragma omp parallel for
		for (int i = 0; i < rot_to_align.size(); i++)
		{
			//Partition matrix into 2 parts, along the known mirror line
			cv::Rect roi_left = cv::Rect(0, 0, rot_to_align[i].cols, refl_lines[i]);
			cv::Rect roi_right = cv::Rect(0, refl_lines[i], rot_to_align[i].cols, rot_to_align[i].rows - refl_lines[i]);

			//Rotate the rightmost region of interest 180 degrees
			cv::Mat rot_180;
			cv::flip(rot_to_align[i](roi_right), rot_180, -1);

			cv::Mat left = rot_to_align[i](roi_left);

			//Quantify the symmetry
			std::vector<float> sym(3);
			sym_param[i] = pearson_corr(rot_180, left,  rot_to_align[i].rows - 2*refl_lines[i], 0);
		}

		return sym_param;
	}

	/*Get the centers of symmetry in each of the surveys from the quantified symmetries by solving the overdetermined set of simultaneous
	**equations created during symmetry registration
	**Inputs:
	**mirror: std::vector<std::vector<float>> &, Mirror symmetry quantification
	**rot_between: std::vector<std::vector<float>> &, Rotational symmetry between surveys quantification
	**rot_in: std::vector<std::vector<float>> &, 180 deg rotational symmetry in surveys quantification
	**symmetries: std::vector<bool> &, Symmetries present. By index: 0 - mir_in, 1 - mir_between, 2 - mir_between_2, 3 - rot_between, 
	**4 - rot_in, 5 - mir_in_rad
	**num_surveys: const int, Number of surveys
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<cv::Point2f>, Centers of symmetry for each of the surveys (column, then row)
	*/
	std::vector<cv::Point2f> get_sym_centers(std::vector<std::vector<float>> &mirror, std::vector<std::vector<float>> &rot_between,
		std::vector<std::vector<float>> &rot_in, std::vector<bool> &symmetries, const int num_surveys, std::vector<cv::Mat> &rot_to_align)
	{
		//Number of inter-comparisons between surveys for between symmetries
		int num_inter_comp = (num_surveys-1) * num_surveys / 2;

		//Get the number of equations
		int num_eqn;
		num_eqn = symmetries[0] ? num_surveys : 0;
		num_eqn += symmetries[1] ? 2*num_inter_comp : 0;
		num_eqn += symmetries[1] ? 0 : (symmetries[2] ? num_inter_comp : 0); //Only use mirrors between spots 2 apart if there are none 1 apart
		num_eqn += symmetries[3] ? num_inter_comp : 0;
		num_eqn += symmetries[4] ? 2*num_surveys : 0;

		//Create simultaneous equations describing the symmetry centers
		Eigen::MatrixXf eqns = Eigen::MatrixXf::Constant(num_eqn, 2*num_surveys, 0.0f); //Express the coordinates of surveys being compared...
		Eigen::VectorXf diff(num_eqn); //...in terms of the differences in symmetry center positions of the surveys or otherwise

	   //Keep track of how many equations have beed added to the matrix
		int eqn_num = 0;

		//Add equations describing mirror symmetry in surveys
		if (symmetries[0])
		{
			for (int i = 0, k = 0; i < num_surveys; i++, eqn_num++)
			{
				//Row estimate
				eqns(eqn_num, i) = 1.0f;
				diff(eqn_num) = (rot_to_align[0].rows - mirror[k][2] - 1) / 2;
				k += i;
			}
		}

		//Add equations describing mirror symmetry between surveys
		if (symmetries[1])
		{
			for (int i = 0, k = 0; i < num_surveys; i++)
			{
				k++;

				for (int j = i+1; j < num_surveys; j++, k++, eqn_num++)
				{
					//Row estimate
					eqns(eqn_num, j) = -1.0f;
					eqns(eqn_num, i) = -1.0f;
					diff(eqn_num) = mirror[k][2] - rot_to_align[j].rows;
					
					eqn_num++;

					//Column estimate
					eqns(eqn_num, j+num_surveys) = 1.0f;
					eqns(eqn_num, i+num_surveys) = -1.0f;
					diff(eqn_num) = mirror[k][1];
				}
			}
		}
		//Mirror symmetry between every second survey, not between all surveys
		else
		{
			if (symmetries[2])
			{
				for (int i = 0, k = 0; i < num_surveys; i++)
				{
					k += 2;

					for (int j = i+2; j < num_surveys; j += 2, k += 2, eqn_num++)
					{
						//Row estimate
						eqns(eqn_num, j) = -1.0f;
						eqns(eqn_num, i) = -1.0f;
						diff(eqn_num) = mirror[k][2] - rot_to_align[j].rows;

						eqn_num++;

						//Column estimate
						eqns(eqn_num, j+num_surveys) = 1.0f;
						eqns(eqn_num, i+num_surveys) = -1.0f;
						diff(eqn_num) = mirror[k][1];
					}
				}
			}
		}

		//Rotational symmetry between surveys
		if (symmetries[3])
		{
			for (int i = 0, k = 0; i < num_surveys; i++)
			{
				for (int j = i+1; j < num_surveys; j++, k++, eqn_num++)
				{
					//Row estimate
					eqns(eqn_num, j) = 1.0f;
					eqns(eqn_num, i) = -1.0f;
					diff(eqn_num) = rot_between[k][2];

					eqn_num++;

					//Column estimate
					eqns(eqn_num, j+num_surveys) = 1.0f;
					eqns(eqn_num, i+num_surveys) = -1.0f;
					diff(eqn_num) = rot_between[k][1];
				}
			}
		}

		if (symmetries[4])
		{
			for (int i = 0; i < num_surveys; i++, eqn_num++)
			{
				//Row estimate
				eqns(eqn_num, i) = 1.0f;
				diff(eqn_num) = (rot_to_align[0].rows - rot_in[i][2] - 1) / 2;

				eqn_num++;

				//Column estimate
				eqns(eqn_num, i+num_surveys) = 1.0f;
				diff(eqn_num) = (rot_to_align[0].cols - rot_in[i][1] - 1) / 2;
			}
		}

		//Solve the simultaneous equations to find the symmetry center
		Eigen::VectorXf solution = eqns.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(diff);

		//Express the symmetry centers in an easier to use form
		std::vector<cv::Point2f> sym_centers(num_surveys);
		for (int i = 0; i < num_surveys; i++)
		{
			//Column, then row
			sym_centers[i] = cv::Point2f(solution(i), solution(i+num_surveys));
		}

		return sym_centers;
	}
}