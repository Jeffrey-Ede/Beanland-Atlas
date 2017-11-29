#include <beanland_atlas.h>

/*Align the diffraction patterns using their known relative positions and average over the aligned px
**mats: std::vector<cv::Mat> &, Diffraction patterns to average over the aligned pixels of
**positions: std::vector<cv::Vec3f> &, Relative positions of the images
**Return:
**std::vector<cv::Mat>, The first OpenCV mat is the average of the aligned diffraction patterns, the 2nd is the number of OpenCV mats
**that contributed to each pixel
*/
std::vector<cv::Mat> align_and_avg(std::vector<cv::Mat> &mats, std::vector<std::array<float, 5>> &positions)
{
	//Assign memory to store the average of the aligned images and to store the number of contributions to each of its elements
	std::vector<cv::Mat> aligned_avg(2);

	//Refine the relative position combinations to get the positions relative to the first image
	//Index 0 - rows, Index 1 - cols
	std::vector<std::vector<int>> refined_pos = refine_rel_pos(positions);

	//Get the minimum and maximum relative positions of rows and columns
	int row_min = refined_pos[0][std::distance(refined_pos[0].begin(), std::min_element(refined_pos[0].begin(), refined_pos[0].end()))];
	int row_max = refined_pos[0][std::distance(refined_pos[0].begin(), std::max_element(refined_pos[0].begin(), refined_pos[0].end()))];
	int col_min = refined_pos[1][std::distance(refined_pos[1].begin(), std::min_element(refined_pos[1].begin(), refined_pos[1].end()))];
	int col_max = refined_pos[1][std::distance(refined_pos[1].begin(), std::max_element(refined_pos[1].begin(), refined_pos[1].end()))];
	
	//Additional number of rows and columns needed so that all the images fit in the aligned pattern
	int extra_rows = row_max - row_min;
	int extra_cols = row_max - row_min;

	//Assign memory to accumulate the images in and count the number of images contributing to each element of the accumulator
	cv::Mat acc = cv::Mat(mats[0].rows+extra_rows, mats[0].cols+extra_cols, CV_32FC1, cv::Scalar(0.0));
	cv::Mat num_overlap = cv::Mat(mats[0].rows+extra_rows, mats[0].cols+extra_cols, CV_16UC1, cv::Scalar(0));
	cv::Rect roi;

	//Accumulate each image
    #pragma omp parallel for
	for (int i = 0; i < mats.size(); i++) {
			
		//Position of image in larger image
		roi = cv::Rect(row_max-refined_pos[0][i], col_max-refined_pos[1][i],  mats[i].rows, mats[i].cols);	

		//Add the image's contribution to the accumulator and increment the contribution count for the elements it contributed to
		acc(roi) += mats[i];
		num_overlap(roi) = num_overlap(roi)+1;
	}
	
	//Divide non-zero accumulator matrix pixel values by number of overlapping contributing images
	float *p;
	ushort *q;
    #pragma omp parallel for
	for (int i = 0; i < num_overlap.rows; i++) {
	
		p = acc.ptr<float>(i);
		q = num_overlap.ptr<ushort>(i);
		for (int j = 0; j < num_overlap.cols; j++) {
	
			//Divide pixels contributed to by the number of contributing pixels
			if (q[j]) {
				p[j] /= q[j];
			}
		}
	}

	//Contain the average of the aligned images and the number of images contributing to each of its elements so that they can be returned
	aligned_avg[0] = acc;
	aligned_avg[1] = num_overlap;

	return aligned_avg;
}

/*Refine the relative positions of the images using all the known relative positions
**positions: std::vector<cv::Vec2f>, Relative image positions and their weightings
**Return:
**std::vector<std::array<int, 2>>, Relative positions of the images, including the first image, to the first image in the same order as the
**images in the image stack
*/
std::vector<std::vector<int>> refine_rel_pos(std::vector<std::array<float, 5>> &positions)
{
	/* Assume that images are all compared against the first image for now. Refine this later */

	//Assign memory to store the refined positions
	std::vector<std::vector<int>> refined_pos(2);

	std::vector<int> x_pos(positions.size());
	std::vector<int> y_pos(positions.size());

	for (int i = 0; i < positions.size(); i++)
	{
		x_pos[i] = positions[i][0];
		y_pos[i] = positions[i][1];
	}

	refined_pos[0] = x_pos;
	refined_pos[1] = y_pos;

	return refined_pos;
}