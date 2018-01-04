#pragma once

#include <defines.h>
#include <includes.h>

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
		cv::Mat &acc, const int radius, const int ns_radius, const int inpainting_method = cv::INPAINT_NS);

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
		int inpainting_method, int col_max, int row_max, int ns_radius);
}