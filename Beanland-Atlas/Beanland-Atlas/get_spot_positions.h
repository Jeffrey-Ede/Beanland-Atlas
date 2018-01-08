#pragma once

#include <includes.h>

#include <commensuration_utility.h>
#include <kernel_launchers.h>
#include <utility.h>

namespace ba
{
	/*Find the positions of the spots in the aligned image average pattern. Most of these spots are made from the contributions of many images
	**so it can be assumed that they are relatively featureless
	**Inputs:
	**align_avg: cv::Mat &, Average values of px in aligned diffraction patterns
	**initial_radius: int, Radius of the spots
	**initial_thickness: int, Thickness of the annulus to convolve with spots
	**annulus_creator: cl_kernel, OpenCL kernel to create padded unblurred annulus to cross correlate the aligned image average pattern Sobel
	**filtrate with
	**circle_creator: cl_kernel, OpenCL kernel to create padded unblurred circle to cross correlate he aligned image average pattern with
	**gauss_creator: cl_kernel, OpenCL kernel to create padded Gaussian to blur the annulus and circle with
	**af_queue: cl_command_queue, ArrayFire command queue
	**align_avg_cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**align_avg_rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**ewald_rad: cv::Vec2f &, Reference to a float to store the Ewald sphere radius and orientation estimated by this function in
	**discard_outer: const int, Discard spots within this distance from the boundary. Defaults to discarding spots within 1 radius
	**Return:
	std::vector<cv::Point> Positions of spots in the aligned image average pattern
	*/
	std::vector<cv::Point> get_spot_pos(cv::Mat &align_avg, int initial_radius, int initial_thickness, cl_kernel annulus_creator,
		cl_kernel circle_creator, cl_kernel gauss_creator, cl_command_queue af_queue, int align_avg_cols, int align_avg_rows, cv::Vec2f &ewald_rad,
		const int discard_outer = DISCARD_SPOTS_DEFAULT);

	/*Blackens the circle of pixels within a certain radius of a point in a floating point OpenCV mat
	**Inputs:
	**mat: cv::Mat &, Reference to a floating point OpenCV mat to blacken a circle on
	**col: const int, column of circle origin
	**row: const int, row of circle origin
	**rad: const int, radius of the circle to blacken
	*/
	void blacken_circle(cv::Mat &mat, const int col, const int row, const int rad);

	/*Uses a set of know spot positions to extract approximate lattice vectors for a diffraction pattern
	**Input:
	**positions: std::vector<cv::Point> &, Known positions of spots in the diffraction pattern
	**Returns:
	**std::vector<cv::Vec2f>, The lattice vectors
	*/
	std::vector<cv::Vec2i> get_lattice_vectors(std::vector<cv::Point> &positions);

	/*Uses lattice vectors to search for spots in the diffraction pattern that have not already been recorded
	**Input:
	**positions: std::vector<cv::Point> &, Known positions of spots in the diffraction pattern
	**lattice_vectors: std::vector<cv::Vec2f> &, Lattice vectors describing the positions of the spots
	**cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rad: int, radius of the spots
	*/
	void find_other_spots(std::vector<cv::Point> &positions, std::vector<cv::Vec2f> &lattice_vectors, 
		int cols, int rows, int rad);

	/*Remove or correct any spot positions that do not fit on the spot lattice very well
	**Input:
	**positions: std::vector<cv::Point>, Positions of located spots
	**lattice_vectors: std::vector<cv::Vec2i> &, Initial lattice vector estimate
	**cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rad: int, radius of the spots
	**tol: float, The maximum fraction of the circle radius that a circle can be away from a multiple of the initial
	**lattice vectors estimate and still be considered a valid spot
	**Returns:
	**std::vector<cv::Point>, Spots that lie within tolerance of the lattice defined by the lattice vectors
	*/
	std::vector<cv::Point> correct_spot_pos(std::vector<cv::Point> &positions, std::vector<cv::Vec2i> &lattice_vectors,	int cols, int rows,
		int rad, float tol = SPOT_POS_TOL);

	/*Refine the lattice vectors by finding the 2 that best least squares fit the data
	**Input:
	**positions: std::vector<cv::Point>, Positions of located spots. Outlier positions have been removed
	**latt_vect: std::vector<cv::Vec2i> &, Original estimate of the lattice vectors
	**cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**range: float, Lattice vectors are varied over +/- this range
	**step: float, Step to incrementally move across the component ranges with
	**Returns:
	**std::vector<cv::Vec2f> &, Refined estimate of the lattice vectors
	*/
	std::vector<cv::Vec2f> refine_lattice_vectors(std::vector<cv::Point> &positions, std::vector<cv::Vec2i> &latt_vect,	
		int cols, int rows, float range, float step);

	/*Get the spot positions as a multiple of the lattice vectors
	**Input:
	**positions: std::vector<cv::Point>, Positions of located spots. Outlier positions have been removed
	**latt_vect: std::vector<cv::Vec2i> &, Lattice vectors
	**cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**Returns:
	**std::vector<cv::Point2i> &, Spot positions as the nearest integer multiples of the lattice vectors
	*/
	std::vector<cv::Point2i> get_pos_as_mult_latt_vect(std::vector<cv::Point> &positions, std::vector<cv::Vec2i> &latt_vect, 
		int cols, int rows);

	/*The positions described by some lattice vectors that lie in a finite plane
	**Input:
	**positions: std::vector<cv::Point>, Positions of located spots. Outlier positions have been removed
	**lattice_vectors: std::vector<cv::Vec2i> &, Lattice vectors
	**cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**origin: cv::Point &, Origin of the lattice that linear combinations of the lattice vectors will be measured relative to
	**Returns:
	**std::vector<cv::Vec2i> &, Linear additive combinations of integer multiples of the lattice vectors that lie in a finite plane
	**with a specified origin. Indices are: 0 - multiple of first lattice vector, 1 - multiple of second lattice vector
	*/
	std::vector<cv::Vec2i> gen_latt_pos(std::vector<cv::Vec2i> &lattice_vectors, int cols, int rows, cv::Point &origin);

	/*Discard the outer spots on the Beanland atlas. Defaults to removing those that are not fully on the aligned diffraction pattern
	**Input:
	**pos: std::vector<cv::Point>, Positions of spots
	**cols: const int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: const int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**dst: const int, Minimum distance of spots from the boundary for them to not be discarded
	**Returns:
	**std::vector<cv::Point> &, Spots that are at least the minimum distance from the boundary
	*/
	std::vector<cv::Point> discard_outer_spots(std::vector<cv::Point> &pos, const int cols, const int rows, const int dst);
}