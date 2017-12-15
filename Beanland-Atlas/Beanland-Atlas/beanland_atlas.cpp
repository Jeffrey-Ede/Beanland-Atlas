#include <beanland_atlas.h>

//Use Beanland Atlas library functions
using namespace ba;

int main()
{
	//Get number of concurrent processors. Use same number of threads
	int NUM_THREADS;
	unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
	if (concurentThreadsSupported) {
		NUM_THREADS = concurentThreadsSupported;
	}
	else {
		NUM_THREADS = 1; //OpenMP will use 1 thread
	}

	omp_set_num_threads(NUM_THREADS);
	omp_set_nested(1); //Enable nested parallelism

	//Read in the image stack
	std::vector<cv::Mat> mats;
	imreadmulti(inputImagePath, mats, CV_LOAD_IMAGE_UNCHANGED);

	//cv::Mat rot = in_plane_rotate(mats[0], 0.1, 0);

	//Preprocess the image stack. At the moment, this just involves median filtering, resizing the images and converting them 
	preprocess(mats, PREPROC_MED_FILT_SIZE);

	//Create OpenCL context and queue for GPU acceleration 
	cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	cl::Device device = devices[0];
	cl::CommandQueue queue(context, device);
	
	//Instruct ArrayFire to use the OpenCL. First, create a device from the current OpenCL device + context + queue
	afcl::addDevice(device(), context(), queue());
	
	//Switch ArrayFire to the device using the device and context as identifiers:
	afcl::setDevice(device(), context());

	//Get ArrayFire device, context and command queue
	static cl_context af_context = afcl::getContext();
	static cl_device_id af_device_id = afcl::getDeviceId();
	static cl_command_queue af_queue = afcl::getQueue();

	//ArrayFire arrays store images in memory transpositionally to OpenCV mats
	int mats_cols_af = mats[0].rows;
	int mats_rows_af = mats[0].cols;

	//Create extended Gaussian creating kernel
	cl_kernel gauss_kernel = create_kernel(gauss_kernel_ext_source, gauss_kernel_ext_kernel, af_context, af_device_id);

	//Create the extended Gaussian
	af::array ext_gauss = extended_gauss(mats[0].rows, mats[0].cols, 0.25*UBOUND_GAUSS_SIZE+0.75, gauss_kernel, af_queue);

	//Fourier transform the Gaussian
	af_array gauss_fft2_af;
	af_fft2_r2c(&gauss_fft2_af, ext_gauss.get(), 1.0f, mats_rows_af, mats_cols_af);
	af::array gauss_fft = af::array(gauss_fft2_af);

	//Use Fourier analysis to place upper bound on the size of the circles
	int ubound = circ_size_ubound(mats, mats_rows_af, mats_cols_af, gauss_fft, MIN_CIRC_SIZE, std::min((int)mats.size(), MAX_AUTO_CONTRIB), 
		af_context, af_device_id, af_queue, NUM_THREADS);

	//Set lower bound, assuming that spots in data will have at least a few pixels diameter
	int lbound = MIN_CIRC_SIZE;

	//Create annulus making kernel
	cl_kernel create_annulus_kernel = create_kernel(annulus_source, annulus_kernel, af_context, af_device_id);

	//Calculate annulus radius and thickness that describe the gradiation of the spots best
	std::vector<int> annulus_param = get_annulus_param(mats[0], lbound, ubound, INIT_ANNULUS_THICKNESS, MAX_SIZE_CONTRIB, 
		mats_rows_af, mats_cols_af, gauss_fft, create_annulus_kernel, af_queue, NUM_THREADS);

	//Number of times to recursively cross correlate annulus with itself
	int order = ubound/(2*annulus_param[0]);

	//Use best annulus parameters to create the annulus to perform cross correlations with
	af::array best_annulus = create_annulus(mats_cols_af*mats_rows_af, mats_cols_af, mats_cols_af/2, mats_rows_af, mats_rows_af/2, 
		annulus_param[0], annulus_param[1], create_annulus_kernel, af_queue);

	//Gaussian blur the annulus in the Fourier domain
	af_array annulus_fft_c;
	af_fft2_r2c(&annulus_fft_c, best_annulus.get(), 1.0f, mats_rows_af, mats_cols_af);
	af::array annulus_fft = recur_conv(gauss_fft*af::array(annulus_fft_c), order);

	//Precalculate the Hann window, ready for repeated application
	cv::Mat hann_window_LUT = create_hann_window(mats[0].rows, mats[0].cols, NUM_THREADS);

	//Create circle creating kernel
	cl_kernel circle_creator = create_kernel(circle_source, circle_kernel, af_context, af_device_id);

	//Create circle
	af::array circle = create_circle(mats_cols_af*mats_rows_af, mats_cols_af, mats_cols_af/2, mats_rows_af, mats_rows_af/2, 
		annulus_param[0], circle_creator, af_queue);

	//Gaussian blur the circle in the Fourier domain
	af_array circle_c;
	af_fft2_r2c(&circle_c, circle.get(), 1.0f, mats_rows_af, mats_cols_af);
	af::array circle_fft = gauss_fft*af::array(circle_c);

	//Find alignment of successive images
	std::vector<std::array<float, 5>> rel_pos = img_rel_pos(mats, hann_window_LUT, annulus_fft, circle_fft, mats_rows_af, mats_cols_af);

	//Refine the relative position combinations to get the positions relative to the first image
	//Index 0 - rows, Index 1 - cols
	std::vector<std::vector<int>> refined_pos = refine_rel_pos(rel_pos);

	//Align the diffraction patterns to create average diffraction pattern
	struct align_avg_mats aligned_avg = align_and_avg(mats, refined_pos);

	//Get the positions of the spots in the aligned images average
	cv::Vec2f samp_to_detect_sphere;
	std::vector<cv::Point> spot_pos = get_spot_pos(aligned_avg.acc, annulus_param[0], annulus_param[0], create_annulus_kernel, 
		circle_creator, gauss_kernel, af_queue, aligned_avg.acc.cols, aligned_avg.acc.rows, samp_to_detect_sphere);

	//Combine the compendiums of maps mapped out by each spot to create maps showing the whole k spaces surveyed by each of the spots,
	//then combine these surveys into an atlas to show the whole k space mapped out
	std::vector<cv::Mat> surveys = create_spot_maps(mats, spot_pos, refined_pos, 0.8*annulus_param[0], 
		annulus_param[0]+2*annulus_param[1], -1);

	struct atlas_sym atlas_symmetry = identify_symmetry(surveys, spot_pos, EQUIDST_THRESH);

	//display_CV(raw_atlas, 7.5e-4);

	//Free OpenCL resources
	clFlush(af_queue);	
	clFinish(af_queue);
	clReleaseKernel(gauss_kernel);
	clReleaseKernel(create_annulus_kernel);
	clReleaseKernel(circle_creator);
	clReleaseCommandQueue(af_queue);
	clReleaseContext(af_context);

	return 0;
}