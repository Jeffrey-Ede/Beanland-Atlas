#include <beanland_atlas.h>

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
	
	//Create OpenCL context and queue for GPU acceleration 
	cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	cl::Device device = devices[0];
	cl::CommandQueue queue(context, device);
	
	//Instruct ArrayFire to use the OpenCL. First, create a device from the current OpenCL device + context + queue
	afcl::addDevice(device(), context(), queue());
	
	//Next switch ArrayFire to the device using the device and context as identifiers:
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
	af::array gauss = af::array(gauss_fft2_af);

	//Use Fourier analysis to place upper bound on the size of the circles
	int ubound = circ_size_ubound(mats, mats_rows_af, mats_cols_af, gauss, MIN_CIRC_SIZE, std::min((int)mats.size(), MAX_AUTO_CONTRIB), 
		af_context, af_device_id, af_queue, NUM_THREADS);

	//Set lower bound, assuming that spots in data will have at least a few pixels diameter
	int lbound = MIN_CIRC_SIZE;

	//Create annulus making kernel
	cl_kernel create_annulus_kernel = create_kernel(annulus_source, annulus_kernel, af_context, af_device_id);

	//Calculate annulus radius and thickness that describe the gradiation of the spots best
	std::vector<int> annulus_param = get_annulus_param(mats[0], lbound, ubound, INIT_ANNULUS_THICKNESS, MAX_SIZE_CONTRIB, 
		mats_rows_af, mats_cols_af, gauss, create_annulus_kernel, af_queue, NUM_THREADS);

	//Use best annulus parameters to create the annulus to perform cross correlations with
	af::array best_annulus = create_annulus(mats_cols_af*mats_rows_af, mats_cols_af, mats_cols_af/2, mats_rows_af, mats_rows_af/2, 
		annulus_param[0], annulus_param[1], create_annulus_kernel, af_queue);

	//Precalculate the Hann window, ready for repeated application
	cv::Mat hann_window_LUT = create_hann_window(mats[0].rows, mats[0].cols, NUM_THREADS);

	//Number of times to recursively cross correlate
	int order = std::max(ubound/(2*annulus_param[0]), 1); //Take max for special case of overlapping Bragg peaks

	//Create circle creating kernel
	cl_kernel circle_creator = create_kernel(circle_source, circle_kernel, af_context, af_device_id);

	//Create circle
	af::array circle = create_circle(mats_cols_af*mats_rows_af, mats_cols_af, mats_cols_af/2, mats_rows_af, mats_rows_af/2, 
		annulus_param[0], circle_creator, af_queue);

	//Find alignment of successive images
	std::vector<cv::Vec3f> rel_pos = img_rel_pos(mats, hann_window_LUT, best_annulus, circle, gauss, order,
		mats_rows_af, mats_cols_af, NUM_THREADS);

	//Free OpenCL resources
	clFlush(af_queue);	
	clFinish(af_queue);
	clReleaseKernel(gauss_kernel);
	clReleaseKernel(create_annulus_kernel);
	clReleaseCommandQueue(af_queue);
	clReleaseContext(af_context);

	return 0;
}