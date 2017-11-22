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

	//Read in the image stack
	cv::Mat image32F;
	std::vector<cv::Mat> mats;
	imreadmulti(inputImagePath, mats, CV_LOAD_IMAGE_UNCHANGED);

	const int num_imgs = mats.size();
	
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

	//Use Fourier analysis to place upper bound on the size of the circles
	int ubound = circ_size_ubound(mats, mats_rows_af, mats_cols_af, MIN_CIRC_SIZE, AUTOCORR_REQ_CONV, 
		std::max(num_imgs, MAX_AUTO_CONTRIB), 0.25*UBOUND_GAUSS_SIZE+0.75, af_context, af_device_id, af_queue,
		NUM_THREADS);

	//Get limits 
	int lbound = MIN_CIRC_SIZE;
	if (!ubound) {
		ubound = std::min(mats_rows_af, mats_cols_af);
	}

	//Calculate size of 1st circle
	

	//Create higher circle depending on the size of the image

	return 0;
}