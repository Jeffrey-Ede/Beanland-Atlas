#define _GATAN_USE_STL_STRING		// Provide conversion from 'DM::String' to 'std::string'

#define _GATANPLUGIN_USES_LIBRARY_VERSION 2
#include "DMPlugInBasic.h"

#define _GATANPLUGIN_USE_CLASS_PLUGINMAIN
#include "DMPlugInMain.h"

using namespace Gatan;

#include <cassert>
#include <string>

void getCoordsFromNTilts(const ulong &nTilts, const ulong &pt, ulong &x, ulong &y);

class SampleImageProcessingPlugIn : public Gatan::PlugIn::PlugInMain
{
	virtual void Start();
	virtual void Run();
	virtual void Cleanup();
	virtual void End();
};

/*
**Get the coordinates of the disk at the current point
*/
void getCoordsFromNTilts(const ulong &nTilts, const ulong &pt, ulong &x, ulong &y)
{
	ulong side = 2*nTilts+1;

	y = (ulong)std::floor((float)(pt/side))-nTilts;
	x = ((pt % side)-nTilts)*(y % 2 ? -1 : 1); //Flip sign every row
}

/*
**Extract disks from CBED patterns to create full LACBED patterns
*/
void cbed_stitcher( const DM::Image &stack, DM::Image &patterns_out, long &nG1, long &nG2,
				   ulong &nTilts, ulong &pX, ulong &pY, ulong &tInc, ulong &g1X, ulong &g1Y, 
				   ulong &g2X, ulong &g2Y, ulong &Rr2, ulong &IsizX, ulong &IsizY )
{
	//Dimensions and area of CBED images
	ulong xsize = stack.GetDimensionSize( 0 );
	ulong ysize = stack.GetDimensionSize( 1 );
	ulong zsize = stack.GetDimensionSize( 2 );

	ulong cbed_area = xsize*ysize;
	ulong pattern_area = IsizX*IsizY;
	ulong montage_size = (2*nG1+1)*(2*nG2+1);

	DM::Image patterns = DM::RealImage( "Contributions", 4, IsizX, IsizY, montage_size ); //Count number of contributions to each pixel
	DM::Image contributions = DM::RealImage( "D-LACBED Stack", 4, IsizX, IsizY);

	//Lock data down so that it can be accessed from C++
	PlugIn::ImageDataLocker stack_l( stack, PlugIn::ImageDataLocker::lock_data_CONTIGUOUS );
	PlugIn::ImageDataLocker patterns_l( patterns, PlugIn::ImageDataLocker::lock_data_CONTIGUOUS );
	PlugIn::ImageDataLocker contributions_l( contributions, PlugIn::ImageDataLocker::lock_data_CONTIGUOUS );

	//Make sure images have the expected data type
	assert( stack_l.get_image_data().get_data_type() == REAL4_DATA );
	assert( patterns_l.get_image_data().get_data_type() == REAL4_DATA );
	assert( contributions_l.get_image_data().get_data_type() == REAL4_DATA );

	// Get pointers to the data
	float32 *stack_data         = reinterpret_cast<float32 *>( stack_l.get_image_data().get_data() );
	float32 *patterns_data      = reinterpret_cast<float32 *>( patterns_l.get_image_data().get_data() );
	float32 *contributions_data = reinterpret_cast<float32 *>( contributions_l.get_image_data().get_data() );

	{
		//Loop over the LACBED stack to build the patterns
		float32 *cbed_rollback = &stack_data[0];
		for(long i = -nG1; i <= nG1; i++)
		{
			for(long j = -nG2; j <= nG2; j++)
			{
				//Loop over CBED stacks
				for(ulong pt = 0; pt < zsize; pt++)
				{
					//Get the coordinates of the point
					ulong x, y;
					getCoordsFromNTilts(nTilts, pt, x, y);

					ulong X = (ulong)std::round((float)(pX+x*tInc+(i*g1X)+(j*g2X)));
					ulong Y = (ulong)std::round((float)(pY+y*tInc+(i*g1Y)+(j*g2Y)));

					//Check that the disk is fully inside the image
					if( !((X<Rr2) && (X+Rr2>IsizX) && (Y<Rr2) && (Y+Rr2>IsizY)) )
					{
						//Extract and add the disk to the patterns and increment the contribution counters
						//for pixels that it contributes to
						for(ulong itery = 0, k = 0; itery < ysize; itery++)
						{
							for(ulong iterx = 0; iterx < xsize; iterx++, k++)
							{
								//Extract the points in the disk
								if((ulong)std::sqrt((float)(iterx*iterx + itery*itery)) < Rr2)
								{
									patterns_data[k] += stack_data[k];
									contributions_data[k]++;
								}
							}
						}
					}

					stack_data += cbed_area;
				}

				//Divide D-LACBED pattern by contributions
				for(ulong itery = 0, k = 0; itery < ysize; itery++)
				{
					for(ulong iterx = 0; iterx < xsize; iterx++, k++)
					{
						//If the pixel has been contributed to
						if(contributions_data[k])
						{
							patterns_data[k] /= contributions_data[k];
							contributions_data[k] = 0; //Reset for next iteration (trading more processing for RAM)
						}
					}
				}

				stack_data = cbed_rollback; //Back to start of stack
				patterns_data += pattern_area; //Go to next D-LACBED pattern
			}
		}

		patterns_l.MarkDataChanged();
		contributions_l.MarkDataChanged(); //Not sure if this line is necessary to free the memory or not
	}

	patterns_out = patterns;
}

/*
** Provide a proxy for 'ImgProc_Horizontal_Derivative' that can be installed
** as a script function. Note that C++ class references cannot be passed to the
** script language, only raw pointers, so 'DM::Image' cannot be used as an argument
** to a script function. Instead, 'DM_ImageToken' is used for image rvalues, and
** 'DM_ImageToken *' for lvalues. The following function shows how to convert.
*/
void SF_cbed_stitcher( DM_ImageToken stack, DM_ImageToken *patterns_out, long nG1, long nG2,
				   ulong nTilts, ulong pX, ulong pY, ulong tInc, ulong g1X, ulong g1Y, ulong g2X, 
				   ulong g2Y, ulong Rr2, ulong IsizX, ulong IsizY )
{
	DM::Image patterns_input;
	cbed_stitcher( stack, patterns_input, nG1, nG2, nTilts, pX, pY, tInc, g1X, g1Y, g2X, g2Y, Rr2, IsizX, IsizY);
	DM::Image::assign_ptr( *patterns_out, patterns_input.get() );
}

///
/// This is called when the plugin is loaded.  Whenever DM is
/// launched, it calls 'Start' for each installed plug-in.
/// When it is called, there is no guarantee that any given
/// plugin has already been loaded, so the code should not
/// rely on scripts installed from other plugins.  The primary
/// use is to install script functions.
///
void SampleImageProcessingPlugIn::Start()
{
	AddFunction( "void cbed_stitcher( BasicImage *, BasicImage *, long, long, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong)", &SF_cbed_stitcher );
}

///
/// This is called when the plugin is loaded, after the 'Start' method.
/// Whenever DM is launched, it calls the 'Run' method for
/// each installed plugin after the 'Start' method has been called
/// for all such plugins and all script packages have been installed.
/// Thus it is ok to use script functions provided by other plugins.
///
void SampleImageProcessingPlugIn::Run()
{
}

///
/// This is called when the plugin is unloaded.  Whenever DM is
/// shut down, the 'Cleanup' method is called for all installed plugins
/// before script packages are uninstalled and before the 'End'
/// method is called for any plugin.  Thus, script functions provided
/// by other plugins are still available.  This method should release
/// resources allocated by 'Run'.
///
void SampleImageProcessingPlugIn::Cleanup()
{
}

///
/// This is called when the plugin is unloaded.  Whenever DM is shut
/// down, the 'End' method is called for all installed plugins after
/// all script packages have been unloaded, and other installed plugins
/// may have already been completely unloaded, so the code should not
/// rely on scripts installed from other plugins.  This method should
/// release resources allocated by 'Start', and in particular should
/// uninstall all installed script functions.
///
void SampleImageProcessingPlugIn::End()
{
}

SampleImageProcessingPlugIn gSampleImageProcessingPlugIn;