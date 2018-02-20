// $BACKGROUND$
  //***  Collect Digital Diffraction Pattern  ***\\
 //** Richard Beanland r.beanland@warwick.ac.uk **\\
//*        Coding started March 2011              *\\
 
// Version 1.4a, 13 Apr 2012
// Orius SC600A on 2100 LaB6
// 1.5 - serpentine raster, 17 May 2012
// 1.7 - collect into memory and process, 26 Feb 2013
// 1.8 - added image shift calibration Summer 2014
// 1.8a - created Interp function

//Realignment 11/4/14 
//TEM mode Camera length 25cm IL1=520A ***for all Alphas***
//Alpha1, Camera length 20 cm, IL1=4AE5
//Alpha9, Camera length 20 cm, IL1=5040
//Alpha1, Camera length 25 cm, IL1=4B25
//Alpha3, Camera length 20 cm, IL1=5171(51A7)


//Global variables
 number true = 1, false = 0;
 number Camb=2672, Camr=2688//Global values, for Orius SC600 camera
 number binning=4;//Could be a dialogue
 number expo=0.04;//Could be a dialogue
 number sleeptime=0.02;//delay while waiting for microsope to respond
 number _t = 0;
 number _l = 0;
 number _b = Camb/binning; 
 number _r = Camr/binning;
 image img1;//live image
 object img_src

//*******************************//
///////////////////////////
// Subroutines.
///////////////////////////
//*******************************//
//GetCoordsFromNTilts
//UpdateCameraImage
//Interp

//Function GetCoordsFromNTilts
//Gets x-ycoords from the index currentPoint
void GetCoordsFromNTilts(number nTilts, number currentPoint, number &i, number &j)
{
  number side = 2*nTilts+1;
  j = floor(currentPoint/side)-nTilts;
  i = ((currentPoint % side)-nTilts)*((-1)**(j % 2));//NB % means modulo, flips sign every row
}//End of GetCoordsFromNTilts


//Function UpdateCameraImage
//Gets next available frame from the camera 
void UpdateCameraImage(object img_src, image img)
{
  // Wait for next frame
  number acq_params_changed=0;
  number max_wait = 0.1;
  if (!img_src.IMGSRC_AcquireTo(img,true,max_wait,acq_params_changed))
  {   
	while (!img_src.IMGSRC_AcquireTo(img,false,max_wait,acq_params_changed))
	{
	}
  }	
}//End of UpdateCameraImage


//Function Interp
//Gives linear interpolation between four values in a 2x2 image A
number Interp(image A, number dX, number dY)
{
  //Coefficients _a,_b,_c,_d for linear interpolation
  //f=a+bx+cy+dxy
  number _a=A.GetPixel(0,0);
  number _b=(A.GetPixel(1,0)-_a);
  number _c=(A.GetPixel(0,1)-_a);
  number _d=(_a-A.GetPixel(0,1)-A.GetPixel(1,0)+A.GetPixel(1,1));
  number corA=_a+_b*dX+_c*dY+_d*dX*dY;
  return corA
}//End of Interp

//*******************************//
///////////////////////////////////
// Main program
///////////////////////////////////
//*******************************//

////////////////////
//Check microscope status before starting
string mode=EMGetImagingOpticsMode();
if (!(mode=="DIFF"))//mag mode
{
  throw ("TEM not in diffraction mode")
}

////////////////////
//Set up parameters with user input
number f;
string date_;
GetDate(f,date_);
string time_;
GetTime(f,time_);
string datetime=date_+"_"+time_;
result("Starting acquisition "+datetime+"\n");
datetime="";
// Get alpha
number alpha=3;
if (!GetNumber("Alpha?",alpha,alpha));
exit(0);
//Check Calibration
number mag=EMGetMagnification(); //Sometimes gives null answer, why??
number camL=EMGetCameraLength();
string Tag_Path="DigitalDiffraction:Alpha="+alpha+":Binning="+binning+":CamL="+CamL+"mm:";
GetPersistentStringNote(Tag_Path+"Date",datetime);
if (datetime=="")
{
  throw("No tilt calibration - please recalibrate");
}
else
{
  result("\nLast calibration "+datetime+"\n");
}
////////////////////
//Load calibration
number Rr,spot,nCals,xTpX,xTpY,yTpX,yTpY,xShpX,xShpY,yShpX,yShpY;
string pathname,file_name,material;
GetPersistentNumberNote(Tag_Path+"Disc Radius",Rr);
GetPersistentNumberNote(Tag_Path+"Spot size",spot);
GetPersistentNumberNote(Tag_Path+"nCals",nCals);
GetPersistentNumberNote(Tag_Path+"xTpX",xTpX);
GetPersistentNumberNote(Tag_Path+"xTpY",xTpY);
GetPersistentNumberNote(Tag_Path+"yTpX",yTpX);
GetPersistentNumberNote(Tag_Path+"yTpY",yTpY);
GetPersistentNumberNote(Tag_Path+"xShpX",xShpX);
GetPersistentNumberNote(Tag_Path+"xShpY",xShpY);
GetPersistentNumberNote(Tag_Path+"yShpX",yShpX);
GetPersistentNumberNote(Tag_Path+"yShpY",yShpY);
GetPersistentStringNote(Tag_Path+"Calibration file path",pathname);
file_name=pathname+"D-ED_TiltCal_A"+Alpha+"_B"+binning+"_C"+CamL+".dm4";
GetPersistentStringNote(Tag_Path+"Material",material);
if (material=="")
{
if (!GetString("Material?",material,material));
exit(0);
}
//Set tilt increment to give a displacement of 40% of disc diameter
number tInc=Rr*0.8;//in pixels
SetPersistentNumberNote(Tag_Path+"TiltIncrement",tInc);
//NB sets a limit on the max number of tilts since the beam shift calibration
//is only correct for up to 80% of the diffraction pattern width.
image TiltCal := NewImageFromFile(file_name);
image Xsh=TiltCal[0,0,0,(2*nCals)+1,(2*nCals)+1,1]+TiltCal[0,0,2,(2*nCals)+1,(2*nCals)+1,3];
Xsh.SetName("X-shift with tilt");
//Xsh.DisplayAt(655,30);
//Xsh.SetWindowSize(200,200);
image Ysh=TiltCal[0,0,1,(2*nCals)+1,(2*nCals)+1,2]+TiltCal[0,0,3,(2*nCals)+1,(2*nCals)+1,4];
Ysh.SetName("Y-shift with tilt");
//Ysh.DisplayAt(875,30);
//Ysh.SetWindowSize(200,200);
number xCal=(0.8*_r)/(2*nCals);//distance between calibration points
number yCal=(0.8*_b)/(2*nCals);//in units of pixels in the diffraction pattern
//Get nTilts
number nTilts=99;
string prompt = "Number of beam tilts (+ & -): ";
result(prompt+" ");
if (!GetNumber(prompt,nTilts,nTilts))
exit(0)
//check that max number of beam tilts doesn't go outside 80% of image
if (nTilts*tInc > nCals*xCal)//NB work on smallest camera dimension
{
  nTilts=floor((nCals*xCal)/tInc);
}
result(nTilts+"\n")
number nPts = (2*nTilts+1)**2;
////////////////////
//Get initial state
number TiltX0,TiltY0,ShiftX0,ShiftY0
EMGetBeamTilt(TiltX0,TiltY0)
result("Initial beam tilts: TiltX = "+TiltX0+", TiltY = "+TiltY0+"\n")
EMGetBeamShift(ShiftX0,ShiftY0);
result("Initial beam shift: ShiftX = "+ShiftX0+", ShiftY = "+ShiftY0+"\n")

////////////////////
// Stop any current camera viewer
number close_view=1, stop_view=0;
cm_stopcurrentcameraViewer(stop_view);

////////////////////
//Start the camera running in fast mode
//Use current camera
object camera = CMGetCurrentCamera();
// Create standard parameters
number kUnprocessed = 1;
number kDarkCorrected = 2;
number kGainNormalized = 3;
number processing = kUnprocessed;	
// Define camera parameter set
object acq_params = camera.CM_CreateAcquisitionParameters_FullCCD(processing,expo,binning,binning);
acq_params.CM_SetDoContinuousReadout(true);
acq_params.CM_SetQualityLevel(0);//what does this do?
object acquisition = camera.CM_CreateAcquisition(acq_params);
object frame_set_info = acquisition.CM_ACQ_GetDetector().DTCTR_CreateFrameSetInfo();
img_src = alloc(CM_AcquisitionImageSource).IMGSRC_Init(acquisition,frame_set_info,0);
CM_ClearDarkImages()//why?
// Create and display fast image
img1:=acquisition.CM_CreateImageForAcquire("Acquired");
img1.DisplayAt(15,30);
img1.SetWindowSize(800,800);
number data_type=img1.GetDataType();
number imgX,imgY
img1.Get2DSize(imgX,imgY);

////////////////////
// Create 3D destination data stack
image CBED_stack := NewImage("CBED Stack",data_type,imgX,imgY,nPts);
CBED_stack = 0;
CBED_stack.DisplayAt(835,30)
CBED_stack.SetWindowSize(200,200)

////////////////////
//Go to first point
number i,j,pX,pY;
number pt=0;
GetCoordsFromNTilts(nTilts,pt,i,j);
//tilts to be applied, in pixels
pX=i*tInc;
pY=j*tInc;
//tilt to be applied, in DAC numbers
number tX=TiltX0 + xTpX*pX + yTpX*pY;
number tY=TiltY0 + xTpY*pX + yTpY*pY;
//linear interpolation for beam shift correction
number dX=pX/xCal % 1;//fractional coords 
number dY=pY/yCal % 1;
//The 4 surrounding shift calibrations
image xX=XSh[floor(pY/yCal)+nCals,floor(pX/xCal)+nCals,floor(pY/yCal)+nCals+2,floor(pX/xCal)+nCals+2];
image yY=YSh[floor(pY/yCal)+nCals,floor(pX/xCal)+nCals,floor(pY/yCal)+nCals+2,floor(pX/xCal)+nCals+2];
//interpolated values
number corX=Interp(Xx,dX,dY);
number corY=Interp(yY,dX,dY);
//shift correction vector
number sX=ShiftX0 + corX*xShpX + corY*yShpX;
number sY=ShiftY0 + corX*xShpY + corY*yShpY;
//set tilt and shift
EMSetBeamShift(sX,sY);
EMSetBeamTilt(tX,tY);
sleep(sleeptime);

////////////////////
// Start acquisition 
//NB define variables outside try/catch
number prog
try
{
 img_src.IMGSRC_BeginAcquisition()

 while(pt < Npts)
 {
   prog=round(100*(pt+1)/nPts);
   OpenAndSetProgressWindow("Data collection","Image "+(pt+1)+" of "+nPts," "+prog+" %");
   GetCoordsFromNTilts(nTilts,pt,i,j);
   //tX and tY as a fraction of image width
   pX=i*tInc;//in pixels
   pY=j*tInc;
   //tilt correction vector
   tX=TiltX0 + xTpX*pX + yTpX*pY;
   tY=TiltY0 + xTpY*pX + yTpY*pY;
   //linear interpolation for beam shift correction
   dX=pX/xCal % 1;
   dY=pY/yCal % 1;
   xX=XSh[floor(pY/yCal)+nCals,floor(pX/xCal)+nCals,floor(pY/yCal)+nCals+2,floor(pX/xCal)+nCals+2];
   yY=YSh[floor(pY/yCal)+nCals,floor(pX/xCal)+nCals,floor(pY/yCal)+nCals+2,floor(pX/xCal)+nCals+2];
   corX=Interp(Xx,dX,dY);
   corY=Interp(yY,dX,dY);
   //shift correction vector
   sX=ShiftX0 + corX*xShpX + corY*yShpX;
   sY=ShiftY0 + corX*xShpY + corY*yShpY;
   //set tilt and shift
   EMSetBeamShift(sX,sY);
   EMSetBeamTilt(tX,tY);
   sleep(sleeptime);
   UpdateCameraImage(img_src,img1);
   CBED_stack[0,0,pt,imgX,imgY,pt+1]=img1;
   pt++;
 }
}
catch
{// We're here because an error happened, stop the acquisition
 img_src.IMGSRC_FinishAcquisition();
}
//reset tilts to original values
EMSetBeamTilt(TiltX0,TiltY0);
EMSetBeamShift(ShiftX0,ShiftY0);

//Put tags in stack
CBED_stack.CM_WriteAcquisitionTagsToImage(camera,acq_params)
CBED_stack.SetStringNote("Info:Date",datetime)
CBED_stack.SetStringNote("Info:Path",pathname)
CBED_stack.SetNumberNote("Info:Camera Length",CamL)
CBED_stack.SetNumberNote("Info:Magnification",mag)
CBED_stack.SetNumberNote("Info:Alpha",Alpha)
CBED_stack.SetNumberNote("Info:Spot size",spot);
CBED_stack.SetNumberNote("Info:Disc Radius",Rr);
CBED_stack.SetStringNote("Info:Material",material)
CBED_stack.SetNumberNote("Tilts:xTpX",xTpX);
CBED_stack.SetNumberNote("Tilts:xTpY",xTpY);
CBED_stack.SetNumberNote("Tilts:yTpX",yTpX);
CBED_stack.SetNumberNote("Tilts:yTpY",yTpY);
CBED_stack.SetNumberNote("Tilts:Increment",tInc);
CBED_stack.SetNumberNote("Shifts:xShpX",xShpX);
CBED_stack.SetNumberNote("Shifts:xShpY",xShpY);
CBED_stack.SetNumberNote("Shifts:yShpX",yShpX);
CBED_stack.SetNumberNote("Shifts:yShpY",yShpY);

//Gain and Dark ref correction
pathname = PathConcatenate( GetApplicationDirectory("common_app_data",0),"Reference Images\\");
string refpath=pathname+"D-ED_DarkReference_Binning"+binning+".dm4";
image dark := NewImageFromFile(refpath);
refpath=pathname+"D-ED_GainReference_Binning"+binning+".dm4";
image gain := NewImageFromFile(refpath);
pt=0;
while (pt<nPts)
{
CBED_stack[0,0,pt,imgX,imgY,pt+1]-=dark;
CBED_stack[0,0,pt,imgX,imgY,pt+1]/=gain;
pt++
}
CBED_stack.SetLimits(CBED_stack.min(),CBED_stack.max())



//tidy up
img_src.IMGSRC_FinishAcquisition();
img1.DeleteImage();
//End of main program
GetDate(f,date_);
GetTime(f,time_);
datetime=date_+"_"+time_;
result("Acquisition complete: "+datetime+" ding, dong\n\n")










