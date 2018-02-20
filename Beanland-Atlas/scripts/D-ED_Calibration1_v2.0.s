// $BACKGROUND$
  //***            Calibrate tilts1            ***\\
 //** Richard Beanland r.beanland@warwick.ac.uk **\\
//*         Coding started June 2012              *\\
 
// 1.0, 23 June 2012
// 1.1 - added fast data collection March 2013
// 1.2 - fast data collection as a subroutine April 2013
// 1.3 - bug fixes April/May 2013
// 1.4 - added image shift calibration Summer 2014
// 1.5 - bug fixes Oct 2016
// 2.0 - split off from shift calibration Nov 2016

//Global variables
 number true=1, false=0;
 number Camb=2672, Camr=2688//Global values, for Orius SC600 camera
 number binning=4;//Could be a dialogue
 number expo=0.02;//Could be a dialogue
 number _t=0;
 number _l=0;
 number _b=Camb/binning; 
 number _r=Camr/binning;
 number sleeptime=0.05;//delay while waiting for microsope to respond
 image img1,img0,img2,imgCC;//live image, reference image, sobel filtered and cross-correlation respectively
 object img_src

//*******************************//
///////////////////////////
// Subroutines.
///////////////////////////
//*******************************//
//UpdateCameraImage
//RemoveOutliers
//GetCoordsFromNTilts
//EMChangeMode
//EMBeamCentre
//Tiltsize
//Shiftsize
//DiscSize
//Sobel


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


//FUNCTION remove outliers
//Remove outliers in the selection by comparing with an identical image with median filer applied
void RemoveOutliers(image img, number thr)
{
  image medImg:=img[];
  medImg=medianFilter(img,3,1);
  // replace ONLY those pixels in IMG which are >thr in comparison with median image
  img=tert((abs(img-medImg)>thr),medImg,img);
  //tidy up
  medImg.DeleteImage();
}//end of RemoveOutliers


//Function GetCoordsFromNTilts
//Gets x-y coords from the index currentPoint
void GetCoordsFromNTilts(number nTilts, number currentPoint, number &i, number &j)
{
  number side=2*nTilts+1;
  j=floor(currentPoint/side)-nTilts;
  i=((currentPoint%side)-nTilts)*((-1)**(j%2));//NB % means modulo, flips sign every row
}//End of GetCoordsFromNTilts


//Function EMChangeMode
//Asks user to change mode - will keep doing so until they comply
//Space bar to exit loop
void EMChangeMode(string mode_want)
{
  string mode_is=EMGetImagingOpticsMode();
  number clickedOK=true;
  while (!(mode_is==mode_want))//not in diffraction mode
  {
   clickedOK=true;
   external sem=NewSemaphore();
   try
   {
     ModelessDialog("Please put the microscope in "+mode_want+" mode","OK",sem);
     GrabSemaphore(sem);
     ReleaseSemaphore(sem);
     FreeSemaphore(sem);
   }
   catch
   {
     FreeSemaphore(sem);
     clickedOK=false;
     break;
   }
   mode_is=EMGetImagingOpticsMode();
  }
  // Give user some way out 
  if (spacedown()) throw("User aborted")
}//End of EMChangeMode


//Function EMBeamCentre
//Puts the beam in the centre of the image given measured position x0,y0
void EMBeamCentre(string Tag_Path)
{
  number ShiftX0,ShiftY0,xShpX,xShpY,yShpX,yShpY; 
  EMGetBeamShift(ShiftX0,ShiftY0);
  //Get beam shift calibration
  GetPersistentNumberNote(Tag_Path+"xShpX",xShpX);
  GetPersistentNumberNote(Tag_Path+"xShpY",xShpY);
  GetPersistentNumberNote(Tag_Path+"yShpX",yShpX);
  GetPersistentNumberNote(Tag_Path+"yShpY",yShpY);
  number x0,y0;//coords of untilted beam
  number maxval=img1.Max(x0,y0);
  number x1=(_r/2)-x0;//x1,y1 are the no. of pixels to move [x,y]
  number y1=(_b/2)-y0;
  number xCentre=round(ShiftX0+x1*xShpX+y1*yShpX);
  number yCentre=round(ShiftY0+x1*xShpY+y1*yShpY);
 EMSetBeamShift(xCentre,yCentre);
 sleep(sleeptime);
}//End of EMBeamCentre


//Function Tiltsize
//Changes tilt increment dTilt to be 1/4 of image height
number TiltSize(number dTilt, number &T1X, number &T1Y, number &T2X, number &T2Y)
{
 //auto-correlation to find the coords of untilted beam x0y0
 number tiltX0,tiltY0; 
 EMGetBeamTilt(tiltX0,tiltY0);
 UpdateCameraImage(img_src,img1);//zero tilt image
 img0=img1;//put into img0
 imgCC=img1.CrossCorrelate(img0);
 number x0,y0;//coords of untilted beam
 number maxval=imgCC.Max(x0,y0);
 //tilt the input guessed dTilt along X-DAC
 EMSetBeamTilt(tiltX0+dTilt,tiltY0);
 sleep(sleeptime);//give the microscope time to respond
 UpdateCameraImage(img_src,img1);//X-DAC tilted image
 imgCC=img1.CrossCorrelate(img0);
 number x,y;//coords of tilted beam
 maxval=imgCC.Max(x,y);
 number tPix=((x-x0)**2+(y-y0)**2)**0.5;//the spot movement in pixels
 //calculate accurate dTilt
 dTilt=dTilt*0.25*_b/tPix;
 
 //Now measure tilt(pixels) per DAC
 EMSetBeamTilt(tiltX0+dTilt,tiltY0);
 sleep(sleeptime);
 UpdateCameraImage(img_src,img1);//X-DAC tilted image
 imgCC=img1.CrossCorrelate(img0);
 maxval=imgCC.max(x,y);
 T1X=(x-x0)/dTilt;
 T1Y=(y-y0)/dTilt;
 EMSetBeamTilt(tiltX0,tiltY0+dTilt);
 sleep(sleeptime);
 UpdateCameraImage(img_src,img1);//Y-DAC tilted image
 imgCC=img1.CrossCorrelate(img0);
 imgCC.UpdateImage();
 maxval=imgCC.Max(x,y);
 T2X=(x-x0)/dTilt;
 T2Y=(y-y0)/dTilt;
 
 //reset tilt to zero again
 EMSetBeamTilt(tiltX0,tiltY0);
 sleep(sleeptime);
 
 return dTilt
}//End of TiltSize


//Function Shiftsize
//Changes shift increment dShift to be 1/4 image height
number ShiftSize(number dShift, number &Sh1X, number &Sh1Y, number &Sh2X, number &Sh2Y)
{
 //auto-correlation to find the coords of unshifted beam x0y0
 number shiftX0,shiftY0; 
 EMGetBeamShift(shiftX0,shiftY0);
 UpdateCameraImage(img_src,img1);//zero shift image
 img0=img1;//put into img0
 imgCC=img0.CrossCorrelate(img0);
 number x0,y0;//coords of unshifted beam
 number maxval=imgCC.Max(x0,y0);
 //Shift the input guessed dShift along X-DAC
 EMSetBeamShift(shiftX0+dShift,shiftY0);
 sleep(sleeptime);//have to give the microscope time to respond
 UpdateCameraImage(img_src,img1);//shifted image
 imgCC=img1.CrossCorrelate(img0);
 number x,y;//coords of shifted beam
 maxval=imgCC.Max(x,y);
 number tPix=((x-x0)**2+(y-y0)**2)**0.5;//the spot movement in pixels
 //calculate accurate dShift
 dShift=round(dShift*0.25*_b/tPix);
 
 //now measure shift(pixels) per DAC
 EMSetBeamShift(shiftX0+dShift,shiftY0);
 sleep(sleeptime);
 UpdateCameraImage(img_src,img1);//X-DAC shifted image
 imgCC=img1.CrossCorrelate(img0);
 maxval=imgCC.Max(x,y);
 Sh1X=(x-x0)/dShift;
 Sh1Y=(y-y0)/dShift;
 EMSetBeamShift(shiftX0,shiftY0+dShift);
 sleep(sleeptime);
 UpdateCameraImage(img_src,img1);//Y-DAC shifted image
 imgCC=img1.CrossCorrelate(img0);
 maxval=imgCC.Max(x,y);
 Sh2X=(x-x0)/dShift;
 Sh2Y=(y-y0)/dShift;

 //reset tilt to zero again
 EMSetBeamShift(shiftX0,shiftY0);
 sleep(sleeptime);
 
 return dShift
}//End of ShiftSize


//Function DiscSize
//Gives radius of CBED disc
number DiscSize(image cbed)//, number &x0, number &y0)
{
 result("Finding radius of CBED disc...\n");
 number imgX,imgY;
 cbed.GetSize(imgX,imgY);
 image disc = cbed.ImageClone()*0;
 number Rmax=round(min(imgX,imgY)/10);
 number Rr=0,Cc1=0,Cc2,dCc=1;
 
 while (dCc>0)
 {
  Rr+=1
  disc=tert(iradius<Rr,1,0);//trial image
  Cc2=max(disc.CrossCorrelate(cbed));//cross correlate
  dCc=Cc2-Cc1;//if better than the last one, dCc is +ve, otherwise exit
  Cc1=Cc2;
 }
 return Rr-1
}//End of DiscSize


//Function Sobel
//3x3 Sobel filter (should be ^0.5 at the end, but not done for speed)
void Sobel(image img)
{
 number imgX,imgY;
 img.GetSize(imgX,imgY);
 image diffX=img.ImageClone()*0;
 image diffY=diffX;
 //x gradient
 diffX[1,1,imgY-1,imgX-1]  =  3*img[0,0,imgY-2,imgX-2];
 diffX[1,1,imgY-1,imgX-1] += 10*img[1,0,imgY-1,imgX-2];
 diffX[1,1,imgY-1,imgX-1] +=  3*img[2,0,imgY  ,imgX-2];
 diffX[1,1,imgY-1,imgX-1] -=  3*img[0,2,imgY-2,imgX  ];
 diffX[1,1,imgY-1,imgX-1] -= 10*img[1,2,imgY-1,imgX  ];
 diffX[1,1,imgY-1,imgX-1] -=  3*img[2,2,imgY  ,imgX  ];
 //y gradient
 diffY[1,1,imgY-1,imgX-1]  =  3*img[0,0,imgY-2,imgX-2];
 diffY[1,1,imgY-1,imgX-1] += 10*img[0,1,imgY-2,imgX-1];
 diffY[1,1,imgY-1,imgX-1] +=  3*img[0,2,imgY-2,imgX  ];
 diffY[1,1,imgY-1,imgX-1] -=  3*img[2,0,imgY  ,imgX-2];
 diffY[1,1,imgY-1,imgX-1] -= 10*img[2,1,imgY  ,imgX-1];
 diffY[1,1,imgY-1,imgX-1] -=  3*img[2,2,imgY  ,imgX  ];
 img=diffX*diffX+diffY*diffY;
}//End of Sobel


//*******************************//
///////////////////////////////////
// Main program
///////////////////////////////////
//*******************************//
number tstart = GetHighResTickCount();

//Get Basic stuff to start
number spot=EMGetSpotSize()+1;
number mag=EMGetMagnification(); //Sometimes gives null answer, why??
number camL=EMGetCameraLength();
//And prompt input for alpha
number alpha=3;
if (!GetNumber("Alpha?",alpha,alpha)) exit(0)
//set up tag and file paths
string Tag_Path="DigitalDiffraction:Alpha="+alpha+":Binning="+binning+":CamL="+CamL+"mm:";
string pathname=PathConcatenate( GetApplicationDirectory("common_app_data",0),"Reference Images\\");
string file_name=pathname+"D-ED_TiltCal_A"+Alpha+"_B"+binning+"_C"+CamL+".dm4";
string datetime;
//number of points +/- to measure
number nTilts=8;
//make or load calibration data, image TiltCal
GetPersistentStringNote(Tag_Path+"Date",datetime);
image TiltCal;
number data_type=2;
if (datetime=="")
{
  result("No tilt calibration: making new calibration file");
  TiltCal := NewImage("Tilt/shift calibration",data_type,(2*nTilts)+1,(2*nTilts)+1,4);
}
else
{
  result("\nLast calibration "+datetime+"\n");
  TiltCal := NewImageFromFile(file_name);
}

//update tags
number f;
string date_;
GetDate(f,date_);
string time_;
GetTime(f,time_);
datetime=date_+"_"+time_;
result ("\nStarting tilt calibration "+datetime+"\n");
SetPersistentStringNote(Tag_Path+"Date",datetime);
SetPersistentNumberNote(Tag_Path+"Spot size",spot);
SetPersistentNumberNote(Tag_Path+"nCals",nTilts);

// Stop any current camera viewer
number close_view=1, stop_view=0;
try
{
  cm_stopcurrentcameraViewer(stop_view);
}
catch
{
  throw("Couldn't stop camera properly, try again!");
}

//set up images to contain data
number nPts=((2*nTilts)+1)**2;
//set up arrays holding shift correction
image Xsh=RealImage("X-shift with tilt",4,(2*nTilts)+1,(2*nTilts)+1 );
//Xsh.DisplayAt(655,30);
//Xsh.SetWindowSize(200,200);
image Ysh=RealImage("Y-shift with tilt",4,(2*nTilts)+1,(2*nTilts)+1 );
//Ysh.DisplayAt(875,30);
//Ysh.SetWindowSize(200,200);

//Start in diffraction mode 
EMChangeMode("DIFF")

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

// Create and display live image
img1:=acquisition.CM_CreateImageForAcquire("Live");
img1.DisplayAt(10,30);
img1.SetWindowSize(200,200);

//Set up reference image img0
img0:=img1.ImageClone();
//img0.DisplayAt(15,30);
//img0.SetWindowSize(200,200);
//img0.SetName("Reference");
//and img2 to display the measured beam positions
img2:=img1.ImageClone()*0;
img2.DisplayAt(225,30);
img2.SetWindowSize(200,200);
img2.SetName("Progress");
//and cross correlation image imgCC
imgCC:= img1.CrossCorrelate(img0);
//imgCC.DisplayAt(445,30);
//imgCC.SetWindowSize(200,200);
//imgCC.SetName("Cross correlation"); 


////////////////////
// Start acquisition 
//NB define variables outside try/catch
number Rr,T1X,T1Y,T2X,T2Y,dTilt,detT,xTpX,xTpY,yTpX,yTpY; 
number Sh1X,Sh1Y,Sh2X,Sh2Y,dShift,detSh,xShpX,xShpY,yShpX,yShpY;
number ShiftX0,ShiftY0,TiltX0,TiltY0,pt,maxval; 
number i,j,tX,tY,x,y,dx,dy,tIncX,tIncY,prog;
try
{
  img_src.IMGSRC_BeginAcquisition()
  UpdateCameraImage(img_src,img1);
  //put the untilted image into img0
  img0=img1;
 
  //Get radius of CBED disc
  Rr=DiscSize(img0);
  result ("Disc radius="+Rr+" pixels\n");
  SetPersistentNumberNote(Tag_Path+"Disc Radius",Rr);

  //Set magnitude of tilt
  // dTilt gives diffraction pattern shift 1/4 of camera height
  //[T1X,T1Y] is the beam tilt in pixels [X,Y] per x DAC
  dTilt=TiltSize(40000/CamL,T1X,T1Y,T2X,T2Y);//40000/CamL is initial guess
  detT=T1X*T2Y-T2X*T1Y;//determinant
  xTpX= T2Y/detT;//EMSetBeamTilt(xTpX,xTpY) shifts the disc 1 x-pixel 
  xTpY=-T1Y/detT;
  yTpX=-T2X/detT;//EMSetBeamTilt(yTpX,yTpY) shifts the disc 1 y-pixel 
  yTpY= T1X/detT;
  //Save them to the global tag group
  SetPersistentNumberNote(Tag_Path+"xTpX",xTpX);
  SetPersistentNumberNote(Tag_Path+"xTpY",xTpY);
  SetPersistentNumberNote(Tag_Path+"yTpX",yTpX);
  SetPersistentNumberNote(Tag_Path+"yTpY",yTpY);
 
  //Go to imaging mode
  EMChangeMode("MAG1")
  
  //Calibrate beam shift - assume it's linear so a single measurement of x and y DAC shift is fine.
  //[Sh1X,Sh1Y] is the beam shift in pixels [X,Y] per x DAC 
  dShift=ShiftSize(20,Sh1X,Sh1Y,Sh2X,Sh2Y);//first guess at shift increment is 20, changed to 1/4 of camera height
  detSh=Sh1X*Sh2Y-Sh2X*Sh1Y;//determinant
  xShpX= Sh2Y/detSh;//EMSetBeamShift(xShpX,xShpY) shifts the beam 1 x-pixel 
  xShpY=-Sh1Y/detSh;                   
  yShpX=-Sh2X/detSh;//EMSetBeamShift(yShpX,yShpY) shifts the beam 1 y-pixel 
  yShpY= Sh1X/detSh;
  //Save them to the global tag group
  SetPersistentNumberNote(Tag_Path+"xShpX",xShpX);
  SetPersistentNumberNote(Tag_Path+"xShpY",xShpY);
  SetPersistentNumberNote(Tag_Path+"yShpX",yShpX);
  SetPersistentNumberNote(Tag_Path+"yShpY",yShpY);
  
  //Centre beam shift and display start point
  UpdateCameraImage(img_src,img1);
  EMBeamCentre(Tag_Path);
  EMGetBeamShift(ShiftX0,ShiftY0);
  result ("Centred beam shift: "+ShiftX0+","+ShiftY0+"\n");
  SetPersistentNumberNote(Tag_Path+"ShiftX0",ShiftX0);
  SetPersistentNumberNote(Tag_Path+"ShiftY0",ShiftY0);
  EMGetBeamTilt(TiltX0,TiltY0);
  result ("Initial beam tilt: "+TiltX0+","+TiltY0+"\n")
  SetPersistentNumberNote(Tag_Path+"TiltX0",TiltX0);
  SetPersistentNumberNote(Tag_Path+"TiltY0",TiltY0); 

  /////////////////
  //Map beam shift as a function of tilt
  //Go to first point
  pt=0;
  GetCoordsFromNTilts(nTilts,pt,i,j);
  //increment in position of disc, in pixels
  tIncX=(_r/(2*nTilts))*0.8;//max beam tilt is 80% of diffraction pattern width from centre
  tIncY=(_b/(2*nTilts))*0.8;//max beam tilt is 80% of diffraction pattern height from centre
  tX=TiltX0 + (i*xTpX + j*yTpX)*tIncX;//convert from pixels to DAC for point i,j
  tY=TiltY0 + (i*xTpY + j*yTpY)*tIncY;
  EMSetBeamTilt(tX,tY);
  sleep(sleeptime);
  UpdateCameraImage(img_src,img1);//throw this image away
  //measure beam shifts and put into XSh,ySh
  while (pt<nPts)
  {
    //set tilt
    GetCoordsFromNTilts(nTilts,pt,i,j);
    tX=TiltX0 + (i*xTpX + j*yTpX)*tIncX;
    tY=TiltY0 + (i*xTpY + j*yTpY)*tIncY;
    EMSetBeamTilt(tX,tY);
    sleep(sleeptime);
    //get image
    UpdateCameraImage(img_src,img1);
    img0=img1;
    img0.RemoveOutliers(50);
    //measure beam position
    maxval=img0.max(x,y);
    //mark&save it
    img2[y-5,x-5,y+5,x+5]=100;
    img2.UpdateImage();
    dx=x-(_r/2);
    dy=y-(_b/2);
    Xsh.SetPixel(i+nTilts,j+nTilts,-dx);//NB negative of measured value so shift cancels tilt
    Ysh.SetPixel(i+nTilts,j+nTilts,-dy); 
    pt++;
  }
  //Put XSh and YSh into TiltCal layers 0 and 1 for saving
  TiltCal[0,0,0,(2*nTilts)+1,(2*nTilts)+1,1]=Xsh; 
  TiltCal[0,0,1,(2*nTilts)+1,(2*nTilts)+1,2]=Ysh;
  result ("Beam shift calibration collected\n")
}
catch
{// We're here because an error happened, stop the acquisition
  img_src.IMGSRC_FinishAcquisition();
}
//stop acquisition
img_src.IMGSRC_FinishAcquisition();

//Save calibration
SetPersistentStringNote(Tag_Path+"Calibration file path",pathname);
TiltCal.SetStringNote("Info:Path",pathname)
TiltCal.SaveAsGatan(file_name)
//Add tags & save Calibration (stack of 4 images)
TiltCal.CM_WriteAcquisitionTagsToImage(camera,acq_params)
TiltCal.SetStringNote("Info:Date",datetime)
TiltCal.SetNumberNote("Info:Camera Length",CamL)
TiltCal.SetNumberNote("Info:Magnification",mag)
TiltCal.SetNumberNote("Info:Alpha",Alpha)
TiltCal.SetNumberNote("Info:Spot size",spot);
TiltCal.SetNumberNote("Info:Disc Radius",Rr);
TiltCal.SetNumberNote("Tilts:xTpX",xTpX);
TiltCal.SetNumberNote("Tilts:xTpY",xTpY);
TiltCal.SetNumberNote("Tilts:yTpX",yTpX);
TiltCal.SetNumberNote("Tilts:yTpY",yTpY);
TiltCal.SetNumberNote("Shifts:xShpX",xShpX);
TiltCal.SetNumberNote("Shifts:xShpY",xShpY);
TiltCal.SetNumberNote("Shifts:yShpX",yShpX);
TiltCal.SetNumberNote("Shifts:yShpY",yShpY);

//tidy up
//reset tilts to original values
EMSetBeamTilt(TiltX0,TiltY0);
img0.DeleteImage();
img1.DeleteImage();
img2.DeleteImage();
imgCC.DeleteImage();
//End of main program

number tend = GetHighResTickCount();
result ("Elapsed time = "+CalcHighResSecondsBetween(tstart,tend)+" seconds\n");
result ("Calibration complete, ding dong\n\n")