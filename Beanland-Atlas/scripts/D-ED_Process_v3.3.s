// $BACKGROUND$
  //***  Process Digital Diffraction Pattern  ***\\
 //** Richard Beanland r.beanland@warwick.ac.uk **\\
//*        Coding started March 2013              *\\
 
// 2.8, 17 Apr 2013
// Orius SC600A on 2100 LaB6
// Complete new version to match compensated data collection
// 2.9 bug fixes April/May 2013
// 2.10 added image shift compensation
// 3.0 added several subroutines for cubic spline of background
// 3.1 background fit done on average CBED image rather than central CBED pattern

//Global variables
number true = 1, false = 0;
number pi=3.1415926535897932384626433832795;


//*******************************//
///////////////////////////
// Subroutines.
///////////////////////////
//*******************************//
//CalculateSplineConstants
//CubicSpline
//SplineRows
//SplineInterp
//DiscMask
//ROIpos
//UserG
//okGdialog
//GetCoordsFromNTilts
//AddYellowArrow
//GetMeanG
//GetG_vectors


// Function CalculateSplineConstants
image CalculateSplineConstants(image dataset)
// Calculates the cubic spline constants - by David Mitchell
{
 number n, sizex, sizey, minx, maxx, yspline, i, prevval, thisx,m,j
 dataset.getsize(sizex,sizey)
 // the number of data points
 n=sizex-1
 minmax(dataset[0,0,1,sizex], minx, maxx)

 // Arrays to store the data points
 // note the data start at pixel position 1 - pixel position 0 is not used
 image x=realimage("",4,sizex+1, 1)
 image a=realimage("",4,sizex+1, 1)
 image xa=realimage("",4,sizex+1, 1)
 image h=realimage("",4,sizex+1, 1)
 image xl=realimage("",4,sizex+1, 1)
 image xu=realimage("",4,sizex+1, 1)
 image xz=realimage("",4,sizex+1, 1)
 image b=realimage("",4,sizex+1, 1)
 image c=realimage("",4,sizex+1, 1)
 image d=realimage("",4,sizex+1, 1)
 image constantarray=realimage("",4,sizex+1,5) // stores all the constants and the x values

 // set the x and y arrays to the respective values in the dataset passed in
 x[0,1,1,sizex+1]=dataset[0,0,1,sizex]
 a[0,1,1,sizex+1]=dataset[1,0,2,sizex]
 m=n-1
 for(i=0; i<m+1;i++)
 {
   h[0,i+1,1,i+2]=getpixel(x,i+2,0)-getpixel(x, i+1,0)
 }
 for(i=1; i<m+1;i++)
 {
   xa[0,i+1,1,i+2]=3*(getpixel(a,i+2,0)*getpixel(h,i,0)-getpixel(a,i+1,0)*(getpixel(x,i+2,0)-getpixel(x,i,0))+getpixel(a,i,0)*getpixel(h,i+1,0))/(getpixel(h,i+1,0)*getpixel(h,i,0))
 }
 setpixel(xl,1,0,1)
 setpixel(xu,1,0,0)
 setpixel(xz,1,0,0)
 for(i=1;i<m+1;i++)
 {
   xl[0,i+1,1,i+2]=2*(getpixel(x,i+2,0)-getpixel(x,i,0))-getpixel(h,i,0)*getpixel(xu,i,0)
   xu[0,i+1,1,i+2]=getpixel(h,i+1,0)/getpixel(xl,i+1,0)
   xz[0,i+1,1,i+2]=(getpixel(xa,i+1,0)-getpixel(h,i,0)*getpixel(xz,i,0))/getpixel(xl,i+1,0)
 }
 setpixel(xl,n+1,0,1)
 setpixel(xz,n+1,0,0)
 setpixel(c,n+1,0,getpixel(xz,n+1,0))
 for(i=0;i<m+1;i++)
 {
   j=m-i
   c[0,j+1,1,j+2]=getpixel(xz,j+1,0)-getpixel(xu,j+1,0)*getpixel(c,j+2,0)
   b[0,j+1,1,j+2]=(getpixel(a,j+2,0)-getpixel(a,j+1,0))/getpixel(h,j+1,0)-getpixel(h,j+1,0)*(getpixel(c,j+2,0)+2*getpixel(c,j+1,0))/3
   d[0,j+1,1,j+2]=(getpixel(c,j+2,0)-getpixel(c,j+1,0))/(3*getpixel(h,j+1,0))
 }
 // Copy the a, b, c and d images to the array image and return it
 constantarray[0,0,1,sizex+1]=a[0,0,1,sizex+1]
 constantarray[1,0,2,sizex+1]=b[0,0,1,sizex+1]
 constantarray[2,0,3,sizex+1]=c[0,0,1,sizex+1]
 constantarray[3,0,4,sizex+1]=d[0,0,1,sizex+1]
 constantarray[4,0,5,sizex+1]=x[0,0,1,sizex+1]
 
 return constantarray
 
}//End of calculatesplineconstants


// Function CubicSpline
number CubicSpline(image constantarray, number xvalue, number extrapolate)
//Input an array of spline constants, an x value
// from which to interpolate a yvalue, and a boolean (interpolate). If interpolate is set to 1 then
// interpolation outside the range will be performed, otherwise values outside this range return a zero.
// the function returns the interpolated y value. As the constants are only calculated once in the
// other function, the computation in this function is very fast and suitable for use in a loop
// to calculate a range of y values
{
 number minx, maxx, sizex, sizey, yspline, i, n
 getsize(constantarray, sizex, sizey)
 // Get the minimum and maximum values in the x data -bottom row of the array from position 1 to n
 minmax(constantarray[4,1,5,sizex], minx, maxx) // ignore position 0
 // Check that the passed in xvalue is within the range of the xvalues in the data set supplied
 // If the extrapolate option is turned off (0) then the function returns a y value of zero
 if(extrapolate==0) // do not extrapolate, any out of range values of x result a y value of zero
 {
   if(xvalue<minx || xvalue>maxx)
   {
     yspline=0
     return yspline
   }
 }
 // loop through the x data (row 5 - position 4 in the constantarray) to find which interval the passed in xvalue lies in
 n=sizex-2 // note - 2 because pixel position 0 is unused.
 for(i=1;i<n;i++)
 {
   if(xvalue<getpixel(constantarray,1,4)) break
   if(xvalue>=getpixel(constantarray,i,4) && xvalue<getpixel(constantarray,i+1,4)) break
 }
 // Calculate the distance between the lower bound x data point for the interval and the passed in xvalue
 number xcalc=xvalue-getpixel(constantarray,i,4)
 // Compute the spline a=row 0, b=row 1, c=row 2 and d=row 3
 // y=a +bxcalc+c*ccalc^2+dxdcalc^3
 yspline=getpixel(constantarray,i,0)+getpixel(constantarray,i,1)*xcalc+getpixel(constantarray,i,2)*xcalc**2+getpixel(constantarray,i,3)*xcalc**3

 return yspline

}//end of CubicSpline


// Function SplineRows
image SplineRows(image RowsOut, image BackNumbers, number gmag)
// Calculates a set of cubic splines for several rows of measurement
//Input is the array of background measurements and an image to be filled with
//a cubic spline in one dimension. gmag the magnitude of the g-vector
//
{
 number nMeas1,nMeas2,LenSp,ind,jnd,yspline,s0,s1,s,ra;
 image SplineConstants;
 BackNumbers.GetSize(nMeas1,nMeas2);
 RowsOut.GetSize(LenSp,nMeas2);
 image Spline := RealImage("SplineFit",4,LenSp,1);
 //data to send to spline constant calculation routine, top row
 image SplineInput := RealImage("spline input",4,nMeas1,2);
 SplineInput[0,0,1,nMeas1]=round(icol*gmag);
 //Calculate cubic spline for rows and put into RowsOut
 for (jnd=0; jnd<nMeas2; jnd++)
 {
   //fill in the bottom row of the input array and calculate the spline
   SplineInput[1,0,2,nMeas1]=BackNumbers[icol,jnd];
   SplineConstants=SplineInput.CalculateSplineConstants()
   //SplineConstants.DisplayAt(100,100);
   for(ind=0; ind<LenSp;ind++)
   {
     yspline=SplineConstants.CubicSpline(ind,0)
     Spline.setpixel(ind,0,yspline)
   } 
   //Spline.UpdateImage();
   //put it in the appropriate row
    RowsOut[icol,jnd]=Spline;
  }

 return RowsOut
 
}//End of SplineRows


// Function SplineInterp
image SplineInterp(image BackOut, image SplineRow, image SplineCol, number g1mag, number g2mag )
//Takes two SplineRow images and produces a 2D interpolation of them
{
 number LenSp1,LenSp2,nMeas1,nMeas2,ind,jnd,knd,l,r,row,ra,Ishift;
 image avSpline:=RealImage("average spline",4,round(g1mag),1);
 //avSpline.DisplayAt(565,30);
 //avSpline.SetWindowSize(200,200);
 
 BackOut.GetSize(LenSp1,LenSp2);
 nMeas1=round(LenSp1/g1mag)+1;
 nMeas2=round(LenSp2/g2mag)+1;
 //result("nMeas:"+nMeas1+","+nMeas2+"\n")
 
 for (jnd=0; jnd<nMeas2-1; jnd++)
 { 
   for (ind=0; ind<nMeas1-1; ind++)
   {
     l=round(ind*g1mag);
     r=l+round(g1mag);
     if (ind==nMeas1-2)//shift for last column
     {
       r=r-1;
       l=l-1;
     }
     //Work out average spline curve row by row
     for (knd=0; knd<round(g2mag); knd++)
     {
       //current row in BackOut
       row=round(jnd*g2mag)+knd;
       row=(row>LenSp2)*LenSp2+(row<LenSp2)*row;//don't go outside the image
       //result("row="+row+"\n");
       //linear ratio
       ra=((round(g2mag)-knd)/round(g2mag));
       //Weighted average
       avSpline=ra*SplineRow[jnd,l,jnd+1,r]+(1-ra)*SplineRow[jnd+1,l,jnd+2,r];
       //Shift in intensity for this row
       Ishift=(SplineCol.GetPixel(ind,row)+SplineCol.GetPixel(ind+1,row)-avSpline.GetPixel(0,0)-avSpline.GetPixel(round(g1mag)-1,0))/2;
       BackOut[row,l,row+1,r]=avSpline+Ishift
     }
   }
 }
 return BackOut
}//End of SplineInterp


// Function DiscMask
image DiscMask(image LocalMask, number Rdisc, number g1X, number g1Y, number g2X, number g2Y)
// Makes a mask to cover the CBED discs
{
 LocalMask=1;
 number LrX=round((abs(g1X)+abs(g2X))/2);
 number LrY=round((abs(g1Y)+abs(g2Y))/2);
 image Disc:= RealImage("Disc",4,2*Rdisc,2*Rdisc);
 Disc=tert( (iradius<Rdisc), 0,1);//black disc
 number t,l,b,r,tt,ll,bb,rr
 //Note two possible cases, origin on the left (g1Y<0) or the top (g1Y>0)
 
 //half disc on left
 tt=(round(LrY-(g1Y+g2Y)/2)-Rdisc)*(g1Y<0)+(round(LrY-(g1Y-g2Y)/2)-Rdisc)*(g1Y>0);
 bb=(round(LrY-(g1Y+g2Y)/2)+Rdisc)*(g1Y<0)+(round(LrY-(g1Y-g2Y)/2)+Rdisc)*(g1Y>0);
 t=(tt>0)*tt+(tt<0)*0;
 l=0;
 b=(bb<(2*LrY+1))*bb+(bb>(2*LrY))*2*LrY;
 r=Rdisc;
 LocalMask[t,l,b,r]=LocalMask[t,l,b,r]*Disc[t-tt,RDisc,b-bb+2*RDisc,2*RDisc];
 
 //half disc on right
 tt=(round(LrY+(g1Y+g2Y)/2)-Rdisc)*(g1Y<0)+(round(LrY+(g1Y-g2Y)/2)-Rdisc)*(g1Y>0);
 bb=(round(LrY+(g1Y+g2Y)/2)+Rdisc)*(g1Y<0)+(round(LrY+(g1Y-g2Y)/2)+Rdisc)*(g1Y>0);
 t=(tt>0)*tt+(tt<0)*0;
 l=2*LrX-Rdisc;
 b=(bb<(2*LrY+1))*bb+(bb>(2*LrY))*2*LrY;
 r=2*LrX;
 LocalMask[t,l,b,r]=LocalMask[t,l,b,r]*Disc[t-tt,0,b-bb+2*RDisc,RDisc];
 
 //half disc on top
 ll=(round(LrX+(g1X-g2X)/2)-Rdisc)*(g1Y<0)+(round(LrX-(g1X+g2X)/2)-Rdisc)*(g1Y>0);
 rr=(round(LrX+(g1X-g2X)/2)+Rdisc)*(g1Y<0)+(round(LrX-(g1X+g2X)/2)+Rdisc)*(g1Y>0);
 t=0;
 l=(ll>0)*ll+(ll<0)*0;
 b=Rdisc;
 r=(rr<(2*LrX)+1)*rr+(rr>(2*LrX))*2*LrX;
 LocalMask[t,l,b,r]=LocalMask[t,l,b,r]*Disc[RDisc,l-ll,2*RDisc,r-rr+2*RDisc];
 
 //half disc on bottom
 ll=(round(LrX-(g1X-g2X)/2)-Rdisc)*(g1Y<0)+(round(LrX+(g1X+g2X)/2)-Rdisc)*(g1Y>0);
 rr=(round(LrX-(g1X-g2X)/2)+Rdisc)*(g1Y<0)+(round(LrX+(g1X+g2X)/2)+Rdisc)*(g1Y>0);
 t=2*LrY-Rdisc;
 l=(ll>0)*ll+(ll<0)*0;
 b=2*LrY;
 r=(rr<(2*LrX)+1)*rr+(rr>(2*LrX))*2*LrX;
 //result("t,l,b,r="+t+","+l+":"+b+","+r+","+"\n");
 LocalMask[t,l,b,r]=LocalMask[t,l,b,r]*Disc[0,l-ll,RDisc,r-rr+2*RDisc];
 
 //Radius
 LocalMask=tert(iradius<1*Rdisc,LocalMask,0)
 
 return LocalMask
}//End of DiscMask


//Function ROIpos
//Gives top,left,bottom,right of an ROI drawn by a user
void ROIpos(image img, string prompt, number &t, number &l, number &b, number &r)
{
 number IsizX,IsizY,clickedOK;
 img.GetSize(IsizX,IsizY);
 ImageDisplay img_disp = img.ImageGetImageDisplay(0);
 roi theROI = NewROI();
 theROI.ROISetRectangle(t,l,b,r);
 img_disp.ImageDisplayAddROI(theROI);
 clickedOK = true;
 external sem = NewSemaphore();
 try
 {
  ModelessDialog(prompt,"OK",sem);
  GrabSemaphore(sem);
  ReleaseSemaphore(sem);
  FreeSemaphore(sem);
 }
 catch
 {
  FreeSemaphore(sem);
  clickedOK = false;
  break;
 }
 img_disp.ImageDisplayDeleteROI(theROI);
 if(clickedOK)
 {
  theROI.ROIGetRectangle(t,l,b,r);
 }
}//End of function ROIpos


//Function GetCoordsFromNTilts
//Gets x-y coords from the index currentPoint
void GetCoordsFromNTilts(number nTilts, number currentPoint, number &i, number &j)
{
  number side=2*nTilts+1;
  j=floor(currentPoint/side)-nTilts;
  i=((currentPoint%side)-nTilts)*((-1)**(j%2));//NB % means modulo, flips sign every row
}//End of GetCoordsFromNTilts


//Function AddText
void AddText(image img, number x, number y, string text)
{
 number bigly=50
 component imgdisp=img.imagegetimagedisplay(0);
 component words=NewTextAnnotation((x+5),(y-bigly/2),text,bigly)
 words.ComponentSetForegroundColor(1,1,0)
 words.componentsetfontfacename("Microsoft Sans Serif")
 imgdisp.ComponentAddChildAtEnd(words)
}//end of AddText


//Function AddBlueCircle
void AddBlueCircle(image img, number t, number l, number b, number r)
{
 component imgdisp=img.imagegetimagedisplay(0);
 component ov=NewOvalAnnotation(t,l,b,r); //create a circle defined by top, left etc.
 ov.ComponentSetForegroundColor(0,0,1);   //make the circle blue
 imgdisp.ComponentAddChildAtEnd(ov);      //add the circle to the image
}//end of AddBlueCircle


//Function AddYellowArrow
void AddYellowArrow(image img, number x0, number y0, number x1, number y1)
{
 component imgdisp=imagegetimagedisplay(img,0);
 component arrowannot=NewArrowAnnotation(y0,x0,y1,x1); // create a single-headed arrow bounded by the rectangle defined by top, left etc.
 arrowannot.ComponentSetForegroundColor(1,1,0); // make the arrow yellow
 arrowannot.ComponentSetDrawingMode(1); // turn background fill to on
 arrowannot.ComponentSetBackgroundColor(0,0,0); // set the background fill to black
 imgdisp.ComponentAddChildAtEnd(arrowannot); // add the arrow to the image
}//end of AddYellowArrow


//Function DeleteStuff
//Deletes circles, lines and text from an image
void DeleteStuff(image img)
{
 component imgdisp=imagegetimagedisplay(img,0);
 number i,n,j
 for (i=0; i<15; i++)
 {
   n=imgdisp.componentcountchildrenoftype(i);
//   result(i+": "+n+" kids\n");
   for (j=0; j<n; j++)
   {
     component id=imgdisp.componentgetnthchildoftype(i,0)
     id.componentremovefromparent()
   }
 }
}//End of DeleteStuff


//Function UserG
//Gives top,left,bottom,right of an ROI drawn by a user
void UserG(image Avg, number Rr, number &g1X, number &g1Y, number &g2X, number &g2Y, number &pXavg, number &pYavg,)
{
 number t,l,b,r,IsizX,IsizY;
 string prompt;
 image AvgTemp
 Avg.GetSize(IsizX,IsizY);
 //Clean up average image
 Avg.DeleteStuff();
 //Get g-vectors manually
 prompt = "Position ROI on central beam and hit OK";
 result(prompt+"...");
 number i=0;
 number j=0;
 t=pYavg+g1Y*i+g2Y*j-Rr;
 l=pXavg+g1X*i+g2X*j-Rr;
 b=pYavg+g1Y*i+g2Y*j+Rr;
 r=pXavg+g1X*i+g2X*j+Rr;
 Avg.ROIpos(prompt,t,l,b,r);
 result("  done\n")
 pXavg=(l+r)/2
 pYavg=(t+b)/2
 //First diffraction vector
 prompt = "Position ROI on 1st g and hit OK"
 result(prompt+"...")
 i=2;
 j=0;
 t=pYavg+g1Y*i+g2Y*j-Rr;
 l=pXavg+g1X*i+g2X*j-Rr;
 b=pYavg+g1Y*i+g2Y*j+Rr;
 r=pXavg+g1X*i+g2X*j+Rr;
 Avg.ROIpos(prompt,t,l,b,r);
 number order=2;
 if (!GetNumber("Diffraction order?",order,order)) exit(0)
 result("  done\n")
 g1X=(((l+r)/2)-pXavg)/order;
 g1Y=(((t+b)/2)-pYavg)/order;
 //Second diffraction vector
 prompt = "Position ROI on 2nd g and hit OK"
 result(prompt+"...")
 i=0;
 j=2;
 t=pYavg+g1Y*i+g2Y*j-Rr;
 l=pXavg+g1X*i+g2X*j-Rr;
 b=pYavg+g1Y*i+g2Y*j+Rr;
 r=pXavg+g1X*i+g2X*j+Rr;
 Avg.ROIpos(prompt,t,l,b,r);
 order=2;
 if (!GetNumber("Diffraction order?",order,order)) exit(0)
 result("  done\n")
 g2X=(((l+r)/2)-pXavg)/order;
 g2Y=(((t+b)/2)-pYavg)/order;
 //show discs and g-vectors on average image
 for (i=-2; i<3; i++)
 {
  for(j=-2; j<3; j++)
  {
   Avg.AddBlueCircle(pYavg+g1Y*i+g2Y*j-Rr,pXavg+g1X*i+g2X*j-Rr,pYavg+g1Y*i+g2Y*j+Rr,pXavg+g1X*i+g2X*j+Rr);
  }
 }
 Avg.AddYellowArrow( pXavg,pYavg,(pXavg+g1X),(pYavg+g1Y) );
 Avg.AddYellowArrow( pXavg,pYavg,(pXavg+g2X),(pYavg+g2Y) );
 Avg.AddText((pXavg+g1X),(pYavg+g1Y),"1");
 Avg.AddText((pXavg+g2X),(pYavg+g2Y),"2");
}//End of function UserG


//Function GetMeanG
void GetMeanG(image MagTheta, image &Cluster, number &MeanMag, number &MeanTheta, number &MeanVx, number &MeanVy, number &nMeas)
{
 //incoming mean values are a single vector
 //outgoing mean values are an average of the vectors deemed to be in the cluster
 //incoming mag/theta is the same for +/- vectors [since tan(q+pi)=tan(q)]
 //so flip the vectors if the x-component has an opposite sign
 number tolMag=5;//tolerance in pixels to say it's in a cluster
 number tolAng=5;//tolerance in degrees to say it's in a cluster
 number n,nVecs,i,dTheta,dMag,ThetaSum,VxSum,VySum,x,y,signX;
 Cluster.Get2DSize(n,nVecs);
 nMeas=0;
 number MagSum=0;
 for (i=0; i<nVecs; i++)
 {
  dMag=abs(MeanMag - MagTheta.GetPixel(0,i));
  dTheta=abs(MeanTheta - MagTheta.GetPixel(1,i) );
  if ( (dTheta<tolAng )&&(dMag<tolMag ) )
  {
    nMeas++;
    MagSum+=MagTheta.GetPixel(0,i);
    ThetaSum+=MagTheta.GetPixel(1,i);
    x=MagTheta.GetPixel(2,i);
    y=MagTheta.GetPixel(3,i);//
    signX=abs(x)/x;
    VxSum+=x*(abs(MeanVx)/MeanVx)*signX;//second part here reverses
    VySum+=y*(abs(MeanVx)/MeanVx)*signX;//sign if x is opposite sign
    Cluster[i,0,i+1,1]=1;//it is in the cluster
    MeanMag=MagSum/nMeas;
    MeanTheta=ThetaSum/nMeas;
    MeanVx=VxSum/nMeas;
    MeanVy=VySum/nMeas;
  }
 }
}//End of GetMeanG


//Function GetG_vectors
//Gets the two smallest g-vectors [g1X,g1Y],[g2X,g2Y], given an image Avg and disc radius Rr
//will return null values if there aren't enough peaks to analyse
void GetG_vectors(image Avg, number Rr, number &g1X, number &g1Y, number &g2X, number &g2Y, number &pXavg, number &pYavg,)
{
 number npeaks=25;//maximum number of peaks to measure in the cross correlation
 number IsizX,IsizY;
 Avg.GetSize(IsizX,IsizY);
 IsizX=IsizX/2;//using an average image twice the width of the CBED stack
 IsizY=IsizY/2;//using an average image twice the height of the CBED stack
 //start by getting a cross-correlation image
 //image of a blank disk, radius Rr
 image Disc:=realimage("Disc",4,2*IsizX,2*IsizY);
 Disc=tert(iradius<Rr,1,0);
 //Disc.DisplayAt(225,30);
 //Disc.SetWindowSize(200,200);
 
 //Cross correlation between average image and the blank disc image
 //gives the position of the central beam
 image AvCC:=Avg.CrossCorrelate(Disc);
 number maxval,xp,yp;
 maxval=AvCC.max(xp,yp);
 pXavg=xp;
 pYavg=yp;
 result("000 beam is at ["+pXavg+","+pYavg+"]\n");
 
 //////////THIS WILL PROBABLY HAVE TO BE A USER INPUT/////////
 number dSize=3.5//a multiplying factor for the disc size when deleting peaks which have already been measured. Ideally dSize*Rr should be half of the smallest g-vector.
 number DdSize=round(dSize*Rr);
 Disc.deleteimage();//tidy up
 //delete all info below 2% of best correlation
 number _top=max(AvCC)*0.02;
 AvCC=tert( (AvCC>_top),AvCC,0.001);//make background not quite zero (mainly for debug, so can see the deleted peaks)
 //AvCC.DisplayAt(445,30);
 //AvCC.SetWindowSize(200,200);

 //x- and y-coords as column vectors
 image X:= RealImage("X-coords",4,1,npeaks);
 //X.DisplayAt(665,30);
 //X.SetDisplayType(5);//show as spreadsheet
 image Y:= RealImage("Y-coords",4,1,npeaks);
 //Y.DisplayAt(885,30);
 //Y.SetDisplayType(5);//show as spreadsheet
 image TempImg=realimage("Deleting disc",4,2*DdSize,2*DdSize);//A dark...
 TempImg=tert(iradius<dSize*Rr,0,1);//...circle
 //TempImg.DisplayAt(225,230);
 number i;
 number Nmax=nPeaks;
 number flag=0;
 for (i=0; i<nPeaks; i++)
 {//get peak position, in descending order of correlation/intensity
  if (max(AvCC)>_top)
  {//only keep going if there are peaks to be found
    maxval=AvCC.max(xp,yp);
    X.SetPixel(0,i,xp);//x-coord of peak
    Y.SetPixel(0,i,yp);//y-coord of peak
    Avg.AddBlueCircle(yp-Rr,xp-Rr,yp+Rr,xp+Rr);
    //result("Spot "+i+" = "+maxval+", at "+xp+","+yp+"\n");
    SetSelection(AvCC,(yp-DdSize),(xp-DdSize),(yp+DdSize),(xp+DdSize));
    AvCC[]*=TempImg;//this peak is done, delete it
    ClearSelection(AvCC);
  }
  else
  {  //No peaks left
    if (flag==0)
    {
      Nmax=i;//reduce number of peaks
      flag=1;
    } 
  }
 }
 nPeaks=nMax;
  if (nPeaks<3)
 {//there aren't enough detected spots, return null values
   //g1X=0;
   //g1Y=0;
   //g2X=0;
   //g2Y=0;
   return
 }
 //Find difference vectors Vx, Vy, by replicating X and Y into square matrices Xx, Xy, and subtracting the transpose
 image Xx:= RealImage("Xx",4,npeaks,npeaks)//
 image Vx:= RealImage("Vx",4,npeaks,npeaks)
 image Yy:= RealImage("Yy",4,npeaks,npeaks)
 image Vy:= RealImage("Vy",4,npeaks,npeaks)
 Xx=X[0,irow];
 Vx=Xx[irow,icol]-Xx;
 Yy=Y[0,irow];
 Vy=Yy[irow,icol]-Yy;
 //Polar coordinates, Vmag and Vtheta
 image Vmag:= RealImage("Vmag",4,npeaks,npeaks);
 Vmag=(( (Vx*Vx)+(Vy*Vy) )**0.5)*(irow>=icol);//irow>icol gives bottom left diagonal half;
 image Vtheta:= RealImage("Vtheta",4,npeaks,npeaks);
 number big=1000000;//an arbitrary number larger than anything else
 Vx=tert((Vx==0),big,Vx);//get rid of divide by zero error
 Vtheta=atan(Vy/Vx)*(irow>=icol);
 Vtheta=Vtheta*180/pi;
 //Sort by magnitude ascending into new column vector MagTheta
 number nVecs=(npeaks*npeaks-npeaks)/2;//number of different vectors
 image MagTheta:= RealImage("Mag-Theta-X-Y",4,4,nVecs);
 //MagTheta.DisplayAt(645,30);
 //MagTheta.SetName("Mag-Theta-X-Y");
 //MagTheta.SetDisplayType(5);//show as spreadsheet
 //MagTheta.SetWindowSize(120,500);
 Vmag=tert((Vmag==0),big,Vmag);//replace zeroes with this big number
 number Mag=Vmag.min(xp,yp);//lowest magnitude in the list
 i=0;
 while (Mag < big)//go through list until all are replaced by the large number 
 {
  MagTheta[i,0,i+1,1]=Vmag[yp,xp,yp+1,xp+1];//first col=magnitude
  MagTheta[i,1,i+1,2]=Vtheta[yp,xp,yp+1,xp+1];//second column=theta
  MagTheta[i,2,i+1,3]=Vx[yp,xp,yp+1,xp+1];//third column=Vx
  MagTheta[i,3,i+1,4]=Vy[yp,xp,yp+1,xp+1];//fourth column=Vy
  Vmag[yp,xp,yp+1,xp+1]=big;//this point is done, eliminate from Vmag
  i=i+1;
  Mag=Vmag.min(xp,yp);
 }
 //set sign of theta [not needed, just flip them in the sum
 //MagTheta[0,1,nVecs,2]=MagTheta[0,1,nVecs,2]*MagTheta[0,3,nVecs,3]/abs(MagTheta[0,3,nVecs,4]);//second column=theta

 //Find clusters - similar g-vectors in mag-theta space
 image Cluster:= RealImage("Cluster",4,1,nVecs);
 //Cluster.DisplayAt(645,30);
 //Cluster.SetName("Cluster");
 //Cluster.SetDisplayType(5);//show as spreadsheet
 //Cluster.SetWindowSize(120,500);
 image gVectors:= RealImage("g's",4,5,nVecs);
 //gVectors.DisplayAt(855,30);
 //gVectors.SetName("g-vectors");
 //gVectors.SetDisplayType(5);//show as spreadsheet
 //gVectors.SetWindowSize(120,500);
 number MeanMag=MagTheta.GetPixel(0,0);//start with mag of first point
 number MeanTheta=MagTheta.GetPixel(1,0);//and with angle of first point
 number MeanVx=MagTheta.GetPixel(2,0);//and with X of first point
 number MeanVy=MagTheta.GetPixel(3,0);//and with Y of first point
 number j=1;
 number k=0;
 number nMeas=1;//number of measured points to give an average g-vector
 //Go through and get clusters
 while (sum(Cluster)<nVecs)
 {
  GetMeanG(MagTheta,Cluster,MeanMag,MeanTheta,MeanVx,MeanVy,nMeas);
  gVectors[k,0,k+1,1]=MeanMag;
  gVectors[k,1,k+1,2]=MeanTheta;
  gVectors[k,2,k+1,3]=MeanVx;
  gVectors[k,3,k+1,4]=MeanVy;
  gVectors[k,4,k+1,5]=nMeas;
  //Find next unmatched point
  i=0;
  while (j==1)
  {
   i++
   j=Cluster.GetPixel(0,i);
  }
  j=1;
  MeanMag=MagTheta.GetPixel(0,i);//next point
  MeanTheta=MagTheta.GetPixel(1,i);//next point
  Cluster[i,0,i+1,1]=1;//next cluster
  k++
}
 //Output - the two smallest g-vectors
 g1X=gVectors.GetPixel(2,0);
 g1Y=gVectors.GetPixel(3,0);
 g2X=gVectors.GetPixel(2,1);
 g2Y=gVectors.GetPixel(3,1);
 result("Found "+nPeaks+" different CBED disks,\n")
 result("giving "+k+" different g-vectors\n")
 //show g-vectors on average image
 Avg.AddYellowArrow( pXavg,pYavg,(pXavg+g1X),(pYavg+g1Y) );
 Avg.AddYellowArrow( pXavg,pYavg,(pXavg+g2X),(pYavg+g2Y) );

}//End of GetG_vectors


//Function GetGids
//Gets HKL of g-vectors from user
void GetGids(number &g1H, number &g1K, number &g1L, number &g2H, number &g2K, number &g2L, number &g1Ag2, number &g1Mg2, number &gC)
{
 if (!GetNumber("First g index H?",g1H,g1H)) exit(0);
 if (!GetNumber("First g index K?",g1K,g1K)) exit(0);
 if (!GetNumber("First g index L?",g1L,g1L)) exit(0);
 if (!GetNumber("Second g index H?",g2H,g2H)) exit(0);
 if (!GetNumber("Second g index K?",g2K,g2K)) exit(0);
 if (!GetNumber("Second g index L?",g2L,g2L)) exit(0);
 result("g1: "+g1H+","+g1K+","+g1L+"\n");
 result("g2: "+g2H+","+g2K+","+g2L+"\n");
 if (!GetNumber("Angle between them?",g1Ag2,g1Ag2)) exit(0);
 if (!GetNumber("Ratio of magnitudes g1/g2?",g1Mg2,g1Mg2)) exit(0);
 result("Ratio of g-vector magnitudes = "+g1Mg2+", angle= "+g1Ag2+" degrees\n");
 if (!GetNumber("Centring g-vector (0,1,2)?",gC,gC)) exit(0);
 if (gC==0) result("The pattern is not face-centred\n");
 if (gC==1) result("g1 is a face-centring vector\n");
 if (gC==2) result("g2 is a face-centring vector\n");
}//End of function GetGids

//*******************************//
///////////////////////////////////
// Main program
///////////////////////////////////
//*******************************//

number f_;
string date_;
GetDate(f_,date_);
string time_;
GetTime(f_,time_);
string datetime=date_+"_"+time_;
result("\nStarting processing "+datetime+"\n")

///////////////
// Get 3D data stack
image CBED_stack := GetFrontImage();
number IsizX,IsizY,nPts;
CBED_stack.Get3DSize(IsizX,IsizY,nPts);
number data_type = CBED_stack.GetDataType();
number Rr,tInc,CamL,mag,Alpha,spot,xTpX,xTpY,yTpX,yTpY,xShpX,xShpY,yShpX,yShpY;
string material;
//get image tags
{
CBED_stack.GetStringNote("Info:Date",datetime);
CBED_stack.GetNumberNote("Info:Camera Length",CamL);
CBED_stack.GetNumberNote("Info:Magnification",mag);
CBED_stack.GetNumberNote("Info:Alpha",Alpha);
CBED_stack.GetNumberNote("Info:Spot size",spot);
CBED_stack.GetNumberNote("Info:Disc Radius",Rr);
CBED_stack.GetStringNote("Info:Material",material);
CBED_stack.GetNumberNote("Tilts:xTpX",xTpX);
CBED_stack.GetNumberNote("Tilts:xTpY",xTpY);
CBED_stack.GetNumberNote("Tilts:yTpX",yTpX);
CBED_stack.GetNumberNote("Tilts:yTpY",yTpY);
CBED_stack.GetNumberNote("Tilts:Increment",tInc);
CBED_stack.GetNumberNote("Shifts:xShpX",xShpX);
CBED_stack.GetNumberNote("Shifts:xShpY",xShpY);
CBED_stack.GetNumberNote("Shifts:yShpX",yShpX);
CBED_stack.GetNumberNote("Shifts:yShpY",yShpY);
if (material=="")
{
if (!GetString("Material?",material,material));
exit(0);
}
result("Material is "+material+"\n");
//Disc movement between images in pixels
result("tilt increment="+tInc+"\n");
}

///////////////
//Make sum of all individual images
number nTilts=((nPts**0.5)-1)/2;//***//***//
result("Data contains +/- "+nTilts+" beam tilts\n")
result("CBED disc radius is "+Rr+" pixels\n")
number _i,_j,prog;
//average CBED is Avg
image Avg:=RealImage("Average_CBED",4,2*IsizX,2*IsizY);
//number of CBED patterns contibuting to each pixel of Avg is stored in AvgC
image AvgC:=RealImage("counts of CBED image",4,2*IsizX,2*IsizY);
result("Creating average CBED pattern...")
//run through CBED stack
number disX,disY,pt;//vector describing displacement of a tilted CBED pattern
number minX=IsizX,minY=IsizY;//most negative displacement values
for (pt=0; pt<nPts; pt++)
{
  prog=round(100*(pt+1)/nPts)
  OpenAndSetProgressWindow("Average CBED pattern","Image "+(pt+1)+" of "+nPts," "+prog+" %");
  GetCoordsFromNTilts(nTilts,pt,_i,_j);
  disX=-round(_i*tInc)+round(0.5*IsizX);
  if (disX<minX) minX=disX;
  disY=-round(_j*tInc)+round(0.5*IsizY);
  if (disY<minY) minY=disY;
  Avg[disY,disX,IsizY+disY,IsizX+disX]+=CBED_stack[0,0,pt,IsizX,IsizY,pt+1];
  AvgC[disY,disX,IsizY+disY,IsizX+disX]+=1;
  Avg.UpdateImage();
}
//make average
Avg=tert( (Avg>0), Avg/AvgC,0);
//fill outside with representative background value
number rBack=Avg.GetPixel(minX+5,minY+5);
//result("\nrBack="+rBack+"\n");
//result("minX="+minX+", minY="+minY+"\n");
Avg=tert( (Avg>0), Avg, rBack);
result("  done\n");
Avg.DisplayAt(30,30);
Avg.SetWindowSize(600,600);

///////////////
//Find g-vectors
number g1X,g1Y,g1mag,g2X,g2Y,g2mag,pXavg,pYavg,ratio,theta;
Avg.GetG_vectors(Rr,g1X,g1Y,g2X,g2Y,pXavg,pYavg);
number pX=pXavg-round(0.5*IsizX);//there's a difference between the centre 
number pY=pYavg-round(0.5*IsizY);//in the avg image and the CBED stack
//Check for failure to get g-vectors
if (g1X**2 > (IsizX**2)/4) g1X=0
if (g1Y**2 > (IsizX**2)/4) g1Y=0
if ( ((g1X+g1Y+g2X+g2Y)==0) )
{
  result("Cannot find g-vectors!");
}
else
{//output g-vector statistics
  Avg.AddText((pXavg+g1X),(pYavg+g1Y),"1");
  Avg.AddText((pXavg+g2X),(pYavg+g2Y),"2");
  g1mag=(g1X**2+g1Y**2)**0.5;
  g2mag=(g2X**2+g2Y**2)**0.5;
  result("g1 = ["+g1X+","+g1Y+"], magnitude "+g1mag+"\n");
  result("g2 = ["+g2X+","+g2Y+"], magnitude "+g2mag+"\n");
  ratio=g1mag/g2mag;;
  theta=180*acos((g1X*g2X+g1Y*g2Y)/(g1mag*g2mag))/pi;
  result("Ratio of g-vector magnitudes = "+ratio+", angle= "+theta+"\n");
}

///////////////
//Check g-vectors are acceptable, if not do them manually
number t,l,b,r;
string prompt;
while(!TwoButtonDialog("Are the measured g-vectors good?","Yes","No") )
{
  Avg.UserG(Rr,g1X,g1Y,g2X,g2Y,pXavg,pYavg);
  pX=pXavg-round(0.5*IsizX);
  pY=pYavg-round(0.5*IsizY);
  g1mag=(g1X**2+g1Y**2)**0.5;
  g2mag=(g2X**2+g2Y**2)**0.5;
  result("000 beam is at ["+pX+","+pY+"]\n");
  result("g1: "+g1X+", "+g1Y+", magnitude "+g1mag+"\n");
  result("g2: "+g2X+", "+g2Y+", magnitude "+g2mag+"\n");
  ratio=g1mag/g2mag;;
  theta=180*acos((g1X*g2X+g1Y*g2Y)/(g1mag*g2mag))/pi;
  result("Ratio of g-vector magnitudes = "+ratio+", angle= "+theta+"\n");
}

///////////////
//g-vector calculations
//make sure a perfect alignment (g1Y=0) never happens
if (g1Y==0) g1Y=g1Y+0.000000001
number dot=(g1X*g2X+g1Y*g2Y)/(g1mag*g2mag);//gives cos(theta)
number cross=(g1X*g2Y-g1Y*g2X)/(g1mag*g2mag);//g1 x g2 gives sin(theta)
number swap;
number redo=0;//flag to redraw annotations
if (sgn(dot)<0)//90<theta<270
{//if the angle between g1 and g2 is not clockwise 0<theta<180 degrees, swap them
  result("Swapping g-vectors to make a right-handed pair\n")
  result("Note the new values: x=right, y=down\n");
  if (sgn(cross)<0)//180<theta<270
  {//change the sign of g2
    g2X=-g2X;
    g2Y=-g2Y;
    //result("changed sign of g2\n")
  }
  else//90<theta<180
  {//g1=-g2(old),g2=g1(old)
    swap=g1X;
	g1X=-g2X;
	g2X=swap;
	swap=g1Y;
	g1Y=-g2Y;
	g2Y=swap;
	swap=g1mag;
	g1mag=g2mag;
	g2mag=swap;
    //result("g1(new)=-g2(old),g2(new)=g1(old)\n")
  }
  redo=1;
}
else
{
  if (sgn(cross)<0)//270<theta<360
  {//swap g1 and g2
    result("Swapping g-vectors to make a right-handed pair\n");
    result("Note the new values: x=right, y=down\n");
    swap=g1X;
	g1X=g2X;
	g2X=swap;
	swap=g1Y;
	g1Y=g2Y;
	g2Y=swap;
	swap=g1mag;
	g1mag=g2mag;
	g2mag=swap;
    //result("g1(new)=g2(old),g2(new)=g1(old)\n")
    redo=1;
  }
}
//make g1X positive
if (g1X<0)
{
  result("Swapping g-vectors to make a right-handed pair\n");
  result("Note the new values: x=right, y=down\n");
  g1X=-g1X;
  g1Y=-g1Y;
  g2X=-g2X;
  g2Y=-g2Y;
  redo=1;
}
//redraw discs and g-vectors on average image
if (redo==1)
{
 Avg.DeleteStuff();
 number i,j
 for (i=-2; i<3; i++)
 {
  for(j=-2; j<3; j++)
  {
   Avg.AddBlueCircle(pYavg+g1Y*i+g2Y*j-Rr,pXavg+g1X*i+g2X*j-Rr,pYavg+g1Y*i+g2Y*j+Rr,pXavg+g1X*i+g2X*j+Rr);
  }
 }
 Avg.AddYellowArrow( pXavg,pYavg,(pXavg+g1X),(pYavg+g1Y) );
 Avg.AddYellowArrow( pXavg,pYavg,(pXavg+g2X),(pYavg+g2Y) );
 Avg.AddText((pXavg+g1X),(pYavg+g1Y),"1");
 Avg.AddText((pXavg+g2X),(pYavg+g2Y),"2");
}
//recalculate dot & cross products
dot=(g1X*g2X+g1Y*g2Y)/(g1mag*g2mag);//gives cos(theta)
cross=(g1X*g2Y-g1Y*g2X)/(g1mag*g2mag);//g1 x g2 gives sin(theta)
theta=(acos(dot)*sgn(cross));
//result("theta="+180*theta/pi+"\n");
result("g1 = ["+g1X+","+g1Y+"]:  g2 = ["+g2X+","+g2Y+"]\n");
//Angle phi between g1 and the x-axis
number dotX=g1X/g1mag;//gives cos(phi)
number crossX=-g1Y/g1mag;//g1 x [100] gives sin(phi)
number phi=(acos(dotX)*sgn(crossX))
//result("phi="+180*phi/pi+"\n");

//Get g-vector details
number nG1=3;
if (!GetNumber("Number of spots for 1st g (+/-)?",nG1,nG1)) exit(0);
number nG2=3;
if (!GetNumber("Number of spots for 2nd g (+/-)?",nG2,nG2)) exit(0);
//Diameter of circular selection
number Wfrac=75
if (!GetNumber("Percentage of spot used, 67-100%?",Wfrac,Wfrac))
exit(0);
number g1H=1;
number g1K=0;
number g1L=0;
number g2H=0;
number g2K=1;
number g2L=0;
number g1Ag2=90;//nominal angle between g1 and g2
number g1Mg2=1;//nominal ratio of magnitudes, g1/g2
number gC=0;//for centred patterns, 0=no centring, otherwise the g-vector in the centre
GetGids(g1H,g1K,g1L,g2H,g2K,g2L,g1Ag2,g1Mg2,gC);
while(!TwoButtonDialog("Are the g-vector HKLs correct?","Yes","No") )
{
 result("Repeating g-vector HKLs\n") 
 GetGids(g1H,g1K,g1L,g2H,g2K,g2L,g1Ag2,g1Mg2,gC);
}
//Put tags into Avg
{
Avg.SetStringNote("Info:Date",datetime);
Avg.SetNumberNote("Info:Camera Length",CamL);
Avg.SetNumberNote("Info:Magnification",mag);
Avg.SetNumberNote("Info:Alpha",Alpha);
Avg.SetNumberNote("Info:Spot size",spot);
Avg.SetNumberNote("Info:Disc Radius",Rr);
Avg.SetStringNote("Info:Material",material);
Avg.SetNumberNote("g-vectors:nG1",nG1);
Avg.SetNumberNote("g-vectors:nG2",nG2);
Avg.SetNumberNote("g-vectors:g1X",g1X);
Avg.SetNumberNote("g-vectors:g1Y",g1Y);
Avg.SetNumberNote("g-vectors:g2X",g2X);
Avg.SetNumberNote("g-vectors:g2Y",g2Y);
Avg.SetNumberNote("g-vectors:g1H",g1H);
Avg.SetNumberNote("g-vectors:g1K",g1K);
Avg.SetNumberNote("g-vectors:g1L",g1L);
Avg.SetNumberNote("g-vectors:g2H",g2H);
Avg.SetNumberNote("g-vectors:g2K",g2K);
Avg.SetNumberNote("g-vectors:g2L",g2L);
Avg.SetNumberNote("g-vectors:g1Ag2",g1Ag2);
Avg.SetNumberNote("g-vectors:g1Mg2",g1Mg2);
Avg.SetNumberNote("g-vectors:gC",gC);
Avg.SetNumberNote("g-vectors:pXavg",pXavg);
Avg.SetNumberNote("g-vectors:pYavg",pYavg);
}

///////////////
// Background calculation
//the number of background measurements needed for g1 and g2
number nMeas1=2*nG1+2;//from -nG1 to +nG1, plus the zero column and the final row
number nMeas2=2*nG2+2;
//size of spline fits are |g1|*(nMeas1-1) & (nMeas2-1)*|g2|
number LenSp1=round(g1mag*(nMeas1-1));
number LenSp2=round(g2mag*(nMeas2-1));
//size of LocalBackImg
number LrX=round((abs(g1X)+abs(g2X))/2);
number LrY=round((abs(g1Y)+abs(g2Y))/2);
number Rdisc=round(min(g1mag,g2mag)/2);

//images for calculation of 2D background
image LocalBackImg := RealImage("Local area",4,2*LrX,2*LrY);//image for measurement of local background
image LocalMask:= RealImage("Local mask",4,2*LrX,2*LrY);//same size image LocalMask
image BackNumbers := RealImage("Background measurements",4,nMeas1,nMeas2);//image to store array of background measurements
image BackNumbers1 := RealImage("Background flip",4,nMeas1,nMeas2);//to generate numbers for zero measurements
image BackNumbersTr:= RealImage("Transposed measurements",4,nMeas2,nMeas1);//transposed version for column calculation
image Rows := RealImage("Spline rows",4,LenSp1,nMeas2);//interpolated on rows
image RowsTr := RealImage("Tr(Spline rows)",4,nMeas2,LenSp1);//transposed rows
image Cols := RealImage("Spline columns",4,nMeas1,LenSp2);//interpolated on columns
image ColsTr := RealImage("Tr(Spline columns)",4,LenSp2,nMeas1);//transposed to use row calc
image Back := RealImage("Background",4,LenSp1,LenSp2);//average
image BackTr := RealImage("Background (Cols only)",4,LenSp2,LenSp1);//transposed interpolated
number LenW1=round(LenSp1+abs(LenSp2*cos(theta)));
image BackShear := RealImage("Background sheared",4,LenW1,LenSp2);//first shear the image
image BackFlip := RealImage("Background flipped",4,LenW1,LenSp2);//horizontal flip needed for theta>90
number LenW2=round(LenSp2*sin(theta));
image BackWarp := RealImage("Background warped",4,LenW1,LenW2);//second compress it, maintains g2 length
image BackRot = BackWarp.Rotate(phi);//rotated to match the image
BackRot.SetName("Background rotated");
number RsizX,RsizY;
BackRot.GetSize(RsizX,RsizY);
//Make the mask
DiscMask(LocalMask,Rdisc,g1X,g1Y,g2X,g2Y);
number X,Y,ind,jnd,knd,t1,l1,b1,r1,inside,row,ra,Ishift
//LocalMask.DisplayAt(550,50);
//Back.DisplayAt(30,30);
//Back.SetWindowSize(300,300);
//BackTr.DisplayAt(30,350);
//BackTr.SetWindowSize(300,300);
//BackShear.DisplayAt(330,30);
//BackShear.SetWindowSize(300,300);
//BackFlip.DisplayAt(330,350);
//BackFlip.SetWindowSize(300,300);
//BackWarp.DisplayAt(660,30);
//BackWarp.SetWindowSize(300,300);
//BackRot.DisplayAt(660,350);
//BackRot.SetWindowSize(300,300);


///////////////
// Measure background values from array of areas between discs in average image
BackNumbers=0;//reset measurements
number gNo=0;
for (ind=-nG1; ind<nG1+2; ind++)
{ 
  for (jnd=-nG2; jnd<nG2+2; jnd++)
  {
    prog=round(100*( gNo/((2*nG1+1)*(2*nG2+1)) ));
    OpenAndSetProgressWindow("Measure background...","Image "+gNo+" of "+(2*nG1+1)*(2*nG2+1)," "+prog+" %");
   //GetCoordsFromNTilts(nTilts,pt,_i,_j);
    //appropriate point -(g1+g2)/2 from disk jnd,ind
    X=round(pXavg+ind*g1X+jnd*g2X-(g1X+g2X)/2);
    Y=round(pYavg+ind*g1Y+jnd*g2Y-(g1Y+g2Y)/2);
    t=Y-LrY;
    b=Y+LrY;
    l=X-LrX
    r=X+LrX;
    inside=!((l<0)+(r>2*IsizX)+(t<0)+(b>2*IsizY));//could be more sophisticated here and use part of the mask when it goes outside the image
    if (inside)//mask off CBED discs and get mean value of what remains
    {
      LocalBackImg=Avg[t,l,b,r]*LocalMask;
      //Put the measurement into BackNumbers
      BackNumbers.SetPixel(ind+nG1,jnd+nG2,sum(LocalBackImg)/sum(LocalMask));
    }
    gNo++;
  }
}

///////////////
// Make 2D cubic Splines
{
OpenAndSetProgressWindow("Making 2D cubic spline...","rows"," ");
//Construct a background in an orthogonal image
//Make an set of spline rows
Rows.SplineRows(BackNumbers,g1mag);
//Make a (transposed) set of spline columns
BackNumbersTr=BackNumbers[irow,icol];
OpenAndSetProgressWindow("Making 2D cubic spline...","columns"," ");
ColsTr.SplineRows(BackNumbersTr,g2mag);
//transpose back to columns again
Cols=ColsTr[irow,icol];
//Combine the two 1D solutions for each reflection
OpenAndSetProgressWindow("Making 2D cubic spline...","rows+columns"," ");
//use row splines for the image and column splines to match intensities
Back.SplineInterp(Rows,Cols,g1mag,g2mag);
//use column splines for the image and row splines to match intensities (all transposed)
RowsTr=Rows[irow,icol];
OpenAndSetProgressWindow("Making 2D cubic spline...","columns+rows"," ");
BackTr.SplineInterp(ColsTr,RowsTr,g2mag,g1mag);
//take the average of the two solutions
Back=(Back+BackTr[irow,icol])/2;
OpenAndSetProgressWindow("Making 2D cubic spline...","scaling"," ");
}

///////////////
// Warp to match g-vectors
//Deform the background image to match the g-vectors
{
if (theta<pi/2)
{//deformations leave the top left pixel [0,0] unchanged
  BackShear=Back[icol-irow*cos(theta),irow];//shear
  BackWarp=BackShear[icol,irow/sin(theta)];//+squash=rotation
}
else
{//the same with flips to leave the top right pixel unchanged
  BackFlip=Back[LenSp1-icol,irow];
  BackShear=BackFlip[icol-irow*cos(pi-theta),irow];
  BackFlip=BackShear[LenSp1+LenSp2*cos(pi-theta)-icol,irow];
  BackWarp=BackFlip[icol,irow/sin(theta)];
}
BackRot=BackWarp.Rotate(phi);
}

///////////////
// Subtract from CBED stack
{
for (pt=0; pt<nPts; pt++)
{
  prog=round(100*(pt+1)/nPts);
  OpenAndSetProgressWindow("Background removal","Image "+(pt+1)+" of "+nPts," "+prog+" %");
  //get the coordinates to put the backround into.
  GetCoordsFromNTilts(nTilts,pt,_i,_j);
  //Datum background point is given by -nG1,-nG2
  X=floor(pX+_i*tInc-(nG1+0.5)*g1X-(nG2+0.5)*g2X);
  Y=floor(pY+_j*tInc-(nG1+0.5)*g1Y-(nG2+0.5)*g2Y);
  //offset of origin in rotated background
  if (phi<0)//NB we avoid rotations larger than +/-90 by choosing g-vectors correctly
  {
    X=X-round(LenW2*abs(sin(phi)));
  }
  else
  {
    Y=Y-round(LenW1*abs(sin(phi)));
  }
  //check edges of ROI in CBED stack
  t=(Y>0)*Y+(Y<0)*0;
  l=(X>0)*X+(X<0)*0;
  b=((Y+RsizY)<IsizY)*(Y+RsizY)+((Y+RsizY)>=IsizY)*(IsizY);
  r=((X+RsizX)<IsizX)*(X+RsizX)+((X+RsizX)>=IsizX)*(IsizX);
  //check edges of ROI in BackRot
  t1=(Y>0)*0+(Y<0)*(-Y);
  l1=(X>0)*0+(X<0)*(-X);
  b1=((Y+RsizY)<IsizY)*RsizY+((Y+RsizY)>=IsizY)*(IsizY-Y);
  r1=((X+RsizX)<IsizX)*RsizX+((X+RsizX)>=IsizX)*(IsizX-X);
  //result("t,l,b,r="+t+","+l+":"+b+","+r+","+"\n");
  //result("t,l,b,r1="+t1+","+l1+":"+b1+","+r1+","+"\n");
  CBED_stack[l,t,pt,r,b,pt+1]=CBED_stack[l,t,pt,r,b,pt+1]-BackRot[t1,l1,b1,r1];
}
//subtract from avg
X=floor(pXavg-(nG1+0.5)*g1X-(nG2+0.5)*g2X)-round(LenW2*abs(sin(phi)))*(phi<0);
Y=floor(pYavg-(nG1+0.5)*g1Y-(nG2+0.5)*g2Y)-round(LenW1*abs(sin(phi)))*(phi>0);
//check edges of ROI in CBED stack
t=(Y>0)*Y+(Y<0)*0;
l=(X>0)*X+(X<0)*0;
b=((Y+RsizY)<2*IsizY)*(Y+RsizY)+((Y+RsizY)>2*IsizY)*(2*IsizY);
r=((X+RsizX)<2*IsizX)*(X+RsizX)+((X+RsizX)>2*IsizX)*(2*IsizX);
//check edges of ROI in BackRot
t1=(Y>0)*0+(Y<0)*(-Y);
l1=(X>0)*0+(X<0)*(-X);
b1=((Y+RsizY)<2*IsizY)*RsizY+((Y+RsizY)>2*IsizY)*(2*IsizY-Y);
r1=((X+RsizX)<2*IsizX)*RsizX+((X+RsizX)>2*IsizX)*(2*IsizX-X);
//Avg[t,l,b,r]=Avg[t,l,b,r]-BackRot[t1,l1,b1,r1];
}

///////////////
// Create 3D data stack for D-LACBED images
result("Creating stack of D-LACBED images...");
image DLACBEDimg:=NewImage("D-LACBED Stack",data_type,IsizX,IsizY,((2*nG1+1)*(2*nG2+1)));
DLACBEDimg=0;
DLACBEDimg.DisplayAt(0,625);
DLACBEDimg.SetWindowSize(200,200);

//Create scratch image for calculation of average
image ScratImg := RealImage("Average",4,IsizX,IsizY)
number Rr2=round(Wfrac*Rr/100);
//other images for cut and copy
image TempImg := RealImage("Disk",4,2*Rr2,2*Rr2);
image vTempImg := RealImage("Temp",4,2*Rr2,2*Rr2);
vTempImg=tert((iradius<Rr2), 1,0);

//Compiled stitcher
cbed_stitcher(CBED_stack, DLACBEDimg, nG1, nG2, nTilts, pX, pY, tInc, g1X, g1Y, g2X, g2Y, Rr2, IsizX, IsizY);

/*//loop over DLACBED stack and build the patterns
gNo=0;
for (ind=-nG1; ind<nG1+1; ind++)
{ 
  for (jnd=-nG2; jnd<nG2+1; jnd++)
  { 
    prog=round(100*( (gNo+1)/((2*nG1+1)*(2*nG2+1)) ))
    OpenAndSetProgressWindow("DLACBED calculation","Image "+gNo+" of "+(2*nG1+1)*(2*nG2+1)," "+prog+" %");
    
    //loop over CBED stack
    for (pt=0; pt<nPts; pt++)
    {
      GetCoordsFromNTilts(nTilts,pt,_i,_j);
      //appropriate vector for disk
      X=round(pX+_i*tInc+(ind*g1X)+(jnd*g2X));
      Y=round(pY+_j*tInc+(ind*g1Y)+(jnd*g2Y));
      inside=!((X-Rr2<0)+(X+Rr2>IsizX)+(Y-Rr2<0)+(Y+Rr2>IsizY));
      if (inside)
      {
        TempImg=CBED_stack[X-Rr2,Y-Rr2,pt, X+Rr2,Y+Rr2,pt+1];//The disk of interest
        TempImg=tert( (iradius<Rr2), TempImg,0);//Cropped to be circular
        DLACBEDimg[X-Rr2,Y-Rr2,gNo, X+Rr2,Y+Rr2,gNo+1] += TempImg;//Add it to the LACBED pattern
        ScratImg[Y-Rr2,X-Rr2,Y+Rr2,X+Rr2] += vTempImg//Update mask which keeps count of the number of images in one pixel
        TempImg=tert( (vTempImg>TempImg), vTempImg, TempImg)
      }
    }
   ScratImg+=(ScratImg==0);//make pixels with zero values equal 1
   DLACBEDimg[0,0,gNo, IsizX,IsizY,gNo+1] /= ScratImg;//divide by mask
   ScratImg=0;
   gNo++;
  }
}*/

DLACBEDimg.SetLimits(DLACBEDimg.min(),DLACBEDimg.max())
//Tidy up
TempImg.DeleteImage();
vTempImg.DeleteImage();
CBED_stack.DeleteImage();

//Put tags into LACBED stack
{
DLACBEDimg.SetStringNote("Info:Date",datetime);
DLACBEDimg.SetNumberNote("Info:Camera Length",CamL);
DLACBEDimg.SetNumberNote("Info:Magnification",mag);
DLACBEDimg.SetNumberNote("Info:Alpha",Alpha);
DLACBEDimg.SetNumberNote("Info:Spot size",spot);
DLACBEDimg.SetNumberNote("Info:Disc Radius",Rr);
DLACBEDimg.SetStringNote("Info:Material",material);
DLACBEDimg.SetNumberNote("g-vectors:nG1",nG1);
DLACBEDimg.SetNumberNote("g-vectors:nG2",nG2);
DLACBEDimg.SetNumberNote("g-vectors:g1X",g1X);
DLACBEDimg.SetNumberNote("g-vectors:g1Y",g1Y);
DLACBEDimg.SetNumberNote("g-vectors:g2X",g2X);
DLACBEDimg.SetNumberNote("g-vectors:g2Y",g2Y);
DLACBEDimg.SetNumberNote("g-vectors:g1H",g1H);
DLACBEDimg.SetNumberNote("g-vectors:g1K",g1K);
DLACBEDimg.SetNumberNote("g-vectors:g1L",g1L);
DLACBEDimg.SetNumberNote("g-vectors:g2H",g2H);
DLACBEDimg.SetNumberNote("g-vectors:g2K",g2K);
DLACBEDimg.SetNumberNote("g-vectors:g2L",g2L);
DLACBEDimg.SetNumberNote("g-vectors:g1Ag2",g1Ag2);
DLACBEDimg.SetNumberNote("g-vectors:g1Mg2",g1Mg2);
DLACBEDimg.SetNumberNote("g-vectors:gC",gC);
}
result("  done\n");

///////////////////////////////////
//Montage of D-LACBED images
result("Creating Montage of D-LACBED images...");
//each D-LACBED image is (2*nTilts+3)*tInc*Rr wide
number wid=2*(nTilts+1)*tInc+2*Rr;//border of Rr
//F is the relative size of D-LACBED vs original disc size
//smallest g is sG (=1 or 2)
number sG = (2-(g2X>g1X))*(abs(g1X)>abs(g1Y))*(abs(g1X)>abs(g1Y));//both g's are closer to horizontal
sG=sG+(2-(g2Y>g1Y))*(abs(g1X)<abs(g1Y))*(abs(g1X)<abs(g1Y));//both g's are closer to vertical
sG=sG+(2-(g2mag>g1mag))*((abs(g1X)>abs(g1Y))*(abs(g1X)<abs(g1Y))+(abs(g1X)<abs(g1Y))*(abs(g1X)>abs(g1Y)))
number F=(wid-2*Rr)/max(abs(g1X),abs(g1Y))*(sG==1)+(wid-2*Rr)/max(abs(g2X),abs(g2Y))*(sG==2);//Scaling factors between CBED image and montage image
number Fsiz=((4*nG1+1.5)*(wid-2*Rr))*(sG==1)+((4*nG2+1.5)*(wid-2*Rr))*(sG==2);//Image size
//result("Scaling factors "+wid+":"+F+"\n")
//The 000 image will be in the centre
number Lx=round(Fsiz/2);
number Ly=round(Fsiz/2);
//result("F,Lx,Ly: "+F+", "+Lx+", "+Ly+"\n")
image Montage := RealImage("D-LACBED montage",4,Fsiz,Fsiz);
Montage.displayat(440,30);
Montage.SetWindowSize(0.75*IsizX,0.75*IsizY);
number a2X,a2Y,t2,l2,b2,r2,a1X,a1Y;
gNo=0;
for (ind=-nG1; ind<nG1+1; ind++)
{ 
  for (jnd=-nG2; jnd<nG2+1; jnd++)
  {
    //a2 is the centre of the rectangle where the D-LACBED image comes from in the stack
    a2X=round(pX+ind*g1X+jnd*g2X);
    a2Y=round(pY+ind*g1Y+jnd*g2Y);
    //result("centre: "+a2X+","+a2Y+"\n")
    //Bounding rectangle for each D-LACBED image
    t2=round( (a2Y-wid)*(1-((a2Y-wid)<0)) );//could also be *(!((a2Y-wid)<0))
    l2=round( (a2X-wid)*(1-((a2X-wid)<0)) );
    b2=round( (a2Y+wid)*(1-((a2Y+wid)>IsizY)) + ((a2Y+wid)>IsizY)*IsizY);
    r2=round( (a2X+wid)*(1-((a2X+wid)>IsizX)) + ((a2X+wid)>IsizX)*IsizX);
    //result("copy from: "+t2+","+l2+","+b2+","+r2+"\n")
    //a1 is the location of the rectangle where the D-LACBED image will go in the montage
    a1X=round(Lx + (ind*g1X*F+jnd*g2X*F));
    a1Y=round(Ly + (ind*g1Y*F+jnd*g2Y*F));
    //result("centre: "+a1X+","+a1Y+"\n")
    t1=round(a1Y-a2Y+t2);
    l1=round(a1X-a2X+l2);
    b1=round(a1Y+b2-a2Y);
    r1=round(a1X+r2-a2X);
    //result("paste to: "+t1+","+l1+","+b1+","+r1+"\n")
    inside=!((l1<0)+(r1>Fsiz)+(t1<0)+(b1>Fsiz));
    //inside=(t1>0)*(l1>0)*(b1<FsizY)*(r1<FsizX);
    if (inside)
    {
      //Montage[t1,l1,b1,r1] = DLACBEDimg[l2,t2,gNo ,r2,b2,gNo+1];
      Montage[t1,l1,b1,r1] = tert( (DLACBEDimg[l2,t2,gNo ,r2,b2,gNo+1]==0),Montage[t1,l1,b1,r1],DLACBEDimg[l2,t2,gNo ,r2,b2,gNo+1])
    }
    gNo++
  }
}
Montage.SetLimits(Montage.min(),Montage.max())
//Put tags into Montage
{
Montage.SetStringNote("Info:Date",datetime);
Montage.SetNumberNote("Info:Camera Length",CamL);
Montage.SetNumberNote("Info:Magnification",mag);
Montage.SetNumberNote("Info:Alpha",Alpha);
Montage.SetNumberNote("Info:Spot size",spot);
Montage.SetNumberNote("Info:Disc Radius",Rr);
Montage.SetStringNote("Info:Material",material);
Montage.SetNumberNote("g-vectors:nG1",nG1);
Montage.SetNumberNote("g-vectors:nG2",nG2);
Montage.SetNumberNote("g-vectors:g1X",g1X);
Montage.SetNumberNote("g-vectors:g1Y",g1Y);
Montage.SetNumberNote("g-vectors:g2X",g2X);
Montage.SetNumberNote("g-vectors:g2Y",g2Y);
Montage.SetNumberNote("g-vectors:g1H",g1H);
Montage.SetNumberNote("g-vectors:g1K",g1K);
Montage.SetNumberNote("g-vectors:g1L",g1L);
Montage.SetNumberNote("g-vectors:g2H",g2H);
Montage.SetNumberNote("g-vectors:g2K",g2K);
Montage.SetNumberNote("g-vectors:g2L",g2L);
Montage.SetNumberNote("g-vectors:g1Ag2",g1Ag2);
Montage.SetNumberNote("g-vectors:g1Mg2",g1Mg2);
Montage.SetNumberNote("g-vectors:gC",gC);
}
result("  done\n")

GetDate(f_,date_);
GetTime(f_,time_);
datetime=date_+"_"+time_;
result("Processing complete: "+datetime+" ding, dong\n\n")


