// $BACKGROUND$
  //***            D-LACBED aligner           ***\\
 //** Richard Beanland r.beanland@warwick.ac.uk **\\
//*         Coding started Jan 2018               *\\
 
// 1.0, 27 Jan 2018

//Global variables
number true = 1, false = 0;
number pi=3.1415926535897932384626433832795;
number tiny=0.000001//a small number
number window=100;// size of window around pattern centre for cross correlation


///////////////////////////
// Subroutines.
///////////////////////////
//Merge2
//Parabola
//PatternCentre
//SymmetryMatrix
//ApplySym
//SymmetryAdd
//Median


//Function Merge2
//Merges image 2 onto image 1 with sub-pixel accuracy, linear interpolation
image Merge2(image img1, image img2, number x, number y, number SumNo)
{
 number sizX,sizY;
 img1.GetSize(sizX,sizY);
 img1*=SumNo;//weight the destination image by the number averaged
 image countSum=SumNo*tert((img1==0),0,1);//mask for averaging
 number fx=floor(x);
 number fy=floor(y);
 //bounding rectangle for copy from img2
 number t1=abs(round( fy*(fy>0) ));//abs needed to avoid problems with -0(?!)
 number l1=abs(round( fx*(fx>0) ));
 number b1=round( sizY+fy*(fy<0) );
 number r1=round( sizX+fx*(fx<0) );
 //bounding rectangle for paste to img1
 number t2=abs(round( (sizY-b1)*(!(t1>0)) ));
 number l2=abs(round( (sizX-r1)*(!(l1>0)) ));
 number b2=round( sizY-t1);
 number r2=round( sizX-l1);
 //numbers for linear interpolation
 number xinterp = x - floor(x);
 number yinterp = y - floor(y);
 number tla = (1-xinterp) * (1-yinterp);
 number bra = xinterp * yinterp;
 number bla = (1-xinterp)*yinterp;
 number tra = xinterp * (1-yinterp);
 //Add 4x image2 into temp image using linear interpolation 
 image temp=img1[t2,l2,b2-1,r2-1]*0 + tla*img2[t1,l1,b1-1,r1-1]+tra*img2[t1,l1+1,b1-1,r1]+bla*img2[t1+1,l1,b1,r1-1]+bra*img2[t1+1,l1+1,b1,r1];
 image tempSum=tert((temp==0),0,1);//mask for subtracting border and intensity correction
 number h=b2-1-t2;
 number w=r2-1-l2;
 //border to remove aliased pixels
 number bor=2;
 //delete one-pixel border
 temp[0,0,h-bor,w-bor]*=tempsum[bor,bor,h,w];
 temp[bor,0,h,w-bor]*=tempsum[0,bor,h-bor,w];
 temp[0,bor,h-bor,w]*=tempsum[bor,0,h,w-bor];
 temp[bor,bor,h,w]*=tempsum[0,0,h-bor,w-bor];
 img1[t2,l2,b2-1,r2-1] += temp;
 //Calcluate mask
 tempSum=tert((temp==0),0,1);
 countSum[t2,l2,b2-1,r2-1]+=tempsum;
 img1=tert( (img1==0), 0, (img1/countSum));

 return img1
}//End of Merge2

//Function Parabola
//Gives sub-pixel peak position in 2D using Kramer's rule
void Parabola(image img, number &x, number &y)
{
  number maxval=img.max(x,y);
  number sizX,sizY;
  img.GetSize(sizX,sizY);
  number x1=-1,x2=0,x3=1;
  number y1,y3
  if (x==0)
  {
    y1=img.GetPixel(sizX-1,y);
  }
  else
  {
    y1=img.GetPixel(x-1,y);
  }
  number y2=maxval;
  if (x==sizX-1)
  {
    y3=img.GetPixel(0,y);
  }
  else
  {
    y3=img.GetPixel(x+1,y);
  }
  number denom=(x1-x2)*(x1-x3)*(x2-x3);
  number A=(x3*(y2-y1)+x2*(y1-y3)+x1*(y3-y2))/denom;
  number B=(x3*x3*(y1-y2)+x2*x2*(y3-y1)+x1*x1*(y2-y3))/denom;
  x-=B/(2*A);
  if (y==0)
  {
    y1=img.GetPixel(sizY-1,y);
  }
  else
  {
    y1=img.GetPixel(x,y-1);
  }
  if (y==sizY-1)
  {
    y3=img.GetPixel(0,y);
  }
  else
  {
    y3=img.GetPixel(x,y+1);
  }
  denom=(x1-x2)*(x1-x3)*(x2-x3);
  A=(x3*(y2-y1)+x2*(y1-y3)+x1*(y3-y2))/denom;
  B=(x3*x3*(y1-y2)+x2*x2*(y3-y1)+x1*x1*(y2-y3))/denom;
  y-=B/(2*A);
}//end of Parabola

//Function PatternCentre
//finds the displacement [x,y] of the pattern centre from the image centre
//by applying a 180 degree rotation and cross-correlating
void PatternCentre(image img, number &x, number &y)
{
 image img180=img.rotate(pi);
 image imgCC=img180.CrossCorrelate(img);
 //imgCC.DisplayAt(500,30)
 imgCC.Parabola(x,y)
 number sizX,sizY
 img.GetSize(sizX,sizY);
 x-=sizX/2;
 y-=sizY/2;
}//end of PatternCentre

//Function symmetry matrix
//returns the matrix relating equivalent patterns in the LACBED stack
void SymmetryMatrix(string SymType, number &m11, number &m12, number &m21, number &m22, string &SymName)
{
  m11=0;
  m12=0;
  m21=0;
  m22=0;
  if (SymType=="2")
  {
    m11=-1;
    m22=-1;
    SymName="2-fold";
  }
  if (SymType=="4+")
  {
    m12=-1;
    m21=1;
    SymName="4-fold(+)";
  }
  if (SymType=="4-")
  {
    m12=1;
    m21=-1;
    SymName="4-fold(-)";
  }
  if (SymType=="mx")
  {
    m11=-1;
    m22=1;
    SymName="x mirror";
  }
  if (SymType=="mx1")
  {
    m11=-1;
    m21=1;
    m22=1;
    SymName="x mirror";
  }
  if (SymType=="mx2")
  {
    m11=-1;
    m12=-1;
    m22=1;
    SymName="x mirror";
  }
  if (SymType=="my")
  {
    m11=1;
    m22=-1;
    SymName="y mirror";
  }
  if (SymType=="my1")
  {
    m11=1;
    m21=-1;
    m22=-1;
    SymName="y mirror";
  }
  if (SymType=="my2")
  {
    m11=1;
    m12=1;
    m22=-1;
    SymName="y mirror";
  }
  if (SymType=="mxy")
  {
    m12=-1;
    m21=-1;
    SymName="x-y mirror";
  }
  if (SymType=="myx")
  {
    m12=-1;
    m21=-1;
    SymName="y-x mirror";
  }

}//end of symmetry matrix

//Function ApplySym
//Applies the symmetry operation given by SymOp to a square image
void  ApplySym( image &img2, string SymOp)
{
 image img0=img2;
 //img2.DisplayAt(225,30);
 //img2.SetName("img2");
 //img2.SetWindowSize(200,200);
 if (SymOp=="4+")
 {
   img2=img0.rotate(pi/2);
 }
 if (SymOp=="4-")
 {
   img2=img0.rotate(-pi/2);
 }
 if (SymOp=="2")
 {
   img2=img0.rotate(pi);//no need if individual 2-folds have been applied
 }
 if (SymOp=="mx")
 {
   img2=img0[iwidth-icol,irow];
 }
 if (SymOp=="my")
 {
   img2=img0[icol,iheight-irow];
 }
 if (SymOp=="mxy")
 {
   img2=img0[iwidth-irow,iheight-icol];
 }
 if (SymOp=="myx")
 {
   img2=img0[irow,icol];
 }

}//End of ApplySym

//Function SymmetryAdd
//Adds symmetrically related patterns
//requires window defined as a global variable, assumes a SQUARE input image [siz x siz]
void SymmetryAdd(image &Lstack, string SymOp, number nG1, number nG2, number g1X, number g1Y, number g2X, number g2Y, number SumNo)
{
 image Lstack2=Lstack.ImageClone();
 number data_type = Lstack.GetDataType();
 number siz,nPatt;
 Lstack.Get3DSize(siz,siz,nPatt);
 number med=round((siz/2));
 image img1=NewImage("img1",data_type,2*siz,2*siz);
 image img2=img1;
 image imgCC=img1*0;
 //img1.DisplayAt(5,30);
 //img1.SetName("img1");
 //img1.SetWindowSize(200,200);
 //img2.DisplayAt(220,30);
 //img2.SetName("img2");
 //img2.SetWindowSize(200,200);
 //imgCC.DisplayAt(440,30);
 //imgCC.SetName("imgCC");
 //imgCC.SetWindowSize(200,200);
 //Get the matrix describing equivalent patterns
 string SymName;
 number m11,m12,m21,m22
 SymOp.SymmetryMatrix(m11,m12,m21,m22,SymName);
 if (SymOp=="my1" || SymOp=="my2") SymOp="my";
 if (SymOp=="mx1" || SymOp=="mx2") SymOp="mx";
 number ind,jnd,indS,jndS,gNoS,prog,t1,l1,b1,r1,t2,l2,b2,r2,x,y;
 number it=0.1;
 number gNo=0;
 for (ind=-nG1; ind<nG1+1; ind++)
 {
   for (jnd=-nG2; jnd<nG2+1; jnd++)
   {
     prog=round(100*( (gNo+1)/Npatt ))
     OpenAndSetProgressWindow(SymName+" symmetry averaging","Image "+(gNo+1)+" of "+Npatt," "+prog+" %");
     img1=0;
     img2=0;
     indS=m11*ind+m12*jnd;//get equivalent slice 
     jndS=m21*ind+m22*jnd;
     gNoS=(jndS+nG2)+((2*nG2)+1)*(indS+nG1);//the slice number of the equivalent pattern
     //result("i,j=["+ind+","+jnd+"] ("+gNo+"), iS,jS=["+indS+","+jndS+"] ("+gNoS+")\n");
     if (indS>(-nG1-it) && indS<(nG1+it) && jndS>(-nG2-it) && jndS<(nG2+it) )
     {//it's in the array, so combine them
       //make a mask around the pattern centre for the cross-correlation
       t1=round((ind*g1Y+jnd*g2Y+siz-window)/2);//in Lstack
       l1=round((ind*g1X+jnd*g2X+siz-window)/2);
       b1=t1+window;
       r1=l1+window;
       t2=t1+med;//in img1
       l2=l1+med;
       b2=b1+med;
       r2=r1+med;
       img1[t2,l2,b2,r2]=Lstack[l1,t1,gNo,r1,b1,gNo+1];
       // the same in the symmetrically related pattern
       t1=round((indS*g1Y+jndS*g2Y+siz-window)/2);
       l1=round((indS*g1X+jndS*g2X+siz-window)/2);
       b1=t1+window;
       r1=l1+window;
       t2=t1+med;//in img1
       l2=l1+med;
       b2=b1+med;
       r2=r1+med;
       img2[t2,l2,b2,r2]=Lstack2[l1,t1,gNoS,r1,b1,gNoS+1];
       //Apply symmetry operation to image
       img2.ApplySym(SymOp)
       imgCC=img2.CrossCorrelate(img1);
       imgCC.Parabola(x,y);
       x-=siz;
       y-=siz;
       //result("x,y:"+x+","+y+"\n");
       //use the full patterns to make the average
       img1[med,med,med+siz,med+siz]=Lstack[0,0,gNo,siz,siz,gNo+1];
       img2[med,med,med+siz,med+siz]=Lstack2[0,0,gNoS,siz,siz,gNoS+1];
       //Apply symmetry operation to image
       img2.ApplySym(SymOp)
       img1=Merge2(img1,img2,x,y,SumNo);
       Lstack[0,0,gNo,siz,siz,gNo+1]=img1[med,med,med+siz,med+siz];
     }
     gNo++
  }
}  

}//end of SymmetryAdd

//Function median
//gives the median value of an image
number median(image img)
{
 number sizX,sizY
 img.GetSize(sizX,sizY);
 number nPix=sizX*sizY;
 number odd=round(nPix % 2);
 number midpoint=round(1+(nPix-odd)/2);//e.g. gives 6 if nPix=10 or 11
 image sorted = NewImage("list",2,1,midpoint)
 //fill up the list to the midpoint
 number ind,x,y;
 for (ind=0; ind<midpoint; ind++)
 {
   sorted[ind,0,ind+1,1]=img.min(x,y);
   img.SetPixel(x,y,img.max());
 }
 number ibar
 if(odd)
 {
   ibar=sorted.getPixel(0,(midpoint-1))
 }
 else
 {
   ibar=(sorted.getPixel(0,(midpoint-1))+sorted.getPixel(0,(midpoint-2)))/2;
 }
 return ibar
}
//End of median


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
OpenAndSetProgressWindow("Starting processing","","");

///////////////
// Get 3D data stack
image L_Instack := GetFrontImage();
number sizX,sizY,nPatt,siz;
L_Instack.Get3DSize(sizX,sizY,nPatt);
//check for image type
if (nPatt==0) throw("Exiting: input should be a 3D LACBED stack");
number data_type = L_Instack.GetDataType();
//result("data type="+data_type+"\n");
siz=round(1.4*max(sizX,sizY));//1.4 is big enough to accomodate a 45 deg rotation
//Lstack will be square, [siz x siz], to contain the averaged patterns
image Lstack := NewImage("Averaged Stack",data_type,siz,siz,nPatt);
Lstack.DisplayAt(5,30);
Lstack.SetWindowSize(600,600);
number med=round((siz/2));//a useful number
//remove negative pixels
L_Instack=tert((L_Instack<0),0,L_Instack);
      
//get image tags
number Rr,tInc,CamL,mag,Alpha,spot,nG1,nG2,g1X,g1Y,g2X,g2Y,g1H,g1K,g1L,g2H,g2K,g2L,g1Ag2,g1Mg2,gC;
string material;
{
L_Instack.GetStringNote("Info:Date",datetime);
L_Instack.GetNumberNote("Info:Camera Length",CamL);
L_Instack.GetNumberNote("Info:Magnification",mag);
L_Instack.GetNumberNote("Info:Alpha",Alpha);
L_Instack.GetNumberNote("Info:Spot size",spot);
L_Instack.GetNumberNote("Info:Disc Radius",Rr);
L_Instack.GetStringNote("Info:Material",material);
L_Instack.GetNumberNote("g-vectors:nG1",nG1);
L_Instack.GetNumberNote("g-vectors:nG2",nG2);
L_Instack.GetNumberNote("g-vectors:g1X",g1X);
L_Instack.GetNumberNote("g-vectors:g1Y",g1Y);
L_Instack.GetNumberNote("g-vectors:g2X",g2X);
L_Instack.GetNumberNote("g-vectors:g2Y",g2Y);
L_Instack.GetNumberNote("g-vectors:g1H",g1H);
L_Instack.GetNumberNote("g-vectors:g1K",g1K);
L_Instack.GetNumberNote("g-vectors:g1L",g1L);
L_Instack.GetNumberNote("g-vectors:g2H",g2H);
L_Instack.GetNumberNote("g-vectors:g2K",g2K);
L_Instack.GetNumberNote("g-vectors:g2L",g2L);
L_Instack.GetNumberNote("g-vectors:g1Ag2",g1Ag2);
L_Instack.GetNumberNote("g-vectors:g1Mg2",g1Mg2);
L_Instack.GetNumberNote("g-vectors:gC",gC);
if (material=="")
{
  if (!GetString("Material?",material,material));
  exit(0);
  L_Instack.SetStringNote("Info:Material",material)
}
result("Material is "+material+"\n");
}
//g-vector outputs
number g1mag=sqrt(g1X*g1X+g1Y*g1Y);
number g2mag=sqrt(g2X*g2X+g2Y*g2Y);
number dot=(g1X*g2X+g1Y*g2Y)/(g1mag*g2mag);//gives cos(theta)
number cross=(g1X*g2Y-g1Y*g2X)/(g1mag*g2mag);//g1 x g2 gives sin(theta)
number theta=(acos(dot)*sgn(cross));
result("g1: "+g1H+","+g1K+","+g1L+", +/-"+nG1+": ["+g1X+","+g1Y+"], magnitude "+g1mag+"\n");
result("g2: "+g2H+","+g2K+","+g2L+", +/-"+nG2+": ["+g2X+","+g2Y+"], magnitude "+g2mag+"\n");
//result("Nominal ratio of magnitudes g1/g2 = "+g1Mg2+"\n");
//result("Actual ratio of magnitudes g1/g2 = "+(g1mag/g2mag)+"\n");
//result("Nominal angle between g1 and g2 ="+g1Ag2+" degrees\n");
//result("Actual angle between g1 and g2 ="+(180*theta/pi)+" degrees\n");

///////////////
//montage
number montage=0;
if (TwoButtonDialog("Make montage?","Yes","No") )
{
  result("Will make montage\n"); 
  montage=1;
}
else
{
  result("Will not make montage\n"); 
  montage=0;
}

///////////////
//Get symmetry
string symmetry;
if (!GetString("Rotational symmetry?","1",symmetry)) exit(0);
result("Applying rotational symmetry "+symmetry+"\n");

///////////////
//use 000 image to find centre and set up averaged stack
number N000=round((nPatt-1)/2);
number x,y,x0,y0;//cross correlation peak for 000
image avL000=L_Instack[0,0,N000,sizX,sizY,N000+1];
avL000.PatternCentre(x,y);
avL000.merge2(avL000.rotate(pi),x,y,1);
x0=x;//we can only expect a rough estimate without using a window
y0=y;
//put central part of 000 pattern into Lstack
number t1=round((siz-window)/2);
number l1=round((siz-window)/2);
number b1=t1+window;
number r1=l1+window;
number t2=round((sizY-y-window)/2);
number l2=round((sizX-x-window)/2);
number b2=t2+window;
number r2=l2+window;
Lstack[l1,t1,N000,r1,b1,N000+1]=avL000[t2,l2,b2,r2];
avL000.DeleteImage();
//fine correction using windowed pattern
Lstack[0,0,N000,siz,siz,N000+1].PatternCentre(x,y);
x0+=x;//correct [x0,y0]
y0+=y;
result("Whole pattern centre displaced by ["+(x0/2)+","+(y0/2)+"]\n");
//roi in instack
number height=round(sizY-abs(y0));
number width=round(sizX-abs(x0));
t1=0*(y0>0)+(sizY-height)*(y0<=0);
l1=0*(x0>0)+(sizX-width)*(x0<=0);
b1=height*(y0>0)+sizY*(y0<=0);
r1=width*(x0>0)+sizX*(x0<=0);
//roi in Lstack
t2=round(med-height/2);
l2=round(med-width/2);
b2=t2+height;
r2=l2+width;
//Copy instack across to Lstack
Lstack[l2,t2,0,r2,b2,Npatt]=L_Instack[l1,t1,0,r1,b1,Npatt];
//from now on everything is done with Lstack, which is large, square and has 000 centred
image Limg=NewImage("Current pattern",data_type,siz,siz);
//Limg.DisplayAt(0,30);
//Limg.SetName("Current pattern");
//Limg.SetWindowSize(200,200);

///////////////
//Measure all the pattern centres in Lstack
result("Measuring pattern centres...");
number n1=2*nG1+1;
number n2=2*nG2+1;
image PattCen=NewImage("Centres",data_type,n1,n2,2);
//PattCen.DisplayAt(630,30);
//PattCen.SetName("Centres");
//PattCen.SetWindowSize(200,200);
number ind,jnd,prog,gNo;
gNo=0;
for (ind=-nG1; ind<nG1+1; ind++)
{ 
  for (jnd=-nG2; jnd<nG2+1; jnd++)
  {
    prog=round(100*( gNo/Npatt ))
    OpenAndSetProgressWindow("Measuring pattern centres","Image "+(gNo+1)+" of "+Npatt," "+prog+" %");
    t1=round((ind*g1Y+jnd*g2Y+siz-window)/2);//window around pattern centre
    l1=round((ind*g1X+jnd*g2X+siz-window)/2);
    b1=t1+window;
    r1=l1+window;
    Limg=0;
    Limg[t1,l1,b1,r1]=Lstack[l1,t1,gNo,r1,b1,gNo+1];
    Limg.PatternCentre(x,y);
    PattCen[(ind+nG1),(jnd+nG2),0,(ind+nG1+1),(jnd+nG2+1),1]=x;
    PattCen[(ind+nG1),(jnd+nG2),1,(ind+nG1+1),(jnd+nG2+1),2]=y;
    gNo++
  }
}
result("done\n");

//update g-vectors
{//median is probably better (less sensitive to bad measurements)
g1X=median(PattCen[0,0,0,(n1-1),n2,1]-PattCen[1,0,0,n1,n2,1]);
g2X=median(PattCen[0,0,0,n1,(n2-1),1]-PattCen[0,1,0,n1,n2,1]);
g1Y=median(PattCen[0,0,1,(n1-1),n2,2]-PattCen[1,0,1,n1,n2,2]);
g2Y=median(PattCen[0,0,1,n1,(n2-1),2]-PattCen[0,1,1,n1,n2,2]);
//alternative using mean
//g1X=sum(PattCen[0,0,0,(n1-1),n2,1]-PattCen[1,0,0,n1,n2,1])/((n1-1)*n2);
//g2X=sum(PattCen[0,0,0,n1,(n2-1),1]-PattCen[0,1,0,n1,n2,1])/((n2-1)*n1);
//g1Y=sum(PattCen[0,0,1,(n1-1),n2,2]-PattCen[1,0,1,n1,n2,2])/((n1-1)*n2);
//g2Y=sum(PattCen[0,0,1,n1,(n2-1),2]-PattCen[0,1,1,n1,n2,2])/((n2-1)*n1);
g1mag=sqrt(g1X*g1X+g1Y*g1Y);
g2mag=sqrt(g2X*g2X+g2Y*g2Y);
dot=(g1X*g2X+g1Y*g2Y)/(g1mag*g2mag);//gives cos(theta)
cross=(g1X*g2Y-g1Y*g2X)/(g1mag*g2mag);//g1 x g2 gives sin(theta)
theta=(acos(dot)*sgn(cross));
result("Accurate g1: ["+g1X+","+g1Y+"]\n");
result("Accurate g2: ["+g2X+","+g2Y+"]\n");
result("Nominal ratio of magnitudes g1/g2 = "+g1Mg2+"\n");
result("Actual ratio of magnitudes g1/g2 = "+(g1mag/g2mag)+"\n");
result("Nominal angle between g1 and g2 ="+g1Ag2+" degrees\n");
result("Actual angle between g1 and g2 ="+(180*theta/pi)+" degrees\n");
}

///////////////
//centred lattice: default is that vectors h1 and h2 are the same as g1 and g2
number h1X=g1X;
number h1Y=g1Y;
number h1H=g1H;
number h1K=g1K;
number h1L=g1L;
number h1mag=g1mag;
number h2X=g2X;
number h2Y=g2Y;
number h2H=g2H;
number h2K=g2K;
number h2L=g2L;
number h2mag=g2mag;
number h1Ah2=g1Ag2;
number h1Mh2=g1Mg2;
if (gC==0) result("The pattern is not face-centred\n");//and so h=g
if (gC==1)
{//set up new vectors h1 and h2 to describe the centred pattern
  result("g1 is a face-centring vector: centred lattice is\n");
  h1X=2*g1X-g2X;
  h1Y=2*g1Y-g2Y;
  h1mag=sqrt(h1X*h1X+h1Y*h1Y);
  h1H=round(2*g1H-g2H);//relying on the user here to have put in indices that work
  h1K=round(2*g1K-g2K);
  h1L=round(2*g1L-g2L);
  h1Ah2=90;//centred patterns are always rectangular
  h1Mh2=g1Mg2*2*sin(g1Ag2*pi/180);
}
if (gC==2)
{
  result("g2 is a face-centring vector: centred lattice is\n");
  h2X=2*g2X-g1X;
  h2Y=2*g2Y-g1Y;
  h2mag=sqrt(h2X*h2X+h2Y*h2Y);
  h2H=round(2*g2H-g1H);
  h2K=round(2*g2K-g1K);
  h2L=round(2*g2L-g1L);
  h1Ah2=90;//centred patterns are always rectangular
  h1Mh2=g1Mg2/(2*sin(g1Ag2*pi/180));
}
if (abs(h1X)<tiny) h1X=0;
if (abs(h1Y)<tiny) h1Y=0;
if (abs(h2X)<tiny) h2X=0;
if (abs(h2Y)<tiny) h2Y=0;
result("h1: "+h1H+","+h1K+","+h1L+" : ["+h1X+","+h1Y+"], magnitude "+h1mag+"\n");
result("h2: "+h2H+","+h2K+","+h2L+" : ["+h2X+","+h2Y+"], magnitude "+h2mag+"\n");

///////////////
//correct the stretch and skew
number sizXr,sizYr;
if (!(symmetry=="0"))
{
  //Calculate angles and new g-vectors
  //Angle phi between h1 and the x-axis
  number dotX=h1X/h1mag;//gives cos(phi)
  number crossX=-h1Y/h1mag;//g1 x [100] gives sin(phi)
  number phi=-(acos(dotX)*sgn(crossX))
  //Rotated stack with h1 horizontal, 
  image Lcorrected=Lstack.rotate(phi);
  Lcorrected.Get3DSize(sizXr,sizYr,nPatt);
  //Lcorrected.DisplayAt(225,30);
  //Lcorrected.SetName("Corrected pattern");
  //Lcorrected.SetWindowSize(200,200);
  result("Correcting distortions...\n");
  //rotated vectors
  number g1Xr=g1X*cos(phi)+g1Y*sin(phi);
  number g1Yr=g1Y*cos(phi)-g1X*sin(phi);
  number g2Xr=g2X*cos(phi)+g2Y*sin(phi);
  number g2Yr=g2Y*cos(phi)-g2X*sin(phi);
  number h1Xr=h1X*cos(phi)+h1Y*sin(phi);
  number h1Yr=0;//by definition
  number h2Xr=h2X*cos(phi)+h2Y*sin(phi);
  number h2Yr=h2Y*cos(phi)-h2X*sin(phi);
  g1X=g1Xr;
  g1Y=g1Yr;
  h1X=h1Xr;
  h1Y=h1Yr;
  g2X=g2Xr;
  g2Y=g2Yr;
  h2Y=h2Yr;
  h2X=h2Xr;
  //stretch of y-component
  number stretch=sin(h1Ah2*pi/180)*h1mag/(h2Y*h1Mh2);
  //skew to compensate y distortion
  number skew=(cos(h1Ah2*pi/180)*h1mag - h2X*h1Mh2)/(h2Y*h1Mh2);
  result("Rotate h1 to be horizontal, "+(phi*180/pi)+" degrees\n");
  //result("New g1: ["+g1X+","+g1Y+"]\n");
  //result("New g2: ["+g2X+","+g2Y+"]\n");
  result("New h1: ["+h1X+","+h1Y+"]\n");
  result("New h2: ["+h2X+","+h2Y+"]\n");
  number stretchshow=(round((stretch-1)*10000))/100;
  number skewshow=(round(skew*10000))/100;
  result("stretch & skew = "+stretchshow+"% & "+skewshow+"%\n");
  //some dummy images in which the distortions will be applied
  image distort=Lcorrected[0,0,0,sizXr,sizYr,1]*0;
  image distort1=distort;
  //distort.DisplayAt(225,230);
  //distort.SetName("corrected pattern");
  //distort.SetWindowSize(200,200);
  //skew=0.1//to test
  number dx=round(sizYr*skew/2);//displacement of pattern centre
  //stretch=1.1//to test
  number dy=round(sizYr*(stretch-1)/2);
  //result("[dx,dy] = ["+dx+","+dy+"]\n");
  //offset selection by displacement
  t1=round(dy+(sizYr-siz)/2);
  l1=round(dx+(sizXr-siz)/2);
  b1=t1+siz;
  r1=l1+siz;
  //border to remove aliased pixels
  number bor=2;
  //work through stack correcting distortions
  gNo=0;
  for (ind=-nG1; ind<nG1+1; ind++)
  { 
    for (jnd=-nG2; jnd<nG2+1; jnd++)
    {
      prog=round(100*( gNo/Npatt ))
      OpenAndSetProgressWindow("Correcting distortions","Image "+(gNo+1)+" of "+Npatt," "+prog+" %");
      distort=Lcorrected[0,0,gNo,sizXr,sizYr,gNo+1];
      distort1=distort[icol,irow/stretch];//need 2 dummy images since can't do it all in one go
      distort=distort1[icol-irow*skew,irow];
      //pixels at the edge of the patterns have been aliased, delete them
      distort1=tert((distort==0),0,1);//now using this image as a mask
      //delete border pixels all round
      distort[0,0,sizYr-bor,sizXr-bor]*=distort1[bor,bor,sizYr,sizXr];
      distort[bor,0,sizYr,sizXr-bor]*=distort1[0,bor,sizYr-bor,sizXr];
      distort[0,bor,sizYr-bor,sizXr]*=distort1[bor,0,sizYr,sizXr-bor];
      distort[bor,bor,sizYr,sizXr]*=distort1[0,0,sizYr-bor,sizXr-bor];
      Lstack[0,0,gNo,siz,siz,gNo+1]=distort[t1,l1,b1,r1];
      gNo++
    }
  }
  //tidy up
  Lcorrected.DeleteImage();
  distort.DeleteImage();
  distort1.DeleteImage();
}

///////////////****
//Averaging using user-input symmetry
//to keep track of the number of images that have been added
number SumNo=1; 

//Apply individual 2-fold pattern symmetries and put into Lstack
if (!(symmetry=="0"))
{
  result("2-fold pattern averaging...");
  gNo=0;
  for (ind=-nG1; ind<nG1+1; ind++)
  { 
    for (jnd=-nG2; jnd<nG2+1; jnd++)
    {
      prog=round(100*( (gNo+1)/Npatt ))
      OpenAndSetProgressWindow("2-fold pattern averaging","Image "+(gNo+1)+" of "+Npatt," "+prog+" %");
      t1=round((ind*g1Y+jnd*g2Y+siz-window)/2);//window around pattern centre
      l1=round((ind*g1X+jnd*g2X+siz-window)/2);
      b1=t1+window;
      r1=l1+window;
      Limg=0;
      Limg[t1,l1,b1,r1]=Lstack[l1,t1,gNo,r1,b1,gNo+1];
      Limg.PatternCentre(x,y);
      //result("x,y:"+x+","+y+"\n");
      //use the full patterns to make the average
      Limg=Lstack[0,0,gNo,siz,siz,gNo+1];
      Limg=Merge2(Limg,(Limg.rotate(pi)),x,y,SumNo);
      Lstack[0,0,gNo,siz,siz,gNo+1]=Limg;
      gNo++
    }
  }
  SumNo++
  result("done\n");
}

//Averaging using rotational symmetry
//number m11,m12,m21,m22;
string SymOp;
//2-fold symmetry
if (symmetry=="2" || symmetry=="4" || symmetry=="6")
{
  result("2-fold symmetry averaging...");
  SymmetryAdd(Lstack,"2",nG1,nG2,g1X,g1Y,g2X,g2Y,SumNo);
  SumNo++
  result("done\n");
}
//4+ symmetry
if (symmetry=="4")
{//NB don't need 4(-) averaging if we have already done a 2-fold
  result("4-fold symmetry averaging...");
  SymmetryAdd(Lstack,"4+",nG1,nG2,g1X,g1Y,g2X,g2Y,SumNo);
  SumNo++
  result("done\n");
}
//3 and 6 fold needed here!


///////////////
//Apply mirrors
if (!(symmetry=="0"))
{
  if (!GetString("mirror symmetry?","mx",symmetry)) exit(0);
  result("Applying mirror symmetry "+symmetry+"\n");
  if (symmetry=="mx" || symmetry=="mxy")
  {
    result("x-mirror symmetry averaging...");
    if (gC==0) SymmetryAdd(Lstack,"mx",nG1,nG2,g1X,g1Y,g2X,g2Y,SumNo);
    if (gC==1) SymmetryAdd(Lstack,"mx1",nG1,nG2,g1X,g1Y,g2X,g2Y,SumNo);
    if (gC==2) SymmetryAdd(Lstack,"mx2",nG1,nG2,g1X,g1Y,g2X,g2Y,SumNo);
    SumNo++
    result("done\n");
  }
  if (symmetry=="my")
  {
    result("y-mirror symmetry averaging...");
    if (gC==0) SymmetryAdd(Lstack,"my",nG1,nG2,g1X,g1Y,g2X,g2Y,SumNo);
    if (gC==1) SymmetryAdd(Lstack,"my1",nG1,nG2,g1X,g1Y,g2X,g2Y,SumNo);
    if (gC==2) SymmetryAdd(Lstack,"my2",nG1,nG2,g1X,g1Y,g2X,g2Y,SumNo);
    SumNo++
    result("done\n");
  }
  if (symmetry=="mxy")
  {
    result("xy-mirror symmetry averaging...");//this can only be present in 4mm symmetry
    if (gC==0) SymmetryAdd(Lstack,"mxy",nG1,nG2,g1X,g1Y,g2X,g2Y,SumNo);
    if (gC==1) result("xy-mirror symmetry incompatible with centred lattice, redefine basis vectors\n"); 
    SumNo++
    result("done\n");
  }
  //other mirrors for 3m and 6mm needed here
}

//additional rotation
{
//image Lrotated2=Lstack.rotate(-45*pi/180);
//Lrotated2.Get3DSize(sizXr,sizYr,nPatt);
//t1=round((sizYr-siz)/2);
//l1=round((sizXr-siz)/2);
//b1=t1+siz;
//r1=l1+siz;
//Lstack[0,0,0,siz,siz,nPatt]=Lrotated2[l1,t1,0,r1,b1,nPatt];
//Lrotated2.DeleteImage();
}

///////////////
//cropped stack
number w=232;
if (!GetNumber("Cropped pattern radius?",w,w)) exit(0);
result("Cropping to ["+(2*w)+"x"+(2*w)+"]\n");
t1=med-w;
l1=med-w;
b1=med+w;
r1=med+w;
image LACBED_reduced_stack=Lstack[l1,t1,0,r1,b1,nPatt];
LACBED_reduced_stack.SetName("Cropped Stack");
LACBED_reduced_stack.DisplayAt(5,30);
LACBED_reduced_stack.SetWindowSize(200,200);
//LACBED_reduced_stack.SetLimits(LACBED_reduced_stack.min(),LACBED_reduced_stack.max())

///////////////
//montage
if (montage==1)
{
  image Dmontage=NewImage("Montage",data_type,(n1*2*w),(n2*2*w));
  image temp=LACBED_reduced_stack[0,0,0,2*w,2*w,1];//temp image for normalising
  number back=5;//a background to subtract, if desired
  Dmontage.DisplayAt(225,30);
  Dmontage.SetWindowSize(500,500);
  gNo=nPatt-1;
  for (ind=-nG1; ind<nG1+1; ind++)
  { 
    for (jnd=-nG2; jnd<nG2+1; jnd++)
    {
      prog=round(100*( gNo/Npatt ))
      OpenAndSetProgressWindow("Making montage","Image "+(gNo+1)+" of "+Npatt," "+prog+" %");
      t1=round((jnd+nG2)*2*w);
      l1=round((ind+nG1)*2*w);
      b1=t1+2*w;
      r1=l1+2*w;
      //result("tlbr:"+t1+","+l1+","+b1+","+r1+"\n");
      temp=LACBED_reduced_stack[0,0,gNo,2*w,2*w,gNo+1]-back;
      temp=tert((temp<0),0,temp);
      temp-=temp.min();
      temp/=temp.max();
      Dmontage[t1,l1,b1,r1]=temp;
      gNo--
    }
  }
  Dmontage.SetLimits(Dmontage.min(),Dmontage.max())
  result("done\n");
}


//set image tags
{
LACBED_reduced_stack.SetStringNote("Info:Date",datetime);
LACBED_reduced_stack.SetNumberNote("Info:Camera Length",CamL)
LACBED_reduced_stack.SetNumberNote("Info:Magnification",mag)
LACBED_reduced_stack.SetNumberNote("Info:Alpha",Alpha);
LACBED_reduced_stack.SetNumberNote("Info:Spot size",spot);
LACBED_reduced_stack.SetNumberNote("Info:Disc Radius",Rr);
LACBED_reduced_stack.SetStringNote("Info:Material",material);
LACBED_reduced_stack.SetNumberNote("g-vectors:nG1",nG1);
LACBED_reduced_stack.SetNumberNote("g-vectors:nG2",nG2);
LACBED_reduced_stack.SetNumberNote("g-vectors:g1X",g1X);
LACBED_reduced_stack.SetNumberNote("g-vectors:g1Y",g1Y);
LACBED_reduced_stack.SetNumberNote("g-vectors:g2X",g2X);
LACBED_reduced_stack.SetNumberNote("g-vectors:g2Y",g2Y);
LACBED_reduced_stack.SetNumberNote("g-vectors:g1H",g1H);
LACBED_reduced_stack.SetNumberNote("g-vectors:g1K",g1K);
LACBED_reduced_stack.SetNumberNote("g-vectors:g1L",g1L);
LACBED_reduced_stack.SetNumberNote("g-vectors:g2H",g2H);
LACBED_reduced_stack.SetNumberNote("g-vectors:g2K",g2K);
LACBED_reduced_stack.SetNumberNote("g-vectors:g2L",g2L);
LACBED_reduced_stack.SetNumberNote("g-vectors:g1Ag2",g1Ag2);
LACBED_reduced_stack.SetNumberNote("g-vectors:g1Mg2",g1Mg2);
}
GetDate(f_,date_);
GetTime(f_,time_);
datetime=date_+"_"+time_;
result("Processing complete: "+datetime+" dingdong\n\n")