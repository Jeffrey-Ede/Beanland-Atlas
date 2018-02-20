// $BACKGROUND$
  //***            D-LACBED saver             ***\\
 //** Richard Beanland r.beanland@warwick.ac.uk **\\
//*         Coding started Jan 2018               *\\
 
// 1.0, 24 Jan 2018
// Saves all images in a D-LACBED stack as files with
// names suitable for input to felix bloch wave simulation/refinement
// see https://github.com/RudoRoemer/Felix
// The text in the Output window can be copied to use as felix.hkl
number pi=3.1415926535897932384626433832795;
number tiny=0.0001


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
image Lstack:=GetFrontImage();
number sizX,sizY,nPatt;
Lstack.Get3DSize(sizX,sizY,nPatt);
//put in a check for image type here
number data_type = Lstack.GetDataType();

//get image tags
number Rr,tInc,CamL,mag,Alpha,spot,nG1,nG2,g1X,g1Y,g2X,g2Y,g1H,g1K,g1L,g2H,g2K,g2L;
string material;
Lstack.GetStringNote("Info:Date",datetime);
Lstack.GetNumberNote("Info:Camera Length",CamL);
Lstack.GetNumberNote("Info:Magnification",mag);
Lstack.GetNumberNote("Info:Alpha",Alpha);
Lstack.GetNumberNote("Info:Spot size",spot);
Lstack.GetNumberNote("Info:Disc Radius",Rr);
Lstack.GetStringNote("Info:Material",material);
Lstack.GetNumberNote("g-vectors:nG1",nG1);
Lstack.GetNumberNote("g-vectors:nG2",nG2);
Lstack.GetNumberNote("g-vectors:g1X",g1X);
Lstack.GetNumberNote("g-vectors:g1Y",g1Y);
Lstack.GetNumberNote("g-vectors:g2X",g2X);
Lstack.GetNumberNote("g-vectors:g2Y",g2Y);
Lstack.GetNumberNote("g-vectors:g1H",g1H);
Lstack.GetNumberNote("g-vectors:g1K",g1K);
Lstack.GetNumberNote("g-vectors:g1L",g1L);
Lstack.GetNumberNote("g-vectors:g2H",g2H);
Lstack.GetNumberNote("g-vectors:g2K",g2K);
Lstack.GetNumberNote("g-vectors:g2L",g2L);
result("Material is "+material+"\n");
result("g1: ["+g1H+","+g1K+","+g1L+"], +/-"+nG1+"\n");
result("g2: ["+g2H+","+g2K+","+g2L+"], +/-"+nG2+"\n");

string path,dir,file_name,plusH,plusK,plusL;
if (!SaveAsDialog("","Choose directory",path)) exit(0)
dir=pathextractdirectory(path,2);
result("Saving to "+dir+"\n");

image lacbed=Lstack[0,0,0,sizX,sizY,1]
lacbed.SetStringNote("Info:Date",datetime);
lacbed.SetNumberNote("Info:Camera Length",CamL)
lacbed.SetNumberNote("Info:Magnification",mag)
lacbed.SetNumberNote("Info:Alpha",Alpha);
lacbed.SetNumberNote("Info:Spot size",spot);
lacbed.SetNumberNote("Info:Disc Radius",Rr);
lacbed.SetStringNote("Info:Material",material);
lacbed.SetNumberNote("g-vectors:nG1",nG1);
lacbed.SetNumberNote("g-vectors:nG2",nG2);
lacbed.SetNumberNote("g-vectors:g1X",g1X);
lacbed.SetNumberNote("g-vectors:g1Y",g1Y);
lacbed.SetNumberNote("g-vectors:g2X",g2X);
lacbed.SetNumberNote("g-vectors:g2Y",g2Y);
lacbed.SetNumberNote("g-vectors:g1H",g1H);
lacbed.SetNumberNote("g-vectors:g1K",g1K);
lacbed.SetNumberNote("g-vectors:g1L",g1L);
lacbed.SetNumberNote("g-vectors:g2H",g2H);
lacbed.SetNumberNote("g-vectors:g2K",g2K);
lacbed.SetNumberNote("g-vectors:g2L",g2L);

number prog,H,K,L,ind,jnd
number gNo=nPatt-1;
for (ind=-nG1; ind<nG1+1; ind++)
{ 
  for (jnd=-nG2; jnd<nG2+1; jnd++)
  {
    prog=round(100*( gNo/Npatt ))
    OpenAndSetProgressWindow("Saving","Image "+(gNo+1)+" of "+Npatt," "+prog+" %");
    H=round(ind*g1H+jnd*g2H+tiny);
    if (H>=0)
    {
      plusH="+";
    }
    else
    {
      plusH="";
    }
    K=round(ind*g1K+jnd*g2K+tiny);
    if (K>=0)
    {
      plusK="+";
    }
    else
    {
      plusK="";
    }
    L=round(ind*g1L+jnd*g2L+tiny);
    if (L>=0)
    {
      plusL="+";
    }
    else
    {
      plusL="";
    }
    lacbed=Lstack[0,0,gNo,sizX,sizY,gNo+1];//.rotate(pi/2);
    file_name=dir+material+"_"+plusH+H+plusK+K+plusL+L;
    //lacbed.SaveAsGatan(file_name);
    lacbed.SaveAsGatan3(file_name);
    result("["+H+","+K+","+L+"]\n")
    gNo--
  }
 }


GetDate(f_,date_);
GetTime(f_,time_);
datetime=date_+"_"+time_;
result("Saving complete: "+datetime+" dingdong\n\n")