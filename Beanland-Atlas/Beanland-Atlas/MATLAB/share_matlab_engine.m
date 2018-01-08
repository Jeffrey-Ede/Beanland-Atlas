function [ success ] = share_matlab_engine( name )
%Name and share a MATLAB engine
matlab.engine.shareEngine(name);
success = 1;
end