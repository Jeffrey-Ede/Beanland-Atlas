function [ success ] = share_matlab_engine( name )
%Name and share a MATLAB engine

if ~matlab.engine.isEngineShared
    matlab.engine.shareEngine(name);
end

success = 1;
end