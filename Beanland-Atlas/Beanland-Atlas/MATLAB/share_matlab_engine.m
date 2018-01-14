function [ success ] = share_matlab_engine( name )
%Name and share a MATLAB engine

%Check if the engine is already shared
shared = matlab.engine.isEngineShared;

if ~shared
    matlab.engine.shareEngine(name);
end
success = 1;
end