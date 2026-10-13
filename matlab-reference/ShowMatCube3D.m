%% Clear workspace and load data
clear all              % Remove all variables from the workspace

% Load the .mat file containing 'imnData' (3D volumetric data)
load("PP.mat")

%% Normalize the volumetric data
% Normalize the data so that its values range between 0 and 1.
% 'imnData' contains the raw volumetric data, and 'max(imnData,[],'all')' 
% finds the maximum value across all elements in the array.
cube = imnData / max(imnData,[],'all');

%% Display the volume data
% 'volshow' is used to visualize 3D volumetric data interactively in MATLAB.
% It displays the normalized volume stored in 'cube'.
v = volshow(cube);
