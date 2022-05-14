%//////////////////////////////////////////////////////////////////////////
% SVM-based automatic cell segmentation and counting for histology data
% version: stable release v1.0
% author: Quentin RV. Ferry
% license: MIT license
%—————————————————————————————————
% function: Use additional metrics to further refine the pool of detected
% cells (ouput of script 6).
%//////////////////////////////////////////////////////////////////////////

clear all; close all; clc; % clear session
script_path = pwd(); % grab path to working directory

%% USER DEFINED
param_pathToTrainingFolder = '../SET/trainingSet_20211118_4164/';  
param_nbInstancesToRun = 200; % number of ROIs to run the program on. Set to 1 for debugging.
param_displayPlot = false; % boolean decides whether or not to display ROIs with bounding boxes overlay.
% Given that cells appear brighter than the background on the image, we can
% further refine the bounding box selection by removing all bounding boxes
% (1) whose top intensities are below param_pixelMin:
param_pixelMin_cutoff = 10; %70
% (2) whose delta between top intensity (corresponds to the cell) and
% bottom intensity (corresponds to background) is bellow
% param_deltaMin_cutoff:
param_deltaMin_cutoff = 20; % default: 20.
% top and bottom intensities are determined using the following percentiles
% to avoid suprious outliers:
param_prctile_top = 95; % default: 95. 
param_prctile_bottom = 10; % default: 10.

%% MAIN
% load list of images to analyse.
load('FileList_test.mat');

% load the models, list radii and padding.
load(strcat(param_pathToTrainingFolder, 'SVM_consensus.mat'));
radii = [];
for i = 1:size(MODELS,2)
    radii = [radii, MODELS{1,i}];
end
radii = unique(radii);
padding = 2 * max(radii);

for file_index = 1:min(size(fileList,1), param_nbInstancesToRun) % loop over ROIs.

    fprintf('\n-------------------------------- index %i of %i\n\n', file_index, size(fileList,1));
    
    % find path to image.
    path_to_image = strcat(fileList(file_index).folder, '/', fileList(file_index).name);
    fprintf('image path: %s\n', path_to_image);
    name = strsplit(fileList(file_index).name, '.');
    name = name{1};
   
    % find path to bounding boxes data.
    path_to_bb = strcat(fileList(file_index).folder, '/', name, '_SVMStable_bb.mat');
    fprintf('bb path: %s\n', path_to_bb);
    
    % create path for new outputs.
    path_to_save_image = strcat(fileList(file_index).folder, '/', name, '_SVMStable_labelled_postprocessed.jpg');
    path_to_save_bb = strcat(fileList(file_index).folder, '/', name, '_SVMStable_bb_postprocessed.mat');

    %----------------------------------------------------------------------
    
    img = imread(path_to_image); % load the image 
    img = padarray(img, [padding, padding], 0); % pad image
    load(path_to_bb); % load bounding boxe's coordinates
    
    % post-processing:
    DELTA = zeros(size(boundingBoxes,1),1); % stores the delta max(pixel intensity)-min(pixel intensity) for each BB. max and min correspond to specific percentiles.
    TOP = zeros(size(boundingBoxes,1),1); % stores max(pixel intensity)-min(pixel intensity) for each BB. 

    for idx_bb = 1:size(boundingBoxes,1)

        current_bb = boundingBoxes(idx_bb,:);
        crop = double(imcrop(img,current_bb));
        crop = crop(:);

        DELTA(idx_bb) = prctile(crop,param_prctile_top) - prctile(crop,param_prctile_bottom);
        TOP(idx_bb) = prctile(crop,param_prctile_top);
    end

    boundingBoxes_kept = boundingBoxes(DELTA > param_deltaMin_cutoff & TOP > param_pixelMin_cutoff,:);
        
    img_show = insertShape(img,'rectangle',boundingBoxes,'LineWidth',1, 'Color', 'white', 'Opacity',0.1);
    img_show = insertShape(img_show,'rectangle',boundingBoxes_kept,'LineWidth',1, 'Color', 'green');
    
    imwrite(img_show, path_to_save_image);
    boundingBoxes = boundingBoxes_kept;
    save(path_to_save_bb, 'boundingBoxes');
end



