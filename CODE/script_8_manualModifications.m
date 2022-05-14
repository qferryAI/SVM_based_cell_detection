%//////////////////////////////////////////////////////////////////////////
% SVM-based automatic cell segmentation and counting for histology data
% version: stable release v1.0
% author: Quentin RV. Ferry
% license: MIT license
%—————————————————————————————————
% function: Manual correction of automatic annotations (img by img).
%//////////////////////////////////////////////////////////////////////////

clear all; close all; clc; % clear session
script_path = pwd(); % grab path to working directory

%% USER DEFINED PARAMETERS
param_pathToTrainingFolder = '../SET/trainingSet_20211118_4164/';
param_usePostProcessedAnnotations = true; % false: use SVM annotations, true: use post processed annotations
param_skipBBFiles = false; % ignore the automatic annotation all together

%% MAIN
% load model and find padding/radii
load(strcat(param_pathToTrainingFolder,'SVM_consensus.mat'));
radii = [];
for i = 1:size(MODELS,2)
    radii = [radii, MODELS{1,i}];
end
radii = unique(radii);
padding = 2 * max(radii);


% Choose and load image
[file,path] = uigetfile('../*.jpg');

img_path = strcat(path, file);
img = imread(img_path);
img = rescale(img, 0,1);
img = padarray(img, [padding, padding], 0);

% Load the corresponding cells detected

% check to see if the '_bb_manual.mat' file exist
bb_file = strsplit(file, '.jpg');
if param_usePostProcessedAnnotations
    bb_file_auto = strcat(path, bb_file{1},'_SVMStable_bb_postprocessed.mat');
    bb_file_manual = strcat(path, bb_file{1},'_SVMStable_bb_postprocessed_manual.mat');
else
    bb_file_auto = strcat(path, bb_file{1},'_SVMStable_bb.mat');
    bb_file_manual = strcat(path, bb_file{1},'_SVMStable_bb_manual.mat');
end

if param_skipBBFiles
    points = [];
else
    if isfile(bb_file_manual) % already maually modified
        load(bb_file_manual); % load points
    else % load automatic detection results
        load(bb_file_auto); % load boundingBoxes
        % convert bounding boxes into points
        points = boundingBoxes(:,1:2) + boundingBoxes(:,3:4)/2;
    end
end

%%
loop = 1;
pos = [500 500 1000 1000];

while(loop == 1)
    
    figure(1)
    imshow(img);
    set(gcf,'position',pos);
    
    if(size(points,1)>0)
        hold on
        scatter(points(:,1),points(:,2), 'MarkerEdgeColor', 'green')
        hold off
    end

    % display image and prompt annotations:
    % '1' key = 49 > add a point
    % '2' key = 50 > remove neirest point
    % any other key exit
    
    [x,y, key] = ginput(1);
   
    if(key == 49 || key == 1)
        
        points = [points; x ,y];
        
    elseif(key == 50)
        
        if(size(points,1)>0)
            
            idx = knnsearch(points,[x,y]);
            mask = 1:size(points, 1);
            points = points(mask ~= idx,:);
            
        end
        
    else
        loop = 0;
    end
        
    pos = get(gcf, 'Position'); %// gives x left, y bottom, width, height

    
end

save(bb_file_manual, 'points');
close all;