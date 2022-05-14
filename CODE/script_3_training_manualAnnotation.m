%//////////////////////////////////////////////////////////////////////////
% SVM-based automatic cell segmentation and counting for histology data
% version: stable release v1.0
% author: Quentin RV. Ferry
% license: MIT license
%—————————————————————————————————
% function: let user manually annotate cells in the cropped ROIs define
% using script_1.
%//////////////////////////////////////////////////////////////////////////

clear all; close all; clc; % clear session
script_path = pwd(); % grab path to working directory

%% USER DEFINED PARAMETERS 
param_nbImagesToAnalyze = 50; % from the FileList_training.mat
param_trainingDate = '20211118'; % user provided. Should match existing date on training folder.
param_trainingKey = 4164; % user provided. Should match existing key. Set to 0 to creates a new key.
param_invertColors = 0; % 0,1 boolean.
param_markerColor = 'red'; % use 'cyan', 'red', 'blue', etc.

%% MAIN
% load list of training images
load('FileList_training.mat');
trainingSetFolder = strcat('../SET/trainingSet_',param_trainingDate,'_',int2str(param_trainingKey));

% choose param_nbImagesToAnalyze out of all file in fileList
if (param_trainingKey == 0)
    
    % create a new training set
    fprintf('...> creating new training set\n');
    
    % create a key
    trainingKey = datasample(1000:9999,1);
    trainingSetFolder = strcat('../SET/trainingSet_',param_trainingDate,'_',int2str(trainingKey));
    status = mkdir(trainingSetFolder);
    
    % choose param_nbImagesToAnalyze amongst FileList_training
    fileShortList = datasample(fileList,min(param_nbImagesToAnalyze, length(fileList)), 'Replace',false);
    
    % save list to the new training folder
    save(strcat(trainingSetFolder, '/', 'FileList_training_short.mat'), 'fileShortList');
    
elseif (exist(trainingSetFolder, 'dir')) 
    
    % user provided an exisiting key. Load FileList_training_short.mat
    fprintf('...> openning existing training set\n');
    load(strcat(trainingSetFolder, '/', 'FileList_training_short.mat'));

else 
    
    % the provided key does not match any existing training folders
    fprintf('...> ERROR the provided date/key combination does not match any existing training folders\n');
end

for index_file = 1:length(fileShortList) % loop over training images
    
    % Get path to the image.
    path = strcat(fileShortList(index_file).folder,'/');
    file = fileShortList(index_file).name;
    fprintf('...> working on %s\n',file);
    
    % Find path to annotation data.
    dataFileName = strcat(trainingSetFolder, '/data_', int2str(index_file), '.mat');
    
    img_path = strcat(path, file);
    fprintf('...> imgpath %s\n',img_path)
    img = imread(img_path);
    img = rescale(img, 0,1); % rescale pixel intensities to [0,1] range.
    if param_invertColors == 1
        img = imcomplement(img); % invert color scheme to see cells better.
    end
    
    
    points = [];
    if isfile(dataFileName) % grab existing annotations if exists.
        fprintf('......> loading existing annotations\n');
        load(dataFileName); 
        points = data.points;
    end
    
    % display image and prompt annotations:
    % '1' key = 49 > add a point
    % '2' key = 50 > remove neirest point
    % any other key exit

    loop = 1;
    pos = [500 500 1000 1000];

    while(loop == 1)

        figure(1)
        imshow(img);
        set(gcf,'position',pos);

        if(size(points,1)>0)
            hold on
            scatter(points(:,1),points(:,2), '+', 'MarkerEdgeColor', param_markerColor)
            hold off
        end

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
    
    % save annotation and path to the cropped image to the disk.
    data.imgPath = img_path;
    data.points = points;
    save(dataFileName, 'data');

end

close all; close all;