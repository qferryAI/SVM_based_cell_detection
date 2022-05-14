%//////////////////////////////////////////////////////////////////////////
% SVM-based automatic cell segmentation and counting for histology data
% version: stable release v1.0
% author: Quentin RV. Ferry
% license: MIT license
%—————————————————————————————————
% function: Simple example script showing how to get stats on the automatic annotations.
% This script computes the cell densities for each ROI and export its
% results in root/RESULTS/
%//////////////////////////////////////////////////////////////////////////

clear all; close all; clc; % clear session
script_path = pwd(); % grab path to working directory

%% USER DEFINED PARAMETERS
param_BBExtension = '_SVMStable_bb_postprocessed.mat'; % type of cell annotation to use
param_resultFile_path = '../RESULTS/results.csv'; % path to the result file

%% MAIN
% prompt user to get select a root folder
selected_path = uigetdir('../IMG/');

% list all subdirectories
files = dir(selected_path); % Get a list of all files and folders in this directory.
dirFlags = [files.isdir]; % Get a logical vector that tells which is a directory.
subFolders = files(dirFlags); % Extract only those that are directories.

% init result file
fileID = fopen(param_resultFile_path,'w');
fprintf(fileID,'section_name,nb_cells,area,density\n');

for k = 1 : length(subFolders) % loop over subdirectories
    
    fprintf('subdirectory #%d = %s\n', k, subFolders(k).name);
   
    if(~any(strcmp(subFolders(k).name, {'.', '..'})))

        cd(script_path);
        cd(strcat(selected_path,'/', subFolders(k).name)); % go to subdirectory
        fprintf('...> parsing folder...%s \n', pwd());
        
        % find all images with matching extensions and add to fileList
        image_nonFiltered = dir('*cropped.jpg');
        img_path= [];
        
        for i = 1:size(image_nonFiltered, 1)

            fprintf('......> found %s...', image_nonFiltered(i).name);
            tmp_name = strsplit(image_nonFiltered(i).name, '_');
            
            if strcmp(tmp_name{1},'.')
                fprintf('ignored.\n');
            else
                img_path = image_nonFiltered(i);
                fprintf('added.\n');
                
                
                % get nb of cells in the ROI.
                img_name = strsplit(img_path.name, '.');
                bb_filename = strcat(img_name{1}, param_BBExtension);
                load(bb_filename); % load boundingBoxes
                nb_cells = size(boundingBoxes,1);
                
                % get the ROI area in pixels.
                img = imread(img_path.name);
                area = sum(img(:) ~= 0);

                % write info to file.
                fprintf(fileID,'%s,%i,%i,%f\n', subFolders(k).name, nb_cells, area, nb_cells/area);
            end
        end         
    end
end

fclose(fileID);
cd(script_path)