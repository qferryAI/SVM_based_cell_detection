%//////////////////////////////////////////////////////////////////////////
% SVM-based automatic cell segmentation and counting for histology data
% version: stable release v1.0
% author: Quentin RV. Ferry
% license: MIT license
%—————————————————————————————————
% function: list all _cropped.jpg files in the subdirectories of the root
% folders and store the list as a FileList_training.mat file in the CODE
% folder.
%//////////////////////////////////////////////////////////////////////////

clear all; close all;
script_path = pwd();

%% USER DEFINED PARAMETERS
% list of ROOT folders
FOLDERS = {'../IMG/TRAINING/'};
listOfExtensions = {'*cropped.jpg'}; %type of images to list in each folder

%% MAIN
% create a fileList variable with the path to all images to analyze:

fileList = [];

for index_folder = 1:length(FOLDERS)
   
    files = dir(FOLDERS{index_folder}); % Get a list of all files and folders in this directory.
    dirFlags = [files.isdir]; % Get a logical vector that tells which is a directory.
    subFolders = files(dirFlags); % Extract only those that are directories.

    for k = 1 : length(subFolders) % loop over subdirectories
        
        fprintf('subdirectory #%d = %s\n', k, subFolders(k).name);
       
        if(~any(strcmp(subFolders(k).name, {'.', '..'})))

            cd(script_path);
            cd(strcat(FOLDERS{index_folder}, subFolders(k).name)); % go to subdirectory
            fprintf('...> parsing folder...%s \n', pwd());
            
            % find all images with matching extensions and add to fileList
            for index_extension = 1:length(listOfExtensions)
                
                image_nonFiltered = dir(listOfExtensions{index_extension});
                img_path= [];
                
                for i = 1:size(image_nonFiltered, 1)

                    fprintf('......> found %s...', image_nonFiltered(i).name);
                    tmp_name = strsplit(image_nonFiltered(i).name, '_');
                    
                    if strcmp(tmp_name{1},'.')
                        fprintf('ignored.\n');
                    else
                        img_path = image_nonFiltered(i);
                        fprintf('added.\n');
                    end

                end
                
                fileList = [fileList;img_path];
                
            end             
        end
    end
    
    cd(script_path);
end

% save to disk in the CODE folder
save('FileList_training.mat', 'fileList');

    