%//////////////////////////////////////////////////////////////////////////
% SVM-based automatic cell segmentation and counting for histology data
% version: stable release v1.0
% author: Quentin RV. Ferry
% license: MIT license
%—————————————————————————————————
% function: define ROIs on all sections of subdirectories of the
% path_to_root folder. ROIs are used to crop original sections and yield a
% new cropped image.
%//////////////////////////////////////////////////////////////////////////

clear all; close all; % clear session
script_path = pwd(); % grab path to working directory

%% USER DEFINED PARAMETERS
path_to_root = '../IMG/your_folder/'; % to parent directory containing all sections to crop.
param_nbInstancesRan = 100; % number of subdirectories visited. Change to 1 when piloting.
img_ext = '.jpg';
overwrite_INFO_file = 1; % 1: overwrite, 0: try loading existing INFO.mat file 

%% MAIN
files = dir(path_to_root); % Get a list of all files and directoris in the root directory.
dirFlags = [files.isdir]; % Get a logical vector that tells which is a directory.
subFolders = files(dirFlags); % Extract only those that are directories.

counter = 0; % counts number of subdirectories visited.

for k = 1 : length(subFolders) % loop over subdirectories
    
  fprintf('subdirectory #%d = %s\n', k, subFolders(k).name);
  
  if(~any(strcmp(subFolders(k).name, {'.', '..'})) && counter < param_nbInstancesRan)
  
      counter = counter + 1;
      current_path = strcat(path_to_root, subFolders(k).name);
      cd(current_path);
        
      % find image
      image_name = subFolders(k).name;
      image_file = strcat(subFolders(k).name, img_ext);
      
      % open image and find ROI
      img = imread(image_file);
      figure (1); imshow(img);
      
      if overwrite_INFO_file == 1 % define a new ROI
          poly = drawpolygon;
          poly = poly.Position;
      else
          load('INFO.mat');
          poly = INFO.poly;
      end

      % delineate bounding box
      bb.xmin = min(poly(:,1));
      bb.ymin = min(poly(:,2));
      bb.width = max(poly(:,1)) - bb.xmin;
      bb.height = max(poly(:,2)) - bb.ymin;

      % crop image
      % convert to B&W if RGB
      if size(img,3) == 3
        imgBW = rgb2gray(img);
      else
        imgBW = img;
      end
      imgBW_crop = imcrop(imgBW, [bb.xmin bb.ymin bb.width bb.height]);
      poly_offset = poly;
      poly_offset(:,1) = poly_offset(:,1) - bb.xmin;
      poly_offset(:,2) = poly_offset(:,2) - bb.ymin;
        
      % zero all pixel outside of the polygon          
      [w,h] = size(imgBW_crop);
      poly_left_mask = poly2mask(poly_offset(:,1),poly_offset(:,2),w,h);
      imgBW_crop(~poly_left_mask) = 0;
    
      path_imgBW_crop = strcat(strcat(image_name, '_cropped.jpg'));
      imwrite(imgBW_crop, path_imgBW_crop);

      % create a data file with all the crop info
      INFO.name = image_name;
      INFO.img_merge_name = image_file;
      INFO.poly = poly;
      INFO.timeStamp = date;
      
      save('INFO.mat', 'INFO');
      cd(script_path);
      
      
  end
  
end

close all;