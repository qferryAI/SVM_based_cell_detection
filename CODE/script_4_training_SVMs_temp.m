%//////////////////////////////////////////////////////////////////////////
% SVM-based automatic cell segmentation and counting for histology data
% version: stable release v1.0
% author: Quentin RV. Ferry
% license: MIT license
%—————————————————————————————————
% function: Uses manual cell annotations from script 3 to learn a series of SVM linear
% classifiers that tell apart cell versus background in the images.
%//////////////////////////////////////////////////////////////////////////

clear all; close all; clc; % clear session
script_path = pwd(); % grab path to working directory

%% USER DEFINED PARAMETERS
param_pathToTrainingFolder = '../SET/trainingSet_20211118_4164/'; % path to the training folder created by script 3.
param_bb_radii = [8, 10]; % vector of radii corresponding to the radius of the bounding box surrounding each annotation.
param_jitter = [-2:2]; % amount of jitter in pixels allowed around ground truth annotations.
param_nbClassifiers = 2; % number of classifiers to train for each radius size.
param_PosNegOverlap_max = 40; % maximum % overlap between bounding boxes of positive (cell) and negative (background) instances.
param_PosNegOverlap_min = 5; % minimum % overlap between bounding boxes of positive (cell) and negative (background) instances.
param_pca_precision = 75;

%% MAIN
trainingSet_files = dir(strcat(param_pathToTrainingFolder,'data_*')); % get list of annotations files
trainingSet_DATA = cell(1,length(trainingSet_files));

for i = 1:length(trainingSet_files)
    
    %load data and append to trainingSet_DATA
    trainingSet_file = trainingSet_files(i);
    load(strcat(trainingSet_file.folder,'/', trainingSet_file.name));
    trainingSet_DATA{i} = data;
    
end

MODELS = cell(5, length(param_bb_radii) * param_nbClassifiers); % varaible that will contain classifiers' parameters
model_counter = 0;

for r = 1:length(param_bb_radii) % loop over radii
   
    fprintf('\nWorking on radius %i\n', r)
    bb_radius = param_bb_radii(r); % set current bounding box radius
        
    for k = 1:param_nbClassifiers % loop over number of classifier/radius
        fprintf('... classifier nb %i\n', k)
        % gather training data
        FEATURES = [];
        LABELS = {};
        
        fprintf('...... extracting feature vectors\n')
        for i = 1:length(trainingSet_DATA)
            
            % get cells' center (manual annotations)
            C = trainingSet_DATA{i}.points;
            % add jitter
            C(:,1) = C(:,1) + randsample(param_jitter, size(C,1), true)';
            C(:,2) = C(:,2) + randsample(param_jitter, size(C,1), true)';

            % extract corresponding feature vectors for positive and
            % negative instances
            [features, labels] = fn_extractDataset(trainingSet_DATA{i}.imgPath, C, bb_radius, param_PosNegOverlap_max, param_PosNegOverlap_min, 1, i);
            FEATURES = [FEATURES; features];
            LABELS = [LABELS; labels];

        end

        % perform PCA to reduce dimensionality
        fprintf('...... reducing dimensionality with PCA\n')
        [PCA_coeff,FEATURES_pca, latent, ~, explained] = pca(FEATURES);
        PCA_nb = sum(cumsum(explained) < param_pca_precision);

        % train linear SVM classifier
        fprintf('...... computing SVM classifier\n')
        SVMModel = fitcsvm(FEATURES_pca(:,1:PCA_nb),LABELS, 'ClassNames',{'negative','positive'});
        CompactSVMModel = compact(SVMModel);
        CompactSVMModel = fitPosterior(CompactSVMModel, FEATURES_pca(:,1:PCA_nb), LABELS);

        % add current classifier to MODELS
        model_counter = model_counter + 1;
        MODELS{1,model_counter} = bb_radius;
        MODELS{2,model_counter} = PCA_nb;
        MODELS{3,model_counter} = PCA_coeff;
        MODELS{4,model_counter} = SVMModel;
        MODELS{5,model_counter} = CompactSVMModel; % NEW
        
    end
        
end

% save all models to disk as a SVM_consensus.mat file
svmModelSave = strcat(param_pathToTrainingFolder, 'SVM_consensus.mat');
save(svmModelSave,'MODELS');
close all;

%//////////////////////////////////////////////////////////////////////////
%% functions

function [FEATURES, LABELS] = fn_extractDataset(path_to_image, centers, bb_radius, param_PosNegOverlap_max, param_PosNegOverlap_min, bool_display, imageIndex)
    
    % extracts feature vectors for positive (cell) and negative
    % (background) instances by cropping the image.

    bb_corner = floor(centers - bb_radius); % find top left corner of all bounding boxes
    bb = [bb_corner, 2*bb_radius*ones(size(bb_corner,1),2)]; % get coordinates of all bounding boxes
    FEATURES = [];
    LABELS = {};

    % read image and add padding to allow cropping at the edges
    img = imread(path_to_image);
    img = padarray(img, [2* bb_radius, 2*bb_radius], 0);
    img = imgaussfilt(img,1); % optional smoothing of the image
    
    % correct bb's coordinates for image padding
    bb(:,1) = bb(:,1) + 2* bb_radius;
    bb(:,2) = bb(:,2) + 2* bb_radius;
    
    if(bool_display) % display padded image and positive bounding boxes
        figure(1)
        img_disp = double(img);
        img_disp = rescale(img_disp, 0,1);
        img_show = insertObjectAnnotation(img_disp,'Rectangle',bb,'p','LineWidth',1, 'Color', 'green');
        imshow(img_show)
        % imwrite(img_show, strcat('img_', int2str(imageIndex),'_positive.jpg'));
        % pause;
    end
    
    % create crop masks
    CROP = img ~= 0;
    MASK_1 = [];
    for k = 1:size(img,1)
        
        MASK_1 = [MASK_1; k * ones(1, size(img,2))];
        
    end
    MASK_1 = MASK_1(CROP);
    
    MASK_2 = [];
    for k = 1:size(img,2)
        
        MASK_2 = [MASK_2, k * ones(size(img,1),1)];
        
    end
    MASK_2 = MASK_2(CROP);
    MASK = [MASK_2, MASK_1]; % image reference
    
    % get feature vectors for the positive instances
    for j = 1:size(bb,1)
        
        crop = double(imcrop(img,bb(j,:)));
        crop = crop(:);
        crop = (crop - mean(crop))/std(crop); % normalize feature vector

        FEATURES = [FEATURES; crop'];
        LABELS{length(LABELS) + 1,1} = 'positive';
    end
    
    % get feature vectors for the negative instances
    counter = 1;
    index = 1;
    MASK = MASK(randsample(size(MASK,1),size(MASK,1)),:);
    bb_negative = [];
    
    while(counter <= size(bb,1) && index < size(MASK,1))
        
        % check if the current sampling bb overlap with one of the bb
        index = index + 1;
        current_bb = [MASK(index, :), 2 * bb_radius, 2 * bb_radius];
        
        if max(rectint(current_bb, bb)) <= param_PosNegOverlap_max && max(rectint(current_bb, bb)) > param_PosNegOverlap_min
            
            counter = counter + 1;
            
            bb_negative = [bb_negative; current_bb];
            
            crop = double(imcrop(img,current_bb));
            crop = crop(:);
            crop = (crop - mean(crop))/std(crop); % normalize feature vector

            FEATURES = [FEATURES; crop'];
            LABELS{length(LABELS) + 1,1} = 'negative';
            
        end
    end
   
    if(bool_display) % display padded image and negative bounding boxes
        figure(1)
        img_disp = double(img);
        img_disp = rescale(img_disp, 0,1);
        img_show = insertObjectAnnotation(img_disp,'Rectangle',bb_negative,'n','LineWidth',1, 'Color', 'red');
        imshow(img_show);
        %imwrite(img_show, strcat('img_', int2str(imageIndex),'_negative.jpg'));
        % pause;
    end
end

