%//////////////////////////////////////////////////////////////////////////
% SVM-based automatic cell segmentation and counting for histology data
% version: stable release v1.0
% author: Quentin RV. Ferry
% license: MIT license
%—————————————————————————————————
% function: Uses SVM linear classifiers trained with script 4 to
% automatically annotate cells in the cropped images listed by script 5.
%//////////////////////////////////////////////////////////////////////////

clear all; close all; clc; % clear session
script_path = pwd(); % grab path to working directory

%% USER DEFINED PARAMETERS
param_pathToTrainingFolder = '../SET/trainingSet_20211118_4164/'; % path to the training folder created by script 3.
param_windowStep = 2; % pixel offset between two consecutive 'sliding' windows.
param_postProba_threshold = 0.99; % classification threshold (default > 0.99 likelihood).
param_overlapThreshold = 0.5; % percentage overlap below which two bounding boxes are considered representing the same cell. 
param_pathToLog = 'log.txt'; % name of log text file used to report images with very little cell detection (this images could be problematic for the classifiers and are key for parameter tuning).
param_displayPlot = true; % boolean to display visual of the segmentation process.

%% MAIN
% open the log file
fileID = fopen(param_pathToLog,'w');

% load list of images to analyse:
load('FileList_test.mat'); % load fileList variable

% load the models
load(strcat(param_pathToTrainingFolder, 'SVM_consensus.mat'));
 
for file_index = 1:size(fileList,1) % loop over images

    fprintf('\n-------------------------------- index %i of %i\n\n', file_index, size(fileList,1));
    path_to_image = strcat(fileList(file_index).folder, '/', fileList(file_index).name);
    fprintf('image path: %s\n', path_to_image);
    
    % get image name
    name = strsplit(fileList(file_index).name, '.');
    name = name{1};
    
    % create paths to output files
    path_to_save_image = strcat(fileList(file_index).folder, '/', name, '_SVMStable_labelled.jpg');
    path_to_save_bb = strcat(fileList(file_index).folder, '/', name, '_SVMStable_bb.mat');

    % load image and generate windows per radius
    % find number of unique radii and corresponding padding
    radii = [];
    for i = 1:size(MODELS,2)
        radii = [radii, MODELS{1,i}];
    end
    radii = unique(radii);
    padding = 2 * max(radii);

    % container for features from sliding window
    fprintf('> sampling images\n');
    SAMPLINGS = cell(3,length(radii));
    
    %% for each unique radii get FEATURES and BBs
    for r = 1:length(radii)

        fprintf('... for radius: %i\n', r);

        bb_radius = radii(r);
        [FEATURES, BB, img] = slideWindowOverImage(path_to_image, bb_radius, param_windowStep, padding);

        % display image and a random k bounding boxes
        if param_displayPlot
            figure(1)
            img_show = insertObjectAnnotation(img,'Rectangle',BB(randsample(size(BB,1), 200),:),'i','LineWidth',1, 'Color', 'cyan');
            imshow(img_show)
        end

        SAMPLINGS{1,r} = img;
        SAMPLINGS{2,r} = BB;
        SAMPLINGS{3,r} = FEATURES;

    end

    %% run each all SVM models on sampled bounding boxes
    fprintf('> running SVMs\n');
    VERDICT = cell(2, size(MODELS,2));

    for m = 1:size(MODELS,2)

        fprintf('... for model: %i\n', m);

        % find radius used
        index_radius = find(radii == MODELS{1,m});
        img = SAMPLINGS{1,index_radius};
        BB = SAMPLINGS{2,index_radius};
        FEATURES = SAMPLINGS{3,index_radius};

        % transform to PCA and predict
        FEATUREScentered = FEATURES - mean(FEATURES,1); % center feature vectors
        FEATUREScentered_pca = FEATUREScentered * MODELS{3,m}; % project in PCA basis
        FEATUREScentered_pca_short = FEATUREScentered_pca(:, 1:MODELS{2,m}); % reduce dimensionality
        [prediction_label,PostProbs] = predict(MODELS{5,m},FEATUREScentered_pca_short); % PostProbs(:,1) = negative, PostProbs(:,2) = positive
        BB_positive = BB(PostProbs(:,2) > param_postProba_threshold,:);
        
        if param_displayPlot
            figure(2)
            img_show = insertShape(img,'rectangle',BB_positive,'LineWidth',1, 'Color', 'green');
            imshow(img_show)
        end

        % filter for overlapping positive BBs
        % current radius MODELS{1,m}
        DATA.img = img;
        DATA.BB = BB_positive;
        DATA.threshold = param_overlapThreshold * MODELS{1,m};
        save(strcat(script_path,'/temp.mat'), 'DATA');
        
        [BB_refined, representation] = refine_BB_point(BB_positive, param_overlapThreshold * MODELS{1,m});
        
        if param_displayPlot
            figure(3)
            img_show = insertShape(img,'rectangle',BB_refined(representation == 1,:),'LineWidth',1, 'Color', 'white');
            img_show = insertShape(img_show,'rectangle',BB_refined(representation > 1,:),'LineWidth',1, 'Color', 'green');
            imshow(img_show)
        end

        % new: remove all BB that don't have more than 1 representation,
        % set other representations to 1, before getting over to verdict

        VERDICT{1, m} = BB_refined(representation > 1,:);
        VERDICT{2, m} = representation(representation > 1);

    end


    %% combine verdicts to reach consensus
    fprintf('> Combining votes\n');

    BB = [];
    for m = 1:size(VERDICT,2)
        BB = [BB;VERDICT{1,m}];
    end
    
    % bring all to same radius
    r = max(radii);
    BB(:,1) = BB(:,1) + BB(:,3)/2 - r;
    BB(:,2) = BB(:,2) + BB(:,4)/2 - r;
    BB(:,3:4) = 2 * r;
        
    [BB_refined_final, representation_final] = refine_BB_point(BB, param_overlapThreshold * r);
    boundingBoxes = BB_refined_final(representation_final >= size(MODELS,2) - 1,:); 

    img_show = insertShape(img,'rectangle',boundingBoxes,'LineWidth',1, 'Color', 'green');
    if param_displayPlot
        figure(4)
        imshow(img_show)
    end
    imwrite(img_show, path_to_save_image);
    save(path_to_save_bb, 'boundingBoxes');
    
    if(size(boundingBoxes,1)<10)
        fprintf(fileID,'%i,%s\n',size(boundingBoxes,1), path_to_image);
    end

end
close all;
fclose(fileID);


%% FUNCTIONS

function [BB_refined, representation, Idx] = refine_BB_point(BB, distanceThreshold)
    
    N = size(BB,1);
    [Idx, Dist] = knnsearch(BB(:,1:2),BB(:,1:2),'K',N);

    % sort 
    [x, sort_idx] = sort(Dist(:,2), 'ascend');
    Idx = Idx(sort_idx,:);
    Dist = Dist(sort_idx,:);
    
    % form clusters
    belongToCluster = zeros(N,1);
    isSingle = zeros(N,1);
    cluster_idx = 0;
    indicesOfClusters = [];
    
    for i = 1:N
        current_idx = Idx(i,1);
        if belongToCluster(current_idx) == 0 % has not be assigned to a cluster yet
            
            cluster_idx = cluster_idx + 1;
            inCluster = Idx(i,Dist(i,:) <= distanceThreshold);
            % check what is left after removing assign bbs:
            inCluster = inCluster(belongToCluster(inCluster) == 0);
            
            if length(inCluster) > 1
                indicesOfClusters = [indicesOfClusters, cluster_idx];
            else
                isSingle(current_idx) = 1;
            end
            belongToCluster(inCluster) = cluster_idx;
        end
    end
    
    % create BB with all singles
    BB_refined = BB(isSingle == 1,:);
    representation = ones(size(BB_refined,1),1);
    
%     fprintf('nb singles: %i of %i\n',size(BB_refined,1), N);
%     fprintf('nb clusters: %i of %i\n',length(indicesOfClusters), N);
    
    % add cluster centers BB one by one
    if ~isempty(indicesOfClusters)
        
        start = size(BB_refined,1);
        representation = [representation; zeros(length(indicesOfClusters),1)];
        BB_refined = [BB_refined; zeros(length(indicesOfClusters),4)];
        
        
        for c = 1:length(indicesOfClusters)
            bb_cluster = BB(belongToCluster == indicesOfClusters(c),:);
            bb_avg = mean(bb_cluster);
            BB_refined(start + c,:) = bb_avg;
            representation(start + c) = size(bb_cluster,1);
        end
    end
    
end

function [FEATURES, BB, img] = slideWindowOverImage(path_to_image, bb_radius, param_windowStep, padding)

    % read image and add padding
    img = imread(path_to_image);
    img = padarray(img, [padding, padding], 0);
    img = imgaussfilt(img,1); % optional smoothing of the image. Should match what was used for training.
    %--------------------------
    
    i = 1;
    
    allocationSize = (floor(size(img,2)/param_windowStep)+1) * (floor(size(img,1)/param_windowStep)+1);
    FEATURES = zeros(allocationSize, (2*bb_radius+1)^2);
    BB = zeros(allocationSize, 4);
    current_pointer = 1;
    
    %fprintf('> %i\n', size(img,2) - 2 * bb_radius);
    
    while(i <= (size(img,2) - 2 * bb_radius))
        
        %fprintf('> column %i\n', i);
        j = 1;
        
        while(j <= (size(img,1) - 2 * bb_radius))
            
            current_bb = [i, j, 2 * bb_radius, 2 * bb_radius];
            
            crop = double(imcrop(img,current_bb));
            crop = crop(:);
            
            if(sum(crop ~= 0) > length(crop)/10) % discard bb which have no pixel content
                norm_factor = std(crop);
                if norm_factor == 0 
                    norm_factor = 1;
                end
                crop = (crop - mean(crop))/norm_factor; % normalize feature vector
                FEATURES(current_pointer, :) = crop';
                BB(current_pointer, :) = current_bb;
                current_pointer = current_pointer + 1;
            end

            j = j + param_windowStep;
            
        end
        
        i = i + param_windowStep;
        
    end
    
    FEATURES = FEATURES(1:current_pointer,:);
    BB = BB(1:current_pointer,:);
    
    %--------------------------
    img = rescale(img, 0,1);
    %--------------------------

end

