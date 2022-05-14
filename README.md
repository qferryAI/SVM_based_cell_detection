# SVM-based cell segmentation

Log file for the making of the GitHub repository for Afif’s first submission 
(last modification 20220512)

# Code local repos

- Last code I used is at (**LV**): `/Users/quentinferry/Dropbox/TONEGAWA/PROJECTS/P11_concept/BATCH_8_DIO-DREADD/HISTOLOGY/20211115_DIO-DREADD_manip/cfos_experiment/ANALYSIS/HPC/20200129_SVM_stable`
- The implementation for large images is at: `-`
- All original codes are at (last **XZ**): `/Users/quentinferry/Dropbox/TONEGAWA/RESOURCE/SOFTWARES_lab/CellDetection`

# Log

- I decided to work on the last version that I have used.
- So there is a difference on how channels are handled between the last version for XZ (uses three channels) and LV, which uses only one channel.
    - I decided to go with the system with a single channel because it can be scaled up easily to each channel.
    - I am basing my work here on: `/Users/quentinferry/Dropbox/TONEGAWA/PROJECTS/P11_concept/BATCH_8_DIO-DREADD/HISTOLOGY/20211115_DIO-DREADD_manip/cfos_experiment/ANALYSIS/HPC/20200129_SVM_stable`

# Header for each file

%////////////////////////////////////////////////////////////////////////////////////////////////////

% SVM-based cells segmentation and counting for histology data

% version: stable release v1.0

% author: Quentin RV. Ferry

% license: MIT license

%—————————————————————————————————

% function: 

%////////////////////////////////////////////////////////////////////////////////////////////////////

---

# README (draft 1.0)

- Notes
    - add a script to compute a #cells/pixel (count the number of cells and divide by the ROI area in pixels)
    - ⚠️ All scripts use relative path with “/” for Unix architectures. Windows users will need to modify the paths to match their own architecture. Of note, some errors can be caused by a missing “/” at the end of the path. Make sure that the paths that you create are not corrupted.

- Repository’s architecture:
    - `root`
        - `CODE`
            - script_1
            - script_2
            - ...
        - `IMG` contains all images and results from the segmentation
            - `ALL` contains all images. Use one subfolder per histological section.
                - `section_1`
                - `section_2`
                - ...
            - `TRAINING` contains a subset of images used for training the SVM classifiers. Use one subfolder per histology section.
                - `section_i`
                - `section_j`
                - ...
        - `SET` contains a folder for each instance of training
            - `trainingSet_date_seed` contains the `data_i.mat` files with the coordinate of the manual annotation over the training images, and a `SVM_consensus.mat` file with the parameter of the trained SVM classifiers.

---

- Step by step manual
    - **Step 0**: Acquire histology images and export to *.jpg format.
        - Each section should have its own folder in `root/IMG/ALL` eg. `root/IMG/ALL/section_1`.
        - The image for section_1 should be name `section_1.jpg` so as to match the name of its parent directory.
        - Images should be exported one channel at the time. For example, if trying to detect cells expressing cfos, export the corresponding channel. Note that sometimes it is easier to delineate regions using several markers or a specific channel that is not the channel of interest (e.g. DAPI). In that case, run pipeline steps x to x on these images and copy the x files to the folders with channels of interest.
    - **Step 1**: Train SVM classifiers. The SVM classifiers are train on a subset of manually annotated images.
        - 1.1: Select a subset of representative images from `root/IMG/ALL` and move them to the training directory `root/IMG/TRAINING`.
        - 1.2: Use the `script_1_defineROIs.m` in `root/CODE` to define region of interests (ROIs) containing cells that will be manually segmented. We recommend having > 500 positive examples of cells to train the SVM classifier. Therefore, if you have N sections in `root/IMG/TRAINING`, we recommend selecting ROIs to contain about 500/N cells. When running the script on `root/IMG/TRAINING` the following will happen:
            - For each subfolder (i.e., for each section in the training set):
                - program opens the `section_i.jpg` file of the section in an interface that lets the user define an ROI with a polygon selection: left clic to add a vertex to the polygon and left clic on the first vertex to close the polygon. Once the selection is closed, two files are created in the same subdirectory: `INFO.mat` will contain the coordinate of the ROI, and `section_i_cropped.jpg`, a black and white image of the cropped ROI.
            - Notes:
                - Parameters:
                    - **path_to_root**: relative path to the parent folder where sections to crop are stored. For example, set to `root/IMG/TRAINING/` to crop the sections in the training directory.
                    - **param_nbInstancesRan**: integer that determines the number of sections to run the code on. To run all sections, make sure that this number exceed the number of sections in the parent directory. Set to 1 to debug the code.
                    - **img_ext**: string with the extension of the original image (default = ‘.jpg’).
                    - **overwrite_INFO_file**: 0 or 1. Set to 0 if INFO.mat file already exist in the section directory, and 1 to redefine the ROIs.
                - Only one ROI per section can be created.
        - 1.3: Use `script_2_training_createFileList.m` in `root/CODE` to tell the computer which images to use for SVM training. Upon running this script the following will happen:
            - The program will loop over all subdirectories of `root/IMG/TRAINING`and create a list of all cropped images created in the previous step.
            - This information will be saved to the disk as `FileList_training.mat` in the `root/CODE` folder.
            - Notes:
                - Parameters:
                    - **FOLDERS**: MATLAB Cell variable containing the relative path to all directories containing sections that the user wish to use for training. Note that all these sections should have been cropped according to step 1.2 beforehand.
                    - **listOfExtensions**: MATLAB Cell variable listing the type of extensions that refer to cropped images.
        - 1.4: Use `script_3_training_manualAnnotation.m` in `root/CODE` to manually annotate all training ROIs with the cells they contain. Upon running this script the following will happen:
            - For each subdirectory (i.e., for each section in the training set):
                - programs open the image of the corresponding ROI in an interface that let the user add and remove cell annotations. A cursor should appear. Press the “1”, “2”, or “3” keys to add an annotation at the cursor location, remove the annotation closest to the cursor, or terminate the annotation process.
            - All modifications are saved in a bunch of files in the `root/SET/trainingSet_date_seed` folder:
                - `data_i.mat` contains the coordinates of the annotated cells in the ROI corresponding to image file i.
                - `FileList_training_short.mat` contains a mapping between each `data_i.mat` and its corresponding ROI file.
            - Notes:
                - When running this script, the user can decide to create brand new annotations or modify an existing training set. By default, the script creates a new training set folder in `root/SET` with the name `trainingSet_date_key`. All new annotations are stored in that directory. However the user can also modify existing annotation by setting the param_trainingDate and param_trainingKey parameters (see parameters below) to match an existing training set folder.
                - If the user provide the path to an already existing training folder, then the program will fetch existing annotations when displaying each ROI. The user can then update the current annotations by adding or removing cells.
                - It is really important to annotate all cell instances in the ROI because the SVM program will sample the remaining of the images to create negative examples for training. If cells are not annotated they could potentially be used as negative examples in the SVM program which will lead to less performant classifiers.
                - Parameters:
                    - **param_nbImagesToAnalyze**: Set the number of images to annotate from the list of training images (see previous step). Used to annotate a smaller sample in the case where a large number of training images were selected (user could for example decide to define ROIs on all images, add all images to the FileList_training.mat, and then randomly sample n < N cropped images to annotate).
                    - **param_trainingDate** & **param_trainingKey**: Are used to create/reference a unique training folder (`trainingSet_date_key`). When **param_trainingKey** is set to 0, a new folder is created using a random key sampled between [1000,9999]. When both a date and key are provided, the program will work on existing annotations stored in the matching training set directory.
                    - **param_invertColors**: Boolean used to display B&W images on an inverted color scheme.
        - 1.5: Use `script_4_training_SVMs.m` in `root/CODE` to train several classifiers on the training set.  Upon running this script the following will happen:
            - Program opens each ROI image listed in `root/SET/trainingSet_date_key`, uses the annotations from the previous step to crop small square region around each cells (bounding boxes) and add the flatten pixel content to a matrix of positive instances. Simultaneously, the program samples and crops similar size regions that have less than x% overlap with any positive bounding boxes in the ROI and add the flatten pixel content to a matrix of negative instances.
            - Program concatenates all positive matrices and negative matrices of all ROIs to create a training set matrix with corresponding labels (positive/negative).
            - Program reduces the dimensionality of the training set matrix using PCA.
            - Program trains a linear SVM-classifier on the training set.
            - This process is repeated  for multiple size of bounding boxes (see parameter param_bb_radii). Additionally, for each bounding box size, multiple classifiers are trained on slightly different training data (offset from ground truth annotations).
            - Parameters of the SVM-classifiers are then store as a `SVM_consensus.mat` file in the `root/SET/trainingSet_date_seed` folder.
            - Notes:
                - While a single classifier can usually do a good job at detecting new cells, we recommend training a minimum of 4x linear classifier (2x classifier per radius and 2x radii). At test time, the detection of new cells is based on a consensus over all linear classifiers, thus improving accuracy.
                - Parameters:
                    - **param_pathToTrainingFolder**: path to the training set folder created at the previous step.
                    - **param_bb_radii**: vector of radii corresponding to the radius of the bounding box surrounding each annotation. We recommend using 2x radii chosen such that the biggest cell occupy ~50% and ~70% of the corresponding bounding boxes. This can be assessed visually when running the code: figures will show the ROI with positive and negative bounding boxes.
                    - **param_jitter**: amount of jitter in pixels allowed around ground truth annotations.
                    - **param_nbClassifiers**: number of classifiers to train for each radius.
                    - **param_PosNegOverlap_max**: maximum % overlap between bounding boxes of positive (cell) and negative (background) instances.
                    - **param_PosNegOverlap_min**: minimum % overlap between bounding boxes of positive (cell) and negative (background) instances.
                    - **param_pca_precision**: Percentage of variance explained by the selected reduced PCA basis (real number in [0,100], default 75).
    - **Step 2**: Use trained SVM classifiers from step 1 to annotate cells across a large databank of histology sections.
        - 2.1: Use the `script_1_defineROIs.m` in `root/CODE` to define ROIs for each images in `root/IMG/ALL`:
            - See step 1.2 for details on the script.
        - 2.2: Use `script_5_test_createFileList.m` in `root/CODE` to list all cropped images (corresponding to ROIs) in a `FileList_test.mat` that will be saved in `root/CODE`.
            - See step 1.3 for details on the script.
        - 2.3: Use `script_6_test_svm_concensus_predict.m` in `root/CODE` to detect cells in all test ROIs. Upon running this script, the following will happen:
            - For each ROI in `FileList_test.mat`
                - For each SVM classifier:
                    - Image is sampled by sliding a window across the image. The content of each window is extracted, flattened, and subjected to the SVM classifier that casts a vote as to whether it contains a cell or not.
                    - All positive bounding boxes (predicted by the classifier to contain a cell) are collected and refined: bounding boxes that overlap are being replaced by a single bounding box and a representation count (how many bounding boxes were averaged to create the refined one). Because true positive cells are often detected in several overlapping bounding boxes, we remove any bounding box that does not have a representation number > 1.
                - Finally positive bounding boxes from all SVM classifiers are aggregated and compared. Only those bounding boxes that were flagged as positive by at least N-1 classifiers (where N is the total number of classifier used) are retained for the final selection.
                - Two files are created in the `root/IMG/ALL/section_i` directory: `section_i_cropped_SVMStable_labelled.jpg` that show the ROI and an overlay of the positive bounding boxes, and `section_i_cropped_SVMStable_bb.mat` that contain the coordinate of the positive bounding boxes.
            - Notes:
                - Parameters:
                    - **param_pathToTrainingFolder**: Path to the training set folder created during training (see step 1.4-5).
                    - **param_windowStep**: pixel offset between two consecutive 'sliding' windows.
                    - **param_postProba_threshold**: classification threshold for the SVM classifier. The classifier looks at the pixel content of a bounding box and outputs the likelihood that the bounding box contains a cell (i.e., positive). All bounding boxes with a likelihood greater than the threshold will be flagged as containing a cell.
                    - **param_overlapThreshold**: percentage overlap below which two bounding boxes are considered to contain the same cell.
                    - **param_pathToLog**: name of log text file used to report images with very little cell detection (this images could be problematic for the classifiers and are key for parameter tuning).
                    - **param_displayPlot**: boolean to display visual of the segmentation process.
    - **Step 3**: Refine predictions and count cells.
        - 3.1: Use `script_7_postProcessing.m` in `root/CODE` to further refine the pool of detected cells. Given that cells appear brighter than the background on the image, we can further refine the bounding box selection by removing all bounding boxes (i) whose top intensities are below the user defined threshold **param_pixelMin_cutoff** (i.e., probably does not contain a cell); (ii) whose delta between top intensity (corresponds to the cell in the bounding box) and bottom intensity (corresponds to background in the bounding box) is bellow a user defined **param_deltaMin_cutoff** (see parameters below). Set the parameters to your liking and run the code. It will produce in each `root/IMG/ALL/section_i` directory two new files: `section_i_cropped_SVMStable_labelled_postprocessed.jpg` showing the ROI with bounding boxes overlayed. Bounding boxes that satisfy the user defined criteria are shown in green, the other are shown in white. The coordinates for the refined bounding box pool are stored in `section_i_cropped_SVMStable_bb_postprocessed.mat`.
            - Notes
                - Parameters
                    - **param_pathToTrainingFolder**: path to the training set folder created during training.
                    - **param_nbInstancesToRun**: number of ROIs to run the program on. Set to 1 for debugging.
                    - **param_displayPlot**: boolean decides whether or not to display ROIs with bounding boxes overlay.
                    - **param_pixelMin_cutoff**: see description above.
                    - **param_deltaMin_cutoff**:  see description above.
                    - **param_prctile_top**: see description above.
                    - **param_prctile_bottom**: see description above.
        - 3.2: Use `script_8_manualModifications.m` in `root/CODE` if you wish to further manually correct the automatic annotations. The code is similar to `script_3_training_manualAnnotation.m` (step 1.4 above). This code applies to a single image. Upon running the code, the user is prompted to select the image he/she wish to work on. Always select one of the `section_i_cropped.jpg` images.  If you have run step 3.1, you can choose to modify the post processed annotations by setting `param_usePostProcessedAnnotations` to true. See step 1.4 for details on how to use the program.
            - Notes
                - Parameters:
                    - **param_pathToTrainingFolder**: path to the training set folder created during training.
                    - **param_usePostProcessedAnnotations**: boolean. false: use SVM annotations, true: use post processed annotations.
                    - **param_skipBBFiles**: boolean. ignore the automatic annotation all together.
        - 3.3: Once you are happy with the cell annotations, there are many things that can be done: You can for example count the number of cells and get a measure of cell densities in the all ROIs. If you have ran cell detection on several channels, you can use the coordinates of the cells to quantify the overlap between cell population in a given ROI. As an example, we provide `script_9_computeCellDensities.m` that outputs in `root/RESULTS` a csv file with the cell density for each ROI in the `root/IMG/ALL` folder. Upon running the code you will be prompted to select the parent directory containing all sections you are interested in (for example `root/IMG/ALL`).
            - Note:
                - Parameters:
                    - **param_BBExtension**: type of cell annotations to use.
                    - **param_resultFile_path**: path to the result .csv file.