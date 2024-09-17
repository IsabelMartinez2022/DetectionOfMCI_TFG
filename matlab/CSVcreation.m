% New CSV files for neo4j import

% 1: Subject nodes
file_subjects = 'C:\Users\isama\Documents\MATLAB\ADNI_Complete\CN\CN_sMRIfMRI_9_13_2024.csv'; 
data = readtable(file_subjects);
subject_id = data.Subject;
diagnosis = data.Group;
sex = data.Sex;
age = data.Age;

% Combining into a table
data_extracted = table(subject_id, diagnosis, sex, age);

% Removing duplicats based on subject id
[~, unique_idx] = unique(data_extracted.subject_id);
data_extracted_unique = data_extracted(unique_idx, :);

% New CSV file
newfile_regions = 'C:\Users\isama\OneDrive\Documentos\CEU\CURSO 2023-2024\TFG\subjects.csv';
writetable(data_extracted_unique, newfile_regions);

% 2: Region nodes
file_regions = 'C:\Users\isama\Documents\MATLAB\spm12\toolbox\cat12\templates_MNI152NLin2009cAsym\Schaefer2018_200Parcels_17Networks_order_new.csv'; 
data2 = readtable(file_regions);
roi_id = data2.ROIid;
roi_name = data2.ROIname;

% New CSV file
data2_extracted = table(roi_id, roi_name);
newfile_regions = 'C:\Users\isama\OneDrive\Documentos\CEU\CURSO 2023-2024\TFG\regions.csv';
writetable(data2_extracted, newfile_regions);

% Subject-region edges 
file_subjects = 'C:\Users\isama\Documents\MATLAB\ADNI_Complete\CN\CN_sMRIfMRI_9_13_2024.csv'; 
file_regions = 'C:\Users\isama\Documents\MATLAB\spm12\toolbox\cat12\templates_MNI152NLin2009cAsym\Schaefer2018_200Parcels_17Networks_order_new.csv';

subject_data = readtable(file_subjects);
subject_ids = subject_data.Subject;
region_data = readtable(file_regions);
region_ids = region_data.ROIid;
region_names = region_data.ROIname;

% Ensuring that region_ids are treated as cell array of strings
if isnumeric(region_ids)
    region_ids = cellstr(num2str(region_ids));
end

root_data_dir = 'C:\Users\isama\Documents\MATLAB\ADNI_Complete\CN';

% Initializing table to store combined data
combined_data = table('Size',[0 4], 'VariableTypes', {'cell', 'cell', 'double', 'double'}, ...
    'VariableNames', {'region_id', 'subject_id', 'volume', 'cortical_thickness'});

for i = 1:length(subject_ids)
    subject_id = subject_ids{i};
    
    % Thickness and volume files
    thickness_files = findFiles(fullfile(root_data_dir, subject_id), 'ROI_Schaefer2018_200Parcels_17Networks_order_thickness\.csv$');
    volume_files = findFiles(fullfile(root_data_dir, subject_id), 'ROI_Schaefer2018_200Parcels_17Networks_order_Vgm\.csv$');
    thickness_data = readtable(thickness_files{1});
    volume_data = readtable(volume_files{1});
    
    % Skipping the first three columns for thickness as extra regions of the Schaefer atlas were added 
    % and the first column for volume
    thickness_values = thickness_data{:, 4:end};
    volume_values = volume_data{:, 2:end};
    
    % Creating a table for the current subject's data
    subject_table = table();
    
    % Iterating over all regions for each subject
    for j = 1:length(region_ids)
        region_id = region_ids{j};
        cortical_thickness = thickness_values(1, j);
        volume = volume_values(1, j);
        new_row = table({region_id}, {subject_id}, volume, cortical_thickness, 'VariableNames', {'region_id', 'subject_id', 'volume', 'cortical_thickness'});
        subject_table = [subject_table; new_row];
    end
    
    % Adding the current subject's data to the combined table
    combined_data = [combined_data; subject_table];
end

% Removing duplicates
combined_data = unique(combined_data);

% New CSV file
output_csv = 'C:\Users\isama\OneDrive\Documentos\CEU\CURSO 2023-2024\TFG\has_region_CN.csv';
writetable(combined_data, output_csv);

% To unify CN and MCI results for "has_region"
file1 = 'C:\Users\isama\OneDrive\Documentos\CEU\CURSO 2023-2024\TFG\has_region_CN.csv';
file2 = 'C:\Users\isama\OneDrive\Documentos\CEU\CURSO 2023-2024\TFG\has_region_MCI.csv';
data1 = readtable(file1);
data2 = readtable(file2);

% Concatenating the two tables by appending MCI rows below CN
unified_data = [data1; data2];  

% Sorting the unified data regions are ordered from 1 to 200
unified_data = sortrows(unified_data, 'region_id');

% New CSV file
output_csv = 'C:\Users\isama\OneDrive\Documentos\CEU\CURSO 2023-2024\TFG\has_region.csv';
writetable(unified_data, output_csv);

% Function to recursively find files
function files = findFiles(dirPath, pattern)
    dirData = dir(dirPath); 
    dirIndex = [dirData.isdir]; % Finding the directories index
    files = {dirData(~dirIndex).name}'; % List of the files
    if ~isempty(files)
    % Looping through each file and adding the directory path to the filename
        for i = 1:length(files)
        files{i} = fullfile(dirPath, files{i});
        end
    end
    subDirs = {dirData(dirIndex).name}; % List of subdirectories
    validIndex = ~ismember(subDirs, {'.', '..'}); % Finding index of subdirectories that are not '.' or '..'
    for iDir = find(validIndex)
        nextDir = fullfile(dirPath, subDirs{iDir}); % Getting the subdirectory path
        files = [files; findFiles(nextDir, pattern)]; % Recursively calling findFiles
    end
    % Filtering files by pattern
    files = files(~cellfun('isempty', regexp(files, pattern, 'once')));
end