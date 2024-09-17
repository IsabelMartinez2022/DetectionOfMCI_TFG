% Root directories 
root_dirs = {'C:\Users\isama\Documents\MATLAB\ADNI_Complete\MCI', 
    'C:\Users\isama\Documents\MATLAB\ADNI_Complete\CN'};

% Array with median correlation values for each subject group
median_correlation_values_all = cell(length(root_dirs), 1);

% Loops through each root directory
for dir_idx = 1:length(root_dirs)
    root_data_dir = root_dirs{dir_idx};

    % List of all subject directories
    all_dirs = genpath(root_data_dir);
    split_dirs = strsplit(all_dirs, pathsep);

    % Directories with image data
    subject_dirs = split_dirs(~cellfun('isempty', regexp(split_dirs, '\\I[^\\]*$')));
    % As some samples were preprocessed under another folder pattern
    subject_realigned_dirs = split_dirs(~cellfun('isempty', regexp(split_dirs, 'I\d+_realigned')));

    % Directories containing the fMRI data
    fmri_dirs = split_dirs(~cellfun('isempty', regexp(split_dirs, 'fMRI')));
    fmri_dirs = fmri_dirs(contains(fmri_dirs, subject_dirs));

    total_subjects = length(fmri_dirs);
    total_regions = 200; 
    all_median_correlation_values = zeros(total_regions, total_regions, total_subjects);

% Loops through each subject directory
    for subj_idx = 1:total_subjects
         subj_dir = fmri_dirs{subj_idx}; 

        % List of smoothed functional images
        func_files = dir(fullfile(subj_dir, 'swa*.nii'));

        % Loads coregistered Schaefer atlas
        coregistered_atlas = dir(fullfile(subj_dir, 'rSchaefer*.nii'));
        if isempty(coregistered_atlas)
            warning('No atlas found in %s', subj_dir);
            continue;
        end
        coregistered_atlas_img = spm_vol(fullfile(coregistered_atlas.folder, coregistered_atlas.name));
        coregistered_atlas_data = spm_read_vols(coregistered_atlas_img);
        total_regions = max(coregistered_atlas_data(:));

        % Returns the number of files (= time points in fMRI data)
        timepoints = numel(func_files);
        mean_time_series = zeros(total_regions, timepoints);
        % Parameters for quality check
        global_mean_intensity = zeros(timepoints, 1);
        pairwise_variance = zeros(timepoints - 1, 1);
        head_motion = zeros(timepoints, 6);

        % Loops through each timepoint
        for i = 1:timepoints
            func_img = spm_vol(fullfile(func_files(i).folder, func_files(i).name));
            func_data = spm_read_vols(func_img);

            % Calculates global mean intensity across time
            global_mean_intensity(i) = mean(func_data(func_data > 0), 'all');

            % Calculates variance between voxel intensities in consecutive fMRI timepoints 
            % (starting from the second time point)
            if i > 1
                pairwise_variance(i-1) = var(func_data(:) - prev_func_data(:));
            end
            prev_func_data = func_data;

            % Searches for motion parameters in the other folder 
            %[~, subj_name, ~] = fileparts(subj_dir);
            %motion_folder = fullfile(fileparts(subj_dir), strrep(subj_name, '_realigned', ''));
            %motion_file = dir(fullfile(subj_dir, 'rp_*.txt'));
              
            %Searches for motion parameters in the same folder
            motion_file = dir(fullfile(func_files(i).folder, 'rp_*.txt'));
            if ~isempty(motion_file)
                motion_data = load(fullfile(motion_file.folder, motion_file.name));
                head_motion(i, :) = motion_data(i, :);
            end

            % Mean time series for each ROI
            for j = 1:total_regions
                % Using a binary mask from the atlas to average the signal inside that ROI
                region_mask = (coregistered_atlas_data == j);
                if any(region_mask(:))
                    region_time_series = mean(func_data(region_mask), 'all');
                    mean_time_series(j, i) = region_time_series;
                else
                    warning('Region %d has no voxels in the mask', j);
                end
            end
        end

        % Normalizes mean time series for each region
        %mean_time_series_normalized = mean_time_series ./ mean(mean_time_series, 2);

        % Normalizes global mean intensity
        global_mean_intensity_normalized = global_mean_intensity / mean(global_mean_intensity);
        % Normalizes pairwise variance by global mean instensity
        pairwise_variance_normalized = pairwise_variance / mean(global_mean_intensity(1:end-1));

        Creating a folder to save the FC results
        saving_directory = fullfile(subj_dir, 'result');
        if ~exist(saving_directory, 'dir')
            mkdir(saving_directory);
        end

        % Plot results
        figure(1);
        subplot(2, 2, 1);
        plot(global_mean_intensity_normalized);
        title('Global Mean Intensity');
        xlabel('Timepoints');
        xlim([0, timepoints - 1]);
        ylabel('Average signal (AU)');

        subplot(2, 2, 2);
        plot(pairwise_variance_normalized);
        title('Pairwise Variance');
        xlabel('Timepoints');
        xlim([0, timepoints - 1]);
        ylabel('Signal variability (AU^2)');

        subplot(2, 2, 3);
        % Conversion from radians to degrees for clearer visualization
        plot([head_motion(:,1:3) head_motion(:,4:6)*180/pi]);
        title('Rigid Body Motion');
        xlabel('Timepoints');
        xlim([0, timepoints - 1]);
        ylabel('Motion (mm/degrees)');
        legend({'X-t', 'Y-t', 'Z-t', 'X-r', 'Y-r', 'Z-r'}, 'FontSize', 5);

        % Computes the correlation matrix
        corr_matrix = corr(mean_time_series');
        % Applies Fisher transformation
        fisher_corr_matrix = atanh(corr_matrix);

        subplot(2, 2, 4);
        imagesc(fisher_corr_matrix, [-2.5, 2.5]); 
        colormap(jet);
        colorbar;
        title('Connectivity Matrix');
        xlabel('Region 1');
        ylabel('Region 2');

        % Saving figure
        combined_filename = fullfile(saving_directory, ['QC_' num2str(subj_idx) '.png']);
        saveas(gcf, combined_filename);
        close(gcf);

        % Saving time series and correlation matrix
        timeseries_name = ['TimeSeries_' num2str(subj_idx) '.csv'];
        csv_filename = fullfile(saving_directory, timeseries_name);
        MeanTimeSeriesTable = table((1:total_regions)', mean(mean_time_series, 2), 'VariableNames', {'Region', 'Mean_Time_Series'});
        writetable(MeanTimeSeriesTable, csv_filename);

        correlationmatrix_name = fullfile(saving_directory, ['ConnectivityMatrix_' num2str(subj_idx) '.csv']);
        CorrelationMatrixData = [];
        for i = 1:total_regions
            for j = 1:total_regions
                CorrelationMatrixData = [CorrelationMatrixData; i, j, fisher_corr_matrix(i, j)];
            end
        end
        CorrelationMatrixTable = array2table(CorrelationMatrixData, 'VariableNames', {'Region1', 'Region2', 'Fisher_Correlation'});
        writetable(CorrelationMatrixTable, correlationmatrix_name);
    end
    
    % Directories containing FC data
    result_dirs = split_dirs(~cellfun('isempty', regexp(split_dirs, 'result$')));
    total_subjects = length(result_dirs);

    for subj_idx = 1:total_subjects
        % Loops through each subject directory containing FC data
        result_dir = fullfile(result_dirs{subj_idx});
        fc_file = dir(fullfile(result_dir, 'ConnectivityMatrix*.csv'));
        
        % Checking if the connectivity file is found
        if isempty(fc_file)
            warning('No connectivity matrix for subject: %s', result_dir);
            continue;
        end
        
        % Loading the connectivity matrix file
        connectivity_matrix_path = fullfile(fc_file.folder, fc_file.name);
        CorrelationMatrixData = readtable(connectivity_matrix_path);
        
        % Reshaping the correlation values into a square matrix for median
        % calculation
        corr_matrix = reshape(corr_values, total_regions, total_regions)';
        all_median_correlation_values(:, :, subj_idx) = corr_matrix;
    end
    
    % Calculates the median correlation across the three dimensions
    median_correlation_values = median(all_median_correlation_values, 3);
    
    % Saving the median correlation matrix for each subject group 
    [~, subject_group, ~] = fileparts(root_data_dir);
    median_correlation_name = ['MedianCorrelationValues_' subject_group '.csv'];
    median_correlation_filename = fullfile(root_data_dir, median_correlation_name);
    
    % For CSV conversion
    MedianCorrelationData = [];
    for i = 1:total_regions
        for j = 1:total_regions
            MedianCorrelationData = [MedianCorrelationData; i, j, median_correlation_values(i, j)];
        end
    end
    
    % Saving the median correlation matrix as a CSV file
    MedianCorrelationTable = array2table(MedianCorrelationData, 'VariableNames', {'Region1', 'Region2', 'Median_Correlation'});
    writetable(MedianCorrelationTable, median_correlation_filename);
end
