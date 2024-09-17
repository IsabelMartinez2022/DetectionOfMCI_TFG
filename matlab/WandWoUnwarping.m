%Script to compare realignment (est&res) and realignment & unwarp methods
dataDir = pwd; 

filePatternA = fullfile(dataDir, 'rp_f*.txt'); % Files after realignment
filePatternB = fullfile(dataDir, 'u_rp_f*.txt'); % Files after realignment & unwarping
filesA = dir(filePatternA);
filesB = dir(filePatternB);

if length(filesA) ~= length(filesB)
    error('Mismatch in number of realignment and unwarping files.');
end

for i=1:10
    % File with motion parameters after just realignment
    params_a_i = importdata(fullfile(dataDir, filesA(i).name));

    % File with motion parameters after realignment and unwarping
    params_b_i = importdata(fullfile(dataDir, filesB(i).name));

    % Each column saves one of the six parameter types (translation in x, y, z, rotation in x, y, z) 
    transx_a = params_a_i(:, 1); 
    transy_a = params_a_i(:, 2); 
    transz_a = params_a_i(:, 3); 
    rotx_a = params_a_i(:, 4); 
    roty_a = params_a_i(:, 5); 
    rotz_a = params_a_i(:, 6); 

    transx_b = params_b_i(:, 1); 
    transy_b = params_b_i(:, 2); 
    transz_b = params_b_i(:, 3); 
    rotx_b = params_b_i(:, 4); 
    roty_b = params_b_i(:, 5); 
    rotz_b = params_b_i(:, 6); 

    % Plot of the motion parameters over time 
    time = 1:length(transx_a); % each row corresponds to a time point
    figure; 
    subplot(2, 1, 1); 
    plot(time, transx_a, 'r', time, transy_a, 'g', time, transz_a, 'b'); 
    title('Method 1 (realignment)'); 
    xlabel('Time'); 
    ylabel('Translation (mm)'); 
    legend('X', 'Y', 'Z'); 

    subplot(2, 1, 2); 
    plot(time, rotx_a, 'r', time, roty_a, 'g', time, rotz_a, 'b'); 
    ylabel('Rotation (radians)'); 
    legend('X', 'Y', 'Z'); 

    figure; 
    subplot(2, 1, 1); 
    plot(time, transx_b, 'r', time, transy_b, 'g', time, transz_b, 'b'); 
    title('Method 2 (realignment + unwarping)'); 
    xlabel('Time'); 
    ylabel('Translation (mm)'); 
    legend('X', 'Y', 'Z'); 

    subplot(2, 1, 2); 
    plot(time, rotx_b, 'r', time, roty_b, 'g', time, rotz_b, 'b'); 
    ylabel('Rotation (radians)'); 
    legend('X', 'Y', 'Z'); 

end 