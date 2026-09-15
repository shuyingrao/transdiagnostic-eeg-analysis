clear;
clc;
close all;

ft_defaults;

%% 路径和配置
CodeFolder = '';
Input_Folder = "\features\group_conn_matrix\";
conspctrm = 'powcorr_ortho'; % 'powcorr_ortho', 'icoh'
bands = {'delta', 'theta', 'alpha', 'beta', 'gamma'};
site = 'HX';
groups = {'HC', 'BD', 'MDD', 'SCZ'};
thresh = 5.5; % [2.6, 3.1, 3.8, 4.3, 5.5];

Output_Folder = "\statics\4groups_nbs_cov";
if ~exist(Output_Folder,'dir')
    mkdir(Output_Folder)
end

% NBS 参数                                                                    
primary_thresh  = thresh;
perm_n          = 5000;
sig_level       = 0.05; 
comp_size_type  = 'Extent';
repeats         = 50;      % 重复次数
num = 2;

info_file = ['\info\' site '_age_sex.mat'];
load(info_file);
info_data = cell(1, 4);
for g = 1:4
    group_info = SubjectInfo{1,g};
    nSubjects = numel(group_info);
    age_sex_matrix = zeros(nSubjects, 2);  % 第一列：Age，第二列：Sex
    for i = 1:nSubjects
        age_sex_matrix(i, 1) = group_info(i).Age;
        age_sex_matrix(i, 2) = group_info(i).Sex;
    end
    info_data{g} = age_sex_matrix;
end


%% 遍历频带
for band_idx = 5:5 % length(bands)
    band = bands{band_idx};
    fprintf('=== Band: %s ===\n', band);

    % 加载每组数据
    group_data = cell(1, 4);
    group_sizes = zeros(1, 4);
    for i = 1:4
        temp = load(fullfile(Input_Folder, conspctrm, [site '_' groups{i} '.mat']));
        group_data{i} = squeeze(temp.Group_conn(band_idx, :, :, :));
        group_sizes(i) = size(group_data{i}, 3);
    end
    
    min_n = min(group_sizes);
    nChan = size(group_data{1}, 1);

    con_mat_results = zeros(repeats, nChan, nChan);
    test_stat_results = zeros(repeats, nChan, nChan);

    %% 重复 N 次下采样 + NBS
    for rep = 1:repeats
        rng(42 + rep + repeats * (num - 1)); 
        fprintf('  Repeat %d / %d\n', rep, repeats);

        sampled_data = cell(1, 4);
        samples_info = cell(1, 4);
        for i = 1:4
            idx = randperm(group_sizes(i), min_n);
            sampled_data{i} = group_data{i}(:, :, idx);
            samples_info{i} = info_data{i}(idx, : );
        end
        group_dummy = zeros(min_n * 4, 4);
        for i = 1:4
            group_dummy(((i-1)*min_n+1):(i*min_n), i) = 1;
        end        
        S = cat(1,samples_info{:});
        design_mat = [group_dummy, S];

        all_data = cat(3, sampled_data{:});

        % 设计矩阵

        contrast = [1 1 1 1 0 0];


        assignin('base','myY', all_data);
        assignin('base','myX', design_mat);
        assignin('base','myC', contrast);

        % NBS参数设置
        UI.method.ui   = 'Run NBS';
        UI.test.ui     = 'F-test';
        UI.size.ui     = comp_size_type;
        UI.thresh.ui   = num2str(primary_thresh);
        UI.perms.ui    = num2str(perm_n);
        UI.alpha.ui    = num2str(sig_level);
        UI.contrast.ui = 'myC';
        UI.design.ui   = 'myX';
        UI.matrices.ui = 'myY';
        UI.exchange.ui = '';
        UI.node_coor.ui  = fullfile(CodeFolder,'node_coordinates.txt');
        UI.node_label.ui = fullfile(CodeFolder,'node_labels.txt');

        try
            clear global nbs;
            global nbs
            dummyS = struct();
            NBSrun(UI, dummyS);
            close all;

            if isfield(nbs.NBS, 'con_mat') && ~isempty(nbs.NBS.con_mat)
                con_mat_results(rep, :, :) = nbs.NBS.con_mat{1};
                test_stat_results(rep, :, :) = nbs.NBS.test_stat;
            end
            clear global nbs;
        catch ME
            warning('NBS failed at repetition %d: %s', rep, ME.message);
        end
    end

    % 保存结果
    save(fullfile(Output_Folder, sprintf('%s_NBS_result_thresh_%s_%s_%d.mat', conspctrm, string(thresh), band,num)), ...
        'con_mat_results', "test_stat_results");
    fprintf('%s_NBS_result_thresh_%s_%s_%d.mat saved!\n', conspctrm, string(thresh), band, num);
    % clear con_mat_results
    
    % save(fullfile(Output_Folder, sprintf('%s_NBS_result_thresh_%s_%s.mat', conspctrm, string(thresh), band)), ...
    %     'con_mat_results');
    % fprintf('%s_NBS_result_thresh_%s_%s.mat saved!\n', conspctrm, string(thresh), band);
end

% disp('=== All bands processed. ===');
mask = squeeze(sum(con_mat_results)); 