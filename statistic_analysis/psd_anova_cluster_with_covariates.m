clear all;
clc;
close all;

ft_defaults

CodeFolder = 'E';
load(fullfile(CodeFolder,'EEGLayout.mat'))
load(fullfile(CodeFolder,'EEGNeighbour.mat'))

power_type =  'norm';
num_repeat = 1;

OutputFolder = '\statics\4groups';
if ~exist(OutputFolder,'dir')
    mkdir(OutputFolder)
end


param = lower(power_type);
switch param
    case 'raw'
        param_field = 'powspctrm';
    case 'log'
        param_field = 'logpowspctrm';
    case 'norm'
        param_field = 'normpowspctrm';
    otherwise
        error('Unknown power_type');
end


%%
sites = {'HX', 'HQ'};
groups = {'HC', 'BD', 'MDD', 'SCZ'};
n_sites = numel(sites);
n_groups = numel(groups);

% 初始化存储结构：每组合并后的数据
group_data = cell(1, n_groups);   % 每组拼接后的PSD数据（cell数组）
info_data = cell(1, n_groups);    % 每组拼接后的年龄性别信息（Nx2矩阵）
group_sizes = zeros(1, n_groups);

for g = 1:n_groups
    temp_all_subjects = {};    % 用于临时合并PSD数据
    temp_all_info = [];        % 用于临时合并年龄性别信息
    
    for s = 1:n_sites
        site = sites{s};

        % ------- 导入 PSD 数据 -------
        psd_folder = fullfile('\features\power_spectrum', [site '_' groups{g}]);
        files = dir(fullfile(psd_folder, '*.mat'));
        n = length(files);
        temp_psd = cell(1, n);
        for i = 1:n
            fname = files(i).name;
            load(fullfile(psd_folder, fname), 'FFT_Power');
            temp_psd{i} = FFT_Power;
        end

        % ------- 导入 年龄性别信息 -------
        info_file = fullfile('\info', [site '_age_sex.mat']);
        load(info_file);  % 加载变量 SubjectInfo
        group_info = SubjectInfo{1, g};
        nSubjects = numel(group_info);
        age_sex_matrix = zeros(nSubjects, 2);  % 第一列 Age，第二列 Sex
        for i = 1:nSubjects
            age_sex_matrix(i, 1) = group_info(i).Age;
            age_sex_matrix(i, 2) = group_info(i).Sex;
        end

        % ------- 合并 -------
        temp_all_subjects = [temp_all_subjects, temp_psd];
        temp_all_info = [temp_all_info; age_sex_matrix];
    end

    group_data{g} = temp_all_subjects;
    group_sizes(g) = length(temp_all_subjects);
    info_data{g} = temp_all_info;
end

%%
for rep = 1:num_repeat
    rng(2024 + rep);
    nHC = min(group_sizes);
    sampled_data = cell(1, 4);
    samples_info = cell(1, 4);
    for i = 1:4
        idx = randperm(group_sizes(i), nHC);
        sampled_data{i} = group_data{i}(idx);
        samples_info{i} = info_data{i}(idx, : );
    end

    cfg = [];
    cfg.keepindividual = 'yes';
    cfg.parameter = param_field;

    HC_PSD = ft_freqgrandaverage(cfg, sampled_data{1}{:});
    BD_PSD = ft_freqgrandaverage(cfg, sampled_data{2}{:});
    MDD_PSD = ft_freqgrandaverage(cfg, sampled_data{3}{:});
    SCZ_PSD = ft_freqgrandaverage(cfg, sampled_data{4}{:});


    %%
    % design matrix f-test :  first 4 column groups, the last 2 column covariate
    % (age/gender)
    N_total  = nHC*4;
    X = zeros(N_total,6);
    X(1:nHC,1) = 1;
    X(1+nHC*1:nHC*2,2) = 1;
    X(1+nHC*2:nHC*3,3) = 1;
    X(1+nHC*3:nHC*4,4) = 1;

    % coveraite
    S = cat(1,samples_info{:});
    X(:,[5 6]) = S;   % covariate with age and gender

    contrast = [1 1 1 1 0 0];
    test = 'ftest';
    % Total number of permutations to generate
    GLM  = [];
    GLM.perms=0;
    GLM.X = X;
    GLM.contrast= contrast;
    GLM.test= test;

    %%
    cfg                  = [];
    cfg.parameter        =  param_field;
    cfg.channel          = {'all'};
    cfg.frequency        =  [0 40];
    % cfg.latency          = [-0.1 0.9];
    cfg.method           = 'montecarlo';
    cfg.statistic        = 'GLMtest';    % ft_statfun_GLMtest
    cfg.correctm         = 'cluster';
    cfg.clusterthreshold = 'nonparametric_common';  % https://mailman.science.ru.nl/pipermail/fieldtrip/2021-May/040896.html
    cfg.clusteralpha     = 0.01;
    cfg.clusterstatistic = 'maxsum';
    cfg.minnbchan        = 3;
    cfg.tail             = 0;
    cfg.clustertail      = 0;
    cfg.alpha            = 0.05;
    cfg.numrandomization = 5000;
    % prepare_neighbours determines what sensors may form clusters
    cfg.neighbours       = neighbours;
    cfg.GLM              = GLM;    % GLM design
    % cfg.avgovertime = 'no';
    % cfg.avgoverfreq = 'no';
    % nsubj    = 2*nsubj;
    % cfg.design(1,:) = [1:nsubj 1:nsubj];
    % cfg.uvar        = 1; % row of design matrix that contains unit variable (in this case: subjects)
    cfg.ivar        = 1;   % number or list with indices indicating the independent variable(s)

    %%%%%%%%  HC MDD BD SCZ  %%%%%%%%%
    cfg.design         = [];
    cfg.design(1,:)    = [ones(1,nHC) ones(1,nHC)*2 ones(1,nHC)*3 ones(1,nHC)*4];
    stat= ft_freqstatistics(cfg,HC_PSD, BD_PSD, MDD_PSD, SCZ_PSD);

    if rep == 1
        nChan = length(stat.label);
        nFreq = length(stat.freq);
        mask_all = false(num_repeat, nChan, nFreq);
        stat_all = zeros(num_repeat, nChan, nFreq);
    end

    mask_all(rep,:,:) = stat.mask;
    stat_all(rep,:,:) = stat.stat .* double(stat.mask);
end

mask_avg = squeeze(mean(mask_all,1));
stat_sum = squeeze(sum(stat_all,1));
mask_count = squeeze(sum(mask_all,1));
stat_avg = stat_sum ./ (mask_count + eps);

freqs = stat.freq;
labels = stat.label;

filename = ['ClusterResult_' power_type '.mat'];
save(fullfile(OutputFolder,filename), 'mask_avg', 'stat_avg', 'freqs', 'labels', 'mask_all', 'stat_all');

HX_mask = mask_avg > 0.98;
stat_avg(~HX_mask) = 0;
label_order = {'Fp1','Fp2','F7','F3','F4','F8','T3','C3','C4','T4','T5','P3','P4','T6','O1','O2'};
[~, reorder_idx] = ismember(label_order, labels);
labels = labels(reorder_idx);
stat_avg = stat_avg(reorder_idx,:);
figure('Name', 'HX Center - ANOVA1','Position', [100,100,1000,800]);
subplot(2,1,1);
imagesc(freqs, 1:nChan, stat_avg, [0 max(abs(stat_avg(:)))]);
title('HX Cluster F-values');
ylabel('Channel'); xlabel('Frequency (Hz)'); colorbar;
yticks(1:nChan); set(gca, 'YTickLabel', labels); axis tight;
colorbar; 