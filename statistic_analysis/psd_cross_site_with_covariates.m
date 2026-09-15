% === 初始化与设置 ===
clear all; clc; close all;
ft_defaults

CodeFolder = '';
load(fullfile(CodeFolder,'EEGLayout.mat'))
load(fullfile(CodeFolder,'EEGNeighbour.mat'))

% 参数设置
power_type = 'raw';  % 'raw', 'log', 'norm'
thresh_ratio = 0.95;  % 重复显著比例阈值
nRepeat = 100;        % 重复次数
site = 'HQ';
groups = {'HC', 'BD', 'MDD', 'SCZ'};


% 参数字段
switch lower(power_type)
    case 'raw';  param_field = 'powspctrm';
    case 'log';  param_field = 'logpowspctrm';
    case 'norm'; param_field = 'normpowspctrm';
    otherwise; error('Unknown power_type');
end

OutputFolder = '\statics\4groups';
if ~exist(OutputFolder,'dir')
    mkdir(OutputFolder)
end

 
ResultFolder ='\statics\4groups';

% 加载 HX 聚类统计
HX_data = load(fullfile(ResultFolder, ['ClusterResult_' power_type '.mat']));
freqs = HX_data.freqs;
labels = HX_data.labels;
HX_mask = HX_data.mask_avg > thresh_ratio;
HX_fval = HX_data.stat_avg;
HX_fval(~HX_mask) = 0;

% 加载 HQ 数据
group_data = cell(1,4);
group_sizes = zeros(1, 4);
for g = 1:4
    psd_folder = ['\features\power_spectrum\' site '_' groups{g}];

    files = dir([psd_folder '/*.mat']);
    n = length(files);
    temp = cell(1,n);
    for i = 1:n
        fname = files(i).name;
        load(fullfile(psd_folder, fname));
        temp{i} = FFT_Power;
    end
    group_data{g} = temp;
    group_sizes(g) = length(temp);
end

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


nFreq = length(freqs);
nChan = length(labels);

HQ_mask_all = zeros(nRepeat, nChan, nFreq);
HQ_stat_all = zeros(nRepeat, nChan, nFreq);

for rep = 1:nRepeat
    fprintf('[%d/%d] Running ANOVA repeat...\n', rep, nRepeat);
    rng(2024 + rep);
    nHC = min(group_sizes);
    sampled_data = cell(1, 4);
    samples_info = cell(1, 4);
    for i = 1:4
        idx = randperm(group_sizes(i), nHC);
        sampled_data{i} = group_data{i}(idx);
        samples_info{i} = info_data{i}(idx, : );
    end

    % design matrix f-test :  first 4 column groups, the last 2 column covariate
    % (age/gender)
    % N_total  = nHC*4;
    % X = zeros(N_total,6);
    % X(1:nHC,1) = 1;
    % X(1+nHC*1:nHC*2,2) = 1;
    % X(1+nHC*2:nHC*3,3) = 1;
    % X(1+nHC*3:nHC*4,4) = 1;
    % 
    % % coveraite
    S = cat(1,samples_info{:});
    % X(:,[5 6]) = S;   % covariate with age and gender
    % 
    % contrast = [1 1 1 1 0 0];
    % test = 'ftest';
    % % Total number of permutations to generate
    % GLM  = [];
    % GLM.perms = 1000;
    % GLM.X = X;
    % GLM.contrast= contrast;
    % GLM.test= test;

    cfg = [];
    cfg.keepindividual = 'yes';
    cfg.parameter = param_field;

    HC_PSD = ft_freqgrandaverage(cfg, sampled_data{1}{:});
    BD_PSD = ft_freqgrandaverage(cfg, sampled_data{2}{:});
    MDD_PSD = ft_freqgrandaverage(cfg, sampled_data{3}{:});
    SCZ_PSD = ft_freqgrandaverage(cfg, sampled_data{4}{:});

    HC_full  = HC_PSD.(param_field);
    BD_full  = BD_PSD.(param_field);
    MDD_full = MDD_PSD.(param_field);
    SCZ_full = SCZ_PSD.(param_field);
    mask_anova = zeros(nChan, nFreq);
    mask_fval = zeros(nChan, nFreq);
    for ch = 1:nChan
        for fq = 1:nFreq
            if HX_mask(ch, fq)
                y = [squeeze(HC_full(:,ch,fq)); squeeze(BD_full(:,ch,fq)); ...
                    squeeze(MDD_full(:,ch,fq)); squeeze(SCZ_full(:,ch,fq))];
                g = [ones(nHC,1); 2*ones(nHC,1); 3*ones(nHC,1); 4*ones(nHC,1)];

                % cfg.GLM            = GLM;
                % cfg.ivar           = 1;
                % cfg.design         = [];
                % cfg.design(1,:)    = [ones(1,nHC) ones(1,nHC)*2 ones(1,nHC)*3 ones(1,nHC)*4];
                % test_stat = ft_statfun_GLMtest(cfg, y', cfg.design);  % 返回 (nPerms + 1) x 1
                % F_obs = test_stat.stat(1); % 原始 f 值
                % nGroups = 4;
                % nSubjPerGroup = nHC;
                % nTotal = nGroups * nSubjPerGroup;
                % df1 = nGroups - 1;
                % df2 = nTotal - nGroups;
                % pval = 1 - fcdf(F_obs, df1, df2);
                % if pval < 0.05
                %     mask_anova(ch, fq) = 1;
                %     mask_fval(ch, fq) = T_obs;
                % end

                [p, tbl, stats] = anovan(y, {g, S(:,1), S(:,2)}, ...
                    'model', 'linear', ...
                    'varnames', {'Group','Age','Sex'},'display','off');
                if p(1) < 0.05
                    mask_anova(ch, fq) = 1;
                    mask_fval(ch, fq) = tbl{2,6};
                end

            end
        end
    end

    % === 存入3D矩阵 ===
    HQ_mask_all(rep,:,:) = mask_anova;
    HQ_stat_all(rep,:,:) = mask_fval .* mask_anova;

end

% === 聚合显著比例和阈值掩码 ===
sig_ratio = squeeze(mean(HQ_mask_all, 1));   % [nChan, nFreq]
HQ_mask  = sig_ratio > thresh_ratio;     % [nChan, nFreq]
stat_sum = squeeze(sum(HQ_stat_all,1));
mask_count = squeeze(sum(HQ_mask_all,1));
HQ_fval = stat_sum ./ (mask_count + eps);
HQ_fval(~HQ_mask) = 0;

filename = ['ValidationResult_' power_type '.mat'];
save(fullfile(OutputFolder,filename), 'HQ_mask', 'HQ_fval', 'freqs', 'labels');

% === 可视化聚合结果 ===
label_order = {'Fp1','Fp2','F7','F3','F4','F8','T3','C3','C4','T4','T5','P3','P4','T6','O1','O2'};
[~, reorder_idx] = ismember(label_order, labels);
labels = labels(reorder_idx);
HX_fval = HX_fval(reorder_idx,:);
% HQ_mask = HQ_mask(reorder_idx,:);
HQ_fval = HQ_fval(reorder_idx,:);

figure('Name', 'HQ Center Verification - ANOVA1','Position', [100,100,1000,800]);
subplot(2,1,1);
imagesc(freqs, 1:nChan, HX_fval, [0 max(abs(HX_fval(:)))]);
title('HX Cluster F-values');
ylabel('Channel'); xlabel('Frequency (Hz)'); colorbar;
yticks(1:nChan); set(gca, 'YTickLabel', labels); axis tight;
colorbar; 

subplot(2,1,2);
imagesc(freqs, 1:nChan, HQ_fval, [0 max(abs(HQ_fval(:)))]);
title('HQ  F-statistic Map');
ylabel('Channel'); xlabel('Frequency (Hz)'); yticks(1:nChan);
set(gca, 'YTickLabel', labels);
colorbar;
% hcb = colorbar;
% colormap(gca,[0.2 0.2 0.9; 1 1 1; 0.9 0.2 0.2]);
% hcb.Ticks = [0 1]; hcb.TickLabels = {'no sig', 'sig'};
axis tight;

