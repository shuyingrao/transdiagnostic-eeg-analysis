%% === 脚本：HX聚类置换检验 + HQ重复逐点t检验 ===
clear; clc; close all; ft_defaults

%% 参数设置
power_type = 'norm';  % 'raw' / 'log' / 'norm'
switch lower(power_type)
    case 'raw';  param_field = 'powspctrm';
    case 'log';  param_field = 'logpowspctrm';
    case 'norm'; param_field = 'normpowspctrm';
    otherwise; error('Unknown power_type');
end

groups = {'HC', 'BD', 'MDD', 'SCZ'};
thresh_ratio = 0.95;
num_repeat = 100;

OutputFolder = '\statics\4groups';

if ~exist(OutputFolder,'dir')
    mkdir(OutputFolder)
end

%% 加载显著mask 和邻居结构

neighbours = load('EEGNeighbour.mat');
neighbours = neighbours.neighbours;

ValidationResult = load(['\statics\4groups\ValidationResult_' power_type '.mat']);

sig_mask = ValidationResult.HQ_mask;
freqs = ValidationResult.freqs;
labels = ValidationResult.labels;

%% === HX 聚类置换检验 ===
site = 'HX';
group_data = cell(1,4);
group_sizes = zeros(1, 4);
for g = 1:4
    folder = ['\features\power_spectrum\' site '_' groups{g}];

    files = dir([folder '/*.mat']);
    temp = cell(1,length(files));
    for i = 1:length(files)
        load(fullfile(folder, files(i).name), 'FFT_Power');
        psd = FFT_Power;
        psd.(param_field)(~sig_mask) = 0;   % 显著mask
        temp{i} = psd;
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

% 聚类检验
nHC = length(group_data{1});
comp_names = {'BD', 'MDD', 'SCZ'};
HX_stats = struct();

for c = 1:3
    all_masks = zeros(num_repeat, length(labels), length(freqs));
    all_tvals = zeros(num_repeat, length(labels), length(freqs));
    for rep = 1:num_repeat
        rng(2024 + rep);
        dz_all = group_data{c+1};
        idx = randperm(length(dz_all), nHC);
        dz_sel = dz_all(idx);
        dz_info = info_data{c+1}(idx, : );
        hc = group_data{1};
        hc_info = info_data{1};

        cfg = [];
        cfg.keepindividual = 'yes'; 
        cfg.parameter = param_field;
        PSD1 = ft_freqgrandaverage(cfg, dz_sel{:});        
        PSD2 = ft_freqgrandaverage(cfg, hc{:});

        % design matrix t-test :  first 2 column groups, the last 2 column covariate
        % (age/gender)
        N_total  = nHC*2;
        X = zeros(N_total,4);
        X(1:nHC,1) = 1;
        X(1+nHC*1:nHC*2,2) = 1;

        % coveraite
        S = [dz_info; hc_info];
        X(:,[3 4]) = S;   % covariate with age and gender

        contrast = [1 -1 0 0];  % group1 > group 2 : +; group1 < group 2 : -
        test = 'ttest';
        % Total number of permutations to generate
        GLM  = [];
        GLM.perms=0;
        GLM.X = X;
        GLM.contrast= contrast;
        GLM.test= test;

        cfg                  = [];
        cfg.parameter        =  param_field;
        cfg.channel          = {'all'};
        cfg.frequency        =  [1 40];
        % cfg.latency          = [-0.1 0.9];
        cfg.method           = 'montecarlo';
        cfg.statistic        = 'GLMtest';    % ft_statfun_GLMtest
        cfg.correctm         = 'cluster';
        cfg.clusterthreshold = 'nonparametric_common';  % https://mailman.science.ru.nl/pipermail/fieldtrip/2021-May/040896.html
        cfg.clusteralpha     = 0.001;
        cfg.clusterstatistic = 'maxsum';
        cfg.minnbchan        = 3;
        cfg.tail             = 0;
        cfg.clustertail      = 0;
        cfg.alpha            = 0.001;
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

        cfg.design         = [];
        cfg.design(1,:)    = [ones(1,nHC) ones(1,nHC)*2];

        stat = ft_freqstatistics(cfg, PSD1, PSD2);
        stat_mask = double(stat.mask);

        all_masks(rep,:,:) = stat_mask;
        all_tvals(rep,:,:) = stat.stat .* stat_mask;
    end
    mask_ratio = squeeze(mean(all_masks, 1));
    HX_sig_mask = mask_ratio > thresh_ratio;
    stat_sum = squeeze(sum(all_tvals,1));
    mask_count = squeeze(sum(all_masks,1));
    tval_map = stat_sum ./ (mask_count + eps);
    tval_map(~HX_sig_mask) = 0;

    HX_stats.([comp_names{c} '_vs_HC_mask']) = HX_sig_mask;
    HX_stats.([comp_names{c} '_vs_HC_tval']) = tval_map;
end

% === 疾病 vs 疾病：HX聚类置换（只执行一次） ===
dz_pairs = {[2,3], [2,4], [3,4]};  % BDvsMDD, BDvsSCZ, MDDvsSCZ
for p = 1:length(dz_pairs)
    idx1 = dz_pairs{p}(1); idx2 = dz_pairs{p}(2);
    name1 = groups{idx1}; name2 = groups{idx2};
    group1 = group_data{idx1};
    group2 = group_data{idx2};

    cfg = [];
    cfg.keepindividual = 'yes'; 
    cfg.parameter = param_field;
    PSD1 = ft_freqgrandaverage(cfg, group1{:});
    PSD2 = ft_freqgrandaverage(cfg, group2{:});

    % design matrix t-test :  first 2 column groups, the last 2 column covariate
    % (age/gender)
    N_total  = group_sizes(idx1) + group_sizes(idx2);
    X = zeros(N_total,4);
    X(1:group_sizes(idx1),1) = 1;
    X(1+group_sizes(idx1)*1:N_total,2) = 1;

    % coveraite
    S = [info_data{idx1}; info_data{idx2}];
    X(:,[3 4]) = S;   % covariate with age and gender

    contrast = [1 -1 0 0];
    test = 'ttest';
    % Total number of permutations to generate
    GLM  = [];
    GLM.perms=0;
    GLM.X = X;
    GLM.contrast= contrast;
    GLM.test= test;

    cfg                  = [];
    cfg.parameter        =  param_field;
    cfg.channel          = {'all'};
    cfg.frequency        =  [1 40];
    % cfg.latency          = [-0.1 0.9];
    cfg.method           = 'montecarlo';
    cfg.statistic        = 'GLMtest';    % ft_statfun_GLMtest
    cfg.correctm         = 'cluster';
    cfg.clusterthreshold = 'nonparametric_common';  % https://mailman.science.ru.nl/pipermail/fieldtrip/2021-May/040896.html
    cfg.clusteralpha     = 0.001;
    cfg.clusterstatistic = 'maxsum';
    cfg.minnbchan        = 3;
    cfg.tail             = 0;
    cfg.clustertail      = 0;
    cfg.alpha            = 0.001;
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

    cfg.design         = [];
    cfg.design(1,:)    = [ones(1,group_sizes(idx1)) ones(1,group_sizes(idx2))*2];

    stat = ft_freqstatistics(cfg, PSD1, PSD2);
    HX_stats.([name1 '_vs_' name2 '_mask']) = stat.mask;
    HX_stats.([name1 '_vs_' name2 '_tval']) = stat.stat .* double(stat.mask);
end

%% === HQ重复逐点t检验 ===
site = 'HQ';

ValidationResult = load(['\statics\4groups\ValidationResult_' power_type '.mat']);
sig_mask = ValidationResult.HQ_mask;
freqs = ValidationResult.freqs;
labels = ValidationResult.labels;
group_data = cell(1,4);
group_sizes = zeros(1, 4);
for g = 1:4
    folder = ['\features\power_spectrum\' site '_' groups{g}];
    files = dir([folder '/*.mat']);
    temp = cell(1,length(files));
    for i = 1:length(files)
        load(fullfile(folder, files(i).name), 'FFT_Power');
        psd = FFT_Power;
        psd.(param_field)(~sig_mask) = 0;
        temp{i} = psd;
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

HQ_stats = struct();
nHC = length(group_data{1});

% 疾病 vs HC 重复 t 检验
for c = 1:3
    sig_mask = HX_stats.([comp_names{c} '_vs_HC_mask']);
    all_tvals = zeros(num_repeat, length(labels), length(freqs));
    HQ_all_mask = zeros(num_repeat, length(labels), length(freqs));
    dz_all = group_data{c+1};
    for rep = 1:num_repeat
        rng(2024 + rep);
        dz_all = group_data{c+1};
        idx = randperm(length(dz_all), nHC);
        dz_sel = dz_all(idx);
        dz_info = info_data{c+1}(idx, : );
        hc = group_data{1};
        hc_info = info_data{1};

        % design matrix t-test :  first 2 column groups, the last 2 column covariate
        % (age/gender)
        N_total  = nHC*2;
        X = zeros(N_total,4);
        X(1:nHC,1) = 1;
        X(1+nHC*1:nHC*2,2) = 1;

        % coveraite
        S = [dz_info; hc_info];
        X(:,[3 4]) = S;   % covariate with age and gender

        contrast = [1 -1 0 0];
        test = 'ttest';
        % Total number of permutations to generate
        GLM  = [];
        GLM.perms = 1000;
        GLM.X = X;
        GLM.contrast= contrast;
        GLM.test= test;


        for ch = 1:length(labels)
            for f = 1:length(freqs)
                if sig_mask(ch,f)
                    hc_vals = cellfun(@(x) x.(param_field)(ch,f), hc);
                    dz_vals = cellfun(@(x) x.(param_field)(ch,f), dz_sel);
                    hc_vals = hc_vals(~isnan(hc_vals)); dz_vals = dz_vals(~isnan(dz_vals));
                    if ~isempty(hc_vals) && ~isempty(dz_vals)

                        cfg.GLM            = GLM;  
                        cfg.ivar           = 1; 
                        cfg.design         = [];
                        cfg.design(1,:)    = [ones(1,nHC) ones(1,nHC)*2];
                        data = [dz_vals hc_vals];

                        test_stat = ft_statfun_GLMtest(cfg, data, cfg.design);  % 返回 (nPerms + 1) x 1
                        T_obs = test_stat.stat(1); % 原始 t 值
                        T_null = test_stat.stat(2:end); % null 分布
                        pval = (1 + sum(abs(T_null) >= abs(T_obs))) / (1 + numel(T_null));  % 双尾 p 值
                        if pval < 0.05
                            HQ_all_mask(rep,ch,f)= 1;
                            all_tvals(rep,ch,f) = T_obs;
                        end

                        % [~,pval,~,stats] = ttest2(dz_vals, hc_vals);
                        % if pval < 0.05
                        %     HQ_all_mask(rep,ch,f)= 1;
                        %     all_tvals(rep,ch,f) = stats.tstat;
                        % end

                    end
                end
            end
        end
    end
    mask_ratio = squeeze(mean(HQ_all_mask, 1));
    HQ_sig_mask = mask_ratio > thresh_ratio;
    stat_sum = squeeze(sum(all_tvals,1));
    mask_count = squeeze(sum(HQ_all_mask,1));
    HQ_tval_map = stat_sum ./ (mask_count + eps);
    HQ_tval_map(~HQ_sig_mask) = 0;

    HQ_stats.([comp_names{c} '_vs_HC_mask']) = HQ_sig_mask;
    HQ_stats.([comp_names{c} '_vs_HC_tval']) = HQ_tval_map;

end 

% === 疾病 vs 疾病：HQ逐点 t 检验（只执行一次）===
for p = 1:length(dz_pairs)
    idx1 = dz_pairs{p}(1); idx2 = dz_pairs{p}(2);
    name1 = groups{idx1}; name2 = groups{idx2};
    sig_mask = HX_stats.([name1 '_vs_' name2 '_mask']);
    group1 = group_data{idx1};
    group2 = group_data{idx2};

    % design matrix t-test :  first 2 column groups, the last 2 column covariate
    % (age/gender)
    N_total  = group_sizes(idx1) + group_sizes(idx2);
    X = zeros(N_total,4);
    X(1:group_sizes(idx1),1) = 1;
    X(1+group_sizes(idx1)*1:N_total,2) = 1;

    % coveraite
    S = [info_data{idx1}; info_data{idx2}];
    X(:,[3 4]) = S;   % covariate with age and gender

    contrast = [1 -1 0 0];
    test = 'ttest';
    % Total number of permutations to generate
    GLM  = [];
    GLM.perms = 1000;
    GLM.X = X;
    GLM.contrast= contrast;
    GLM.test= test;


    tval_map = zeros(length(labels), length(freqs));
    HQ_mask = zeros(length(labels), length(freqs));
    for ch = 1:length(labels)
        for f = 1:length(freqs)
            if sig_mask(ch,f)
                vals1 = cellfun(@(x) x.(param_field)(ch,f), group1);
                vals2 = cellfun(@(x) x.(param_field)(ch,f), group2);
                vals1 = vals1(~isnan(vals1)); vals2 = vals2(~isnan(vals2));
                if ~isempty(vals1) && ~isempty(vals2)

                    cfg.GLM            = GLM;
                    cfg.ivar           = 1;
                    cfg.design         = [];
                    cfg.design(1,:)    = [ones(1,group_sizes(idx1)) ones(1,group_sizes(idx2))*2];
                    data = [vals1 vals2];

                    test_stat = ft_statfun_GLMtest(cfg, data, cfg.design);  % 返回 (nPerms + 1) x 1
                    T_obs = test_stat.stat(1); % 原始 t 值
                    T_null = test_stat.stat(2:end); % null 分布
                    pval = (1 + sum(abs(T_null) >= abs(T_obs))) / (1 + numel(T_null));  % 双尾 p 值
                    if pval < 0.05
                        HQ_mask(ch,f)= 1;
                        tval_map(ch,f) = T_obs;
                    end

                    % [~,pval,~,stats] = ttest2(vals1, vals2);
                    % if pval < 0.05
                    %     HQ_mask(ch,f)= 1;
                    %     tval_map(ch,f) = stats.tstat;
                    % end

                end
            end
        end
    end
    HQ_stats.([name1 '_vs_' name2 '_tval']) = tval_map;
    HQ_stats.([name1 '_vs_' name2 '_mask']) = HQ_mask;
end

filename = ['HXClusterHQTStats_'  power_type '.mat'];
save(fullfile(OutputFolder,filename), 'HX_stats','HQ_stats', 'freqs', 'labels');