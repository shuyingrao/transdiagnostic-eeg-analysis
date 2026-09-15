clear; clc;close all;

% ============ 参数设置 =============
conspctrm = 'powcorr_ortho';
band_list = {'delta','theta','alpha','beta','gamma'};
group_names = {'HC', 'BD', 'MDD', 'SCZ'};
pair_list = { ...
    {'BD', 'HC'}, {'MDD', 'HC'}, {'SCZ', 'HC'}, ...
    {'BD', 'MDD'}, {'BD', 'SCZ'}, {'MDD', 'SCZ'}};
pair_idx_list = { [2,1], [3,1], [4,1], [2,3], [2,4], [3,4]};
Input_Folder = '\features\group_conn_matrix\';
thresh = '5.5'; % [2.6, 3.1, 3.8, 4.3, 5.5];
repeat_N = 100;
thresh_ratio = 0.95;

Output_Folder = '\statics\4groups_nbs_cov\2groups\';
if ~exist(Output_Folder,'dir')
    mkdir(Output_Folder)
end

% === 导入 Results 掩码 ===
Result_Folder = 'E:\多中心临床脑电\统计\0709statics\4groups_nbs_cov';
R = load(fullfile(Result_Folder, sprintf('%s_HX_HQ_ANOVA_mask_thresh_%s.mat', conspctrm, thresh)));

info_file = ['\info\' 'HX_age_sex.mat'];
load(info_file);
HX_info_data = cell(1, 4);
for g = 1:4
    group_info = SubjectInfo{1,g};
    nSubjects = numel(group_info);
    age_sex_matrix = zeros(nSubjects, 2);  % 第一列：Age，第二列：Sex
    for i = 1:nSubjects
        age_sex_matrix(i, 1) = group_info(i).Age;
        age_sex_matrix(i, 2) = group_info(i).Sex;
    end
    HX_info_data{g} = age_sex_matrix;
end

info_file = ['\info\' 'HQ_age_sex.mat'];
load(info_file);
HQ_info_data = cell(1, 4);
for g = 1:4
    group_info = SubjectInfo{1,g};
    nSubjects = numel(group_info);
    age_sex_matrix = zeros(nSubjects, 2);  % 第一列：Age，第二列：Sex
    for i = 1:nSubjects
        age_sex_matrix(i, 1) = group_info(i).Age;
        age_sex_matrix(i, 2) = group_info(i).Sex;
    end
    HQ_info_data{g} = age_sex_matrix;
end

HX_stats = struct();
HQ_stats = struct();


for b = 1:length(band_list)
    band = band_list{b};
    band_idx = b;
    fprintf('\n--- Band: %s ---\n', band);

    mask = logical(triu(R.Results.(band).HQ_mask, 1));
    if ~any(mask(:))
        fprintf('跳过频带 %s：无显著掩码\n', band);
        continue; 
    end
    edge_idx = find(mask);

    for p = 1:length(pair_list)
        g1 = pair_list{p}{1}; g2 = pair_list{p}{2};
        g1_idx = pair_idx_list{p}(1); g2_idx = pair_idx_list{p}(2); 
        fprintf('\n>> %s vs %s\n', g1, g2);
        is_disease_vs_HC = strcmp(g2, 'HC');

        % === HX 数据 ===
        D1 = load(fullfile(Input_Folder, conspctrm, ['HX_' g1 '.mat'] ));
        D2 = load(fullfile(Input_Folder, conspctrm, ['HX_' g2 '.mat']));
        X = squeeze(D1.Group_conn(band_idx, :, :, :));
        Y = squeeze(D2.Group_conn(band_idx, :, :, :));
        [nChan, ~, n1] = size(X); n2 = size(Y, 3);


       % === HX t检验筛选显著边 ===
       if is_disease_vs_HC
           fprintf('在HX上进行多次下采样t检验...\n');
           HX_all_mask = zeros(repeat_N, nChan, nChan);
           all_tvals = zeros(repeat_N, nChan, nChan);

           nSub = min(n1, n2);  % 对齐样本数

           for r = 1:repeat_N
               rng(r);
               idx1 = randperm(n1, nSub);
               idx2 = randperm(n2, nSub);
               info1 = HX_info_data{g1_idx}(idx1, : );
               info2 = HX_info_data{g2_idx}(idx2, : );

               % design matrix t-test :  first 2 column groups, the last 2 column covariate
               % (age/gender)
               N_total  = nSub*2;
               X_glm = zeros(N_total,4);
               X_glm(1:nSub,1) = 1;
               X_glm(1+nSub*1:nSub*2,2) = 1;

               % coveraite
               S = [info1; info2];
               X_glm(:,[3 4]) = S;   % covariate with age and gender

               contrast = [1 -1 0 0];
               test = 'ttest';
               % Total number of permutations to generate
               GLM  = [];
               GLM.perms = 1000;
               GLM.X = X_glm;
               GLM.contrast= contrast;
               GLM.test= test;

               for ei = 1:length(edge_idx)
                   [row, col] = ind2sub([nChan, nChan], edge_idx(ei));
                   v1 = squeeze(X(row, col, idx1));
                   v2 = squeeze(Y(row, col, idx2));

                   cfg.GLM            = GLM;
                   cfg.ivar           = 1;
                   cfg.design         = [];
                   cfg.design(1,:)    = [ones(1,nSub) ones(1,nSub)*2];
                   data = [v1; v2]';

                   test_stat = ft_statfun_GLMtest(cfg, data, cfg.design);  % 返回 (nPerms + 1) x 1
                   T_obs = test_stat.stat(1); % 原始 t 值
                   T_null = test_stat.stat(2:end); % null 分布
                   pval = (1 + sum(abs(T_null) >= abs(T_obs))) / (1 + numel(T_null));  % 双尾 p 值
                   if pval < 0.05
                       HX_all_mask(r, row, col) = 1;
                       all_tvals(r, row, col) = T_obs;
                   end

               end
           end
           mask_ratio = squeeze(mean(HX_all_mask, 1));
           HX_sig_mask = mask_ratio > thresh_ratio;
           stat_sum = squeeze(sum(all_tvals,1));
           mask_count = squeeze(sum(HX_all_mask, 1));
           HX_tval_map = stat_sum ./ (mask_count + eps);
           HX_tval_map(~HX_sig_mask) = 0; 
       else
           fprintf('在HX上使用全部数据做t检验...\n');
           HX_sig_mask = zeros(nChan);
           HX_tval_map = zeros(nChan);

           % design matrix t-test :  first 2 column groups, the last 2 column covariate
           % (age/gender)
           N_total  = n1 + n2;
           X_glm = zeros(N_total,4);
           X_glm(1:n1,1) = 1;
           X_glm(1+n1*1:N_total,2) = 1;

           % coveraite
           S = [HX_info_data{g1_idx}; HX_info_data{g2_idx}];
           X_glm(:,[3 4]) = S;   % covariate with age and gender

           contrast = [1 -1 0 0];
           test = 'ttest';
           % Total number of permutations to generate
           GLM  = [];
           GLM.perms = 1000;
           GLM.X = X_glm;
           GLM.contrast= contrast;
           GLM.test= test;

           for ei = 1:length(edge_idx)
               [row, col] = ind2sub([nChan, nChan], edge_idx(ei));
               v1 = squeeze(X(row, col, :));
               v2 = squeeze(Y(row, col, :));

               cfg.GLM            = GLM;
               cfg.ivar           = 1;
               cfg.design         = [];
               cfg.design(1,:)    = [ones(1,n1) ones(1,n2)*2];
               data = [v1; v2]';

               test_stat = ft_statfun_GLMtest(cfg, data, cfg.design);  % 返回 (nPerms + 1) x 1
               T_obs = test_stat.stat(1); % 原始 t 值
               T_null = test_stat.stat(2:end); % null 分布
               pval = (1 + sum(abs(T_null) >= abs(T_obs))) / (1 + numel(T_null));  % 双尾 p 值
               if pval < 0.05
                   HX_sig_mask(row, col) = 1;
                   HX_tval_map(row, col) = T_obs;
               end
           end

       end
       hx_edges = find(triu(HX_sig_mask, 1));
       fprintf('HX验证显著边数：%d\n', length(hx_edges));

  
        % === HQ 验证 ===
        G1 = load(fullfile(Input_Folder, conspctrm, ['HQ_' g1 '.mat']));
        G2 = load(fullfile(Input_Folder, conspctrm, ['HQ_' g2 '.mat']));
        Xq = squeeze(G1.Group_conn(band_idx, :, :, :));
        Yq = squeeze(G2.Group_conn(band_idx, :, :, :));
        nq1 = size(Xq, 3); nq2 = size(Yq, 3);

        if is_disease_vs_HC
            fprintf('在HQ上进行多次下采样t检验...\n');
            HQ_all_mask = zeros(repeat_N, nChan, nChan);
            all_tvals = zeros(repeat_N, nChan, nChan);
            nSub = min(nq1, nq2);
            for r = 1:repeat_N
                rng(r);
                idx1 = randperm(nq1, nSub);
                idx2 = randperm(nq2, nSub);
                info1 = HQ_info_data{g1_idx}(idx1, : );
                info2 = HQ_info_data{g2_idx}(idx2, : );

                % design matrix t-test :  first 2 column groups, the last 2 column covariate
                % (age/gender)
                N_total  = nSub*2;
                X_glm = zeros(N_total,4);
                X_glm(1:nSub,1) = 1;
                X_glm(1+nSub*1:nSub*2,2) = 1;

                % coveraite
                S = [info1; info2];
                X_glm(:,[3 4]) = S;   % covariate with age and gender

                contrast = [1 -1 0 0];
                test = 'ttest';
                % Total number of permutations to generate
                GLM  = [];
                GLM.perms = 1000;
                GLM.X = X_glm;
                GLM.contrast= contrast;
                GLM.test= test;


                for ei = 1:length(hx_edges)
                    [row, col] = ind2sub([nChan, nChan], hx_edges(ei));
                    v1 = squeeze(Xq(row, col, idx1));
                    v2 = squeeze(Yq(row, col, idx2));

                    cfg.GLM            = GLM;
                    cfg.ivar           = 1;
                    cfg.design         = [];
                    cfg.design(1,:)    = [ones(1,nSub) ones(1,nSub)*2];
                    data = [v1; v2]';

                    test_stat = ft_statfun_GLMtest(cfg, data, cfg.design);  % 返回 (nPerms + 1) x 1
                    T_obs = test_stat.stat(1); % 原始 t 值
                    T_null = test_stat.stat(2:end); % null 分布
                    pval = (1 + sum(abs(T_null) >= abs(T_obs))) / (1 + numel(T_null));  % 双尾 p 值
                    if pval < 0.05
                        HQ_all_mask(r, row, col) = 1;
                        all_tvals(r, row, col) = T_obs;
                    end
                end
            end
            mask_ratio = squeeze(mean(HQ_all_mask, 1));
            HQ_sig_mask = mask_ratio > thresh_ratio;
            stat_sum = squeeze(sum(all_tvals,1));
            mask_count = squeeze(sum(HQ_all_mask, 1));
            HQ_tval_map = stat_sum ./ (mask_count + eps);
            HQ_tval_map(~HQ_sig_mask) = 0;
        else
            fprintf('在HQ上使用全部数据做t检验...\n');
            HQ_sig_mask = zeros(nChan);
            HQ_tval_map = zeros(nChan);

            % design matrix t-test :  first 2 column groups, the last 2 column covariate
            % (age/gender)
            N_total  = nq1 + nq2;
            X_glm = zeros(N_total,4);
            X_glm(1:nq1,1) = 1;
            X_glm(1+nq1*1:N_total,2) = 1;

            % coveraite
            S = [HQ_info_data{g1_idx}; HQ_info_data{g2_idx}];
            X_glm(:,[3 4]) = S;   % covariate with age and gender

            contrast = [1 -1 0 0];
            test = 'ttest';
            % Total number of permutations to generate
            GLM  = [];
            GLM.perms = 1000;
            GLM.X = X_glm;
            GLM.contrast= contrast;
            GLM.test= test;

            for ei = 1:length(hx_edges)
                [row, col] = ind2sub([nChan, nChan], hx_edges(ei));
                v1 = squeeze(Xq(row, col, :));
                v2 = squeeze(Yq(row, col, :));

                cfg.GLM            = GLM;
                cfg.ivar           = 1;
                cfg.design         = [];
                cfg.design(1,:)    = [ones(1,nq1) ones(1,nq2)*2];
                data = [v1; v2]';

                test_stat = ft_statfun_GLMtest(cfg, data, cfg.design);  % 返回 (nPerms + 1) x 1
                T_obs = test_stat.stat(1); % 原始 t 值
                T_null = test_stat.stat(2:end); % null 分布
                pval = (1 + sum(abs(T_null) >= abs(T_obs))) / (1 + numel(T_null));  % 双尾 p 值
                if pval < 0.05
                    HQ_sig_mask(row, col) = 1;
                    HQ_tval_map(row, col) = T_obs;
                end
            end

        end
        hq_edges = find(triu(HQ_sig_mask, 1));
        fprintf('HQ验证显著边数：%d\n', length(hq_edges));

        % 保存到结构体中
        HX_stats.(band).(sprintf('%s_vs_%s_mask', g1, g2)) = HX_sig_mask;
        HX_stats.(band).(sprintf('%s_vs_%s_tval', g1, g2)) = HX_tval_map;
        HQ_stats.(band).(sprintf('%s_vs_%s_mask', g1, g2)) = HQ_sig_mask;
        HQ_stats.(band).(sprintf('%s_vs_%s_tval', g1, g2)) = HQ_tval_map;

    end
end
save(fullfile(Output_Folder, sprintf('%s_HXTStatsHQTStats_thresh_%s.mat', conspctrm, thresh)), 'HX_stats','HQ_stats');

fprintf('\n全部频带和组别的分析完成 ✅\n');
