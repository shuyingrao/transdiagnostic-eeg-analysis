clear; clc;

% === 参数设置 ===
bands = {'delta', 'theta', 'alpha', 'beta', 'gamma'};
band_names = {'Delta (1-4Hz)', 'Theta (4-8Hz)', 'Alpha (8-12Hz)', 'Beta (13-30Hz)', 'Gamma (30-40Hz)'};
thresh = '5.5'; % [2.6, 3.1, 3.8, 4.3, 5.5];
conspctrm = 'powcorr_ortho';  % 功能连接类型
conspctrmName = strrep(conspctrm, '_', ' ');
Result_Folder = '\statics\4groups_nbs_cov';
Input_Folder = "\features\group_conn_matrix\";
site = 'HQ';
groups = {'HC', 'BD', 'MDD', 'SCZ'};
N_repeat = 100;
p_thresh = 0.05;
thresh_count = 95; % HX NBS
ratio_thresh = 0.95; % HQ ANOVA

% === 加载 layout 等可视化数据（若后续需要）===
load('EEGLayout.mat');

Output_Folder = "\statics\4groups_nbs_cov";
if ~exist(Output_Folder,'dir')
    mkdir(Output_Folder)
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

Results = struct(); 


for band_idx = 5:5 % 1:length(bands)
    band = bands{band_idx};
    fprintf('--- Band: %s ---\n', band);

    % === 加载显著边的结果 ===
    % stat_file = fullfile(Result_Folder, ...
    %     sprintf('%s_NBS_result_thresh_%s_%s.mat', conspctrm, thresh, band));
    % load(stat_file, 'con_mat_results');  % 加载 mask

    stat_file1 = fullfile(Result_Folder, ...
        sprintf('%s_NBS_result_thresh_%s_%s_1.mat', conspctrm, thresh, band));
    data1 = load(stat_file1, 'con_mat_results');
    con_mat1 = data1.con_mat_results;
    stat_file2 = fullfile(Result_Folder, ...
        sprintf('%s_NBS_result_thresh_%s_%s_2.mat', conspctrm, thresh, band));
    data2 = load(stat_file2, 'con_mat_results');
    con_mat2 = data2.con_mat_results;
    con_mat_results = cat(1, con_mat1, con_mat2);

    mask = squeeze(sum(con_mat_results));  % (nChan x nChan)，逻辑矩阵，显著边为1
    sig_mask = mask >= thresh_count;
    [i_chan, j_chan] = find(triu(sig_mask, 1));
    nEdges = length(i_chan);

    % === 加载每组功能连接矩阵 ===
    group_data = cell(1, 4);
    group_sizes = zeros(1, 4);
    for g = 1:4
        data = load(fullfile(Input_Folder, conspctrm, [site '_' groups{g} '.mat']));
        conn_data = squeeze(data.Group_conn(band_idx, :, :, :));  % (chan x chan x subj)
        group_data{g} = conn_data;
        group_sizes(g) = size(conn_data, 3);
    end

    min_n = min(group_sizes);  % 统一下采样数量
    nChan = size(group_data{1}, 1);
    % === 初始化结果统计矩阵 ===
    
    test_stat_results = zeros(N_repeat, nChan, nChan);
    p_count = zeros(nEdges, 1);  % 每条边显著次数

    % === 重复ANOVA分析 ===
    for r = 1:N_repeat
        edge_pvals = zeros(nEdges, 1);
        rng(r + 42);
        % 每组下采样
        subsampled_data = [];
        group_vector = [];
        subsampled_info = [];
        for g = 1:4
            idx = randperm(group_sizes(g), min_n);
            conn = group_data{g}(:, :, idx);  % chan x chan x min_n
            conn_edge = zeros(nEdges, min_n);
            info = info_data{g}(idx,:);
            for e = 1:nEdges
                conn_edge(e, :) = squeeze(conn(i_chan(e), j_chan(e), :));
            end
            subsampled_data = [subsampled_data, conn_edge];  % e x (min_n * g)
            group_vector = [group_vector; repmat(g, min_n, 1)];
            subsampled_info = [subsampled_info; info];
        end

        % 对每条边做ANOVA
        for e = 1:nEdges
            data_vector = subsampled_data(e, :)';

            [p, tbl, stats] = anovan(data_vector, {group_vector, subsampled_info(:,1), subsampled_info(:,2)}, ...
                'model', 'linear', ...
                'varnames', {'Group','Age','Sex'},'display','off');
            if p(1) < 0.05
                p_count(e) = p_count(e) + 1;
            end
            ii = i_chan(e); jj = j_chan(e);
            test_stat_results(r, ii, jj) = tbl{2,6};
            test_stat_results(r, jj, ii) = tbl{2,6};

        end
    end

    % === 保留显著重复次数超过阈值的边 ===
    stable_thresh = round(N_repeat * ratio_thresh);
    stable_mask = zeros(size(sig_mask));
    for k = 1:nEdges
        if p_count(k) >= stable_thresh
            stable_mask(i_chan(k), j_chan(k)) = 1;
        end
    end

    Results.(band).HX_mask = sig_mask;
    Results.(band).HQ_mask = stable_mask;
    Results.(band).HQ_stat = test_stat_results;

end

save(fullfile(Output_Folder, ...
    sprintf('%s_HX_HQ_ANOVA_mask_thresh_%s.mat', conspctrm, thresh)), ...
    'Results');

%%
h = figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.7], 'Color', 'w');
set(h, 'DefaultAxesFontSize', 12);

for b = 1:length(bands)
    band = bands{b};
    HX_mask = Results.(band).HX_mask;
    HQ_mask = Results.(band).HQ_mask;


    subplot(2, 5, b);
    axis equal off
    hold on;
    pos = layout.pos;

    if isempty(nonzeros(HX_mask))
        node_strength = zeros(size(pos, 1), 1);
        scatter(pos(:,1), pos(:,2), 50, [0.5 0.5 0.5], 'filled');
        ft_plot_topo(pos(:,1), pos(:,2), node_strength, ...
            'mask', layout.mask, 'outline', layout.outline, ...
            'interplim', 'mask', 'style', 'fill', 'gridscale', 150);
        if length(layout.label) < 40
            text(pos(:,1), pos(:,2), layout.label, 'FontSize', 10, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
        title( ['Huaxi ' band_names{b} ' NBS' ]);
        axis off;

        subplot(2, 5, b + 5);
        axis equal off
        hold on;
        node_strength = zeros(size(pos, 1), 1);
        scatter(pos(:,1), pos(:,2), 50, [0.5 0.5 0.5], 'filled');
        ft_plot_topo(pos(:,1), pos(:,2), node_strength, ...
            'mask', layout.mask, 'outline', layout.outline, ...
            'interplim', 'mask', 'style', 'fill', 'gridscale', 150);
        if length(layout.label) < 40
            text(pos(:,1), pos(:,2), layout.label, 'FontSize', 10, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
        title( ['Hangqi ' band_names{b} ' ANOVA']);
        axis off;
        continue;
    end

    axis equal off
    node_strength = zeros(size(pos, 1), 1);
    scatter(pos(:,1), pos(:,2), 50, [0.5 0.5 0.5], 'filled');
    ft_plot_topo(pos(:,1), pos(:,2), node_strength, ...
        'mask', layout.mask, 'outline', layout.outline, ...
        'interplim', 'mask', 'style', 'fill', 'gridscale', 150);
    [r1, c1] = find(triu(HX_mask, 1) > 0);
    for k = 1:length(r1)
        val = HX_mask(r1(k), c1(k));
        line([pos(r1(k),1), pos(c1(k),1)], [pos(r1(k),2), pos(c1(k),2)], ...
             'Color', 'black', 'LineWidth', 2);
    end

    if length(layout.label) < 40
        text(pos(:,1), pos(:,2), layout.label, 'FontSize', 10, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    title( ['Huaxi ' band_names{b} ' NBS' ]);
    axis off;

    % --- 2. 验证方向图 ---
    subplot(2, 5, b + 5);
    axis equal off
    hold on;
    node_strength = zeros(size(pos, 1), 1);
    scatter(pos(:,1), pos(:,2), 50, [0.5 0.5 0.5], 'filled');
    ft_plot_topo(pos(:,1), pos(:,2), node_strength, ...
        'mask', layout.mask, 'outline', layout.outline, ...
        'interplim', 'mask', 'style', 'fill', 'gridscale', 150);
    [r1, c1] = find(triu(HQ_mask, 1) > 0);
    for k = 1:length(r1)
        line([pos(r1(k),1), pos(c1(k),1)], [pos(r1(k),2), pos(c1(k),2)], ...
             'Color', 'black', 'LineWidth', 2);  % 黑线
    end
    if length(layout.label) < 40
        text(pos(:,1), pos(:,2), layout.label, 'FontSize', 10, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    title( ['Hangqi ' band_names{b} ' ANOVA']);
    axis off;
end

sgtitle(sprintf('%s Functional Connectivity: thresh %s ', conspctrmName, thresh), 'FontSize', 16, 'FontWeight', 'bold');
