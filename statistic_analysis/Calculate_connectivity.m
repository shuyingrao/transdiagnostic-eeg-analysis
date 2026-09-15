function Calculate_connectivity(cfgs, subID, con)
% con = 'icoh', 'powcorr_ortho', 'plv'
% 计算五个频带的功能连接，整合到一个结构体中保存

InputFolder   = cfgs.InputFolder;
OutputFolder  = cfgs.OutputFolder;
CodeFolder    = cfgs.CodeFolder;
GroupName     = cfgs.GroupName;
freq_res      = cfgs.freq_res; % 频率分辨率 (Hz)

OutputFolder_Sub = fullfile(OutputFolder, GroupName);
if ~exist(OutputFolder_Sub, 'dir')
    mkdir(OutputFolder_Sub)
end

% 加载数据
load(fullfile(InputFolder, strcat(subID, '_EEG_ECClean')))

% 转换为FieldTrip格式
FT_EEG = eeglab2fieldtrip(EEG_ECClean, 'raw');

% 分段处理 (1秒分段，50%重叠)
cfg = [];
cfg.length  = 1;
cfg.overlap = 0.5;
FTEEG = ft_redefinetrial(cfg, FT_EEG);

% 定义频带
freq_bands = {
    'delta', 1, 4;
    'theta', 4, 8;
    'alpha', 8, 13;
    'beta', 13, 30;
    'gamma', 30, 40
    };

% 初始化输出结构体
Conn = struct();
Conn.freq_resolution = freq_res;
Conn.connectivity_method = con;
Conn.freq_bands = freq_bands;

% 处理每个频带
for i = 1:size(freq_bands, 1)
    band_name = freq_bands{i,1};
    f_low = freq_bands{i,2};
    f_high = freq_bands{i,3};

    fprintf('\nProcessing %s band (%.1f-%.1f Hz)...\n', band_name, f_low, f_high);

    % 设置感兴趣的频率
    fois = f_low:freq_res:f_high;

    % 频域分析配置
    cfg = [];
    cfg.method     = 'mtmfft';
    cfg.pad        = 'nextpow2';
    cfg.tapsmofrq  = freq_res * 2; % 频率平滑
    cfg.taper      = 'dpss';
    cfg.foi        = fois;
    cfg.output     = 'fourier';

    % 计算频域表示
    FFT_band = ft_freqanalysis(cfg, FTEEG);

    % 连接性分析配置
    cfg = [];
    if strcmp(con, 'icoh')
        cfg.method = 'coh';
        cfg.complex = 'absimag';
    else
        cfg.method = con;
    end

    % 计算连接性
    conn = ft_connectivityanalysis(cfg, FFT_band);

    % 获取连接性度量名称
    conname = fieldnames(conn);
    conname = conname{4};

    % 保存到结构体
    Conn.(band_name).freq = FFT_band.freq;
    Conn.(band_name).(conname) = conn.(conname);
    Conn.(band_name).dimord = conn.dimord;

    % 计算并保存频带平均连接矩阵
    if ndims(conn.(conname)) == 3
        Conn.(band_name).([conname '_avg']) = squeeze(mean(conn.(conname), 3));
    else
        Conn.(band_name).([conname '_avg']) = conn.(conname);
    end
end

Conn.label = conn.label;
Conn.elec = conn.elec;

%% Calculate full spectrum connectivity for reference
cfg = [];
cfg.method     = 'mtmfft';
cfg.pad        = 'nextpow2';
cfg.tapsmofrq  = freq_res * 2; % 频率平滑
cfg.taper      = 'dpss';
cfg.foilim        = [0 40];
cfg.output     = 'fourier';
% cfg.keeptrials = 'yes';
FFT_Power = ft_freqanalysis(cfg, FTEEG);
if strcmp(con, 'icoh')
    cfg.method = 'coh';
    cfg.complex = 'absimag';
else
    cfg.method = con;
end
full_conn = ft_connectivityanalysis(cfg, FFT_Power);
conname = fieldnames(full_conn);
conname = conname{4};
Conn.(conname) = full_conn;

% 保存所有频带结果
save_name = sprintf('%s_%s', subID, con);
save(fullfile(OutputFolder_Sub, save_name), 'Conn');

fprintf('\nAll frequency bands processed and saved for subject %s\n', subID);
fprintf('Results saved to: %s\n', fullfile(OutputFolder_Sub, save_name));
end