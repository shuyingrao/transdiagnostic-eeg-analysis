function Calculate_PSD(cfgs, subID)

InputFolder   =  cfgs.InputFolder;
OutputFolder  = cfgs.OutputFolder;
CodeFolder    = cfgs.CodeFolder;
GroupName    = cfgs.GroupName;


OutputFolder_Sub = fullfile(OutputFolder,GroupName);  
   if ~exist(OutputFolder_Sub,'dir')
     mkdir(OutputFolder_Sub)
   end                                                     

% Loading data
load(fullfile(InputFolder,strcat(subID,'_EEG_ECClean')))

% loading layout 
load(fullfile(CodeFolder,'EEGLayout.mat'))
% Transfering to FT style 
FT_EEG = eeglab2fieldtrip(EEG_ECClean,'raw');


% divide data into 1s segment with 0.5s overlaping
cfg = [];
cfg.length  = 1;
cfg.overlap = 0.5;
FTEEG = ft_redefinetrial(cfg,FT_EEG);   

%%  Calculate Power   
cfg               = [];
cfg.method        = 'mtmfft';
cfg.pad           = 'nextpow2';
cfg.tapsmofrq     = 2;
cfg.taper         = 'hanning';
cfg.foilim        = [1 40];
% cfg.foi           = 1:0.5:40;
cfg.output        = 'pow';
% cfg.keeptrials = 'yes';  % 保留单分段数据
FFT_Power         = ft_freqanalysis(cfg, FTEEG);

% Power Normalization
FFT_Power.normpowspctrm =FFT_Power.powspctrm./(repmat(sum(FFT_Power.powspctrm,2),[1 size(FFT_Power.powspctrm,2)]));
% Log10 power 
FFT_Power.logpowspctrm  = log10(FFT_Power.powspctrm);

%% Calculate band-specific averages
% Define frequency bands
freq_bands = {
    'delta', 1, 4;
    'theta', 4, 8;
    'alpha', 8, 13;
    'beta', 13, 30;
    'gamma', 30, 40
};

% Initialize structure for band-specific data
band_power = struct();
band_power.dimord = 'chan';

for i = 1:size(freq_bands,1)
    band_name = freq_bands{i,1};
    f_low = freq_bands{i,2};
    f_high = freq_bands{i,3};
    
    % Find frequencies within the band
    band_idx = FFT_Power.freq >= f_low & FFT_Power.freq <= f_high;
    
    % Calculate average for each type of power
    band_power.(band_name).pow = mean(FFT_Power.powspctrm(:,band_idx), 2);
    band_power.(band_name).normpow = mean(FFT_Power.normpowspctrm(:,band_idx), 2);
    band_power.(band_name).logpow = mean(FFT_Power.logpowspctrm(:,band_idx), 2);
end

% Add frequency band information to the output
FFT_Power.bands = freq_bands;
FFT_Power.band_power = band_power;

% 保存所有频带结果
save_name = sprintf('%s_%s', subID, 'PSD');
save(fullfile(OutputFolder_Sub, save_name), 'FFT_Power');

fprintf('\nAll frequency bands processed and saved for subject %s\n', subID);
fprintf('Results saved to: %s\n', fullfile(OutputFolder_Sub, save_name));
end