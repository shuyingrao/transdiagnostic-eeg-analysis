function Calculate_PSD_Interpolation(cfgs, subID)

InputFolder   =  cfgs.InputFolder;
OutputFolder  = cfgs.OutputFolder;
CodeFolder    = cfgs.CodeFolder;
GroupName     = cfgs.GroupName;

OutputFolder_Sub = fullfile(OutputFolder, GroupName);  
if ~exist(OutputFolder_Sub, 'dir')
    mkdir(OutputFolder_Sub)
end                                                     

% Loading data
load(fullfile(InputFolder, strcat(subID,'_EEG_ECClean')))

% loading layout 
load(fullfile(CodeFolder,'EEGLayout.mat'))

% Transfering to FT style 
FT_EEG = eeglab2fieldtrip(EEG_ECClean, 'raw');

% divide data into 1s segment with 0.5s overlaping
cfg = [];
cfg.length  = 1;
cfg.overlap = 0.5;
FTEEG = ft_redefinetrial(cfg, FT_EEG);   

%%  Calculate Power Spectrum  
cfg = [];
cfg.method     = 'mtmfft';
cfg.pad        = 'nextpow2';
cfg.tapsmofrq  = 2;
cfg.taper      = 'hanning';
cfg.foilim     = [1 40];
cfg.output     = 'pow';
FFT_Power      = ft_freqanalysis(cfg, FTEEG);

% Define common frequency axis
target_freqs = 1:1:40;

% Interpolate PSD to unified frequency grid
interp_psd = zeros(size(FFT_Power.powspctrm,1), length(target_freqs));
for ch = 1:size(FFT_Power.powspctrm, 1)
    interp_psd(ch,:) = interp1(FFT_Power.freq, FFT_Power.powspctrm(ch,:), target_freqs, 'linear', 'extrap');
end

% Replace original PSD with interpolated one
FFT_Power.freq = target_freqs;
FFT_Power.powspctrm = interp_psd;

% Power Normalization (after interpolation)
FFT_Power.normpowspctrm = FFT_Power.powspctrm ./ sum(FFT_Power.powspctrm, 2);

% Log10 power (after interpolation)
FFT_Power.logpowspctrm = log10(FFT_Power.powspctrm);

%% Calculate band-specific averages
freq_bands = {
    'delta', 1, 4;
    'theta', 4, 8;
    'alpha', 8, 13;
    'beta', 13, 30;
    'gamma', 30, 40
};

band_power = struct();
band_power.dimord = 'chan';

for i = 1:size(freq_bands,1)
    band_name = freq_bands{i,1};
    f_low = freq_bands{i,2};
    f_high = freq_bands{i,3};
    
    band_idx = FFT_Power.freq >= f_low & FFT_Power.freq <= f_high;
    
    band_power.(band_name).pow     = mean(FFT_Power.powspctrm(:, band_idx), 2);
    band_power.(band_name).normpow = mean(FFT_Power.normpowspctrm(:, band_idx), 2);
    band_power.(band_name).logpow  = mean(FFT_Power.logpowspctrm(:, band_idx), 2);
end

FFT_Power.bands = freq_bands;
FFT_Power.band_power = band_power;

% Save results
save_name = sprintf('%s_PSD', subID);
save(fullfile(OutputFolder_Sub, save_name), 'FFT_Power');

fprintf('\nAll frequency bands processed and saved for subject %s\n', subID);
fprintf('Results saved to: %s\n', fullfile(OutputFolder_Sub, save_name));

end
