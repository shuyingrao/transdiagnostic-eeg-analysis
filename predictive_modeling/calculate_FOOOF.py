"""
Calculate_FOOOF.py
对Calculate_PSD.m输出的*_PSD.mat文件进行FOOOF分析
"""

import math
import os
import numpy as np
import scipy.io as sio
from fooof import FOOOFGroup
from fooof.analysis import get_band_peak_fg


def load_psd_mat(mat_path):
    """
    载入由MATLAB Calculate_PSD保存的PSD文件
    
    返回:
        freqs    : (n_freqs,)        频率轴
        psd      : (n_chan, n_freqs) 线性功率谱
        labels   : list of str       通道名
    """
    # FieldTrip结构嵌套较深,用squeeze_me和struct_as_record方便访问
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    FFT_Power = mat['FFT_Power']
    
    freqs  = np.asarray(FFT_Power.freq, dtype=float)
    psd    = np.asarray(FFT_Power.powspctrm, dtype=float)  # 线性功率,FOOOF需要的格式
    labels = [str(x) for x in np.atleast_1d(FFT_Power.label)]
    
    # 保证psd是 (n_chan, n_freqs)
    if psd.ndim == 1:
        psd = psd[np.newaxis, :]
    if psd.shape[1] != freqs.size and psd.shape[0] == freqs.size:
        psd = psd.T
    
    return freqs, psd, labels


def calculate_fooof(cfgs, subID):
    """
    对单被试运行FOOOF分析
    
    cfgs: dict, 含
        InputFolder  : PSD .mat文件所在主目录
        OutputFolder : 输出主目录
        GroupName    : 子文件夹名(被试组)
    subID: str, 被试ID
    """
    # ---------- 路径 ----------
    input_folder  = cfgs['InputFolder']
    output_folder = cfgs['OutputFolder']
    group_name    = cfgs['GroupName']
    
    in_file  = os.path.join(input_folder,  group_name, f'{subID}_PSD.mat')
    out_dir  = os.path.join(output_folder, group_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # ---------- 载入PSD ----------
    freqs, psd, labels = load_psd_mat(in_file)
    n_chan = psd.shape[0]
    print(f'[{subID}] Loaded PSD: {n_chan} channels x {len(freqs)} freqs')
    
    # ---------- FOOOF设置 ----------
    freq_range = [3, 35]                  # 拟合范围
    settings = dict(
        peak_width_limits = [1, 8],       # 峰宽限制(Hz)
        max_n_peaks       = 6,            # 最多峰数
        min_peak_height   = 0.1,          # 峰高阈值(log10 power)
        peak_threshold    = 2.0,          # 相对噪声SD的倍数
        aperiodic_mode    = 'fixed',      # 'fixed' / 'knee'
        verbose           = False
    )
    
    # ---------- 并行拟合所有通道 ----------
    fg = FOOOFGroup(**settings)
    fg.fit(freqs, psd, freq_range)
    
    # ---------- 提取aperiodic参数 ----------
    # fixed mode: [offset, exponent]; knee mode: [offset, knee, exponent]
    ap_params = fg.get_params('aperiodic_params')   # (n_chan, 2 or 3)
    r_squared = fg.get_params('r_squared')
    fit_error = fg.get_params('error')
    
    aperiodic = {
        'offset'   : ap_params[:, 0],
        'exponent' : ap_params[:, -1],          # 不论几列,exponent总在最后
        'r_squared': r_squared,
        'error'    : fit_error,
    }
    if settings['aperiodic_mode'] == 'knee':
        aperiodic['knee'] = ap_params[:, 1]
    else:
        aperiodic['knee'] = np.full(n_chan, np.nan)
    
    # ---------- 提取频段内peak (CF, PW, BW) ----------
    bands = {
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
    }
    
    band_peaks = {}
    for bname, brange in bands.items():
        # get_band_peak_fg: 每通道在该频段内取功率最高的peak,无则返回NaN
        bp = get_band_peak_fg(fg, brange)        # (n_chan, 3): [CF, PW, BW]
        band_peaks[bname] = {
            'CF':       bp[:, 0],
            'PW':       bp[:, 1],
            'BW':       bp[:, 2],
            'detected': ~np.isnan(bp[:, 0]),
        }
    
    # ---------- 提取每通道全部peak ----------
    # 用cell-like结构,每通道一个 (n_peaks, 3) 数组
    all_peaks = np.empty(n_chan, dtype=object)
    for i in range(n_chan):
        fm = fg.get_fooof(ind=i, regenerate=False)
        all_peaks[i] = fm.peak_params_.copy() if fm.peak_params_.size else np.empty((0, 3))
    
    # ---------- 重建谱(可选,用于可视化/调试) ----------
    # FOOOFGroup内部存的是log10谱
    fit_freqs = fg.freqs                    # 限制在freq_range内
    fooofed_spectrum = np.array([fg.get_fooof(i).fooofed_spectrum_ for i in range(n_chan)])
    ap_fit           = np.array([fg.get_fooof(i)._ap_fit           for i in range(n_chan)])
    
    # ---------- 组装保存结构 ----------
    FOOOF_results = {
        'subID'            : subID,
        'label'            : np.array(labels, dtype=object),
        'freqs'            : freqs,
        'fit_freqs'        : fit_freqs,
        'freq_range'       : np.array(freq_range),
        'aperiodic'        : aperiodic,
        'band_peaks'       : band_peaks,
        'all_peaks'        : all_peaks,
        'fooofed_spectrum' : fooofed_spectrum,   # log10空间
        'ap_fit'           : ap_fit,             # log10空间
        'settings'         : settings,
    }
    
    # ---------- 保存为.mat ----------
    save_path = os.path.join(out_dir, f'{subID}_FOOOF.mat')
    sio.savemat(save_path, {'FOOOF_results': FOOOF_results}, do_compression=True)
    
    # ---------- 同时保存FOOOF原生格式(便于Python端再分析) ----------
    fg.save(file_name=f'{subID}_FOOOFGroup',
            file_path=out_dir,
            save_results=True, save_settings=True, save_data=True)
    
    # ---------- 打印汇总 ----------
    print(f'\n===== FOOOF done for {subID} =====')
    print(f'Mean R^2     : {np.nanmean(r_squared):.3f}')
    print(f'Mean exponent: {np.nanmean(aperiodic["exponent"]):.3f}')
    print(f'Mean offset  : {np.nanmean(aperiodic["offset"]):.3f}')
    for bname in bands:
        n_det = band_peaks[bname]['detected'].sum()
        print(f'{bname:>5} peak detected in {n_det}/{n_chan} channels')
    print(f'Saved: {save_path}\n')
    
    return FOOOF_results, fg


# ============================================================
# 批处理入口
# ============================================================
def batch_run(cfgs, subID_list):
    """对一组被试批量运行"""
    failed = []
    for sid in subID_list:
        try:
            calculate_fooof(cfgs, sid)
        except Exception as e:
            print(f'[ERROR] {sid}: {e}')
            failed.append((sid, str(e)))
    if failed:
        print('\n===== Failed subjects =====')
        for sid, msg in failed:
            print(f'{sid}: {msg}')
    return failed


# ============================================================
# 单被试可视化检查
# ============================================================
def plot_fooof_check(subID, cfgs, save_fig=True):
    """
    画出所有通道的FOOOF拟合结果,4x4布局,标题为通道名
    """
    import matplotlib.pyplot as plt
    
    out_dir = os.path.join(cfgs['OutputFolder'], cfgs['GroupName'])
    
    # 载入FOOOFGroup对象
    fg = FOOOFGroup()
    fg.load(file_name=f'{subID}_FOOOFGroup', file_path=out_dir)
    
    # 载入通道名(从_FOOOF.mat中读)
    mat_path = os.path.join(out_dir, f'{subID}_FOOOF.mat')
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    labels = [str(x) for x in np.atleast_1d(m['FOOOF_results'].label)]
    
    n_chan = len(fg)
    n_rows = 4
    n_cols = math.ceil(n_chan / n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4* n_rows))
    axes = axes.flatten()
    
    for ch in range(n_rows * n_cols):
        ax = axes[ch]
        if ch < n_chan:
            fm = fg.get_fooof(ind=ch, regenerate=True)
            fm.plot(ax=ax, plot_peaks='shade', add_legend=False)
            
            # 标题:通道名 + 关键指标
            r2  = fm.r_squared_
            exp = fm.aperiodic_params_[-1]
            ax.set_title(f'{labels[ch]}  |  R²={r2:.2f}  exp={exp:.2f}',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)', fontsize=9)
            ax.set_ylabel('log10(Power)',   fontsize=9)
            ax.tick_params(labelsize=8)
        else:
            # 不足16通道时关闭多余子图
            ax.axis('off')
    
    fig.suptitle(f'Example: Subject {subID} — FOOOF fits ', # ({cfgs["GroupName"]})
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_fig:
        fig_path = os.path.join(out_dir, f'{subID}_FOOOF_check.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved: {fig_path}')
    plt.show()
    plt.close(fig)


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    cfgs = {
        'InputFolder' : r'/PSD',
        'OutputFolder': r'/FOOOF',
        'GroupName'   : 'HX_HC',
    }
    
#     file_list = os.listdir(os.path.join(cfgs['InputFolder'], cfgs['GroupName']))
#     sub_list = sorted([f.split('_')[0] for f in file_list if f.endswith('_PSD.mat')])
#     print(f'Found {len(sub_list)} subjects')

#     # 单被试测试
#     # calculate_fooof(cfgs, 'v0024')
    plot_fooof_check('vND005', cfgs, save_fig=False)
    
#     # 批量运行
#     subID_list = [file_list[i][:-len('_PSD.mat')] for i in range( len(file_list))]
#     # subID_list = sub_list
#     batch_run(cfgs, subID_list)

