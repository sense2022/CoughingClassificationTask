import csv
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import librosa.display
import librosa
import os

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

def spec_visualize(input_data):

    data = input_data[:,:1000]
    label = input_data[:,1000]
    
    kk = 0

    for i, c in enumerate(data):
        
        if label[i] != 4:
            continue
        kk += 1
        if kk < 7:
            continue
        
        # normalize
        p = c
        c -= c.min()
        c = ((c / (c.max() + 1e-6)) - 0.5)

        wav = c.cpu().numpy()
        X = librosa.stft(wav, n_fft=128)
        Xdb = librosa.amplitude_to_db(abs(X))
        out = torch.from_numpy(spec_to_image(Xdb)).cuda()
        
        print(label[i])
        
        # Set signal X and Y values
        lw = 4
        x = np.arange(1000)
        x= (x/31)
        
        # Set format
        y_val = c.cpu().numpy()
        ylabel_format = '{:,.1f}'
        xlabel_format = '{:,.0f}'

        # Set plot size
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, hspace=0)
        axs = gs.subplots(sharex=True)

        # Plot signals
        axs[0].plot(
            x,
            y_val,
            color="black",
            lw=lw,
        )
        
        ytick_label = axs[0].get_yticks().tolist()
        xtick_label = (axs[0].get_xticks())[1:].tolist()

        # Plot Spectrogram Image
        spec = spec_to_image(Xdb)
        librosa.display.specshow(spec, sr=30, y_axis='hz',cmap='viridis')

        # Set up labels, titles and ticks
        fig.suptitle("Walking", size=45, fontweight="bold")
        plt.xlabel("Times (s)", size=45, fontweight="bold")
        axs[0].set_ylabel("$\Delta$R/R$_0$ (%)", size=45, fontweight="bold")
        axs[1].set_ylabel("Freq. (Hz)", size=45, fontweight="bold")

        axs[0].yaxis.set_major_locator(mticker.FixedLocator(ytick_label))
        axs[0].set_yticklabels([ylabel_format.format(x) for x in ytick_label], fontsize=40, fontweight="bold")
        plt.yticks(fontsize=40, fontweight="bold")

        axs[1].xaxis.set_major_locator(mticker.FixedLocator(xtick_label))
        axs[1].set_xticklabels([xlabel_format.format(x) for x in xtick_label], fontsize=40, fontweight="bold")

        axs[0].tick_params(width=lw)
        axs[0].spines['top'].set_linewidth(lw)
        axs[0].spines['left'].set_linewidth(lw)
        axs[0].spines['right'].set_linewidth(lw)

        axs[1].spines['top'].set_linewidth(lw)
        axs[1].spines['left'].set_linewidth(lw)
        axs[1].spines['bottom'].set_linewidth(lw)
        axs[1].spines['right'].set_linewidth(lw)

        plt.rcParams["font.weight"] = "bold"
        plt.tick_params(width=lw)

        # Save plot figures
        plt.savefig('Walking.png', bbox_inches = "tight")

        raise SystemExit()

if __name__ == '__main__':

    root_path = 'volunteers'

    print('loading data...')
    A = torch.load(os.path.join(root_path, 'data_A.pth'))
    B = torch.load(os.path.join(root_path, 'data_B.pth'))
    C = torch.load(os.path.join(root_path, 'data_C.pth'))
    D = torch.load(os.path.join(root_path, 'data_D.pth'))

    data = torch.cat((A, B, C, D), 0)
    print('Data Length:', len(data))
    
    # Spectrogram Visualization
    spec_visualize(data)