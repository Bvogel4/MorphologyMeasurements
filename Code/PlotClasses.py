import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typing import List, Dict, Optional, Callable, Tuple


class GeneralPlotter:
    def __init__(self, data: Dict[str, np.ndarray], masks: Dict[str, np.ndarray],
                 labels: Dict[str, str], colors: Dict[str, str]):
        self.data = data
        self.masks = masks
        self.labels = labels
        self.colors = colors

    def lighten_color(self, color: str, amount: float = 0.5) -> str:
        """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        """
        import colorsys
        try:
            c = plt.matplotlib.colors.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*plt.matplotlib.colors.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

    def plot_main(self, ax: plt.Axes, x_key: str, y_key: str, show_scatter: bool = True,
                  show_lines: bool = False, size: int = 30):
        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            dm_color = self.lighten_color(color, 1.5)

            if show_scatter:
                ax.scatter(self.data[f'{x_key}_d'][mask], self.data[f'{y_key}_d'][mask],
                           c=color, marker='o', s=size, alpha=1)
                ax.scatter(self.data[f'{x_key}_s'][mask], self.data[f'{y_key}_s'][mask],
                           c=color, marker='*', s=size, alpha=1)

            if show_lines or not show_scatter:
                for i in np.where(mask)[0]:
                    if show_scatter:
                        ax.plot([self.data[f'{x_key}_s'][i], self.data[f'{x_key}_d'][i]],
                                [self.data[f'{y_key}_s'][i], self.data[f'{y_key}_d'][i]],
                                zorder=0, lw=0.5, c='k', alpha=0.7)
                    else:
                        ax.plot([self.data[f'{x_key}_s'][i], self.data[f'{x_key}_d'][i]],
                                [self.data[f'{y_key}_s'][i], self.data[f'{y_key}_d'][i]],
                                zorder=0, lw=1, c=color, alpha=0.7)

    def plot_histograms(self, ax_histx: plt.Axes, ax_histy: plt.Axes,
                        ax_histx1: plt.Axes, ax_histy1: plt.Axes,
                        x_key: str, y_key: str):
        bins = np.linspace(0, 1, 16)
        lw = 1.5

        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]

            ax_histx.hist(self.data[f'{x_key}_s'][mask], bins=bins, density=True, histtype='step', color=color, lw=lw)
            ax_histy.hist(self.data[f'{y_key}_s'][mask], bins=bins, density=True, histtype='step',
                          orientation='horizontal', color=color, lw=lw)
            ax_histx1.hist(self.data[f'{x_key}_d'][mask], bins=bins, density=True, histtype='step', color=color, lw=lw,
                           ls='--')
            ax_histy1.hist(self.data[f'{y_key}_d'][mask], bins=bins, density=True, histtype='step',
                           orientation='horizontal', color=color, lw=lw, ls='--')

    def plot_data_with_masks(self, x_key: str, y_key: str, show_lines: bool = False,
                             show_scatter: bool = True, filename: str = None):
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(5, 5, height_ratios=[1, 1, 3, 3, 3], width_ratios=[3, 3, 3, 1, 1])
        ax_main = plt.subplot(gs[2:5, 0:3])
        ax_histx = plt.subplot(gs[1, 0:3], sharex=ax_main)
        ax_histx1 = plt.subplot(gs[0, 0:3], sharex=ax_main)
        ax_histy = plt.subplot(gs[2:5, 3], sharey=ax_main)
        ax_histy1 = plt.subplot(gs[2:5, 4], sharey=ax_main)

        self.plot_main(ax_main, x_key, y_key, show_scatter, show_lines)
        self.plot_histograms(ax_histx, ax_histy, ax_histx1, ax_histy1, x_key, y_key)

        # Set up legends, labels, and grid
        for mask_name, color in self.colors.items():
            if show_scatter:
                ax_main.scatter(-1, -1, c=color, marker='o', label=self.labels[mask_name], s=30)
            else:
                ax_main.plot([], [], c=color, label=self.labels[mask_name])

        if show_scatter:
            ax_main.scatter(-1, -1, c='gray', marker='o', label='Dark Matter', s=30)
            ax_main.scatter(-1, -1, c='gray', marker='*', label='Stellar', s=30)

        ax_main.legend(loc='upper left', fontsize=10)
        ax_main.set_xlabel(rf'${self.labels[x_key]} = {x_key[0].upper()}/A$', fontsize=20)
        ax_main.set_ylabel(rf'${self.labels[y_key]} = {y_key[0].upper()}/A$', fontsize=20)
        ax_main.set_xlim([0, 1])
        ax_main.set_ylim([0, 1])
        ax_main.plot([0, 1], [0, 1], c='0.5', linestyle='--')
        ax_main.grid(True)
        ax_main.tick_params(which='both', labelsize=15)

        # Remove tick marks for histograms
        for ax in [ax_histx, ax_histx1, ax_histy, ax_histy1]:
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                           labelbottom=False, labelleft=False)

        # Add legends for histograms
        ax_histx.hist([], bins=1, density=True, histtype='step', color='k', lw=1.5, label='Stellar')
        ax_histx1.hist([], bins=1, density=True, histtype='step', color='k', lw=1.5, ls='--', label='Dark Matter')
        ax_histx.legend(fontsize=8)
        ax_histx1.legend(fontsize=8)

        plt.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)
        else:
            plt.show()

    def calculate_shape_diff(self):
        for key in ['b', 'c', 't']:
            diff = self.data[f'{key.upper()}_d'] / self.data[f'{key.upper()}_s']
            label = f'{key.upper()}_D/{key.upper()}_*'

            for mask_name, mask in self.masks.items():
                print(
                    f'{self.labels[mask_name]} galaxies: {label} mean: {np.mean(diff[mask]):.2f}, std: {np.std(diff[mask]):.2f}')

    def plot_triaxiality_vs_mass(self, filename: str = None):
        f, ax = plt.subplots(2, 1, figsize=(15, 6))
        plt.subplots_adjust(hspace=0)

        for i in [0, 1]:
            ax[i].set_xlim([5.77, 9.5])
            ax[i].set_ylim([0, 1])
            ax[i].plot([4, 9.5], [1 / 3, 1 / 3], c='.75', linestyle='--', zorder=0)
            ax[i].plot([4, 9.5], [2 / 3, 2 / 3], c='.75', linestyle='--', zorder=0)
            ax[i].tick_params(which='both', labelsize=15)
            ax[i].text(5.8, 1 / 6, 'Oblate', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
            ax[i].text(5.8, 3 / 6, 'Triaxial', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
            ax[i].text(5.8, 5 / 6, 'Prolate', fontsize=12, rotation='vertical', verticalalignment='center', c='k')

        ax[0].set_ylabel('T', fontsize=20)
        ax[1].set_ylabel(r'T$_*$', fontsize=20)
        ax[0].set_yticks([0, .5, 1])
        ax[1].set_yticks([0, .5])
        ax[1].set_xlabel(r'Log(M$_*$/M$_\odot$)', fontsize=20)
        ax[0].set_xticks([])

        for i in np.arange(len(self.data['masses'])):
            ax[0].axvline(self.data['masses'][i], ymin=min([self.data['T_d'][i], self.data['T_s'][i]]),
                          ymax=max([self.data['T_d'][i], self.data['T_s'][i]]), c='.5', zorder=0)

        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            ax[0].scatter(self.data['masses'][mask], self.data['T_d'][mask], c=color, marker='o',
                          label=f'{self.labels[mask_name]} Dark Matter')
            ax[0].scatter(self.data['masses'][mask], self.data['T_s'][mask], c=color, marker='v',
                          label=f'{self.labels[mask_name]} Stellar')

        ax[0].legend()

        vmin, vmax = np.min(self.data['mb']), np.max(self.data['mb'])
        norm = plt.Normalize(vmin, vmax)

        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
        for i, (mask_name, mask) in enumerate(self.masks.items()):
            marker = markers[i % len(markers)]
            p = ax[1].scatter(self.data['masses'][mask], self.data['T_s'][mask], c=self.data['mb'][mask],
                              cmap='viridis', norm=norm, marker=marker, s=100, edgecolors='k',
                              label=self.labels[mask_name])

        cbar = f.colorbar(p, cax=f.add_axes([.91, .11, .03, .77]))
        cbar.set_label(r'M$_{bary}$/M$_{vir}(<$R$_{eff}$)', fontsize=25)
        cbar.ax.tick_params(labelsize=15)
        ax[1].legend()

        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)
        else:
            plt.show()




# Usage example:
# data = {
#     'B_s': B_s, 'C_s': C_s, 'T_s': T_s, 'B_d': B_d, 'C_d': C_d, 'T_d': T_d,
#     'masses': masses, 'mb': mb, 'htype': htype, 'reff': reff, 'mvir': mvir
# }
# masks = {'disky': disky_mask, 'non_disky': ~disky_mask}
# labels = {'disky': 'Disky', 'non_disky': 'Non-Disky', 'b': 'S', 'c': 'Q'}
# colors = {'disky': 'green', 'non_disky': 'k'}

# plotter = GeneralPlotter(data, masks, labels, colors)
# plotter.plot_data_with_masks('b', 'c', show_lines=False, show_scatter=True)
# plotter.calculate_shape_diff()
# plotter.plot_triaxiality_vs_mass()