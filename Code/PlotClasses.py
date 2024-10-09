import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typing import List, Dict, Optional, Callable, Tuple
#reset to default matplotlib settings
plt.rcdefaults()


class GeneralPlotter:
    def __init__(self, data: Dict[str, np.ndarray], masks: Dict[str, np.ndarray],
                 labels: Dict[str, str], colors: Dict[str, str]):
        self.data = data
        self.masks = masks
        self.labels = labels
        self.colors = colors
        self.legend_fontsize = 20
        self.axis_fontsize = 33
        self.tick_fontsize = 20
        self.point_size = 120
        self.bin_count = 10
        reff = data['reff_multi']
        if reff != 1:
            self.suptitle = rf'{reff}R$_{{eff}}$'
        else:
            self.suptitle = rf'R$_{{eff}}$'

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
                  show_lines: bool = False):
        size = self.point_size
        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            dm_color = self.lighten_color(color, 1.5)

            if show_scatter:
                ax.scatter(self.data[f'{x_key}_d'][mask], self.data[f'{y_key}_d'][mask],
                           c=color, marker='o', s=size, alpha=1, zorder=10,edgecolors='white')
                ax.scatter(self.data[f'{x_key}_s'][mask], self.data[f'{y_key}_s'][mask],
                           c=color, marker='*', s=size*4, alpha=1,zorder=10,edgecolors='white')

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
        bins = np.linspace(0, 1, self.bin_count + 1)
        lw = 2
        linestyles = ['-', '--', '-.', ':']

        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            density=False
            ax_histx.hist(self.data[f'{x_key}_s'][mask], bins=bins, density=density, histtype='step', color=color, lw=lw,
                          ls=linestyles[0],alpha = 1)

            ax_histy.hist(self.data[f'{y_key}_s'][mask], bins=bins, density=density, histtype='step',
                          orientation='horizontal', color=color, lw=lw, ls = linestyles[0],alpha=1)

            ax_histx1.hist(self.data[f'{x_key}_d'][mask], bins=bins, density=density, histtype='step', color=color, lw=lw,
                           ls=linestyles[1])

            ax_histy1.hist(self.data[f'{y_key}_d'][mask], bins=bins, density=density, histtype='step',
                           orientation='horizontal', color=color, lw=lw, ls=linestyles[1])

    def plot_data_with_masks(self, x_key: str, y_key: str, show_lines: bool = False,
                             show_scatter: bool = True, filename: str = None):
        fig = plt.figure(figsize=(10, 10),dpi=100)
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
                ax_main.scatter(-1, -1, c=color, marker='o', s=self.point_size, label=f'{self.labels[mask_name]}')
            else:
                ax_main.plot([], [], c=color)

        if show_scatter:
            ax_main.scatter(-1, -1, c='gray', marker='o', s=self.point_size, label='Dark Matter')
            ax_main.scatter(-1, -1, c='gray', marker='*', s=self.point_size, label='Stars')


        ax_main.legend(loc='upper left', fontsize=self.legend_fontsize)
        ax_main.set_xlabel(rf'${self.labels[x_key]} = {x_key[0].upper()}/A$ ({self.suptitle})', fontsize=self.axis_fontsize)
        ax_main.set_ylabel(rf'${self.labels[y_key]} = {y_key[0].upper()}/A$ ({self.suptitle})', fontsize=self.axis_fontsize)
        ax_main.set_xlim([0, 1])
        ax_main.set_ylim([0, 1])
        ax_main.plot([0, 1], [0, 1], c='0.5', linestyle='--')
        #make grid last so it doesn't interfere with the plot
        #ax_main.grid(True, zorder = 0)
        ax_main.tick_params(which='both', labelsize=self.tick_fontsize)

        # Remove tick marks for histograms
        for ax in [ax_histx, ax_histx1, ax_histy, ax_histy1]:
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                           labelbottom=False, labelleft=False)

        # Add legends for histograms
        ax_histx.hist([], bins=1, density=True, histtype='step', color='k', lw=1.5, label='Stars')
        ax_histx1.hist([], bins=1, density=True, histtype='step', color='k', lw=1.5, ls='--', label='Dark Matter')
        ax_histx.legend(fontsize=self.legend_fontsize)
        ax_histx1.legend(fontsize=self.legend_fontsize)
        #fig.suptitle(self.suptitle, fontsize=self.axis_fontsize)

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
        #CDM
        f, ax = plt.subplots(1, 1, figsize=(15, 3),dpi=100)
        #ax.set_xlim([5.77, 9.5])
        #plt.subplots_adjust(hspace=0)
        #
        # for i in [0, 1]:
        #     ax[i].set_xlim([5.77, 9.5])
        #     ax[i].set_ylim([0, 1])
        #     ax[i].plot([4, 9.5], [1 / 3, 1 / 3], c='.75', linestyle='--', zorder=0)
        #     ax[i].plot([4, 9.5], [2 / 3, 2 / 3], c='.75', linestyle='--', zorder=0)
        #     ax[i].tick_params(which='both', labelsize=self.tick_fontsize)
        #     ax[i].text(5.8, 1 / 6, 'Oblate', fontsize=8, rotation='vertical', verticalalignment='center', c='k')
        #     ax[i].text(5.8, 3 / 6, 'Triaxial', fontsize=8, rotation='vertical', verticalalignment='center', c='k')
        #     ax[i].text(5.8, 5 / 6, 'Prolate', fontsize=8, rotation='vertical', verticalalignment='center', c='k')
        #     #ax[i].tick_params(axis='y',labelsize=self.tick_fontsize)
        # SIDM vs CDM
        #f, ax = plt.subplots(1, 1, figsize=(8, 3),dpi=100)
        #plt.subplots_adjust(hspace=0)

        xlimit = 5.77
        text_x = 5.8
        ax.set_xlim([xlimit, 9.5])
        ax.set_ylim([0, 1])
        ax.plot([xlimit, 9.5], [1 / 3, 1 / 3], c='.75', linestyle='--', zorder=0)
        ax.plot([xlimit, 9.5], [2 / 3, 2 / 3], c='.75', linestyle='--', zorder=0)
        ax.tick_params(which='both', labelsize=self.tick_fontsize)
        ax.text(text_x, 1 / 6, 'Oblate', fontsize=8, rotation='vertical', verticalalignment='center', c='k')
        ax.text(text_x, 3 / 6, 'Triaxial', fontsize=8, rotation='vertical', verticalalignment='center', c='k')
        ax.text(text_x, 5 / 6, 'Prolate', fontsize=8, rotation='vertical', verticalalignment='center', c='k')
        ax.tick_params(axis='y',labelsize=self.tick_fontsize)


        ax.set_ylabel('T', fontsize=self.axis_fontsize)
        #ax[1].set_ylabel(r'T$_*$', fontsize=self.axis_fontsize)
        ax.set_yticks([0, .5, 1])
        #ax[1].set_yticks([0, .5])
        ax.set_xlabel(r'Log(M$_*$/M$_\odot$)', fontsize=self.axis_fontsize)
        ax.set_xticks([])
        #ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        #ax[1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        #f.suptitle(self.suptitle, fontsize=self.axis_fontsize)



        for i in np.arange(len(self.data['masses'])):
            ax.axvline(self.data['masses'][i], ymin=min([self.data['T_d'][i], self.data['T_s'][i]]),
                          ymax=max([self.data['T_d'][i], self.data['T_s'][i]]), c='.5', zorder=0)

        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            ax.scatter(self.data['masses'][mask], self.data['T_d'][mask], c=color, marker='o', s=self.point_size,edgecolors='white')
            ax.scatter(self.data['masses'][mask], self.data['T_s'][mask], c=color, marker='v', s=self.point_size,edgecolors='white')


        #add dummy points to legend for dark matter and stellar
        ax.scatter(-1, -1, c='gray', marker='o', label='Dark Matter')
        ax.scatter(-1, -1, c='gray', marker='v', label='Stars')
        #add dummy points for mask colors
        for mask_name, color in self.colors.items():
            ax.scatter(-1, -1, c=color, marker='o', label=self.labels[mask_name])
        ax.legend(loc='lower left', fontsize=10, bbox_to_anchor=(0.02, -.05),ncol=2)


        # # Define the range of the colormap slice (0 to 1)
        # start = 0.1  # Start of the colormap slice
        # end = 0.8  # End of the colormap slice
        # from matplotlib.colors import LinearSegmentedColormap
        #
        # # Get the magma colormap
        # magma = plt.cm.get_cmap('magma')
        #
        #
        #
        # #
        # # Create a new colormap from the slice
        # magma_slice = LinearSegmentedColormap.from_list(
        #     'magma_slice', magma(np.linspace(start, end, 256))
        # )
        # color = np.log10(self.data['mb'])
        # color = self.data['mb']
        #
        # vmin, vmax = np.min(color), np.max(color)
        # norm = plt.Normalize(vmin, vmax)
        #
        # markers = ['s', 'o', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
        # for i, (mask_name, mask) in enumerate(self.masks.items()):
        #     #if i == 0:
        #         #continue
        #     marker = markers[i % len(markers)]
        #     p = ax[1].scatter(self.data['masses'][mask], self.data['T_s'][mask], c=color[mask],
        #                       cmap=magma_slice, norm=norm, marker=marker, s=self.point_size, edgecolors='white',
        #                       label=self.labels[mask_name])
        #
        #
        # cbar = f.colorbar(p, cax=f.add_axes([.91, .11, .03, .77]))
        # # cbar.set_label(r'M$_{bary}$/M$_{vir}(<$R$_{eff}$)', fontsize=self.axis_fontsize)
        # cbar.set_label(r'M$_{bary}$/M$_{vir}$', fontsize=self.axis_fontsize)
        # cbar.ax.tick_params(labelsize=self.tick_fontsize)
        # ax[1].legend(loc='lower left', fontsize=10, bbox_to_anchor=(0.02, -.05))

        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)
        else:
            plt.show()

    def plot_T_stellar_vs_dark(self, filename: str = None):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8),dpi=100)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.fill_between([0, 1], [-1 / 3, 2 / 3], [1 / 3, 4 / 3], color='0.75', alpha=.3)
        ax.plot([0, 1], [0, 1], c='0.5', linestyle='--', zorder=0)
        ax.set_ylabel(rf'T$_*$({self.suptitle})', fontsize=self.axis_fontsize)
        ax.set_xlabel(rf'T$_{{DM}}$({self.suptitle})', fontsize=self.axis_fontsize)
        #ax.grid(True)
        ax.tick_params(which='both', labelsize=self.tick_fontsize)

        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            ax.scatter(self.data['T_s'][mask], self.data['T_d'][mask], c=color, label=self.labels[mask_name],
                       s=self.point_size, edgecolors='white')

        ax.legend(loc='upper left', fontsize=self.legend_fontsize)
        #fig.suptitle(self.suptitle, fontsize=self.axis_fontsize)
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)
        else:
            plt.show()

    def plot_Mstar_V_b_d_b_s_V_c_d_c_s(self, filename: str = None):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True,dpi=100)
        #plt.subplots_adjust(wspace=0.1)

        for ax in axs:
            #ax.set_xscale('log')
            ax.set_xlabel(r'Log(M$_*$/M$_\odot)$', fontsize=self.axis_fontsize)
            #ax.set_ylim(0.5, 2.5)
            #ax.grid()
            ax.tick_params(which='both', labelsize=self.tick_fontsize)

        axs[0].set_ylabel(rf'Q$_{{DM}}$/Q$_*$ ({self.suptitle})', fontsize=self.axis_fontsize)
        axs[1].set_ylabel(rf'S$_{{DM}}$/S$_*$ ({self.suptitle})', fontsize=self.axis_fontsize)

        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            axs[0].scatter(self.data['masses'][mask], self.data['B_d'][mask] / self.data['B_s'][mask],
                           c=color, label=self.labels[mask_name], s=self.point_size*1.3, alpha=1, zorder=10,edgecolors='white')
            axs[1].scatter(self.data['masses'][mask], self.data['C_d'][mask] / self.data['C_s'][mask],
                           c=color, s=self.point_size*1.3, alpha=1, zorder=10,edgecolors='white')

        axs[0].axhline(1, c='0.5', linestyle='--', zorder=0)
        axs[1].axhline(1, c='0.5', linestyle='--', zorder=0)

        axs[0].legend(loc='upper left', fontsize=self.legend_fontsize)
        #fig.suptitle(self.suptitle, fontsize=self.axis_fontsize)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)
        else:
            plt.show()

    def plot_q_s_q_d(self, filename: Optional[str] = None):
        fig = plt.figure(figsize=(16, 16),dpi=100)
        gs = fig.add_gridspec(2, 2, left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.15, hspace=0.15)
        axs = gs.subplots(sharex='col', sharey='row')

        for ax in axs.flatten():
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.plot([0, 1], [0, 1], c='0.5', linestyle='--', zorder=0)
            ax.tick_params(which='both', labelsize=self.tick_fontsize)
            ax.set_aspect('equal', adjustable='box')

        # Set y-axis labels
        axs[0, 0].set_ylabel(r'$Q_{*_{_{_{_{_{.}}}}}}$', fontsize=self.axis_fontsize)
        axs[1, 0].set_ylabel(r'$S_{*_{_{_{_{_{.}}}}}}$', fontsize=self.axis_fontsize)


        # Set individual x-axis labels for each subfigure
        axs[0, 0].set_xlabel(r'Q$_{DM}$', fontsize=self.axis_fontsize)
        axs[0, 1].set_xlabel(r'Q$_{DM}$', fontsize=self.axis_fontsize)
        axs[1, 0].set_xlabel(r'S$_{DM}$', fontsize=self.axis_fontsize)
        axs[1, 1].set_xlabel(r'S$_{DM}$', fontsize=self.axis_fontsize)
        fig.suptitle(self.suptitle, fontsize=self.axis_fontsize)

        # Plots colored by mask
        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            axs[0, 0].scatter(self.data['B_d'][mask], self.data['B_s'][mask],
                              c=color, label=self.labels[mask_name], s=self.point_size, edgecolors='white')
            axs[1, 0].scatter(self.data['C_d'][mask], self.data['C_s'][mask],
                              c=color, s=self.point_size, edgecolors='white')

        axs[0, 0].legend(loc='lower left', fontsize=self.legend_fontsize)

        # Plots colored by mb
        from matplotlib.colors import LinearSegmentedColormap
        magma = plt.cm.get_cmap('magma')
        magma_slice = LinearSegmentedColormap.from_list(
            'magma_slice', magma(np.linspace(0.1, 0.8, 256))
        )

        color = self.data['mb']
        #color = self.data['masses']
        vmin, vmax = np.min(color), np.max(color)
        norm = plt.Normalize(vmin, vmax)

        scatter = axs[0, 1].scatter(self.data['B_d'], self.data['B_s'],
                                    c=color, cmap=magma_slice, norm=norm, s=self.point_size, edgecolors='white')
        axs[1, 1].scatter(self.data['C_d'], self.data['C_s'],
                          c=color, cmap=magma_slice, norm=norm, s=self.point_size, edgecolors='white')

        # Adjust colorbar position
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label(r'M$_{bary}$/M$_{vir}$', fontsize=self.axis_fontsize)
        cbar.ax.tick_params(labelsize=self.tick_fontsize)

        # Adjust subplot layouts to accommodate x-labels
        # plt.tight_layout()

        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=100)
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