import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression

import traceback

from typing import List, Dict, Optional, Callable, Tuple
#reset to default matplotlib settings
plt.rcdefaults()

ellipse_data = {
    'KF High Mass': {  # 10^9.0 M_sun < M_sstarf < 10^9.6 M_sun
        '1R_eff': {'center': (0.863, 0.297), 'std': (0.075, 0.103)},
        '2R_eff': {'center': (0.91, 0.352), 'std': (0.04, 0.155)}
    },
    'KF Medium Mass': {  # 10^8.5 M_sun < M_sstarf < 10^9.0 M_sun
        '1R_eff': {'center': (0.857, 0.323), 'std': (0.05, 0.118)},
        '2R_eff': {'center': (0.902, 0.371), 'std': (0.05, 0.163)}
    },
    'KF Low Mass': {  # 10^7.0 M_sun < M_sstarf < 10^8.5 M_sun
        '1R_eff': {'center': (0.753, 0.459), 'std': (0.087, 0.194)},
        '2R_eff': {'center': (0.778, 0.446), 'std': (0.102, 0.213)}
    }
}

# Colors for each mass bin
ell_colors = {
    'KF Low Mass': 'indigo',
    'KF Medium Mass': 'darkred',
    'KF High Mass': 'limegreen'
}


def T(ba,ca):
    return( (1-ba**2)/(1-ca**2) )

def add_regression_analysis(ax, x, y, color, alpha=0.1):
    """
    Add linear regression line and confidence band to a plot, and return statistics.

    Args:
        ax: matplotlib axis
        x: x-values (masses)
        y: y-values (shape parameters)
        color: color for the regression line and band
        alpha: transparency for the confidence band

    Returns:
        dict: Statistical measures including mean, std, and regression info
    """
    # Compute basic statistics
    mean_val = np.mean(y)
    std_val = np.std(y)

    # Prepare data for regression
    X = x.reshape(-1, 1)

    # Perform linear regression
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    # Calculate residuals and their standard deviation
    residuals = y - y_pred
    residual_std = np.std(residuals)

    # Calculate standard error of slope
    n = len(x)
    x_mean = np.mean(x)
    sum_sq_x = np.sum((x - x_mean) ** 2)
    slope_std_error = residual_std / np.sqrt(sum_sq_x)

    # Sort x values for plotting
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    # Plot regression line
    ax.plot(x_sorted, y_pred_sorted, ls='dotted', color=color, alpha=0.5, zorder=5)

    # # Create confidence bands (±2σ)
    # ax.fill_between(x_sorted,
    #                 y_pred_sorted - 2 * residual_std,
    #                 y_pred_sorted + 2 * residual_std,
    #                 color=color, alpha=alpha, zorder=1)

    # Return statistics
    stats = {
        'mean': mean_val,
        'std': std_val,
        'residual_std': residual_std,
        'slope': reg.coef_[0],
        'slope_std_error': slope_std_error,
        'intercept': reg.intercept_,
        'r_squared': reg.score(X, y)
    }

    # Print statistics
    print(f"\nStatistics:")
    print(f"Mean: {mean_val:.3f}")
    print(f"Std: {std_val:.3f}")
    print(f"Residual Std: {residual_std:.3f}")
    print(f"Slope: {reg.coef_[0]:.3f} ± {slope_std_error:.3f}")
    print(f"Intercept: {reg.intercept_:.3f}")
    print(f"R²: {reg.score(X, y):.3f}")

    return stats




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
        self.plot_KF_data = False
        self.plot_regression = False
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
                  show_lines: bool = False, link_sims: bool = False):
        size = self.point_size
        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            dm_color = self.lighten_color(color, 1.5)

            if show_scatter:
                ax.scatter(self.data[f'{x_key}_d'][mask], self.data[f'{y_key}_d'][mask],
                           c=color, marker='o', s=size, alpha=1, zorder=10,edgecolors='white')
                ax.scatter(self.data[f'{x_key}_s'][mask], self.data[f'{y_key}_s'][mask],
                           c=color, marker='*', s=size*3.5, alpha=1,zorder=10,edgecolors='white')

            if show_lines or not show_scatter:
                if not link_sims:
                    for i in np.where(mask)[0]:
                        if show_scatter:
                            ax.plot([self.data[f'{x_key}_s'][i], self.data[f'{x_key}_d'][i]],
                                    [self.data[f'{y_key}_s'][i], self.data[f'{y_key}_d'][i]],
                                    zorder=0, lw=0.5, c='k', alpha=0.7)
                        else:
                            ax.plot([self.data[f'{x_key}_s'][i], self.data[f'{x_key}_d'][i]],
                                    [self.data[f'{y_key}_s'][i], self.data[f'{y_key}_d'][i]],
                                    zorder=0, lw=1, c=color, alpha=0.7)
                if link_sims:
                    # link lines across feedback types for sims with the same name
                    # link lines from _s to _s and _d to _d of different feedbacks

                    sims = np.copy(self.data['sims'])

                    # remove extra characters in names starting with a period
                    #sims = [sim.split('.')[0] for sim in sims]
                    # print(sims)
                    # find unique sims
                    unique_sims = np.unique(sims)

                    # loop over each unique sim and plot lines across feedback types
                    for sim in unique_sims:

                        # find indices of sims that match the current sim
                        indices = []
                        for i in range(len(sims)):
                            if sims[i] == sim:
                                indices.append(i)

                        # for each index, find out which feedback type it is
                        sidm_index = []
                        cdm_index = []
                        for index in indices:
                            if self.data['feedback_type'][index] == 'BWMDC':
                                sidm_index.append(index)
                            elif self.data['feedback_type'][index] == 'SBMarvel':
                                cdm_index.append(index)
                            else:
                                print('Error: feedback type not recognized')
                                print(self.data['feedback_type'][index])
                                # raise an error
                                ValueError('Error: feedback type not recognized')
                            # plot line
                        ax.plot([self.data[f'{x_key}_s'][sidm_index], self.data[f'{x_key}_s'][cdm_index]],
                                [self.data[f'{y_key}_s'][sidm_index], self.data[f'{y_key}_s'][cdm_index]],
                                zorder=0, lw=0.5, c='k', alpha=0.7)
                        ax.plot([self.data[f'{x_key}_d'][sidm_index], self.data[f'{x_key}_d'][cdm_index]],
                                [self.data[f'{y_key}_d'][sidm_index], self.data[f'{y_key}_d'][cdm_index]],
                                zorder=0, lw=0.5, c='k', alpha=0.7)


        #plot shaded region where x > 0.6 and y < 0.4
        ax.fill_between([0.65, 1], 0, 0.4, color='0.75', alpha=0.3)




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
                             show_scatter: bool = True, filename: str = None, link_sims = False):
        # Plot ellipses
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms


        fig = plt.figure(figsize=(10, 10),dpi=100)
        gs = gridspec.GridSpec(5, 5, height_ratios=[1, 1, 3, 3, 3], width_ratios=[3, 3, 3, 1, 1])
        ax_main = plt.subplot(gs[2:5, 0:3])
        ax_histx = plt.subplot(gs[1, 0:3], sharex=ax_main)
        ax_histx1 = plt.subplot(gs[0, 0:3], sharex=ax_main)
        ax_histy = plt.subplot(gs[2:5, 3], sharey=ax_main)
        ax_histy1 = plt.subplot(gs[2:5, 4], sharey=ax_main)

        self.plot_main(ax_main, x_key, y_key, show_scatter, show_lines, link_sims)
        self.plot_histograms(ax_histx, ax_histy, ax_histx1, ax_histy1, x_key, y_key)

        # Set up legends, labels, and grid
        for mask_name, color in self.colors.items():
            if show_scatter:
                ax_main.scatter(-1, -1, c=color, marker='o', s=self.point_size, label=f'{self.labels[mask_name]}')
            else:
                ax_main.plot([], [], c=color)

        if show_scatter:
            ax_main.scatter(-1, -1, c='gray', marker='o', s=self.point_size, label='Dark Matter')
            ax_main.scatter(-1, -1, c='gray', marker='*', s=self.point_size, label='Stellar Matter')


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
        ax_histx.hist([], bins=1, density=True, histtype='step', color='k', lw=1.5, label='Stellar Matter')
        ax_histx1.hist([], bins=1, density=True, histtype='step', color='k', lw=1.5, ls='--', label='Dark Matter')
        ax_histx.legend(fontsize=self.legend_fontsize)
        ax_histx1.legend(fontsize=self.legend_fontsize)
        #fig.suptitle(self.suptitle, fontsize=self.axis_fontsize)






        print('plot_KF_data')
        print(self.plot_KF_data)
        print(self.plot_KF_data == False)
        # Choose which R_eff data to use
        if self.plot_KF_data == False:
            r_eff_key = None
        elif self.data['reff_multi'] == 1:
            r_eff_key = '1R_eff'
        elif self.data['reff_multi']==2:
            r_eff_key = '2R_eff'
        else:
            r_eff_key = None
        #fig, ax = plt.subplots(1, 1, figsize=(8, 8),dpi=100)

        def confidence_ellipse(center, std, ax, color, linestyle='--', lw=3):
            # Create the ellipse with 2 standard deviations (95% confidence)
            pearson = 0  # Assuming no correlation
            scale_x = std[0] *1
            scale_y = std[1] *1


            ellipse = Ellipse(center, width=scale_x * 2, height=scale_y * 2,
                              facecolor='None', alpha=1, edgecolor=color, linestyle=linestyle, lw=3,
                            zorder=100
                              )


            return ax.add_patch(ellipse)

        # Plot each mass bin's ellipse
        if r_eff_key:
            for mass_bin, color in ell_colors.items():
                # if color != 'mediumspringgreen':
                #     continue
                data = ellipse_data[mass_bin][r_eff_key]
                confidence_ellipse(data['center'], data['std'], ax_main, color)
                ax_main.plot(-1, -1, c=color, label=mass_bin, linestyle='--', lw=3)
            ax_main.legend(loc='upper left', fontsize=self.legend_fontsize)


            #print number of stellar points that are within the ellipse
            for mass_bin, color in ell_colors.items():
                for mask_name, mask in self.masks.items():
                    #print(mass_bin)
                    #print(self.masks)
                    data = ellipse_data[mass_bin][r_eff_key]
                    center = data['center']
                    std = data['std']
                    x = self.data[f'{x_key}_s'][mask]
                    y = self.data[f'{y_key}_s'][mask]
                    x_std = std[0]
                    y_std = std[1]
                    x_center = center[0]
                    y_center = center[1]

                    #plot ellipse based on the data in each mask
                    x_c = np.mean(x)
                    y_c = np.mean(y)
                    x_std = np.std(x)
                    y_std = np.std(y)
                    color = self.colors[mask_name]
                    #confidence_ellipse((x_c, y_c), (x_std, y_std), ax_main, color, linestyle='-', lw=3)
                    #print number of points within the ellipse
                    condition = ((x - x_center)**2/x_std**2 + (y - y_center)**2/y_std**2 < 1)
                    print(f'{mass_bin} {mask_name}: {np.sum(condition)}/{np.sum(mask)} points within ellipse')

        #format ax
        # ax.set_xlim([0, 1])
        # ax.set_ylim([0, 1])
        # ax.set_xlabel(rf'${self.labels[x_key]} = {x_key[0].upper()}/A$ ({self.suptitle})', fontsize=self.axis_fontsize)
        # ax.set_ylabel(rf'${self.labels[y_key]} = {y_key[0].upper()}/A$ ({self.suptitle})', fontsize=self.axis_fontsize)
        # ax.plot([0, 1], [0, 1], c='0.5', linestyle='--')
        # ax.tick_params(which='both', labelsize=self.tick_fontsize)
        #ax.grid(True, zorder=0)
        #ax.set_title(self.suptitle, fontsize=self.axis_fontsize)
        # Add legend



        plt.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)
        else:
            plt.show()





    def calculate_shape_diff(self):
        for key in ['b', 'c', 't']:
            diff = self.data[f'{key.upper()}_d'] / self.data[f'{key.upper()}_s']
            label = f'{key.upper()}_D/{key.upper()}_*'
            #print data values
            for mask_name, mask in self.masks.items():
                print(
                    f'{self.labels[mask_name]} galaxies: {label} mean: {np.mean(diff[mask]):.2f}, std: {np.std(diff[mask]):.2f}')
                d = self.data[f'{key.upper()}_d'][mask]
                s = self.data[f'{key.upper()}_s'][mask]
                print(f'{key.upper()}_d: {np.mean(d):.2f} +/- {np.std(d):.2f}')
                print(f'{key.upper()}_s: {s.mean():.2f} +/- {s.std():.2f}')
        #



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
        ax.tick_params(axis='x',labelsize=self.tick_fontsize)
        ax.set_xticks([6, 7, 8, 9])


        ax.set_ylabel(r'$T$', fontsize=self.axis_fontsize)
        #ax[1].set_ylabel(r'T$_*$', fontsize=self.axis_fontsize)
        ax.set_yticks([0, .5, 1])
        #ax[1].set_yticks([0, .5])
        ax.set_xlabel(r'Log(M$_*$/M$_\odot$)', fontsize=self.axis_fontsize)
        #ax.set_xticks([])
        #ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        #ax[1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        #f.suptitle(self.suptitle, fontsize=self.axis_fontsize)



        for i in np.arange(len(self.data['masses'])):
            ax.axvline(self.data['masses'][i], ymin=min([self.data['T_d'][i], self.data['T_s'][i]]),
                          ymax=max([self.data['T_d'][i], self.data['T_s'][i]]), c='.5', zorder=0)

        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            ax.scatter(self.data['masses'][mask], self.data['T_d'][mask], c=color, marker='o', s=self.point_size,edgecolors='white')
            ax.scatter(self.data['masses'][mask], self.data['T_s'][mask], c=color, marker='*', s=3.5*self.point_size,edgecolors='white')


        #add dummy points to legend for dark matter and stellar
        ax.scatter(-1, -1, c='gray', marker='o', label='Dark Matter')
        ax.scatter(-1, -1, c='gray', marker='*', label='Stellar Matter')
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
        ax.set_ylabel(rf'$T_*$({self.suptitle})', fontsize=self.axis_fontsize)
        ax.set_xlabel(rf'$T_{{\mathrm{{DM}}}}$({self.suptitle})', fontsize=self.axis_fontsize)

        #ax.grid(True)
        ax.tick_params(which='both', labelsize=self.tick_fontsize)

        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            ax.scatter(self.data['T_d'][mask], self.data['T_s'][mask], c=color, label=self.labels[mask_name],
                       s=self.point_size, edgecolors='white')
            if self.plot_regression:
                #regression analysis
                print(f"\nAnalysis for {mask_name}:")
                add_regression_analysis(ax, self.data['T_d'][mask], self.data['T_s'][mask], color)
            

        #highlight T where C_d or C_s is greater than 0.8
        for i in np.arange(len(self.data['masses'])):
            if self.data['C_d'][i] > 0.9 or self.data['C_s'][i] > 0.9:
                ax.scatter(self.data['T_s'][i], self.data['T_d'][i], c='r', s=self.point_size, edgecolors='white',alpha=0.5)

        ax.legend(loc='upper left', fontsize=self.legend_fontsize)
        #fig.suptitle(self.suptitle, fontsize=self.axis_fontsize)
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)
        else:
            plt.show()

    #make a plot showing triaxilty histograms for each mask
    # def plot_triaxiality_histograms(self, filename: Optional[str] = None):
    #     """
    #     Plot histograms of the triaxiality parameter for each mask
    #     """
    #     # Create figure with subplots
    #     fig, ax = plt.subplots(1, 1, figsize=(8, 8),dpi=100)
    #     # plot histograms of abs (T_d - T_s) for each mask
    #     T_diff = np.log10((self.data['T_d']/ self.data['T_s']))
    #     T_diff = abs((self.data['T_d'] - self.data['T_s']))
    #     bins = np.linspace(min(T_diff), max(T_diff), 12)
    #     #bins = np.logspace(min(T_diff_log), max(T_diff_log), 12)
    #     #print(bins)
    #
    #     for mask_name, mask in self.masks.items():
    #         color = self.colors[mask_name]
    #         label = self.labels[mask_name]
    #
    #         # Calculate histogram data
    #         hist_values, bin_edges = np.histogram(
    #             T_diff[mask],
    #             bins=bins,
    #             density=False
    #         )
    #         #print(hist_values)
    #         # For the second mask, make the counts negative
    #         if list(self.masks.keys()).index(mask_name) == 1:
    #             hist_values = -hist_values
    #         # Plot histogram
    #         ax.bar(bin_edges[:-1],
    #                hist_values,
    #                width=np.diff(bin_edges),
    #                alpha=1,
    #                color=color,
    #                edgecolor='white',
    #                label=f"{label}",
    #                linewidth=1.5,
    #                align='edge'
    #                )
    #         # Set labels and title
    #         ax.tick_params(labelsize=self.tick_fontsize)
    #
    #         #ax.set_xscale('log')
    #
    #         # Adjust y-axis labels to show absolute values
    #         yticks = ax.get_yticks()
    #         ax.set_yticklabels([str(abs(int(tick))) for tick in yticks])
    #     ax.set_ylabel('Count', fontsize=self.axis_fontsize)
    #     ax.legend(fontsize=self.legend_fontsize)
    #     ax.set_xlabel(rf'$ T_{{\mathrm{{DM}}}} - T_*$ ({self.suptitle})', fontsize=self.axis_fontsize)

    def plot_triaxiality_histograms(self, filename: Optional[str] = None):
        import math
        """
        Plot histograms of the triaxiality parameter for each mask
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)

        T_diff = abs((self.data['T_d'] - self.data['T_s']))
        bins = np.linspace(min(T_diff), max(T_diff), 12)
        #bins from 0 to 0.36 in steps of 0.03
        bins = np.linspace(0, 0.36, 13)
        print(bins)


        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            label = self.labels[mask_name]

            hist_values, bin_edges = np.histogram(
                T_diff[mask],
                bins=bins,
                density=False
            )



            # For the second mask, make the counts negative
            if list(self.masks.keys()).index(mask_name) == 1:
                hist_values = -hist_values

            ax.bar(bin_edges[:-1],
                   hist_values,
                   width=np.diff(bin_edges),
                   alpha=1,
                   color=color,
                   edgecolor='white',
                   label=f"{label}",
                   linewidth=1,
                   align='edge'
                   )

        ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))

        #set major and minor ticks for the x-axis at 0.15 and .03
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.15))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.03))
        
        yticks = ax.get_yticks()


        # Set labels showing absolute values but maintain position
        yticks = [int(tick) for tick in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(abs(int(tick))) for tick in yticks])

        ax.tick_params(labelsize=self.tick_fontsize)
        ax.set_ylabel('Count', fontsize=self.axis_fontsize)
        ax.legend(fontsize=self.legend_fontsize)
        ax.set_xlabel(rf'$ |T_{{\mathrm{{DM}}}} - T_*|$ ({self.suptitle})', fontsize=self.axis_fontsize)

        # #show grid with y values at integers
        #
        #ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

        #save fig if filename is provided
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)
        else:
            plt.show()
        # Add overall statistics
        stats_text = []
        for mask_name, mask in self.masks.items():
            stats_text = []
            n_halos = np.sum(mask)
            stats = f"{self.labels[mask_name]}:\n"
            stats += f"N = {n_halos}\n"
            stats += f"Mean: {np.nanmean(T_diff[mask]):.3f} ± {np.nanstd(T_diff[mask]):.2f}\n"
            stats_text.append(stats)
            #add count of galaxies with less than 0.1 difference
            print(rf'{self.labels[mask_name]}: {np.sum( abs(T_diff[mask]) < 0.1)} less than  $\pm$ 0.1 difference')
            #add count of galaxies with less than 0.15 difference
            print(rf'{self.labels[mask_name]}: {np.sum(abs(T_diff[mask]) < 0.15)} less than $\pm$ 0.15 difference')

            print(stats)



    def plot_Mstar_V_b_d_b_s_V_c_d_c_s(self, filename: str = None):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True, dpi=100)

        for ax in axs:
            ax.set_xlabel(r'Log(M$_*$/M$_\odot)$', fontsize=self.axis_fontsize)
            ax.tick_params(which='both', labelsize=self.tick_fontsize)

        axs[0].set_ylabel(rf'$Q_{{\mathrm{{DM}}}}/Q_*$ ({self.suptitle})', fontsize=self.axis_fontsize)
        axs[1].set_ylabel(rf'$S_{{\mathrm{{DM}}}}/S_*$ ({self.suptitle})', fontsize=self.axis_fontsize)

        stats = {}  # Store statistics for each mask and ratio
        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            # B_d/B_s ratio
            b_ratio = self.data['B_d'][mask] / self.data['B_s'][mask]
            axs[0].scatter(self.data['masses'][mask], b_ratio,
                           c=color, label=self.labels[mask_name],
                           s=self.point_size * 1.3, alpha=1, zorder=10, edgecolors='white')



            # C_d/C_s ratio
            c_ratio = self.data['C_d'][mask] / self.data['C_s'][mask]
            axs[1].scatter(self.data['masses'][mask], c_ratio,
                           c=color, s=self.point_size * 1.3, alpha=1, zorder=10, edgecolors='white')

            if self.plot_regression:
                print(f"\nAnalysis for {mask_name} - Q_DM/Q_*:")
                stats[f"{mask_name}_b_ratio"] = add_regression_analysis(
                    axs[0], self.data['masses'][mask], b_ratio, color
                )

                print(f"\nAnalysis for {mask_name} - S_DM/S_*:")
                stats[f"{mask_name}_c_ratio"] = add_regression_analysis(
                    axs[1], self.data['masses'][mask], c_ratio, color
                )

        axs[0].axhline(1, c='0.5', linestyle='--', zorder=0)
        axs[1].axhline(1, c='0.5', linestyle='--', zorder=0)

        axs[0].legend(loc='upper left', fontsize=self.legend_fontsize)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)
        else:
            plt.show()

        return stats

    def plot_Mstar_V_QvS(self, filename: str = None):

        fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True, dpi=100)

        for ax in axs:
            ax.set_xlabel(r'Log(M$_*$/M$_\odot)$', fontsize=self.axis_fontsize)
            ax.tick_params(which='both', labelsize=self.tick_fontsize)

        axs[0].set_ylabel(rf'$Q_*$ ({self.suptitle})', fontsize=self.axis_fontsize)
        axs[1].set_ylabel(rf'$S_*$ ({self.suptitle})', fontsize=self.axis_fontsize)

        stats = {}  # Store statistics for each mask and parameter
        for mask_name, mask in self.masks.items():
            color = self.colors[mask_name]
            # B_s analysis
            axs[0].scatter(self.data['masses'][mask], self.data['B_s'][mask],
                           c=color, label=self.labels[mask_name],
                           s=self.point_size * 1.5, alpha=1, zorder=10, edgecolors='white', marker='o')



            # C_s analysis
            axs[1].scatter(self.data['masses'][mask], self.data['C_s'][mask],
                           c=color, s=self.point_size * 1.5, alpha=1, zorder=10, edgecolors='white', marker='o')

            if self.plot_regression:
                print(f"\nAnalysis for {mask_name} - Q_*:")
                stats[f"{mask_name}_B_s"] = add_regression_analysis(
                    axs[0], self.data['masses'][mask], self.data['B_s'][mask], color
                )

                print(f"\nAnalysis for {mask_name} - S_*:")
                stats[f"{mask_name}_C_s"] = add_regression_analysis(
                    axs[1], self.data['masses'][mask], self.data['C_s'][mask], color
                )




        # add shaded boxes showing the range of the ellipse values for Q and S for different mass bins
        # ellipse_data = {
        #     'KF High Mass': {  # 10^9.0 M_sun < M_sstarf < 10^9.6 M_sun
        #         '1R_eff': {'center': (0.863, 0.297), 'std': (0.075, 0.103)},
        #         '2R_eff': {'center': (0.91, 0.352), 'std': (0.04, 0.155)}
        #     },
        #     'KF Medium Mass': {  # 10^8.5 M_sun < M_sstarf < 10^9.0 M_sun
        #         '1R_eff': {'center': (0.857, 0.323), 'std': (0.05, 0.118)},
        #         '2R_eff': {'center': (0.902, 0.371), 'std': (0.05, 0.163)}
        #     },
        #     'KF Low Mass': {  # 10^7.0 M_sun < M_sstarf < 10^8.5 M_sun
        #         '1R_eff': {'center': (0.753, 0.459), 'std': (0.087, 0.194)},
        #         '2R_eff': {'center': (0.778, 0.446), 'std': (0.102, 0.213)}
        #     }
        # }
        #
        # # Colors for each mass bin
        # ell_colors = {
        #     'KF Low Mass': 'indigo',
        #     'KF Medium Mass': 'darkred',
        #     'KF High Mass': 'limegreen'
        # }

        #creat dict with extents of shaded boxes in x-axis

        # Choose which R_eff data to use
        if self.plot_KF_data == False:
            r_eff_key = None
        elif self.data['reff_multi'] == 1:
            r_eff_key = '1R_eff'
        elif self.data['reff_multi']==2:
            r_eff_key = '2R_eff'
        else:
            r_eff_key = None

        extents = {'KF High Mass': (9.0, 9.6), 'KF Medium Mass': (8.5, 9.0), 'KF Low Mass': (7.0, 8.5)}
        if r_eff_key:
            for mass_bin, color in ell_colors.items():
                data = ellipse_data[mass_bin][r_eff_key]
                #get the x-axis extent of the shaded box
                extent = extents[mass_bin]
                y_extent = (data['center'][0] - data['std'][0], data['center'][0] + data['std'][0])
                #add shaded box to the plot
                axs[0].fill_between(extent, y_extent[0], y_extent[1], color=color, alpha=0.3, zorder=0,label = mass_bin)

                #get the y-axis extent of the shaded box
                y_extent = (data['center'][1] - data['std'][1], data['center'][1] + data['std'][1])
                #add shaded box to the plot
                axs[1].fill_between(extent, y_extent[0], y_extent[1], color=color, alpha=0.3, zorder=0)

        axs[0].set_ylim([0, 1])
        axs[0].legend(loc='best', fontsize=self.legend_fontsize)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)
        else:
            plt.show()

        return stats

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

    def plot_angle_distributions(self, filename: Optional[str] = None):
        """
        Plot histograms of the orientation angles between dark matter and stellar components
        """
        diffs_at_Reff = self.data['diffs_at_reff']

        # Create figure with subplots
        fig, ((ax2, ax3, ax4)) = plt.subplots(1, 3, figsize=(20, 5), sharey=False, dpi=300)
        # fig.suptitle(
        #     f'Distribution of Alignment Angles between Dark Matter and Stellar Components ({self.suptitle})',
        #     fontsize=self.axis_fontsize)

        # Extract all angles
        abs_angles = np.array([d['absolute_angle'] for d in diffs_at_Reff])
        axis_a_angles = np.array([d['axis_a_angle'] for d in diffs_at_Reff])
        axis_b_angles = np.array([d['axis_b_angle'] for d in diffs_at_Reff])
        axis_c_angles = np.array([d['axis_c_angle'] for d in diffs_at_Reff])
        halo_ids = np.array([d['halo_id'] for d in diffs_at_Reff])

        # Common histogram parameters
        bins = np.arange(0, 91, 10)  # 10-degree bins from 0 to 90
        # bins = 10

        # Set up axes
        axes = [
            # (ax1, abs_angles, 'Absolute Orientation Angle'),
            (ax2, axis_a_angles, 'Major Axis (A) Alignment'),
            (ax3, axis_b_angles, 'Intermediate Axis (B) Alignment'),
            (ax4, axis_c_angles, 'Minor Axis (C) Alignment')
        ]

        # Plot for each mask
        for mask_name, mask in self.masks.items():
            print(mask_name)
            print(np.sum(mask))
            color = self.colors[mask_name]
            label = self.labels[mask_name]

            # Create mask for the angles based on halo IDs
            angle_mask = mask

            for ax, angles, title in axes:
                masked_angles = angles[angle_mask]

                # Calculate histogram data
                hist_values, bin_edges = np.histogram(
                    masked_angles,
                    bins=bins,
                    density=False
                )

                # For the second mask, make the counts negative
                if list(self.masks.keys()).index(mask_name) == 1:
                    hist_values = -hist_values

                # Plot histogram
                ax.bar(bin_edges[:-1],
                       hist_values,
                       width=np.diff(bin_edges),
                       alpha=1,
                       color=color,
                       edgecolor='white',
                       label=f"{label}",  # \nμ={np.nanmean(masked_angles):.1f}°
                       linewidth=1.5,
                       align='edge'
                       )

                # Add mean line (adjust for second mask)
                mean_val = np.nanmean(masked_angles)
                # ax.axvline(mean_val, color=color, linestyle='--', alpha=0.5)

                # Set labels and title
                ax.set_title(title, fontsize=self.legend_fontsize)
                ax.tick_params(labelsize=self.tick_fontsize)

                #set y axis ticks to multiples of 5
                ax.yaxis.set_major_locator(plt.MultipleLocator(5))

                # Adjust y-axis labels to show absolute values
                yticks = ax.get_yticks()
                #if too many labels change to multiples of 10
                if len(yticks) > 10:
                    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
                    yticks = ax.get_yticks()

                yticks = [int(tick) for tick in yticks]
                ax.set_yticks(yticks)
                ax.set_yticklabels([str(abs(int(tick))) for tick in yticks])

        ax2.set_ylabel('Count', fontsize=self.axis_fontsize)
        ax2.legend(fontsize=self.legend_fontsize)
        # delta angle in phi, theta, psi in degrees
        angle_labels = [r'$\Delta \theta^{\circ}$', r'$\Delta \phi^{\circ}$',
                        r'$\Delta \psi^{\circ}$']
        # or alternatively:
        # angle_labels = [r'$\Delta \phi^\circ$', r'$\Delta \theta^\circ$', r'$\Delta \psi^\circ$']

        for ax, angle_label in zip([ax2, ax3, ax4], angle_labels):
            ax.set_xlabel(angle_label, fontsize=self.axis_fontsize)

        # Add overall statistics
        stats_text = []
        for mask_name, mask in self.masks.items():
            n_halos = np.sum(mask)
            stats = f"{self.labels[mask_name]}:\n"
            stats += f"N = {n_halos}\n"
            stats += f"Abs: {np.nanmean(abs_angles[mask]):.1f}° ± {np.nanstd(abs_angles[mask]):.1f}°\n"
            stats += f"a: {np.nanmean(axis_a_angles[mask]):.1f}° ± {np.nanstd(axis_a_angles[mask]):.1f}°\n"
            stats += f"b: {np.nanmean(axis_b_angles[mask]):.1f}° ± {np.nanstd(axis_b_angles[mask]):.1f}°\n"
            stats += f"c: {np.nanmean(axis_c_angles[mask]):.1f}° ± {np.nanstd(axis_c_angles[mask]):.1f}°\n"
            stats_text.append(stats)
            # print number of galaxies with less than 10 degree difference
            print(len(axis_a_angles[mask]))
            print(f'{self.labels[mask_name]}: {np.sum(axis_a_angles[mask] < 10)} a: less than 10 degree difference')
            print(f'{self.labels[mask_name]}: {np.sum(axis_b_angles[mask] < 10)} b: less than 10 degree difference')
            print(f'{self.labels[mask_name]}: {np.sum(axis_c_angles[mask] < 10)} c: less than 10 degree difference')

        # Add statistics text to figure
        # fig.text(0.02, 0.98, '\n\n'.join(stats_text),
        #          fontsize=self.tick_fontsize,
        #          va='top',
        #          ha='left')
        print(stats)

        # Adjust layout
        plt.tight_layout()

        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)
        else:
            plt.show()

        # Set up color dictionary
        colors = {
            'Oblate': 'teal',
            'Prolate': 'red',
            'Spherical': 'mediumspringgreen',
            'Other': 'xkcd:dark grey'
        }
        # Create masks
        spherical_mask = ((self.data['B_s'] > 0.8) & (self.data['C_s'] > 0.8)) | (
                (self.data['B_d'] > 0.8) & (self.data['C_d'] > 0.8))
        oblate_mask = ((self.data['T_s'] < 1 / 3) | (self.data['T_d'] < 1 / 3)) & ~spherical_mask
        prolate_mask = ((self.data['T_s'] > 2 / 3) | (self.data['T_d'] > 2 / 3)) & ~spherical_mask & ~oblate_mask
        other_mask = ~(spherical_mask | prolate_mask | oblate_mask)

        # Set up mask dictionary - split into positive and negative groups
        positive_masks = {
            'Prolate': prolate_mask,
            'Other': other_mask
        }
        negative_masks = {
            'Oblate': oblate_mask,
            'Spherical': spherical_mask
        }

        # Create figure with subplots
        fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(20, 5), sharey=False, dpi=300)
        axes = [
            (ax2, axis_a_angles, 'Major Axis (A) Alignment'),
            (ax3, axis_b_angles, 'Intermediate Axis (B) Alignment'),
            (ax4, axis_c_angles, 'Minor Axis (C) Alignment')
        ]

        # First plot positive masks
        for mask_name in positive_masks:
            mask = positive_masks[mask_name]
            color = colors[mask_name]

            count = np.sum(mask)
            label = f"{mask_name}"  # N = {count}"

            for ax, angles, title in axes:
                masked_angles = angles[mask]

                hist_values, bin_edges = np.histogram(
                    masked_angles,
                    bins=bins,
                    density=False
                )

                # Plot histogram
                ax.bar(bin_edges[:-1],
                       hist_values,
                       width=np.diff(bin_edges),
                       color=color,
                       alpha=1,
                       edgecolor='white',
                       label=label,
                       linewidth=1,
                       align='edge'
                       )

        # Then plot negative masks
        for mask_name in negative_masks:
            mask = negative_masks[mask_name]
            color = colors[mask_name]

            count = np.sum(mask)
            label = f"{mask_name}"  # N = {count}"

            for ax, angles, title in axes:
                masked_angles = angles[mask]

                hist_values, bin_edges = np.histogram(
                    masked_angles,
                    bins=bins,
                    density=False
                )

                # Make these counts negative
                hist_values = -hist_values

                # Plot histogram
                ax.bar(bin_edges[:-1],
                       hist_values,
                       width=np.diff(bin_edges),
                       color=color,
                       alpha=1,
                       edgecolor='white',
                       label=label,
                       linewidth=1.5,
                       align='edge'
                       )

                # Set labels and title
                ax.set_title(title, fontsize=self.legend_fontsize)
                ax.tick_params(labelsize=self.tick_fontsize)

                #set y axis ticks to multiples of 5
                #ax.yaxis.set_major_locator(plt.MultipleLocator(5))
                # Adjust y-axis labels to show absolute values
                yticks = ax.get_yticks()
                #convert yticks to ints
                yticks = [int(tick) for tick in yticks]
                ax.set_yticks(yticks)
                ax.set_yticklabels([str(abs(int(tick))) for tick in yticks])

        # Create two legends: one at the top and one at the bottom
        for ax in [ax2]:
            # Get the current handles and labels
            handles, labels = ax.get_legend_handles_labels()

            # Split them into positive and negative groups
            pos_handles = handles[:len(positive_masks)]
            pos_labels = labels[:len(positive_masks)]
            neg_handles = handles[len(positive_masks):]
            neg_labels = labels[len(positive_masks):]

            # Create two legends
            leg1 = ax.legend(pos_handles, pos_labels,
                             loc='upper right',
                             fontsize=self.legend_fontsize,
                             bbox_to_anchor=(1, 1))
            leg2 = ax.legend(neg_handles, neg_labels,
                             loc='lower right',
                             fontsize=self.legend_fontsize,
                             bbox_to_anchor=(1, 0))

            # Add both legends
            ax.add_artist(leg1)
            ax.add_artist(leg2)

        # Set common labels
        ax2.set_ylabel('Count', fontsize=self.axis_fontsize)

        # Set angle labels
        angle_labels = [r'$\Delta \theta^{\circ}$', r'$\Delta \phi^{\circ}$', r'$\Delta \psi^{\circ}$']
        for ax, angle_label in zip([ax2, ax3, ax4], angle_labels):
            ax.set_xlabel(angle_label, fontsize=self.axis_fontsize)
        plt.tight_layout()
        # save
        if filename:
            # edit filename
            # remove .png at the end
            filename = filename[:-4]
            filename = filename + '_split.png'
            plt.savefig(filename, bbox_inches='tight', pad_inches=.1, dpi=300)

        return fig







