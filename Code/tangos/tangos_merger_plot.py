
from tangos.relation_finding import MultiHopMostRecentMergerStrategy
import tangos
from collections import defaultdict
#import networkx as nx
import matplotlib.pyplot as plt
import time
import mytangosproperty


def smooth_shape(rbins, ba, ca):
    k = 4
    s_factor = 1
    """
    Smooth and filter data, handling a few NaN values gracefully.

    Parameters:
    rbins, ba, ca: array-like, input data
    k: int, degree of the smoothing spline (default 3)
    s_factor: float, smoothing factor as a fraction of len(rbins) (default 0.01)
    residual_threshold, jump_threshold, jump_percentage: unused in this version

    Returns:
    rbins, ba, ca: filtered arrays
    ba_s, ca_s: smoothed spline functions
    """
    import numpy as np
    from scipy.interpolate import UnivariateSpline

    # Remove rows where either ba or ca is NaN
    mask = ~np.isnan(ba) & ~np.isnan(ca)
    rbins_filtered = rbins[mask]
    ba_filtered = ba[mask]
    ca_filtered = ca[mask]

    # if there are no values, return the original values
    if len(rbins_filtered) == 0:
        return rbins, ba, ca, SmoothAxisRatio.nan_func, SmoothAxisRatio.nan_func

    # Calculate smoothing parameter
    s = s_factor * len(rbins_filtered)

    # Create splines
    # print(rbins_filtered, ba_filtered)
    ba_s = UnivariateSpline(rbins_filtered, ba_filtered, k=k, s=s)
    ca_s = UnivariateSpline(rbins_filtered, ca_filtered, k=k, s=s)

    # Print some diagnostic information
    # print(f"Total data points: {len(rbins)}")
    # print(f"Data points after NaN removal: {len(rbins_filtered)}")
    # print(f"NaN percentage: {(1 - len(rbins_filtered)/len(rbins))*100:.2f}%")

    n = len(rbins_filtered)
    # calculate residuals and remove outliers
    ba_residuals = ba_filtered - ba_s(rbins_filtered)
    ca_residuals = ca_filtered - ca_s(rbins_filtered)
    # calculate the standard deviation of the residuals
    ba_std = np.std(ba_residuals)
    ca_std = np.std(ca_residuals)
    # remove outliers
    d = 3

    mask = np.abs(ba_residuals) < d * ba_std

    rbins_filtered = rbins_filtered[mask]
    ba_filtered = ba_filtered[mask]
    ca_filtered = ca_filtered[mask]
    mask = np.abs(ca_residuals[mask]) < d * ca_std
    rbins_filtered = rbins_filtered[mask]
    ba_filtered = ba_filtered[mask]
    ca_filtered = ca_filtered[mask]
    # Recreate splines
    ba_s = UnivariateSpline(rbins_filtered, ba_filtered, k=k, s=s)
    ca_s = UnivariateSpline(rbins_filtered, ca_filtered, k=k, s=s)

    # remove any points that are isolated in space
    # calculate the difference between each point

    diff = np.diff(rbins_filtered, prepend=0)
    # print(diff)
    # mask isolated points
    mask = diff > 1
    # print(mask)
    # print(rbins_filtered[mask])
    # print(diff[mask])
    rbins_filtered = rbins_filtered[~mask]
    ba_filtered = ba_filtered[~mask]
    ca_filtered = ca_filtered[~mask]
    # Recreate splines
    ba_s = UnivariateSpline(rbins_filtered, ba_filtered, k=k, s=s)
    ca_s = UnivariateSpline(rbins_filtered, ca_filtered, k=k, s=s)
    # Print some diagnostic information
    # print(f"Data points after outlier removal: {len(rbins_filtered)}")
    # print(f"Outlier percentage: {(1 - len(rbins_filtered)/len(rbins))*100:.2f}%")

    # def clip_function(func):
    #     def clipped(x):
    #         return np.clip(func(x), 0, 1)
    #
    #     return clipped
    #
    # # clip the function to 0,1
    # ba_s_c = clip_function(ba_s)
    # ca_s_c = clip_function(ca_s)

    return rbins_filtered, ba_filtered, ca_filtered, ba_s, ca_s




class HaloNode:
    def __init__(self, halo):
        self.halo = halo
        # self.child = None
        self.progenitors = []
        self.ndm = halo.NDM
        self.mvir = halo['Mvir']
        self.mstar = halo['Mstar']
        self.reff = halo['reff'] #change this to change where shapes are measured
        print(self.reff)
        if self.reff is not None:
            self.reff = self.reff*2
        self.merger_time = None
        self.merger_ratio = None

        self.is_merger = False
        self.merger_ratio = None

        #try:
        rbins_s, ba_s, ca_s = halo['rbins_s'], halo['ba_s'], halo['ca_s']
        rbins_filtered, ba_filtered, ca_filtered, ba_s, ca_s = smooth_shape(rbins_s, ba_s, ca_s)
        self.ba_s = ba_s(self.reff)
        self.ca_s = ca_s(self.reff)

        rbins_d, ba_d, ca_d = halo['rbins_d'], halo['ba_d'], halo['ca_d']
        rbins_filtered, ba_filtered, ca_filtered, ba_d, ca_d = smooth_shape(rbins_d, ba_d, ca_d)
        self.ba_d = ba_d(self.reff)
        self.ca_d = ca_d(self.reff)


    def calculate_merger_ratio(self):
        if len(self.progenitors) > 1:
            self.is_merger = True
            main_progenitor_mass = max(p.mvir for p in self.progenitors)
            sum_other_masses = sum(p.mvir for p in self.progenitors if p.mvir != main_progenitor_mass)
            if sum_other_masses > 0:
                self.merger_ratio = main_progenitor_mass / sum_other_masses
            else:
                self.merger_ratio = None  # or some other value to indicate division by zero
                self.is_merger = False
        else:
            self.is_merger = False
            self.merger_ratio = None


def build_merger_tree(input_halo, max_depth=10, min_fractional_weight=0.01, min_fractional_NDM=0.01, timeout=600):
    """
    Build a merger tree starting from the input halo.
    :param input_halo: Halo to build the tree from
    :param max_depth: Maximum number of generations to go back
    :param min_fractional_weight: Minimum fractional weight to consider a progenitor
    :param min_fractional_NDM: Minimum fractional NDM to consider a progenitor
    :param timeout: Maximum time (in seconds) to spend building the tree
    :return: Dictionary representing the merger tree
    """
    start_time = time.time()
    tree = defaultdict(list)
    start_node = HaloNode(input_halo)
    tree[0].append(start_node)

    # Use MultiHopAllProgenitorsStrategy to get all progenitors
    strategy = tangos.relation_finding.MultiHopAllProgenitorsStrategy(input_halo, nhops_max=max_depth)
    link_objs = strategy._get_query_all()

    # Create a cache of links
    link_cache = defaultdict(list)
    for obj in link_objs:
        link_cache[obj.halo_from_id].append(obj)

    def build_subtree(node, depth):
        if depth >= max_depth or time.time() - start_time > timeout:
            return

        link_objs = link_cache.get(node.halo.id, [])
        max_NDM = max((o.halo_to.NDM for o in link_objs), default=0)

        for obj in link_objs:
            progenitor = obj.halo_to
            max_weight = max(o.weight for o in link_objs if o.halo_from_id == obj.halo_from_id)

            if (obj.weight > max_weight * min_fractional_weight and
                    progenitor.NDM > min_fractional_NDM * max_NDM):

                progenitor_node = HaloNode(progenitor)
                progenitor_node.weight = obj.weight
                # before adding progenitor halo to the list of progenitors, check if it is already in the list
                # check progenitor_node.halo.halo_number against [p.halo.halo_number for p in node.progenitors]
                if progenitor_node.halo.halo_number not in [p.halo.halo_number for p in node.progenitors]:
                    node.progenitors.append(progenitor_node)
                    tree[depth + 1].append(progenitor_node)

                    build_subtree(progenitor_node, depth + 1)

        # After adding all progenitors, calculate the merger ratio
        node.calculate_merger_ratio()
        if node.is_merger:
            node.merger_time = node.halo.timestep.time_gyr

        # Return the main progenitor (the one with the highest mass)
        return max(node.progenitors, key=lambda x: x.mvir) if node.progenitors else None

    main_line = [start_node]
    current_node = start_node
    while current_node:
        current_node = build_subtree(current_node, len(main_line))
        if current_node:
            main_line.append(current_node)

    return tree, main_line


def print_merger_tree(tree):
    for depth, nodes in tree.items():
        print(f"Depth {depth}:")
        for node in nodes:
            print(f"  Halo {node.halo.halo_number} at time {node.halo.timestep.time_gyr:.2f} Gyr")
            print(f"    Mvir: {node.mvir:.2e} Msun, NDM: {node.ndm}")
            if node.ba_s is not None:
                print(f"    Shape (s): ba={node.ba_s:.2f}, ca={node.ca_s:.2f}")
                print(f"    Shape (d): ba={node.ba_d:.2f}, ca={node.ca_d:.2f}")
            if node.progenitors:
                # print("    Progenitors:", [prog.halo.halo_number for prog in node.progenitors])
                print("    Progenitors:", [prog.halo for prog in node.progenitors])
            if node.is_merger:
                print(f"    Merger at {node.merger_time} Gyr with ratio {node.merger_ratio}")
        print()


def visualize_tree(tree):
    G = nx.DiGraph()
    time_to_nodes = defaultdict(list)

    # Create nodes and organize by time
    for depth, nodes in tree.items():
        for node in nodes:
            time = round(node.halo.timestep.time_gyr, 2)  # Round to 2 decimal places
            G.add_node(node.halo.id, time=time, mvir=node.mvir)
            time_to_nodes[time].append(node.halo.id)
            for prog in node.progenitors:
                G.add_edge(prog.halo.id, node.halo.id)

    # Calculate positions
    pos = {}
    times = sorted(time_to_nodes.keys(), reverse=True)
    for i, time in enumerate(times):
        nodes = time_to_nodes[time]
        y = 1 - (i / (len(times) - 1))  # Normalize y position
        for j, node in enumerate(nodes):
            x = (j + 1) / (len(nodes) + 1)  # Distribute nodes horizontally
            pos[node] = (x, y)

    plt.figure(figsize=(5, 12))

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True, edge_color='gray', arrowsize=20)

    # Draw nodes
    node_sizes = [300 * (G.nodes[node]['mvir'] / max(nx.get_node_attributes(G, 'mvir').values())) ** 0.5 for node in
                  G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_sizes)

    # Draw labels
    nx.draw_networkx_labels(G, pos, {node: f"{node}\n{G.nodes[node]['time']:.2f} Gyr" for node in G.nodes}, font_size=8)

    # Add time labels on the y-axis
    plt.yticks([(1 - (i / (len(times) - 1))) for i in range(len(times))], [f"{time:.2f} Gyr" for time in times])

    plt.title("Time-Organized Merger Tree Visualization")
    plt.xlabel("Halo ID")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


def T(ba, ca):
    return ((1 - ba ** 2) / (1 - ca ** 2))


def plot_ba_ca(main_line, figure_folder, sim_name):
    times = [node.halo.timestep.time_gyr for node in main_line]
    ba_s = [node.ba_s for node in main_line]
    ca_s = [node.ca_s for node in main_line]
    ba_d = [node.ba_d for node in main_line]
    ca_d = [node.ca_d for node in main_line]

    fig = plt.figure(figsize=(12, 10))
    ms = 10
    scatter_s = plt.scatter(ba_s, ca_s, c=times, cmap='viridis', marker='*', label='Stellar')
    scatter_d = plt.scatter(ba_d, ca_d, c=times, cmap='viridis', marker='o', label='Dark Matter')
    plt.colorbar(scatter_s, label='Time (Gyr)')

    # Connect points with lines
    plt.plot(ba_s, ca_s, '-', color='k', alpha=0.5)
    plt.plot(ba_d, ca_d, '-', color='k', alpha=0.5)

    # Mark merger events
    for node in main_line:
        if node.is_merger:
            plt.plot(node.ba_s, node.ca_s, '*', markersize=ms, mfc='none', mec='black')
            plt.plot(node.ba_d, node.ca_d, 'o', markersize=ms, mfc='none', mec='black')

    # label merger events with merger ratio
    for node in main_line:
        if node.is_merger:
            plt.text(node.ba_s, node.ca_s, f'{node.merger_ratio:.1f}', fontsize=12)
            plt.text(node.ba_d, node.ca_d, f'{node.merger_ratio:.1f}', fontsize=12)

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    #plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    plt.xlabel('B/A')
    plt.ylabel('C/A')
    plt.title(sim_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    fig.savefig(figure_folder + 'QvS_mergers.png')
    #plt.close()

    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    ax[0].plot(times, ba_s, label='Stellar', color='blue', ls='--', marker='o')
    ax[0].plot(times, ba_d, label='Dark Matter', color='k', ls='--', marker='o')
    ax[0].set_ylabel('B/A')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[0].set_ylim(0, 1)

    ax[1].plot(times, ca_s, label='Stellar', color='green', ls='--', marker='o')
    ax[1].plot(times, ca_d, label='Dark Matter', color='k', ls='--', marker='o')
    ax[1].set_ylabel('C/A')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    ax[1].set_ylim(0, 1)

    # add vertical lines for merger events label with merger ratio

    for node in main_line:
        if node.is_merger and node.merger_ratio is not None:
            print(node.is_merger, node.merger_time, node.merger_ratio)
            ax[0].axvline(node.merger_time, color='red', linestyle='--')
            ax[1].axvline(node.merger_time, color='red', linestyle='--')
            ax[0].text(node.merger_time + .02, 0.1, f'{node.merger_ratio:.1f}', rotation=90, verticalalignment='center',
                       fontsize=12)
            ax[1].text(node.merger_time + .02, 0.9, f'{node.merger_ratio:.1f}', rotation=90, verticalalignment='center',
                       fontsize=12)

    # triaxiality plot vs time for stellar and dark matter components

    # fig,ax = plt.subplots(1,1,figsize=(10, 4), sharex=True)

    ba_s = np.array(ba_s)
    ca_s = np.array(ca_s)
    ba_d = np.array(ba_d)
    ca_d = np.array(ca_d)

    T_s = T(ba_s, ca_s)
    T_d = T(ba_d, ca_d)
    ax = ax[2]
    ax.plot(times, T_s, label='Stellar', color='blue', ls='--', marker='o')
    ax.plot(times, T_d, label='Dark Matter', color='k', ls='--', marker='o')
    # mark merger events
    for node in main_line:
        if node.is_merger:
            ax.axvline(node.merger_time, color='red', linestyle='--')
            ax.text(node.merger_time + .02, 0.1, f'{node.merger_ratio:.1f}', rotation=90, verticalalignment='center',
                    fontsize=12)

    text_x_position = np.min(times) - .07
    ax.plot([4, 14], [1 / 3, 1 / 3], c='.75', linestyle='--', zorder=0)
    ax.plot([4, 14], [2 / 3, 2 / 3], c='.75', linestyle='--', zorder=0)
    ax.tick_params(which='both', labelsize=15)
    ax.text(text_x_position, 1 / 6, 'Oblate', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
    ax.text(text_x_position, 3 / 6, 'Triaxial', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
    ax.text(text_x_position, 5 / 6, 'Prolate', fontsize=12, rotation='vertical', verticalalignment='center', c='k')
    ax.set_xlim(np.min(times) - .1, np.max(times))
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_ylabel('T', fontsize=20)

    # convert times to redshift for x-tick labels
    redshifts = [node.halo.timestep.redshift for node in main_line]

    ax.set_xticks(times)
    # for the labels display as floats with 2 decimal places
    redshifts = [f'{redshift:.2f}' for redshift in redshifts]

    ax.set_xticklabels(redshifts)
    ax.set_xlabel('Redshift', fontsize=20)

    fig.suptitle(sim_name, fontsize=20)

    fig.savefig(figure_folder + 'TvsTime_mergers.png')
    plt.show(fig)
    print(times, T_s)


def plot_ba_ca_smoothed(main_line, figure_folder, sim_name):
    times = [node.halo.timestep.redshift for node in main_line]
    # get halos
    halos = [node.halo for node in main_line]
    ba_s = [halo['ba_s'] for halo in halos]
    ca_s = [halo['ca_s'] for halo in halos]
    ba_d = [halo['ba_d'] for halo in halos]
    ca_d = [halo['ca_d'] for halo in halos]
    ba_s_smoothed = [halo.calculate('ba_s_smoothed()') for halo in halos]
    ca_s_smoothed = [halo.calculate('ca_s_smoothed()') for halo in halos]
    ba_d_smoothed = [halo.calculate('ba_d_smoothed()') for halo in halos]
    ca_d_smoothed = [halo.calculate('ca_d_smoothed()') for halo in halos]
    reff = [halo['reff'] for halo in halos]
    rbins = [halo['rbins'] for halo in halos]

    # get number of halos
    n = len(halos)
    # create a plot with n rows and 2 columns
    # plot rbins vs ba_s and ba_s_smoothed, and ca_s and ca_s_smoothed
    # set height based on n
    height = 2 * n
    fig, axs = plt.subplots(n, 2, figsize=(15, height), sharex='col', sharey='row')
    axs = axs.T
    rmax = np.max(rbins[-1])

    for i in range(n):
        ax = axs[:, i]
        # mark merger events
        if main_line[i].is_merger:
            # add background color to plot to highlight merger events
            ax[0].set_facecolor('lightgray')
            ax[1].set_facecolor('lightgray')

        ax[0].grid()
        ax[1].grid()
        ax[0].set_title(f'z = {times[i]:.2f}')
        ax[1].set_title(f'z = {times[i]:.2f}')
        ax[0].vlines(reff[i], 0, 1, color='gray')
        ax[0].plot(rbins[i], ba_s[i], label='Stellar B/A', color='r')
        ax[0].plot(rbins[i], ba_s_smoothed[i], color='r', linestyle='--')
        ax[0].plot(rbins[i], ba_d[i], label='Dark B/A', color='k')
        ax[0].plot(rbins[i], ba_d_smoothed[i], label='Smoothed', color='k', linestyle='--')
        ax[0].set_ylim(0, 1)
        ax[1].vlines(reff[i], 0, 1, color='gray')
        ax[1].plot(rbins[i], ca_s[i], label='Stellar C/A', color='b')
        ax[1].plot(rbins[i], ca_s_smoothed[i], color='b', linestyle='--')
        ax[1].plot(rbins[i], ca_d[i], label='Dark C/A', color='k')
        ax[1].plot(rbins[i], ca_d_smoothed[i], label='ca_d_smoothed', color='k', linestyle='--')
        ax[1].set_ylim(0, 1)
        ax[0].set_ylabel('B/A')
        ax[1].set_ylabel('C/A')
        if i == 1:
            ax[0].legend(loc='lower right')
            ax[1].legend(loc='lower right')
        ax[0].set_xlim(0, rmax)
        ax[1].set_xlim(0, rmax)

    ax[0].set_xlabel('r [kpc]')
    ax[1].set_xlabel('r [kpc]')
    fig.suptitle(sim_name)
    plt.show()
    fig.savefig(figure_folder + 'ba_ca_smoothed.png')
    plt.close(fig)


def find_most_recent_merger(main_line):
    for i, node in enumerate(main_line):
        if node.is_merger:
            return i, node
    return None, None


import matplotlib.pyplot as plt
import numpy as np


# def plot_merger_ba_ca(main_lines, figure_folder, link_dm_to_stellar=True, link_timesteps=True):
#     width = 8
#     height = 8
#     fig, ax = plt.subplots(figsize=(width, height))
#
#     for idx, main_line in enumerate(main_lines):
#         merger_idx, merger_node = find_most_recent_merger(main_line)
#         if merger_idx is None:
#             print(f"No merger found for simulation {idx}")
#             continue
#
#         phase_list = main_line[-3:]  # Get the last three nodes
#         labels = [f'z = {abs(node.redshift):.2f}' for node in phase_list]
#         colors = ["#0072B2", "#E69F00", "#009E73"]
#
#         for phase, color, label in zip(phase_list, colors, labels):
#             if phase is not None:
#                 ba_s, ca_s = phase.ba_s, phase.ca_s
#                 ba_d, ca_d = phase.ba_d, phase.ca_d
#
#                 ax.scatter(ba_s, ca_s, c=color, marker='*', s=75, alpha=0.7, zorder=2)
#                 ax.scatter(ba_d, ca_d, c=color, marker='o', s=50, alpha=0.7, zorder=2)
#
#                 if phase.is_merger:
#                     ax.scatter(ba_s, ca_s, c='k', marker='*', s=125, facecolors='none', zorder=1)
#                     ax.scatter(ba_d, ca_d, c='k', marker='o', s=75, facecolors='none', zorder=1)
#
#                 if link_dm_to_stellar and ba_s is not None and ca_s is not None and ba_d is not None and ca_d is not None:
#                     ax.plot([ba_d, ba_s], [ca_d, ca_s], c=color, alpha=0.5, linestyle='--')
#
#         if link_timesteps:
#             ba_s = [phase.ba_s for phase in phase_list if phase is not None]
#             ca_s = [phase.ca_s for phase in phase_list if phase is not None]
#             ba_d = [phase.ba_d for phase in phase_list if phase is not None]
#             ca_d = [phase.ca_d for phase in phase_list if phase is not None]
#             ax.plot(ba_s, ca_s, c='gray', alpha=0.5, linestyle='-')
#             ax.plot(ba_d, ca_d, c='gray', alpha=0.5, linestyle='-')
#
#     # Add dummy points for legend
#     ax.scatter([], [], c='k', marker='*', label='Stellar')
#     ax.scatter([], [], c='k', marker='o', label='Dark Matter')
#     for color, label in zip(colors, labels):
#         ax.scatter([], [], c=color, marker='o', label=label)
#
#     ax.set_aspect('equal')
#     ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel('S = B/A', fontsize=20)
#     ax.set_ylabel('Q = C/A', fontsize=20)
#     ax.legend(loc='upper left', fontsize=15)
#     ax.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig(f'{figure_folder}/merger_ba_ca_all_sims.png', bbox_inches='tight', dpi=300)


import matplotlib.pyplot as plt
import numpy as np


def plot_Mstar_vs_T(main_lines, figure_folder, link_timesteps=True):
    width = 8
    height = 4
    fig, ax = plt.subplots(figsize=(width, height))

    for idx, main_line in enumerate(main_lines):
        merger_idx, merger_node = find_most_recent_merger(main_line)
        if merger_idx is None:
            print(f"No merger found for simulation {idx}")
            continue

        phase_list = main_line[-3:]  # Get the last three nodes
        labels = [f'z = {abs(node.redshift):.2f}' for node in phase_list]
        colors = ["#0072B2", "#E69F00", "#009E73"]

        for phase, color, label in zip(phase_list, colors, labels):
            if phase is not None:
                Mstar = phase.mstar
                T_s = T(phase.ba_s, phase.ca_s)
                T_d = T(phase.ba_d, phase.ca_d)

                ax.scatter(Mstar, T_s, c=color, marker='*', s=75, alpha=0.7, zorder=2)
                ax.scatter(Mstar, T_d, c=color, marker='o', s=50, alpha=0.7, zorder=2)

                if phase.is_merger:
                    ax.scatter(Mstar, T_s, c='k', marker='*', s=125, facecolors='none', zorder=1)
                    ax.scatter(Mstar, T_d, c='k', marker='o', s=75, facecolors='none', zorder=1)

        if link_timesteps:
            Mstar = [phase.mstar for phase in phase_list if phase is not None]
            T_s = [T(phase.ba_s, phase.ca_s) for phase in phase_list if phase is not None]
            T_d = [T(phase.ba_d, phase.ca_d) for phase in phase_list if phase is not None]
            ax.plot(Mstar, T_s, c='gray', alpha=0.5, linestyle='-')
            ax.plot(Mstar, T_d, c='gray', alpha=0.5, linestyle='-')

    ax.set_xlabel('Log Stellar Mass', fontsize=20)
    
    ax.set_ylabel('T', fontsize=20)
    ax.set_xscale('log')
    #dummy points for legend
    ax.scatter([], [], c='k', marker='*', label='Stellar')
    ax.scatter([], [], c='k', marker='o', label='Dark Matter')
    for color, label in zip(colors, labels):
        ax.scatter([], [], c=color, marker='o', label=label)

    ax.legend(loc='best', fontsize=15)
    
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{figure_folder}/Mstar_vs_T_all_sims.png', bbox_inches='tight', dpi=300)


import numpy as np

import numpy as np


def plot_merger_ba_ca(main_lines, figure_folder, link_dm_to_stellar=True, link_timesteps=True):
    width = 8
    height = 8
    axis_font_size = 33
    legend_font_size = 20
    tick_font_size = 20

    fig, ax = plt.subplots(figsize=(width, height))

    def get_min_merger_ratio(line):
        ratios = [node.merger_ratio for node in line if node.merger_ratio is not None and node.merger_ratio != np.inf]
        return min(ratios) if ratios else np.inf

    def scale_size(input_value, min_input=4.2, max_input=6, max_size=800, min_size=50):
        clamped_input = max(min_input, min(input_value, max_input))
        inverse_ratio = 1 - (clamped_input - min_input) / (max_input - min_input)
        scaled_size = min_size + inverse_ratio * (max_size - min_size)
        return round(scaled_size)

    # Define colors outside the loop to maintain consistency
    colors = ["#0072B2", "#E69F00", "#009E73"]

    for idx, main_line in enumerate(main_lines):
        merger_idx, merger_node = find_most_recent_merger(main_line)
        if merger_idx is None:
            print(f"No merger found for simulation {idx}")
            continue

        min_merger_ratio = get_min_merger_ratio(main_line)
        point_size = scale_size(min_merger_ratio)

        phase_list = main_line[-3:]  # Get the last three nodes
        labels = [f'z = {abs(node.redshift):.2f}' for node in phase_list]
        #print the range of ba_s and ca_s, ba_d and ca_d across the three phases
        print(f"Simulation {idx}")
        ba_d_list, ca_d_list, ba_s_list, ca_s_list = [], [], [], []
        for phase in phase_list:
            if phase is not None:
                ba_s_list.append(phase.ba_s)
                ca_s_list.append(phase.ca_s)
                ba_d_list.append(phase.ba_d)
                ca_d_list.append(phase.ca_d)
        print(f"ba_s: {min(ba_s_list):.2f} - {max(ba_s_list):.2f}")
        print(f"ca_s: {min(ca_s_list):.2f} - {max(ca_s_list):.2f}")
        print(f"ba_d: {min(ba_d_list):.2f} - {max(ba_d_list):.2f}")
        print(f"ca_d: {min(ca_d_list):.2f} - {max(ca_d_list):.2f}")



        if link_timesteps:
            ba_s = [phase.ba_s for phase in phase_list if phase is not None]
            ca_s = [phase.ca_s for phase in phase_list if phase is not None]
            ba_d = [phase.ba_d for phase in phase_list if phase is not None]
            ca_d = [phase.ca_d for phase in phase_list if phase is not None]
            ax.plot(ba_s, ca_s, c='gray', alpha=0.5, linestyle='-', zorder=0)
            ax.plot(ba_d, ca_d, c='gray', alpha=0.5, linestyle='-', zorder=0)

        for phase, color, label in zip(phase_list, colors, labels):
            if phase is not None:
                ba_s, ca_s = phase.ba_s, phase.ca_s
                ba_d, ca_d = phase.ba_d, phase.ca_d
                if phase.is_merger:
                    ax.scatter(ba_s, ca_s, edgecolors='k', marker='*', s=point_size, c=color, zorder=2, alpha=0.7, lw=2)
                    ax.scatter(ba_d, ca_d, edgecolors='k', marker='o', s=point_size * .8, c=color, zorder=2, alpha=0.7,
                               lw=2)

                    # Add annotations with arrows for merger events
                    # Stellar component annotation
                    # ax.annotate(
                    #     f'{phase.merger_ratio:.1f}',
                    #     xy=(ba_s, ca_s),  # point to annotate
                    #     xytext=(ba_s - 0.1, ca_s - 0),  # offset text position
                    #     fontsize=15,
                    #     color=color,
                    #     zorder=3,
                    #     arrowprops=dict(
                    #         arrowstyle='->',
                    #         color=color,
                    #         alpha=0.7
                    #     ),
                    #     bbox=dict(
                    #         facecolor='white',
                    #         edgecolor='none',
                    #         alpha=0.7,
                    #         pad=0.5
                    #     )
                    # )
                    print(f'{phase.merger_ratio:.1f}',phase.mstar,phase.redshift)

                    # # Dark matter component annotation
                    # ax.annotate(
                    #     f'{phase.merger_ratio:.1f}',
                    #     xy=(ba_d, ca_d),
                    #     xytext=(ba_d + 0.1, ca_d + 0.1),
                    #     fontsize=15,
                    #     color=color,
                    #     zorder=3,
                    #     arrowprops=dict(
                    #         arrowstyle='->',
                    #         color=color,
                    #         alpha=0.7
                    #     ),
                    #     bbox=dict(
                    #         facecolor='white',
                    #         edgecolor='none',
                    #         alpha=0.7,
                    #         pad=0.5
                    #     )
                    # )
                else:
                    ax.scatter(ba_s, ca_s, c=color, marker='*', s=point_size, zorder=1, alpha=0.7)
                    ax.scatter(ba_d, ca_d, c=color, marker='o', s=point_size * 0.8, zorder=1, alpha=0.7)

                if link_dm_to_stellar and ba_s is not None and ca_s is not None and ba_d is not None and ca_d is not None:
                    ax.plot([ba_d, ba_s], [ca_d, ca_s], c=color, alpha=0.5, linestyle='--')

    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)

    # Add dummy points for legend
    ax.scatter([], [], c='k', marker='*', s=75, label='Stellar')
    ax.scatter([], [], c='k', marker='o', s=60, label='Dark Matter')
    ax.scatter([], [], marker='o', s=90, facecolors='none', edgecolors='k', label='Merger')
    for color, label in zip(colors, labels):
        ax.scatter([], [], c=color, marker='o', s=75, label=label)

    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(False)

    ax.set_xlabel(r'$S = B/A$', fontsize=axis_font_size)
    ax.set_ylabel(r'$Q = C/A$', fontsize=axis_font_size)
    ax.plot([0, 1], [0, 1], c='0.5', linestyle='--')
    ax.legend(loc='upper left', fontsize=legend_font_size)

    plt.tight_layout()
    plt.savefig(f'{figure_folder}/merger_ba_ca_all_sims.png', bbox_inches='tight', dpi=300)

    
# def plot_merger_ba_ca(main_lines, figure_folder, link_dm_to_stellar=True, link_timesteps=True):
#     width = 10
#     height = 10
#     axis_font_size = 33
#     legend_font_size = 20
#     tick_font_size = 20
#
#     fig, ax = plt.subplots(figsize=(width, height))
#
#
#     def get_min_merger_ratio(line):
#         ratios = [node.merger_ratio for node in line if node.merger_ratio is not None and node.merger_ratio != np.inf]
#         return min(ratios) if ratios else np.inf
#
#     def scale_size(input_value, min_input=4.2, max_input=6, max_size=800, min_size=50):
#         # Ensure the input is within the expected range
#         clamped_input = max(min_input, min(input_value, max_input))
#
#         # Calculate the inverse ratio
#         inverse_ratio = 1 - (clamped_input - min_input) / (max_input - min_input)
#
#         # Use this inverse ratio to interpolate between min_size and max_size
#         scaled_size = min_size + inverse_ratio * (max_size - min_size)
#
#         return round(scaled_size)
#
#     for idx, main_line in enumerate(main_lines):
#         merger_idx, merger_node = find_most_recent_merger(main_line)
#         if merger_idx is None:
#             print(f"No merger found for simulation {idx}")
#             continue
#
#         min_merger_ratio = get_min_merger_ratio(main_line)
#         point_size = scale_size(min_merger_ratio)
#
#         phase_list = main_line[-3:]  # Get the last three nodes
#         labels = [f'z = {abs(node.redshift):.2f}' for node in phase_list]
#         colors = ["#0072B2", "#E69F00", "#009E73"]
#
#
#
#         if link_timesteps:
#             ba_s = [phase.ba_s for phase in phase_list if phase is not None]
#             ca_s = [phase.ca_s for phase in phase_list if phase is not None]
#             ba_d = [phase.ba_d for phase in phase_list if phase is not None]
#             ca_d = [phase.ca_d for phase in phase_list if phase is not None]
#             ax.plot(ba_s, ca_s, c='gray', alpha=0.5, linestyle='-',zorder=0)
#             ax.plot(ba_d, ca_d, c='gray', alpha=0.5, linestyle='-',zorder=0)
#
#         for phase, color, label in zip(phase_list, colors, labels):
#             if phase is not None:
#                 ba_s, ca_s = phase.ba_s, phase.ca_s
#                 ba_d, ca_d = phase.ba_d, phase.ca_d
#                 if phase.is_merger:
#                     #print reddshfits, halo numbers, merger ratios
#                     #print(phase.redshift, phase.mstar, phase.merger_ratio)
#                     ax.scatter(ba_s, ca_s, edgecolors = 'k', marker='*', s=point_size, c=color,zorder=2, alpha=0.7,lw = 2)
#                     ax.scatter(ba_d, ca_d, edgecolors = 'k', marker='o', s=point_size*.8, c=color,zorder=2, alpha=0.7,lw = 2)
#                 else:
#                     ax.scatter(ba_s, ca_s, c=color, marker='*', s=point_size, zorder=1,alpha=0.7)
#                     ax.scatter(ba_d, ca_d, c=color, marker='o', s=point_size * 0.8, zorder=1,alpha=0.7)
#
#
#
#                 if link_dm_to_stellar and ba_s is not None and ca_s is not None and ba_d is not None and ca_d is not None:
#                     ax.plot([ba_d, ba_s], [ca_d, ca_s], c=color, alpha=0.5, linestyle='--')
#         #for each simulation, add text with the smallest merger ratio, annotating the merger event
#         #find the largest merger ratio
#         max_merger_ratio = min([node.merger_ratio for node in main_line if node.merger_ratio is not None and node.merger_ratio != np.inf])
#         #find the node with the largest merger ratio
#         max_merger_node = [node for node in main_line if node.merger_ratio == max_merger_ratio]
#         #annotate the merger event with the largest merger ratio
#         ax.annotate(f'{max_merger_ratio:.1f}', (max_merger_node[0].ba_s, max_merger_node[0].ca_s), fontsize=15, color='black', zorder=3)
#         ax.annotate(f'{max_merger_ratio:.1f}', (max_merger_node[0].ba_d, max_merger_node[0].ca_d), fontsize=15, color='black', zorder=3)
#
#
#
#
#     ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
#     # Add dummy points for legend
#     ax.scatter([], [], c='k', marker='*', s=75, label='Stellar')
#     ax.scatter([], [], c='k', marker='o', s=60, label='Dark Matter')
#     ax.scatter([], [], marker='o', s=90, facecolors='none',edgecolors = 'k', label='Merger')
#     for color, label in zip(colors, labels):
#         ax.scatter([], [], c=color, marker='o', s=75, label=label)
#
#     ax.set_aspect('equal')
#     #ax.plot([0, 1], [0, 1], '--', color='gray', alpha=1,lw = 2)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0,1)
#     # ax.set_xlim(0, 1)
#     # ax.set_ylim(0, 1)
#     ax.grid(False)
#
#     ax.set_xlabel('S = B/A', fontsize=axis_font_size)
#     ax.set_ylabel('Q = C/A', fontsize=axis_font_size)
#     ax.legend(loc='upper left', fontsize=legend_font_size)
#
#
#     plt.tight_layout()
#     plt.savefig(f'{figure_folder}/merger_ba_ca_all_sims.png', bbox_inches='tight', dpi=300)


import numpy as np
import matplotlib.pyplot as plt


def plot_merger_T_d_vs_T_s(main_lines, figure_folder, link_dm_to_stellar=True, link_timesteps=True):
    width = 8
    height = 8
    fig, ax = plt.subplots(figsize=(width, height))

    def T(ba, ca):
        return (1 - ba ** 2) / (1 - ca ** 2)

    def get_min_merger_ratio(line):
        ratios = [node.merger_ratio for node in line if node.merger_ratio is not None and node.merger_ratio != np.inf]
        return min(ratios) if ratios else np.inf

    def scale_size(min_ratio, max_size=500, min_size=10):
        if min_ratio == np.inf:
            return min_size
        return max_size * (1 / (1 + min_ratio))  # Adjust this formula as needed

    for idx, main_line in enumerate(main_lines):
        merger_idx, merger_node = find_most_recent_merger(main_line)
        if merger_idx is None:
            print(f"No merger found for simulation {idx}")
            continue

        min_merger_ratio = get_min_merger_ratio(main_line)
        point_size = scale_size(min_merger_ratio)

        phase_list = main_line[-3:]  # Get the last three nodes
        labels = [f'z = {abs(node.redshift):.2f}' for node in phase_list]
        colors = ["#0072B2", "#E69F00", "#009E73"]

        for phase, color, label in zip(phase_list, colors, labels):
            if phase is not None:
                T_s = T(phase.ba_s, phase.ca_s) if phase.ba_s is not None and phase.ca_s is not None else None
                T_d = T(phase.ba_d, phase.ca_d) if phase.ba_d is not None and phase.ca_d is not None else None

                if T_s is not None and T_d is not None:
                    ax.scatter(T_d, T_s, c=color, marker='o', s=point_size, alpha=0.7, zorder=2)

                    if phase.is_merger:
                        ax.scatter(T_d, T_s, c='k', marker='o', s=point_size * 1.5, facecolors='none', zorder=1)

        if link_timesteps:
            T_s_list = [T(phase.ba_s, phase.ca_s) for phase in phase_list if
                        phase is not None and phase.ba_s is not None and phase.ca_s is not None]
            T_d_list = [T(phase.ba_d, phase.ca_d) for phase in phase_list if
                        phase is not None and phase.ba_d is not None and phase.ca_d is not None]
            if len(T_s_list) > 1 and len(T_d_list) > 1:
                ax.plot(T_d_list, T_s_list, c='gray', alpha=0.5, linestyle='-')

    # Add dummy points for legend
    for color, label in zip(colors, labels):
        ax.scatter([], [], c=color, marker='o', s=75, label=label)

    ax.set_aspect('equal')
    ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    max_T = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(0, max_T)
    ax.set_ylim(0, max_T)
    ax.set_ylabel(r'T$_*$', fontsize=20)
    ax.set_xlabel('T$_{DM}$', fontsize=20)
    ax.legend(loc='upper left', fontsize=15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{figure_folder}/merger_T_d_vs_T_s_all_sims.png', bbox_inches='tight', dpi=300)



    
import matplotlib.pyplot as plt
import numpy as np

def plot_merger_ratio_vs_shape(main_lines, figure_folder, link_timesteps=True):
    width = 8
    height = 4

    # Plot merger ratio vs ba_d/ba_s
    fig, ax = plt.subplots(figsize=(width, height))
    for idx, main_line in enumerate(main_lines):
        merger_idx, merger_node = find_most_recent_merger(main_line)
        if merger_idx is None:
            print(f"No merger found for simulation {idx}")
            continue

        phase_list = main_line[-3:]  # Get the last three nodes
        colors = ["#0072B2", "#E69F00", "#009E73"]

        for phase, color in zip(phase_list, colors):
            if phase is not None and phase.is_merger:
                ba_ratio = phase.ba_d / phase.ba_s if phase.ba_s else np.nan
                ax.scatter(phase.merger_ratio, ba_ratio, c=color, marker='o', s=50, alpha=0.7, zorder=2)

        if link_timesteps:
            merger_ratios = [phase.merger_ratio for phase in phase_list if phase.is_merger]
            ba_ratios = [phase.ba_d / phase.ba_s if phase.ba_s else np.nan for phase in phase_list if phase.is_merger]
            ax.plot(merger_ratios, ba_ratios, c='gray', alpha=0.5, linestyle='-')

    ax.set_xlabel('Merger Ratio', fontsize=20)
    ax.set_ylabel('ba_d / ba_s', fontsize=20)
    ax.legend(loc='best', fontsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figure_folder}/merger_ratio_vs_ba_ratio.png', bbox_inches='tight', dpi=300)

    # Plot merger ratio vs ca_d/ca_s
    fig, ax = plt.subplots(figsize=(width, height))
    for idx, main_line in enumerate(main_lines):
        merger_idx, merger_node = find_most_recent_merger(main_line)
        if merger_idx is None:
            print(f"No merger found for simulation {idx}")
            continue

        phase_list = main_line[-3:]  # Get the last three nodes
        colors = ["#0072B2", "#E69F00", "#009E73"]

        for phase, color in zip(phase_list, colors):
            if phase is not None and phase.is_merger:
                ca_ratio = phase.ca_d / phase.ca_s if phase.ca_s else np.nan
                ax.scatter(phase.merger_ratio, ca_ratio, c=color, marker='o', s=50, alpha=0.7, zorder=2)

        if link_timesteps:
            merger_ratios = [phase.merger_ratio for phase in phase_list if phase.is_merger]
            ca_ratios = [phase.ca_d / phase.ca_s if phase.ca_s else np.nan for phase in phase_list if phase.is_merger]
            ax.plot(merger_ratios, ca_ratios, c='gray', alpha=0.5, linestyle='-')

    ax.set_xlabel('Merger Ratio', fontsize=20)
    ax.set_ylabel('ca_d / ca_s', fontsize=20)
    ax.legend(loc='best', fontsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figure_folder}/merger_ratio_vs_ca_ratio.png', bbox_inches='tight', dpi=300)

    # Plot merger ratio vs T_s - T_d
    fig, ax = plt.subplots(figsize=(width, height))
    for idx, main_line in enumerate(main_lines):
        merger_idx, merger_node = find_most_recent_merger(main_line)
        if merger_idx is None:
            print(f"No merger found for simulation {idx}")
            continue

        phase_list = main_line[-3:]  # Get the last three nodes
        colors = ["#0072B2", "#E69F00", "#009E73"]

        for phase, color in zip(phase_list, colors):
            if phase is not None and phase.is_merger:
                T_s = T(phase.ba_s, phase.ca_s)
                T_d = T(phase.ba_d, phase.ca_d)
                T_diff = T_s - T_d
                ax.scatter(phase.merger_ratio, T_diff, c=color, marker='o', s=50, alpha=0.7, zorder=2)

        if link_timesteps:
            merger_ratios = [phase.merger_ratio for phase in phase_list if phase.is_merger]
            T_diffs = [T(phase.ba_s, phase.ca_s) - T(phase.ba_d, phase.ca_d) for phase in phase_list if phase.is_merger]
            ax.plot(merger_ratios, T_diffs, c='gray', alpha=0.5, linestyle='-')

    ax.set_xlabel('Merger Ratio', fontsize=20)
    ax.set_ylabel('T_s - T_d', fontsize=20)
    ax.legend(loc='best', fontsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figure_folder}/merger_ratio_vs_T_diff.png', bbox_inches='tight', dpi=300)



import pickle
import os


def extract_picklable_data(main_lines):
    picklable_data = []
    for main_line in main_lines:
        line_data = []
        for node in main_line:
            node_data = {
                'halo_number': node.halo.halo_number,
                'timestep_number': node.halo.timestep.extension,
                'mvir': node.mvir,
                'mstar': node.mstar,
                'ndm': node.ndm,
                'ba_s': node.ba_s,
                'ca_s': node.ca_s,
                'ba_d': node.ba_d,
                'ca_d': node.ca_d,
                'is_merger': node.is_merger,
                'merger_ratio': node.merger_ratio,
                'merger_time': node.merger_time,
                'redshift': node.halo.timestep.redshift,
                'time_gyr': node.halo.timestep.time_gyr
            }
            line_data.append(node_data)
        picklable_data.append(line_data)
    return picklable_data


def save_picklable_data(main_lines, filename='picklable_main_lines.pkl'):
    new_picklable_data = extract_picklable_data(main_lines)

    existing_data = []
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                existing_data = pickle.load(f)
            print(f"Loaded existing data from {filename}")
        except Exception as e:
            print(f"Error loading existing data: {e}")

    combined_data = existing_data + new_picklable_data

    with open(filename, 'wb') as f:
        pickle.dump(combined_data, f)
    print(f"Saved combined data to {filename}")


def load_picklable_data(filename='picklable_main_lines.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"File {filename} not found.")
        return []


class SimpleHaloNode:
    def __init__(self, data):
        self.__dict__.update(data)


def reconstruct_main_lines(picklable_data):
    reconstructed_lines = []
    for line_data in picklable_data:
        reconstructed_line = [SimpleHaloNode(node_data) for node_data in line_data]
        reconstructed_lines.append(reconstructed_line)
    return reconstructed_lines

# Usage:
# After generating your main_lines:
# save_picklable_data(main_lines)

# When you want to load and use the data later:
# picklable_data = load_picklable_data()
# reconstructed_main_lines = reconstruct_main_lines(picklable_data)