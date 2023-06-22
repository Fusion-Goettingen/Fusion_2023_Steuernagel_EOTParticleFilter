import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.utilities.utils import rot


def plot_elliptic_extent(m, p, ax=None, color='b', alpha=1., label=None, linestyle=None, show_center=True, fill=False):
    """
    Add matplotlib ellipse patch based on location and extent information about vehicle
    :param m: Kinematic information as 4D array [x, y, velocity_x, velocity_y]
    :param p: extent information as 3D array [orientation, length, width]. Orientation in radians.
    :param ax: matplotlib axis to plot on or None (will use .gca() if None)
    :param color: Color to plot the ellipse and marker in
    :param alpha: Alpha value for plot
    :param label: Label to apply to plot or None to not add a label
    :param linestyle: Linestyle parameter passed to matplotlib
    :param show_center: If True, will additionally add an x for the center location
    :param fill: Whether to fill the ellipse
    """
    if ax is None:
        ax = plt.gca()
    theta, l1, l2 = p
    theta = np.rad2deg(theta)
    # patches.Ellipse takes angle counter-clockwise
    el = patches.Ellipse(xy=m[:2], width=l1, height=l2, angle=theta, fill=fill, color=color, label=label,
                         alpha=alpha, linestyle=linestyle)
    if show_center:
        ax.scatter(m[0], m[1], color=color, marker='x')
    ax.add_patch(el)


def plot_rectangle(m, p, ax=None, color='b', linestyle='-', alpha=1.0, label=None, with_orientation=False,
                   show_center=True, fill=False):
    """
    Add a rectangle to the plot

    :param m: Center
    :param p: [orientation, length, width] of rectangle
    :param ax: matplotlib axis to plot on or None (will use .gca() if None)
    :param color: Color to plot the rectangle and marker in
    :param linestyle: Linestyle parameter passed to matplotlib
    :param alpha: Alpha value for plot
    :param label: Label to apply to plot or None to not add a label
    :param with_orientation: add a line indicating the orientation
    :param show_center: If True, will additionally add an x for the center location
    :param fill: Whether to fill the rectangle
    """
    if ax is not None:
        plt.sca(ax)
    center = np.array(m)[:2]
    theta, length, width = p
    p = [
        [-length / 2, -width / 2],
        [-length / 2, width / 2],
        [length / 2, width / 2],
        [length / 2, -width / 2],
        [0, 0],
        [length / 2, 0]
    ]
    p = np.array([rot(theta) @ np.array(point) for point in p])
    p += center
    if fill:
        plt.fill(p[:4, 0], p[:4, 1], color=color, alpha=alpha, label=label)
    else:
        plt.plot(p[[0, 1], 0], p[[0, 1], 1], c=color, linestyle=linestyle, alpha=alpha, label=label)
        plt.plot(p[[1, 2], 0], p[[1, 2], 1], c=color, linestyle=linestyle, alpha=alpha)
        plt.plot(p[[2, 3], 0], p[[2, 3], 1], c=color, linestyle=linestyle, alpha=alpha)
        plt.plot(p[[3, 0], 0], p[[3, 0], 1], c=color, linestyle=linestyle, alpha=alpha)

    if with_orientation:
        plt.plot(p[[4, 5], 0], p[[4, 5], 1], c=color, linestyle=linestyle, alpha=alpha)

    if show_center:
        plt.scatter(m[0], m[1], color=color, marker='x')

    return


def mark_standard_trajectory_turn_in_plot(curve_linewidth: int = 14, color='k'):
    """
    Adds marks for the standard trajectory in the plot (12-17, 23-28, 41-46, 52-57).

    Note that the parameters should generally not be used, except for edge cases, in order to ensure that the
    visualization is the same between all figures.
    """
    xlim = np.array(plt.gca().get_xlim())
    ylim = plt.gca().get_ylim()

    plt.plot([12, 17], [ylim[0], ylim[0]], c=color, linewidth=curve_linewidth)
    plt.plot([23, 28], [ylim[0], ylim[0]], c=color, linewidth=curve_linewidth)
    plt.plot([41, 46], [ylim[0], ylim[0]], c=color, linewidth=curve_linewidth)
    plt.plot([52, 57], [ylim[0], ylim[0]], c=color, linewidth=curve_linewidth)

    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)


def plot_elliptic_state(full_state, ax=None, color='b', alpha=1., label=None, linestyle=None, show_center=True,
                        fill=False):
    """Wraps around plot_elliptic_extent. This method takes a 7D state and extracts m and p from it"""
    m = full_state[:2]
    p = full_state[4:]

    return plot_elliptic_extent(m, p, ax=ax, color=color, alpha=alpha, label=label, linestyle=linestyle,
                                show_center=show_center, fill=fill)


def create_grid_plot(grid_data, xlabel, ylabel, xticks, yticks, vmin=None, vmax=None, cmap="viridis", show_cbar=True,
                     cbar_title=None):
    """
    This function uses imshow to plot some kind of grid data on the current axis, which can be set using plt.sca(AXIS).

    Note that this function does NOT call

    :param grid_data: Data to be plotted
    :param xlabel: label for the x axis
    :param ylabel: label for the y axis
    :param xticks: tick labels for the x axis
    :param yticks: tick labels for the y axis
    :param vmin: Color scaling minimum or None. If None, will use the smallest value in grid_data for this instead
    :param vmax: Color scaling maximum or None. If None, will use the largest value in grid_data for this instead
    :param cmap: Colormap to use. Default is viridis. If your data is consists of a difference in value, the suggested
    standard is "seismic"
    :param show_cbar: Bool indicating whether the colorbar should be shown and added to the plot
    :param cbar_title: If not None, this will be used as the 90° rotated label of the colorbar
    """
    grid_data = np.array(grid_data)

    plt.imshow(grid_data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    plt.xlabel(xlabel)
    plt.xticks(np.arange(start=0, stop=len(xticks)))
    plt.gca().set_xticklabels(xticks)
    plt.ylabel(ylabel)
    plt.yticks(np.arange(start=0, stop=len(yticks)))
    plt.gca().set_yticklabels(yticks)
    if show_cbar:
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        if cbar_title is not None:
            cbar.ax.get_yaxis().labelpad = 5
            cbar.ax.set_ylabel(cbar_title, rotation=90)
