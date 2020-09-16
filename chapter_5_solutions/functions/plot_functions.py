import numpy as np
import math

###


def scatter_plot(ax, g, w, points, xmin, xmax, **kwargs):

    x, y = points
    color = "red"
    if 'color' in kwargs:
        color = kwargs['color']

    x_fit = np.linspace(xmin, xmax, 300).reshape(300, 1)
    y_fit = g(x_fit, w)

    ax.plot(x_fit, y_fit, color=color, linewidth=2)

    ax.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=40)

    # clean up panel
    ymin, ymax = [np.min(y_fit), np.max(y_fit)]
    ymin -= (ymax - ymin)*0.1
    ymax += (ymax - ymin)*0.1
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # label axes
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$y$', rotation=0, fontsize=12)
    ax.set_title('data', fontsize=13)

    ax.axhline(y=0, color='k', zorder=0, linewidth=0.5)
    ax.axvline(x=0, color='k', zorder=0, linewidth=0.5)

###


def surface_plot(ax, g, **kwargs):

    if 'view' in kwargs:
        view = kwargs['view']
        ax.view_init(view[0], view[1])

    xmin = -3.1
    xmax = 3.1
    ymin = -3.1
    ymax = 3.1
    if 'xmin' in kwargs:
        xmin = kwargs['xmin']
    if 'xmax' in kwargs:
        xmax = kwargs['xmax']
    if 'ymin' in kwargs:
        ymin = kwargs['ymin']
    if 'ymax' in kwargs:
        ymax = kwargs['ymax']

    #### define input space for function and evaluate ####
    w1 = np.linspace(xmin, xmax, 200)
    w2 = np.linspace(ymin, ymax, 200)
    w1_vals, w2_vals = np.meshgrid(w1, w2)
    w1_vals.shape = (len(w1)**2, 1)
    w2_vals.shape = (len(w2)**2, 1)
    h = np.concatenate((w1_vals, w2_vals), axis=1)
    func_vals = np.asarray([g(np.reshape(s, (2, 1))) for s in h])

    ### plot function as surface ###
    w1_vals.shape = (len(w1), len(w2))
    w2_vals.shape = (len(w1), len(w2))
    func_vals.shape = (len(w1), len(w2))
    ax.plot_surface(w1_vals, w2_vals, func_vals, alpha=0.1, color='w',
                    rstride=25, cstride=25, linewidth=1, edgecolor='k', zorder=2)

    # plot z=0 plane
    ax.plot_surface(w1_vals, w2_vals, func_vals*0, alpha=0.1, color='w',
                    zorder=1, rstride=25, cstride=25, linewidth=0.3, edgecolor='k')

    # clean up axis
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    ax.set_xlabel('$w_0$', fontsize=14)
    ax.set_ylabel('$w_1$', fontsize=14, rotation=0)
    ax.set_title('$g(w_0,w_1)$', fontsize=14)

### visualize contour plot of cost function ###


def contour_plot(ax, g, wmax, num_contours):

    #### define input space for function and evaluate ####
    w1 = np.linspace(-wmax, wmax, 100)
    w2 = np.linspace(-wmax, wmax, 100)
    w1_vals, w2_vals = np.meshgrid(w1, w2)
    w1_vals.shape = (len(w1)**2, 1)
    w2_vals.shape = (len(w2)**2, 1)
    h = np.concatenate((w1_vals, w2_vals), axis=1)
    func_vals = np.asarray([g(np.reshape(s, (2, 1))) for s in h])

    # func_vals = np.asarray([self.g(s) for s in h])
    w1_vals.shape = (len(w1), len(w1))
    w2_vals.shape = (len(w2), len(w2))
    func_vals.shape = (len(w1), len(w2))

    ### make contour right plot - as well as horizontal and vertical axes ###
    # set level ridges
    levelmin = min(func_vals.flatten())
    levelmax = max(func_vals.flatten())
    cutoff = 0.5
    cutoff = (levelmax - levelmin)*cutoff
    numper = 3
    levels1 = np.linspace(cutoff, levelmax, numper)
    num_contours -= numper

    levels2 = np.linspace(levelmin, cutoff, min(num_contours, numper))
    levels = np.unique(np.append(levels1, levels2))
    num_contours -= numper
    while num_contours > 0:
        cutoff = levels[1]
        levels2 = np.linspace(levelmin, cutoff, min(num_contours, numper))
        levels = np.unique(np.append(levels2, levels))
        num_contours -= numper

    ax.contour(w1_vals, w2_vals, func_vals, levels=levels, colors='k')
    ax.contourf(w1_vals, w2_vals, func_vals, levels=levels, cmap='Blues')

    # clean up panel
    ax.set_xlabel('$w_0$', fontsize=12)
    ax.set_ylabel('$w_1$', fontsize=12, rotation=0)
    ax.set_title(r'$g\left(w_0,w_1\right)$', fontsize=13)

    ax.axhline(y=0, color='k', zorder=0, linewidth=0.5)
    ax.axvline(x=0, color='k', zorder=0, linewidth=0.5)
    ax.set_xlim([-wmax, wmax])
    ax.set_ylim([-wmax, wmax])


def decision_boundary_plot(ax, w, g, points):
    x, y = points

    # generate input range for functions
    xmin = min(min(x[:, 0]), min(x[:, 1]))
    xmax = max(max(x[:, 0]), max(x[:, 1]))
    gapx = (xmax - xmin)*0.1
    xmin -= gapx
    xmax += gapx

    r = np.linspace(xmin, xmax, 400)
    x1_vals, x2_vals = np.meshgrid(r, r)
    x1_vals.shape = (len(r)**2, 1)
    x2_vals.shape = (len(r)**2, 1)
    h = np.concatenate([x1_vals, x2_vals], axis=1)
    g_vals = g(h, w)
    g_vals = np.asarray(g_vals)
    # vals for cost surface
    x1_vals.shape = (len(r), len(r))
    x2_vals.shape = (len(r), len(r))
    g_vals.shape = (len(r), len(r))

    ax.contour(x1_vals, x2_vals, g_vals, colors='k',
               levels=[0], linewidths=3, zorder=1)

    ax.contourf(x1_vals, x2_vals, g_vals, colors=[
                "blue", "red"], alpha=0.1, levels=1)

    # scatter points
    classes = np.unique(y)
    colors = ['cornflowerblue', 'salmon']
    for i, num in enumerate(classes):
        inds = np.argwhere(y == num)
        ax.scatter(x[inds, 0], x[inds, 1], color=colors[i %
                                                        len(colors)], linewidth=1, marker='o', edgecolor='k', s=50)

    # clean up panel
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([xmin, xmax])

    ax.set_xticks(np.arange(round(xmin), round(xmax) + 1, 1.0))
    ax.set_yticks(np.arange(round(xmin), round(xmax) + 1, 1.0))

    # label axes
    ax.set_xlabel(r'$x_1$', fontsize=12, labelpad=0)
    ax.set_ylabel(r'$x_2$', rotation=0, fontsize=12, labelpad=5)


def decision_boundary_3d_plot(ax, w, g, points, **kwargs):

    colors = ["cornflowerblue", "salmon"]
    x, y = points

    # generate input range for functions
    xmin = min(min(x[:, 0]), min(x[:, 1]))
    xmax = max(max(x[:, 0]), max(x[:, 1]))
    gapx = (xmax - xmin)*0.35
    xmin -= gapx
    xmax += gapx
    ymiddle = (np.max(y) - np.min(y)) / 2

    r = np.linspace(xmin, xmax, 400)
    x1_vals, x2_vals = np.meshgrid(r, r)
    x1_vals.shape = (len(r)**2, 1)
    x2_vals.shape = (len(r)**2, 1)
    h = np.concatenate([x1_vals, x2_vals], axis=1)
    g_vals = g(h, w)
    g_vals = np.asarray(g_vals)
    # vals for cost surface
    x1_vals.shape = (len(r), len(r))
    x2_vals.shape = (len(r), len(r))
    g_vals.shape = (len(r), len(r))

    # set view
    if 'view' in kwargs:
        view = kwargs['view']
        ax.view_init(view[0], view[1])

    ax.plot_surface(x1_vals, x2_vals, g_vals, alpha=0.1, rstride=20,
                    cstride=20, linewidth=0.15, color='w', edgecolor='k', zorder=2)

    ax.contour(x1_vals, x2_vals, g_vals, colors='k',
               levels=[0], linewidths=3, zorder=1)
    ax.contourf(x1_vals, x2_vals, g_vals+ymiddle,
                colors=colors, levels=1, zorder=1, alpha=0.1)
    ax.contourf(x1_vals, x2_vals, g_vals, colors=colors,
                levels=1, zorder=1, alpha=0.1)

    class_nums = np.unique(y)

    # scatter points in both panels
    for i, c in enumerate(class_nums):
        inds = np.argwhere(y == c)
        # ind = [v[0] for v in ind]
        ax.scatter(x[inds, 0], x[inds, 1], y[inds], s=80,
                   color=colors[i % len(colors)], edgecolor='k', linewidth=1.5)

    ymax = np.max(y)
    ymin = np.min(y)
    ygap = (ymax - ymin)*0.2
    ymin -= ygap
    ymax += ygap

    # clean up panel
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([xmin, xmax])
    ax.set_zlim([ymin, ymax])

    ax.set_xticks(np.arange(round(xmin) + 1, round(xmax), 1.0))
    ax.set_yticks(np.arange(round(xmin) + 1, round(xmax), 1.0))
    ax.set_zticks([-1, 0, 1])

    # label axes
    ax.set_xlabel(r'$x_1$', fontsize=12, labelpad=5)
    ax.set_ylabel(r'$x_2$', rotation=0, fontsize=12, labelpad=5)
    ax.set_zlabel(r'$y$', rotation=0, fontsize=12, labelpad=-3)

    # clean up panel
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)


def colorspec(points):
    # produce color scheme
    s = np.linspace(0, 1, len(points[:round(len(points)/2)]))
    s.shape = (len(s), 1)
    t = np.ones(len(points[round(len(points)/2):]))
    t.shape = (len(t), 1)
    s = np.vstack((s, t))
    cs = np.concatenate((s, np.flipud(s)), 1)
    cs = np.concatenate((cs, np.zeros((len(s), 1))), 1)
    return cs


### visualize weights steps during optimization ###
def weight_history_plot(ax, g, weight_history, **kargs):

    cost_history = False
    if "cost_history" in kargs:
        cost_history = kargs["cost_history"]

    num_frames = len(weight_history)
    colors = colorspec(weight_history)
    for k in range(num_frames):
        # current color
        color = colors[k]

        # current weights
        w = weight_history[k]

        ###### steps ######
        if k == 0 or k == num_frames - 1:
            points = [w[0], w[1]]
            if cost_history:
                points.append(cost_history[k])
            ax.scatter(*points, s=90, facecolor=color,
                       edgecolor='k', linewidth=0.5, zorder=3)
        else:
            # plot connector between points for visualization purposes
            w_old = weight_history[k-1]
            w_new = weight_history[k]

            points = [[w_old[0], w_new[0]], [w_old[1], w_new[1]]]
            if cost_history:
                points.append(cost_history[k])

            ax.plot(*points, color=color, linewidth=3,
                    alpha=1, zorder=2)      # plot approx
            ax.plot(*points, color='k',
                    linewidth=3 + 1, alpha=1, zorder=1)      # plot approxs
