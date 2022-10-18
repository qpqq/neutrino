import os
import sys

import numpy as np
import pandas as pd
from scipy import stats, ndimage

import matplotlib as mpl
import matplotlib.pyplot as plt

import catalogs
import angles

EMin = catalogs.EMin
EMax = catalogs.EMax

BinsNumber = catalogs.BinsNumber
BestBin = catalogs.BestBin
BinsEdges = catalogs.BinsEdges

Mu = catalogs.Mu
Sigma = catalogs.Sigma

TickSize = 25
TickSizeCbar = 25
LabelSize = 25
LegendSize = 25
TitleSize = 35


def allsky(catalog, show=True, save=True):
    visuals = {
        '2mrs': {
            'par': [5, 1, 3],
            'by': 'DIST',
            'vmin': 1,
            'vmax': 349,
            'extend': 'max',
            'xlabel': 'Distance, Mpc',
            'minMag': 7,
            'maxMag': 12,
            'step': 1,
            'offset_1': 0.17,
            'offset_2': 0.27
        },
        '2mrsg': {
            'par': [5, 1, 3],
            'by': 'DIST',
            'vmin': 1,
            'vmax': 349,
            'extend': 'neither',
            'xlabel': 'Distance, Mpc',
            'minMag': 7,
            'maxMag': 12,
            'step': 1,
            'offset_1': 0.17,
            'offset_2': 0.27
        },
        'cf2': {
            'par': [5, 0.005, 4],
            'by': 'DIST',
            'vmin': None,
            'vmax': 200.2,
            'extend': 'max',
            'xlabel': 'Distance, Mpc',
            'minMag': 8,
            'maxMag': 18,
            'step': 2,
            'offset_1': 0.35,
            'offset_2': 0.5
        },
        'bzcat': {
            'par': [5, 0.1, 3],
            'by': 'Z',
            'vmin': None,
            'vmax': 3.005,
            'extend': 'max',
            'xlabel': 'Redshift',
            'minMag': 14,
            'maxMag': 22,
            'step': 2,
            'offset_1': 0.3,
            'offset_2': 0.5
        },
        'milliquas': {
            'par': [0.5, 0.01, 3],
            'by': 'Z',
            'vmin': None,
            'vmax': 4.005,
            'extend': 'max',
            'xlabel': 'Redshift',
            'minMag': 15,
            'maxMag': 25,
            'step': 2,
            'offset_1': 0.35,
            'offset_2': 0.5
        }
    }

    cat = catalog
    if catalog[-2:] == '_s':
        cat = catalog[:-2]
    elif catalog[-3:] == '_ss':
        cat = catalog[:-3]
    a, b, c = visuals[cat]['par']
    by = visuals[cat]['by']
    vmin = visuals[cat]['vmin']
    vmax = visuals[cat]['vmax']
    extend = visuals[cat]['extend']
    xlabel = visuals[cat]['xlabel']
    min_mag_sample = visuals[cat]['minMag']
    max_mag_sample = visuals[cat]['maxMag']
    step = visuals[cat]['step']
    offset_1 = visuals[cat]['offset_1']
    offset_2 = visuals[cat]['offset_2']

    def calc_size(mag):
        return a + b * (max_mag - mag) ** c

    data = catalogs.read(catalog)

    max_mag = max(data['MAG'].max(), max_mag_sample)
    if catalog[:-2] == '2mrs' or catalog[:-2] == '2mrsg':
        real = catalogs.read(catalog[:-2])
        max_mag = max(real['MAG'].max(), max_mag_sample)

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111, projection='hammer')

    # values increase counterclockwise
    ax.set_xticks(ax.get_xticks(),
                  ['210°', '240°', '270°', '300°', '330°', '0°', '30°', '60°', '90°', '120°', '150°'][::-1])

    lon = angles.to_rad(data['GLON'].to_numpy())
    lat = angles.to_rad(data['GLAT'].to_numpy())

    # reflect the values relative to zero
    lon = np.where(lon < np.pi, -lon, -lon + 2 * np.pi)

    alpha = 0.7
    cmap = mpl.cm.get_cmap("jet").copy()
    color_over = cmap.get_over()
    color_over[-1] = alpha
    cmap.set_over(color_over)
    axrgb = ax.scatter(lon, lat,
                       s=calc_size(data['MAG']),
                       marker='o', alpha=alpha,
                       c=data[by], cmap=cmap,
                       label=fr'$N={len(data.index)}$')
    axrgb.set_clim(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(axrgb, extend=extend, location='bottom', fraction=0.05, shrink=0.3,
                        pad=0.05, anchor=(0, 3))

    ax.grid(True, linestyle=':', linewidth=0.5)

    ax.set_title(fr'{catalogs.FullName[catalog]}', fontsize=TitleSize, pad=45)
    cbar.ax.set_xlabel(xlabel, fontsize=LabelSize, labelpad=-80)
    ax.legend(loc='upper right', fontsize=LegendSize, markerscale=0, frameon=False, fancybox=False)
    ax.tick_params(axis='both', which='major', labelsize=TickSize)
    cbar.ax.tick_params(axis='both', which='major', labelsize=TickSizeCbar)

    # circles
    ax2 = fig.add_axes([0.83, 0.038, 0.13, 0.1])
    ax2.axis('off')
    sample_mags = np.arange(min_mag_sample, max_mag_sample + step, step)
    for _mag in sample_mags:
        ax2.scatter(_mag, 0, s=calc_size(_mag), marker='o', c='black', clip_on=False)
        offset_11 = 0
        if _mag // 10 > 0:
            offset_11 = offset_1
        ax2.text(_mag - offset_1 - offset_11, -0.04, f'{str(_mag)}', fontsize=20)
    ax2.text((min_mag_sample + max_mag_sample) / 2 - offset_2, 0.022, r'$m$', fontsize=LabelSize)

    fig.tight_layout()

    if save:
        os.makedirs('all-sky graphs', exist_ok=True)
        fig.savefig(f'all-sky graphs/{catalog}.png', dpi=120)

    if show:
        plt.show()

    plt.close(fig)


def hist_sum_example(n_particles, z=None):
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    hist = catalogs.normal_pdf_logx_hist(n_particles, z)
    ax.stairs(hist, BinsEdges, fill=False, color='tab:red', label=fr'$N={n_particles}$')

    x = np.linspace(np.log10(EMin), np.log10(EMax), 10 ** 4)
    y = stats.norm.pdf(x, loc=Mu, scale=Sigma) * n_particles * (BinsEdges[1] - BinsEdges[0])
    ax.plot(x, y, linestyle='dotted', color='tab:blue')

    ax.set_xlabel(r'log$_{10}E$', fontsize=LabelSize)
    ax.set_ylabel(r'$N$', fontsize=LabelSize)
    ax.set_xlim(np.log10(EMin), np.log10(EMax))
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=TickSize)
    ax.yaxis.offsetText.set_fontsize(TickSize)

    ax.grid(True, linewidth=0.3)
    ax.grid(True, 'minor', linewidth=0.1)

    title = fr'Normal distribution {n_particles} particles, $\sigma = {Sigma}$, $\mu = {Mu}$'
    ax.set_title(title, fontsize=TitleSize, pad=15)
    leg = ax.legend(loc='upper right', fontsize=LegendSize, frameon=False, fancybox=False)
    for item in leg.legendHandles:
        item.set_visible(False)

    fig.tight_layout()
    plt.show()

    plt.close(fig)


def hist_sum(catalog, show=True, save=True):
    data = catalogs.read(catalog)

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    hist = data.iloc[:, -BinsNumber:]
    hist = np.sum(hist, axis=0)
    ax.stairs(hist, BinsEdges, fill=False, color='tab:red', label=fr'$N={len(data.index)}$')

    x = np.linspace(np.log10(EMin), np.log10(EMax), 10 ** 4)
    n_particles = np.sum(hist)
    y = stats.norm.pdf(x, loc=Mu, scale=Sigma) * n_particles * (BinsEdges[1] - BinsEdges[0])
    ax.plot(x, y, linestyle='dotted', color='tab:blue')

    ax.set_xlabel(r'log$_{10}E$', fontsize=LabelSize)
    ax.set_ylabel(r'Neutrino flux, s$^{-1}$ cm$^{-2}$', fontsize=LabelSize)
    ax.set_xlim(np.log10(EMin), np.log10(EMax))
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=TickSize)
    ax.yaxis.offsetText.set_fontsize(TickSize)

    ax.grid(True, linewidth=0.3)
    ax.grid(True, 'minor', linewidth=0.1)

    ax.set_title(fr'{catalogs.FullName[catalog]} distribution', fontsize=LabelSize, pad=15)
    leg = ax.legend(loc='upper right', fontsize=LegendSize, frameon=False, fancybox=False)
    for item in leg.legendHandles:
        item.set_visible(False)

    fig.tight_layout()

    if save:
        os.makedirs('histogram sum', exist_ok=True)
        fig.savefig(f'histogram sum/{catalog}.png', dpi=120)

    if show:
        plt.show()

    plt.close(fig)


def hist_by(catalog, by, show=True, save=True):
    visuals = {
        'Z': {
            'xlabel': 'Redshift',
            'xlimMod': '0',
            'dirName': 'histogram by redshift',
        },
        'DIST': {
            'xlabel': 'Distance, Mpc',
            'xlimMod': '0',
            'dirName': 'histogram by distance',
        },
        'MAG': {
            'xlabel': 'Magnitude',
            'xlimMod': 'bin',
            'dirName': 'histogram by magnitude',
        },
        'MAG ABS': {
            'xlabel': 'Absolute magnitude',
            'xlimMod': 'bin',
            'dirName': 'histogram by absolute magnitude',
        },
        'NEU': {
            'xlabel': r'log$_{10}$ of neutrino flux',
            'xlimMod': 'bin',
            'dirName': 'histogram by flux',
        },
        'BAR': {
            'xlabel': r'log$_{10}$ of neutrino flux in bin #' + f'{BestBin}',
            'xlimMod': 'bin',
            'dirName': 'histogram by flux in bin',
        }
    }

    xlabel = visuals[by]['xlabel']
    xlim_mod = visuals[by]['xlimMod']
    dir_name = visuals[by]['dirName']

    data = catalogs.read(catalog)

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    if by == 'NEU':
        by = 'NEU LOG'
        data[by] = np.log10(data['NEU'])

    if by == 'BAR':
        by = f'BIN {BestBin} LOG'
        data[by] = np.log10(data[f'BIN {BestBin}'])

    hist, bins, _ = ax.hist(data[by], 25, color='tab:blue', label=fr'$N={len(data.index)}$')

    ax.set_xlabel(xlabel, fontsize=LabelSize)
    ax.set_ylabel('$N$', fontsize=LabelSize)
    xlim = 0
    if xlim_mod == 'bin':
        xlim = bins[0]
    ax.set_xlim(xlim, bins[-1])
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=TickSize)
    ax.yaxis.offsetText.set_fontsize(TickSize)

    ax.grid(True, linestyle=':', linewidth=0.3)
    ax.grid(True, 'minor', linestyle=':', linewidth=0.1)

    ax.set_title(fr'{catalogs.FullName[catalog]}', fontsize=TitleSize, pad=15)
    leg = ax.legend(fontsize=LegendSize, handlelength=0, frameon=False, fancybox=False)
    for item in leg.legendHandles:
        item.set_visible(False)

    fig.tight_layout()

    if save:
        os.makedirs(dir_name, exist_ok=True)
        fig.savefig(dir_name + f'/{catalog}.png', dpi=120)

    if show:
        plt.show()

    plt.close(fig)


def gauss(catalog, glon, glat, dgl, n_grid, fwhm, save=True):
    offset = 5 * fwhm
    fwhm *= n_grid / dgl  # from degrees to pixels

    glon_min = glon - dgl / 2 - offset
    glon_max = glon + dgl / 2 + offset
    glat_min = glat - dgl / 2 - offset
    glat_max = glat + dgl / 2 + offset

    offset_grid = 2 * int(n_grid / dgl * offset)
    n_grid += offset_grid

    data = catalogs.read(catalog)
    data = data.loc[(data['GLON'] > glon_min) &
                    (data['GLON'] < glon_max) &
                    (data['GLAT'] > glat_min) &
                    (data['GLAT'] < glat_max)]

    x = np.linspace(glon_min, glon_max, n_grid + 1)
    y = np.linspace(glat_min, glat_max, n_grid + 1)
    xi = np.searchsorted(x, data['GLON']) - 1
    yi = np.searchsorted(y, data['GLAT']) - 1

    # grid with min float values
    z = np.full(shape=(n_grid, n_grid), fill_value=sys.float_info.min)
    for i in range(len(xi)):
        z[yi[i], xi[i]] += data.iloc[i][f'BIN {BestBin}']

    # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    sigma_gauss = fwhm / 2.355
    z = ndimage.gaussian_filter(z, sigma=sigma_gauss)

    offset_grid //= 2
    x = x[offset_grid:-offset_grid]
    y = y[offset_grid:-offset_grid]
    z = z[offset_grid:-offset_grid, offset_grid:-offset_grid]

    if save:
        os.makedirs('gauss graphs', exist_ok=True)
        data = data.sort_values(f'BIN {BestBin}', ascending=False).reset_index(drop=True)
        data.to_csv(f'gauss graphs/{catalog}.csv', index=False, float_format='%.15f')

    return x, y, z


def gauss_graph(catalog, show=True, save=True):
    visuals = {
        '2mrs': {
            'dgl': 60,
            'vmin': 10 ** -8
        },
        '2mrsg': {
            'dgl': 60,
            'vmin': 10 ** -8
        },
        'cf2': {
            'dgl': 60,
            'vmin': 10 ** -8
        },
        'bzcat': {
            'dgl': 60,
            'vmin': 10 ** -10
        },
        'milliquas': {
            'dgl': 30,
            'vmin': 10 ** -8
        }
    }

    cat = catalog
    if catalog[-2:] == '_s':
        cat = catalog[:-2]
    elif catalog[-3:] == '_ss':
        cat = catalog[:-3]
    dgl = visuals[cat]['dgl']
    vmin = visuals[cat]['vmin']

    fig = plt.figure(figsize=(14, 10.5))

    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    # values increase counterclockwise
    ax.invert_xaxis()

    glon = 90
    glat = 0
    n_grid = 100
    fwhm = 1.5
    x, y, z = gauss(catalog, glon, glat, dgl, n_grid, fwhm, save=save)

    pc = ax.pcolormesh(x, y, z, norm=mpl.colors.LogNorm(vmin=vmin), cmap='jet')

    cbar = fig.colorbar(pc, pad=0.01)
    cbar.ax.tick_params(labelsize=TickSizeCbar)
    cbar.ax.set_ylabel(f'Neutrino flux in bin #{BestBin}, ' + r's$^{-1}$ cm$^{-2}$', fontsize=LabelSize)

    ax.set_xlabel('Galactic longitude, degrees', fontsize=LabelSize)
    ax.set_ylabel('Galactic latitude, degrees', fontsize=LabelSize)

    ax.tick_params(axis='both', which='major', labelsize=TickSize)

    ax.set_title(f'{catalogs.FullName[catalog]}, {n_grid}x{n_grid}, fwhm = {fwhm}', fontsize=30, pad=15)

    fig.tight_layout()

    if save:
        fig.savefig(f'gauss graphs/{catalog}.png', dpi=120)

    if show:
        plt.show()

    plt.close(fig)


def diff(show=True, save=True):
    d1 = pd.read_csv('datasets/2mrsg_full.csv')
    d1 = d1.loc[(d1['Vgp'] > 0) & (d1['K_t'] > 0)]

    d2 = pd.read_csv('datasets/cf2_full.csv')
    d2 = d2.loc[(d2['Dist'] > 0) & (d2['Btot'] > 0) & (d2['Dist'] < 350)]

    d3 = pd.merge(d1, d2, how='inner', on=['pgc'])
    d3['DIFF'] = d3['Vgp'] / catalogs.H - d3['Dist']

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    bins = ax.hist(d3['DIFF'], 50, range=(-50, 50), label=fr'$N={len(d3.index)}$')[1]

    ax.set_xlabel('Difference in distance, Mpc', fontsize=LabelSize)
    ax.set_ylabel(r'$N$', fontsize=LabelSize)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=TickSize)
    ax.yaxis.offsetText.set_fontsize(TickSize)

    ax.grid(True, linestyle=':', linewidth=0.3)
    ax.grid(True, 'minor', linestyle=':', linewidth=0.1)

    ax.set_title(fr'Difference in distance between cf2 and 2mrsg catalogs', fontsize=TitleSize, pad=15)
    ax.legend(fontsize=LegendSize, handlelength=0, frameon=False, fancybox=False)

    fig.tight_layout()

    if save:
        os.makedirs('other graphs', exist_ok=True)
        fig.savefig('other graphs/diff.png', dpi=120)

    if show:
        plt.show()

    plt.close(fig)


def btot_vs_dist(show=True, save=True):
    data = pd.read_csv('datasets/cf2_full.csv')

    data = data.loc[(data['Dist'] > 0) & (data['Btot'] > 0)]

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    ax.scatter(data['Dist'], data['Btot'], s=1)

    ax.set_xlabel('Distance, Mpc', fontsize=LabelSize)
    ax.set_ylabel('$B_{tot}$', fontsize=LabelSize)
    ax.set_xlim(0)
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=TickSize)
    ax.yaxis.offsetText.set_fontsize(TickSize)

    ax.grid(True, linestyle='--', linewidth=0.3)
    ax.grid(True, 'minor', linestyle='--', linewidth=0.1)

    ax.set_title(r'$B_{tot}$' + f' magnitude vs distance in {catalogs.FullName["cf2"]}', fontsize=TitleSize, pad=15)

    fig.tight_layout()

    if save:
        os.makedirs('other graphs', exist_ok=True)
        fig.savefig('other graphs/btotVsDist.png', dpi=120)

    if show:
        plt.show()

    plt.close(fig)


def malmquist(catalog, show=True, save=True):
    visuals = {
        '2mrs': {
            'r offset': -400,
            'mag abs offset': 0
        },
        '2mrsg': {
            'r offset': 0,
            'mag abs offset': -8.5
        },
        'cf2': {
            'r offset': -140,
            'mag abs offset': -10
        },
        'bzcat': {
            'r offset': -2600,
            'mag abs offset': 0
        }
    }

    r_offset = visuals[catalog]['r offset']
    mag_abs_offset = visuals[catalog]['mag abs offset']

    real = catalogs.read(catalog)
    sim = catalogs.read(f'{catalog}_s')
    sel = catalogs.read(f'{catalog}_ss')

    real = real.loc[real['DIST'] < real['DIST'].max() + r_offset]
    real = real.loc[real['MAG ABS'] < real['MAG ABS'].max() + mag_abs_offset]

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    ax.scatter(sim['DIST'], sim['MAG ABS'], c='tab:green', s=1., label='simulated')
    ax.scatter(real['DIST'], real['MAG ABS'], c='tab:blue', s=4., label='real')
    ax.scatter(sel['DIST'], sel['MAG ABS'], c='tab:orange', s=1., label='simulated with selection')

    ax.set_xlabel('Distance, Mpc', fontsize=LabelSize)
    ax.set_ylabel('Absolute magnitude', fontsize=LabelSize)
    xlim = max(real['DIST'].max(), sim['DIST'].max(), sel['DIST'].max())
    ax.set_xlim(0, xlim)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=TickSize)
    ax.yaxis.offsetText.set_fontsize(TickSize)

    ax.grid(True, linewidth=0.3)
    ax.grid(True, 'minor', linewidth=0.1)

    ax.set_title(f'Malmquist bias for {catalogs.FullName[catalog]}', fontsize=TitleSize, pad=15)
    leg = ax.legend(loc='upper right', fontsize=LegendSize, frameon=False, fancybox=False)
    for item in leg.legendHandles:
        item.set_sizes([30])

    fig.tight_layout()

    if save:
        os.makedirs('malmquist', exist_ok=True)
        fig.savefig(f'malmquist/{catalog}.png', dpi=120)

    if show:
        plt.show()

    plt.close(fig)
