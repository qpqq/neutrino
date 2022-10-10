import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import catalogs
import angles

# TODO презентацию со всеми картинками, несколько на слайд
# TODO формулы з.в. и нейтринного потока
# TODO рассказать при показе энергетического распределения: диапазон, параметры гауссианы, одни и те же бины
# TODO 2mrs не нужен

# TODO normal_pdf_logx_graph сложить для всех галактик в каталоге

# constants
EMin = 5 * 10 ** 3
EMax = 30 * 10 ** 6
EMean = 1 * 10 ** 6

binsNumber = 50
binsEdges = np.linspace(np.log10(EMin), np.log10(EMax), binsNumber)

mu = np.log10(EMean)
sigma = 0.6

mSun = -26.74
auInPc = 206265
NSunAu = 7 * 10 ** 10


def allsky(catalog):
    visuals = {
        '2mrs': {
            'par': [5, 1, 3],
            'by': 'DIST',
            'vmin': 1,
            'vmax': 349,
            'extend': 'max',
            'xlabel': 'Distance, Mpc',
            'minMag': 6,
            'maxMag': 11,
            'step': 1,
            'offset_1': 0.13,
            'offset_2': 0.27
        },
        '2mrsg': {
            'par': [5, 1, 3],
            'by': 'DIST',
            'vmin': 1,
            'vmax': 349,
            'extend': 'neither',
            'xlabel': 'Distance, Mpc',
            'minMag': 6,
            'maxMag': 11,
            'step': 1,
            'offset_1': 0.13,
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
            'offset_1': 0.3,
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
            'offset_1': 0.22,
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
            'offset_1': 0.25,
            'offset_2': 0.5
        }
    }

    a, b, c = visuals[catalog]['par']
    by = visuals[catalog]['by']
    vmin = visuals[catalog]['vmin']
    vmax = visuals[catalog]['vmax']
    extend = visuals[catalog]['extend']
    xlabel = visuals[catalog]['xlabel']
    min_mag_sample = visuals[catalog]['minMag']
    max_mag_sample = visuals[catalog]['maxMag']
    step = visuals[catalog]['step']
    offset_1 = visuals[catalog]['offset_1']
    offset_2 = visuals[catalog]['offset_2']

    def calc_size(mag):
        return a + b * (max_mag - mag) ** c

    data = catalogs.read(catalog)
    max_mag = max(data['MAG'].max(), max_mag_sample)

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

    ax.set_title(fr'{catalogs.fullName[catalog]}', fontsize=35, pad=45)
    cbar.ax.set_xlabel(xlabel, fontsize=20, labelpad=-70)
    ax.legend(loc='upper right', fontsize=20, markerscale=0, frameon=False, fancybox=False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    cbar.ax.tick_params(axis='both', which='major', labelsize=15)

    # circles
    ax2 = fig.add_axes([0.83, 0.038, 0.13, 0.1])
    ax2.axis('off')
    sample_mags = np.arange(min_mag_sample, max_mag_sample + step, step)
    for _mag in sample_mags:
        ax2.scatter(_mag, 0, s=calc_size(_mag), marker='o', c='black', clip_on=False)
        offset_11 = 0
        if _mag // 10 > 0:
            offset_11 = offset_1
        ax2.text(_mag - offset_1 - offset_11, -0.033, f'{str(_mag)}', fontsize=15)
    ax2.text((min_mag_sample + max_mag_sample) / 2 - offset_2, 0.022, r'$m$', fontsize=20)

    fig.tight_layout()

    os.makedirs('all-sky graphs', exist_ok=True)
    fig.savefig(f'all-sky graphs/{catalog}.png', dpi=120)
    plt.show()


def normal_pdf_logx_hist(n_particles, z=None):
    # TODO ускорить

    n_distr = 10 ** 5
    rng = np.random.default_rng()

    def histogram(distribution):
        return np.histogram(distribution, binsEdges)[0].astype('float64')

    if np.isscalar(n_particles):
        if z is None:
            z = 0

        _mu = mu - np.log10(z + 1)
        distr = rng.normal(_mu, sigma, n_distr)
        hist = histogram(distr)

    else:
        if z is None:
            distr = rng.normal(mu, sigma, n_distr)
            hist = histogram(distr)
            hist = np.tile(hist, (len(n_particles), 1))  # duplicates array n times

        else:
            _mu = mu - np.log10(z + 1)
            distr = rng.normal(_mu, sigma, (n_distr, len(z)))
            hist = np.apply_along_axis(histogram, 0, distr).T

    hist = (hist.T * n_particles).T
    hist /= n_distr

    return hist


def normal_pdf_logx_graph(n_particles, z=None):
    from scipy.stats import norm

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    hist = normal_pdf_logx_hist(n_particles, z)
    ax.stairs(hist, binsEdges, fill=False, color='tab:red')

    x = np.linspace(np.log10(EMin), np.log10(EMax), 10 ** 4)
    y = norm.pdf(x, loc=mu, scale=sigma) * n_particles * (binsEdges[1] - binsEdges[0])
    ax.plot(x, y, linestyle='dotted', color='tab:blue')

    ax.set_xlabel('$E$, eV', fontsize=20)
    ax.set_ylabel('$N$', fontsize=20)
    ax.set_xlim(np.log10(EMin), np.log10(EMax))
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.offsetText.set_fontsize(18)

    ax.grid(True, linewidth=0.3)
    ax.grid(True, 'minor', linewidth=0.1)

    title = fr'Normal distribution {n_particles} particles, $\sigma = {sigma}$, $\mu = {mu}$'
    ax.set_title(title, fontsize=25, pad=15)

    fig.tight_layout()
    plt.show()


def hist_by(catalog, by):
    visuals = {
        'DIST': {
            'xlabel': 'Distance, Mpc',
            'xlim': '0',
            'dirName': 'histogram by distance',
            'fileName': f'histogram by distance/{catalog}.png'
        },
        'MAG': {
            'xlabel': 'Magnitude',
            'xlim': 'bin',
            'dirName': 'histogram by magnitude',
            'fileName': f'histogram by magnitude/{catalog}.png'
        },
        'Z': {
            'xlabel': 'Redshift',
            'xlim': '0',
            'dirName': 'histogram by redshift',
            'fileName': f'histogram by redshift/{catalog}.png'
        }
    }

    data = catalogs.read(catalog)

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    hist, bins, _ = ax.hist(data[by], 25, color='tab:blue')

    ax.set_xlabel(visuals[by]['xlabel'], fontsize=20)
    ax.set_ylabel('$N$', fontsize=20)
    xlim = 0
    if visuals[by]['xlim'] == 'bin':
        xlim = bins[0]
    ax.set_xlim(xlim, bins[-1])
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.offsetText.set_fontsize(18)

    ax.grid(True, linestyle=':', linewidth=0.3)
    ax.grid(True, 'minor', linestyle=':', linewidth=0.1)

    ax.set_title(fr'{catalogs.fullName[catalog]}, $N={len(data.index)}$', fontsize=25, pad=15)

    fig.tight_layout()

    os.makedirs(visuals[by]['dirName'], exist_ok=True)
    fig.savefig(visuals[by]['fileName'], dpi=120)

    plt.show()


def scaling(data):
    data['NEU'] = NSunAu * 10 ** (0.4 * (mSun - data['MAG']))


def gauss(catalog, glon, glat, dgl, n_grid, fwhm):
    from scipy.ndimage import gaussian_filter

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
    scaling(data)

    # extracts specific column
    data['BAR'] = normal_pdf_logx_hist(data['NEU'].to_numpy(), data['Z'].to_numpy())[:, 25]

    x = np.linspace(glon_min, glon_max, n_grid + 1)
    y = np.linspace(glat_min, glat_max, n_grid + 1)
    xi = np.searchsorted(x, data['GLON']) - 1
    yi = np.searchsorted(y, data['GLAT']) - 1

    # grid with min float values
    z = np.full(shape=(n_grid, n_grid), fill_value=sys.float_info.min)
    for i in range(len(xi)):
        z[yi[i], xi[i]] += data.iloc[i]['BAR']

    # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    sigma_gauss = fwhm / 2.355
    z = gaussian_filter(z, sigma=sigma_gauss)

    offset_grid //= 2
    x = x[offset_grid:-offset_grid]
    y = y[offset_grid:-offset_grid]
    z = z[offset_grid:-offset_grid, offset_grid:-offset_grid]

    os.makedirs('gauss graphs', exist_ok=True)
    data = data.sort_values('BAR', ascending=False).reset_index(drop=True)
    data.to_csv(f'gauss graphs/{catalog}.csv', index=False, float_format='%.15f')

    return x, y, z


def gauss_graph(catalog):
    fig = plt.figure(figsize=(14, 10.5))

    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    # values increase counterclockwise
    ax.invert_xaxis()

    glon = 90
    glat = 0
    dgl = 60
    n_grid = 100
    fwhm = 1.5
    x, y, z = gauss(catalog, glon, glat, dgl, n_grid, fwhm)

    vmin = 10 ** -8
    pc = ax.pcolormesh(x, y, z, norm=mpl.colors.LogNorm(vmin=vmin), cmap='jet')

    cbar = fig.colorbar(pc, pad=0.01)
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel(r'Neutrino flux, [s$^{-1}$ cm$^{-2}$]', fontsize=20)

    ax.set_xlabel('Galactic longitude', fontsize=20)
    ax.set_ylabel('Galactic latitude', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.set_title(f'{catalogs.fullName[catalog]}, {n_grid}x{n_grid}, fwhm = {fwhm}', fontsize=25, pad=15)

    fig.tight_layout()
    fig.savefig(f'gauss graphs/{catalog}.png', dpi=120)
    plt.show()


def diff():
    # TODO сделать более явным (покрупнее шрифт)

    d1 = pd.read_csv('datasets/2mrsG.csv')
    d1 = d1.loc[(d1['Vgp'] > 0) & (d1['K_t'] > 0)]

    d2 = pd.read_csv('datasets/cf2.csv')
    d2 = d2.loc[(d2['Dist'] > 0) & (d2['Btot'] > 0) & (d2['Dist'] < 350)]

    d3 = pd.merge(d1, d2, how='inner', on=['pgc'])
    d3['DIFF'] = d3['Vgp'] / catalogs.H - d3['Dist']

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    ax.hist(d3['DIFF'], 25)

    ax.set_xlabel('Difference in distance, Mpc', fontsize=20)
    ax.set_ylabel('$N$', fontsize=20)
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.offsetText.set_fontsize(18)

    ax.grid(True, linestyle=':', linewidth=0.3)
    ax.grid(True, 'minor', linestyle=':', linewidth=0.1)

    ax.set_title(fr'Difference in distance between cf2 and 2mrsg catalogs, $N={len(d3.index)}$', fontsize=25, pad=15)

    fig.tight_layout()

    os.makedirs('other graphs', exist_ok=True)
    fig.savefig('other graphs/diff.png', dpi=120)

    plt.show()


def k_s_vs_d_l():
    # TODO сделать его для Btot

    data = pd.read_csv('datasets/cf2.csv')

    data = data.loc[(data['Dist'] > 0) & (data['Ks'] > 0)]

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    ax.scatter(data['Dist'], data['Ks'], s=1)

    ax.set_xlabel('Distance, Mpc', fontsize=20)
    ax.set_ylabel('$K_s$', fontsize=20)
    ax.set_xlim(0)
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.offsetText.set_fontsize(18)

    ax.grid(True, linestyle='--', linewidth=0.3)
    ax.grid(True, 'minor', linestyle='--', linewidth=0.1)

    ax.set_title(fr'$K_s$ magnitude vs distance in {catalogs.fullName["cf2"]}', fontsize=25, pad=15)

    fig.tight_layout()

    os.makedirs('other graphs', exist_ok=True)
    fig.savefig('other graphs/kSVsDl.png', dpi=120)

    plt.show()
