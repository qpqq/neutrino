import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import catalogs
import angles

# TODO гистограмма для потоков (логарифм энергии)
# TODO гистограмма для потоков в столбце (логарифм энергии, используемый бин)

# TODO разность звездных величин в разных фильтрах
#  https://vk.com/doc500152640_648627931?hash=z948VlQzUFMxtCxVrb6zvVXIyIIECwjAqzHWRQvDzKT&dl=o0hgqnhtA55KssQgEXoREkaWcw5sihCEaGIILpWZWxg

# TODO промоделировать пуассоновский процесс (обычный генератор) в кубе (сфере) некоторого радиуса
#  у каждого объекта будет абсолютная величина (обычный генератор) (первое приближение генерировать через гауссиану)
#  посчитать видимую звездную
#  сделать два каталога: с селекцией и без (обрубить по видимой звездной)
#  число точек в искусственных каталогах должно быть равно числу точек в реальных
#  (можно потыкаться или моделировать сразу много)
#  картинки:
#  как в смещении Малмквиста (абсолютная от расстояния)
#  все те картинки, что мы делали для реальных каталогов (должно быть видно смещение)
#  ----------------------------------------------------------------------------------
#  должно получится 2 * 4 каталога (сделать сначала по bzcat)

# TODO (со звездочкой) исследовать каталог квазаров на однородные выборки (по Type, по Zcite)

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

tickSize = 25  # old 20
tickSizeCbar = 25  # old 20
labelSize = 25  # old 20
legendSize = 25  # old 20
titleSize = 35  # old 35
magTextSize = 20  # old 15


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
            'offset_1': 0.15,
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
            'offset_1': 0.15,
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

    ax.set_title(fr'{catalogs.fullName[catalog]}', fontsize=titleSize, pad=45)
    cbar.ax.set_xlabel(xlabel, fontsize=labelSize, labelpad=-80)
    ax.legend(loc='upper right', fontsize=legendSize, markerscale=0, frameon=False, fancybox=False)
    ax.tick_params(axis='both', which='major', labelsize=tickSize)
    cbar.ax.tick_params(axis='both', which='major', labelsize=tickSizeCbar)

    # circles
    ax2 = fig.add_axes([0.83, 0.038, 0.13, 0.1])
    ax2.axis('off')
    sample_mags = np.arange(min_mag_sample, max_mag_sample + step, step)
    for _mag in sample_mags:
        ax2.scatter(_mag, 0, s=calc_size(_mag), marker='o', c='black', clip_on=False)
        offset_11 = 0
        if _mag // 10 > 0:
            offset_11 = offset_1
        ax2.text(_mag - offset_1 - offset_11, -0.04, f'{str(_mag)}', fontsize=magTextSize)
    ax2.text((min_mag_sample + max_mag_sample) / 2 - offset_2, 0.022, r'$m$', fontsize=labelSize)

    fig.tight_layout()

    os.makedirs('all-sky graphs', exist_ok=True)
    fig.savefig(f'all-sky graphs/{catalog}.png', dpi=120)
    plt.show()


def scaling(data):
    data['NEU'] = NSunAu * 10 ** (0.4 * (mSun - data['MAG']))


def normal_pdf_logx_hist(n_particles, z=None):
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
        size = len(n_particles)

        if z is None:
            distr = rng.normal(mu, sigma, n_distr)
            hist = histogram(distr)
            hist = np.tile(hist, (size, 1))  # duplicates array n times

        else:
            hist = np.array([[]])
            chunk = 5000
            for i in range(0, size, chunk):
                chunk_cur = min(chunk, size - i)

                mu_diff = np.log10(z[i:i + chunk] + 1)

                distr = rng.normal(mu, sigma, n_distr)
                distr = np.tile(distr, (chunk_cur, 1))
                distr = (distr.T - mu_diff)

                hist_ = np.apply_along_axis(histogram, 0, distr).T
                hist = np.concatenate((hist, hist_), axis=0) if hist.size else hist_

    hist = (hist.T * n_particles).T
    hist /= n_distr

    return hist


def normal_pdf_logx_graph(n_particles, z=None):
    from scipy.stats import norm

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    hist = normal_pdf_logx_hist(n_particles, z)
    ax.stairs(hist, binsEdges, fill=False, color='tab:red', label=fr'$N={n_particles}$')

    x = np.linspace(np.log10(EMin), np.log10(EMax), 10 ** 4)
    y = norm.pdf(x, loc=mu, scale=sigma) * n_particles * (binsEdges[1] - binsEdges[0])
    ax.plot(x, y, linestyle='dotted', color='tab:blue')

    ax.set_xlabel(r'log$_{10}E$', fontsize=labelSize)
    ax.set_ylabel('$N$', fontsize=labelSize)
    ax.set_xlim(np.log10(EMin), np.log10(EMax))
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=tickSize)
    ax.yaxis.offsetText.set_fontsize(18)

    ax.grid(True, linewidth=0.3)
    ax.grid(True, 'minor', linewidth=0.1)

    title = fr'Normal distribution {n_particles} particles, $\sigma = {sigma}$, $\mu = {mu}$'
    ax.set_title(title, fontsize=titleSize, pad=15)
    leg = ax.legend(loc='upper right', fontsize=legendSize, frameon=False, fancybox=False)
    for item in leg.legendHandles:
        item.set_visible(False)

    fig.tight_layout()
    plt.show()


def normal_pdf_logx_graph_all(catalog):
    from scipy.stats import norm

    data = catalogs.read(catalog)
    # data = data.loc[(data['Z'] > 2) & (data['Z'] < 3)]
    scaling(data)

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    hist = normal_pdf_logx_hist(data['NEU'].to_numpy(), data['Z'].to_numpy())
    hist = np.sum(hist, axis=0)
    ax.stairs(hist, binsEdges, fill=False, color='tab:red', label=fr'$N={len(data.index)}$')

    x = np.linspace(np.log10(EMin), np.log10(EMax), 10 ** 4)
    n_particles = np.sum(hist)
    y = norm.pdf(x, loc=mu, scale=sigma) * n_particles * (binsEdges[1] - binsEdges[0])
    ax.plot(x, y, linestyle='dotted', color='tab:blue')

    ax.set_xlabel(r'log$_{10}E$', fontsize=labelSize)
    ax.set_ylabel(r'Neutrino flux, s$^{-1}$ cm$^{-2}$', fontsize=labelSize)
    ax.set_xlim(np.log10(EMin), np.log10(EMax))
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=tickSize)
    ax.yaxis.offsetText.set_fontsize(tickSize)

    ax.grid(True, linewidth=0.3)
    ax.grid(True, 'minor', linewidth=0.1)

    ax.set_title(fr'{catalogs.fullName[catalog]} distribution', fontsize=labelSize, pad=15)
    leg = ax.legend(loc='upper right', fontsize=legendSize, frameon=False, fancybox=False)
    for item in leg.legendHandles:
        item.set_visible(False)

    fig.tight_layout()

    os.makedirs('histogram sum', exist_ok=True)
    fig.savefig(f'histogram sum/{catalog}.png', dpi=120)

    plt.show()


def hist_by(catalog, by):
    visuals = {
        'DIST': {
            'xlabel': 'Distance, Mpc',
            'xlim': '0',
            'dirName': 'histogram by distance',
        },
        'MAG': {
            'xlabel': 'Magnitude',
            'xlim': 'bin',
            'dirName': 'histogram by magnitude',
        },
        'Z': {
            'xlabel': 'Redshift',
            'xlim': '0',
            'dirName': 'histogram by redshift',
        }
    }

    xlabel = visuals[by]['xlabel']
    xlim_mod = visuals[by]['xlim']
    dir_name = visuals[by]['dirName']

    data = catalogs.read(catalog)

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    hist, bins, _ = ax.hist(data[by], 25, color='tab:blue', label=fr'$N={len(data.index)}$')

    ax.set_xlabel(xlabel, fontsize=labelSize)
    ax.set_ylabel('$N$', fontsize=labelSize)
    xlim = 0
    if xlim_mod == 'bin':
        xlim = bins[0]
    ax.set_xlim(xlim, bins[-1])
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=tickSize)
    ax.yaxis.offsetText.set_fontsize(tickSize)

    ax.grid(True, linestyle=':', linewidth=0.3)
    ax.grid(True, 'minor', linestyle=':', linewidth=0.1)

    ax.set_title(fr'{catalogs.fullName[catalog]}', fontsize=titleSize, pad=15)
    leg = ax.legend(fontsize=legendSize, handlelength=0, frameon=False, fancybox=False)
    for item in leg.legendHandles:
        item.set_visible(False)

    fig.tight_layout()

    os.makedirs(dir_name, exist_ok=True)
    fig.savefig(dir_name + f'/{catalog}.png', dpi=120)

    plt.show()


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
    # TODO номер столбца добавить

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

    dgl = visuals[catalog]['dgl']
    vmin = visuals[catalog]['vmin']

    fig = plt.figure(figsize=(14, 10.5))

    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    # values increase counterclockwise
    ax.invert_xaxis()

    glon = 90
    glat = 0
    n_grid = 100
    fwhm = 1.5
    x, y, z = gauss(catalog, glon, glat, dgl, n_grid, fwhm)

    pc = ax.pcolormesh(x, y, z, norm=mpl.colors.LogNorm(vmin=vmin), cmap='jet')

    cbar = fig.colorbar(pc, pad=0.01)
    cbar.ax.tick_params(labelsize=tickSizeCbar)
    cbar.ax.set_ylabel(r'Neutrino flux, s$^{-1}$ cm$^{-2}$', fontsize=labelSize)

    ax.set_xlabel('Galactic longitude, degrees', fontsize=labelSize)
    ax.set_ylabel('Galactic latitude, degrees', fontsize=labelSize)

    ax.tick_params(axis='both', which='major', labelsize=tickSize)

    ax.set_title(f'{catalogs.fullName[catalog]}, {n_grid}x{n_grid}, fwhm = {fwhm}', fontsize=30, pad=15)

    fig.tight_layout()
    fig.savefig(f'gauss graphs/{catalog}.png', dpi=120)
    plt.show()


def diff():
    d1 = pd.read_csv('datasets/2mrsG.csv')
    d1 = d1.loc[(d1['Vgp'] > 0) & (d1['K_t'] > 0)]

    d2 = pd.read_csv('datasets/cf2.csv')
    d2 = d2.loc[(d2['Dist'] > 0) & (d2['Btot'] > 0) & (d2['Dist'] < 350)]

    d3 = pd.merge(d1, d2, how='inner', on=['pgc'])
    d3['DIFF'] = d3['Vgp'] / catalogs.H - d3['Dist']

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    bins = ax.hist(d3['DIFF'], 50, range=(-50, 50), label=fr'$N={len(d3.index)}$')[1]

    ax.set_xlabel('Difference in distance, Mpc', fontsize=labelSize)
    ax.set_ylabel(r'$N$', fontsize=labelSize)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=tickSize)
    ax.yaxis.offsetText.set_fontsize(tickSize)

    ax.grid(True, linestyle=':', linewidth=0.3)
    ax.grid(True, 'minor', linestyle=':', linewidth=0.1)

    ax.set_title(fr'Difference in distance between cf2 and 2mrsg catalogs', fontsize=titleSize, pad=15)
    ax.legend(fontsize=legendSize, handlelength=0, frameon=False, fancybox=False)

    fig.tight_layout()

    os.makedirs('other graphs', exist_ok=True)
    fig.savefig('other graphs/diff.png', dpi=120)

    plt.show()


def btot_vs_dist():
    data = pd.read_csv('datasets/cf2.csv')

    data = data.loc[(data['Dist'] > 0) & (data['Btot'] > 0)]

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    ax.scatter(data['Dist'], data['Btot'], s=1)

    ax.set_xlabel('Distance, Mpc', fontsize=labelSize)
    ax.set_ylabel('$B_{tot}$', fontsize=labelSize)
    ax.set_xlim(0)
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=tickSize)
    ax.yaxis.offsetText.set_fontsize(tickSize)

    ax.grid(True, linestyle='--', linewidth=0.3)
    ax.grid(True, 'minor', linestyle='--', linewidth=0.1)

    ax.set_title(r'$B_{tot}$' + f' magnitude vs distance in {catalogs.fullName["cf2"]}', fontsize=titleSize, pad=15)

    fig.tight_layout()

    os.makedirs('other graphs', exist_ok=True)
    fig.savefig('other graphs/btotVsDist.png', dpi=120)

    plt.show()


normal_pdf_logx_graph(10 ** -6)
