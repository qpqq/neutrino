import os

import numpy as np
import pandas as pd
from scipy import integrate, stats, optimize
from astropy.io import fits
from astropy.table import Table

import angles

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.show_dimensions', True)
pd.set_option('display.precision', 1)

np.set_printoptions(precision=2)

C = 299792.458  # speed of light
H = 70  # Hubble constant

Omega_m = 0.3  # matter density
Omega_v = 0.7  # vacuum density

EMin = 5 * 10 ** 3
EMax = 30 * 10 ** 6
EMean = 1 * 10 ** 6

BinsNumber = 50
BestBin = 25
BinsEdges = np.linspace(np.log10(EMin), np.log10(EMax), BinsNumber + 1)

Mu = np.log10(EMean)
Sigma = 0.6

mSun = -26.74  # Sun apparent magnitude at 1 a.u.
NSunAu = 7 * 10 ** 10  # neutrino flux from Sun at 1 a.u.

# from https://doi.org/10.3847/1538-4365/aabfdf
mVKs = 1.542  # V-2MASS_Ks color index
mBV = 0.629  # B-V color index
mVR = 0.387  # V-R color index

# identical columns of new tables
NewColumnsNames = ['RA', 'DEC', 'GLON', 'GLAT', 'Z', 'DIST', 'MAG']

FullName = {
    '2mrs': '2MASS Redshift Survey',
    '2mrsg': '2MASS Redshift Survey (by groups)',
    'cf2': 'Cosmicflows-2.1',
    'bzcat': 'Roma-BZCAT',
    'milliquas': 'Million Quasars'
}

for Name in list(FullName.keys()):
    FullName[f'{Name}_s'] = FullName[Name] + ' simulated'
    FullName[f'{Name}_ss'] = FullName[Name] + ' simulated with selection'


def r_comoving(z):
    return C / H * integrate.quad(lambda y: 1 / np.sqrt(Omega_v + Omega_m * (1 + y) ** 3), 0, z)[0]


def make_z_dist():
    data = pd.DataFrame()
    data['Z'] = np.linspace(0, 10, 10000)
    data['DIST'] = data['Z'].apply(r_comoving)

    data.to_csv(f'datasets/z_dist.csv', index=False, float_format='%.3f')


def normal_pdf_logx_hist(n_particles, z=None):
    n_particles = np.asarray(n_particles)

    if z is None:
        z = np.zeros(n_particles.size)

    z = np.asarray(z)

    mu = Mu - np.log10(z + 1)
    mu = np.atleast_2d(mu).T  # transpose 1-D array
    mu = np.tile(mu, (1, BinsEdges.size))

    hist = stats.norm.cdf(BinsEdges, loc=mu, scale=Sigma)  # returns points on BinsEdges
    hist = hist[:, 1:] - hist[:, :-1]  # returns points in the center of BinsEdges
    hist = (hist.T * n_particles).T

    return hist


def prepare_and_write(func):
    def wrapper():
        data, name = func()
        data = data.astype('float64')

        data['MAG ABS'] = data['MAG'] - 25 - 5 * np.log10(data['DIST'] / (1 + data['Z']))
        data['NEU'] = NSunAu / (1 + data['Z']) ** 4 * 10 ** (0.4 * (mSun - data['MAG']))

        data = data.sort_values('DIST').reset_index(drop=True)
        data.to_csv(f'datasets/{name}.csv.zip', index=False, float_format='%.15f')

    return wrapper


@prepare_and_write
def prepare_2mrs():
    """
    Main page:
        http://tdc-www.harvard.edu/2mrs/
    Readme:
        http://tdc-www.harvard.edu/2mrs/2mrs_readme.html
    EDD:
        http://edd.ifa.hawaii.edu/dfirst.php?
    """

    hdul = fits.open('datasets/2mrs_full.fits')
    data = Table(hdul[1].data).to_pandas()
    hdul.close()

    data = data.loc[(data['V'] > 0) & (data['MKTC'] > 0)]

    new_data = pd.DataFrame()
    new_data[NewColumnsNames] = data[['RA', 'DEC', 'GLON', 'GLAT', 'MKC', 'V', 'MKTC']]

    # m_V - m_ks = mVKs => m_V = m_ks + mVKs
    # m_V - m_R = mVR => m_R = m_V - mVR = m_ks + mVKs - mVR
    new_data['MAG'] = new_data['MAG'] + mVKs - mVR
    # velocity to redshift
    new_data['Z'] = new_data['DIST'] / C
    # velocity to Mpc
    new_data['DIST'] /= H
    # removing nearby galaxies due to inaccuracy in determining the distance
    new_data = new_data.loc[(new_data['DIST'] > 50)]

    return new_data, '2mrs'


@prepare_and_write
def prepare_2mrsg():
    """
    Combined of two catalogs

    EDD:
        http://edd.ifa.hawaii.edu/dfirst.php?
    """

    data = pd.read_csv('datasets/2mrsg_full.csv')

    data = data.loc[(data['Vgp'] > 0) & (data['K_t'] > 0)]

    data['DEC'] = angles.to_dec(data['GLong'], data['GLat'])
    data['RA'] = angles.to_ra(data['GLong'], data['GLat'], data['DEC'])

    new_data = pd.DataFrame()
    new_data[NewColumnsNames] = data[['RA', 'DEC', 'GLong', 'GLat', 'pgc', 'Vgp', 'K_t']]

    # m_V - m_ks = mVKs => m_V = m_ks + mVKs
    # m_V - m_R = mVR => m_R = m_V - mVR = m_ks + mVKs - mVR
    new_data['MAG'] = new_data['MAG'] + mVKs - mVR
    # velocity to redshift
    new_data['Z'] = new_data['DIST'] / C
    # velocity to Mpc
    new_data['DIST'] /= H

    return new_data, '2mrsG'


@prepare_and_write
def prepare_cf2():
    """
    EDD:
        http://edd.ifa.hawaii.edu/dfirst.php?
    """

    data = pd.read_csv('datasets/cf2_full.csv')

    data = data.loc[(data['Dist'] > 0) & (data['Btot'] > 0) & (data['Dist'] < 350)]

    new_data = pd.DataFrame()
    new_data[NewColumnsNames] = data[['RAJ', 'DeJ', 'Glon', 'Glat', 'pgc', 'Dist', 'Btot']]

    new_data['RA'] = angles.from_hms(new_data['RA'])
    new_data['DEC'] = angles.from_dms(new_data['DEC'])

    # m_B - m_V = mBV => m_V = m_B - mBV
    # m_V - m_R = mVR => m_R = m_V - mVR = m_B - mBV - mVR
    new_data['MAG'] = new_data['MAG'] - mBV - mVR
    # Mpc to redshift
    new_data['Z'] = new_data['DIST'] * H / C

    return new_data, 'cf2'


@prepare_and_write
def prepare_bzcat():
    """
    Main page:
        https://heasarc.gsfc.nasa.gov/W3Browse/all/romabzcat.html
    Catalog page:
        http://cdsarc.u-strasbg.fr/ftp/cats/VII/274/
    Readme:
        http://cdsarc.u-strasbg.fr/ftp/cats/VII/274/ReadMe
    """

    hdul = fits.open('datasets/bzcat_full.fits')
    data = Table(hdul[1].data).to_pandas()
    hdul.close()

    data = data.loc[(data['z'] > 0) & (data['Rmag'] > 0)]

    data['RA'] = 15 * (data['RAh'] + data['RAm'] / 60 + data['RAs'] / 60 / 60)
    sign = np.where(data['DE-'] == '-', -1, 1)
    data['DEC'] = sign * (data['DEd'] + data['DEm'] / 60 + data['DEs'] / 60 / 60)

    new_data = pd.DataFrame()
    new_data[NewColumnsNames] = data[['RA', 'DEC', 'GLON', 'GLAT', 'z', 'Seq', 'Rmag']]

    # redshift to Mpc
    new_data['DIST'] = new_data['Z'].apply(r_comoving)

    return new_data, 'bzcat'


@prepare_and_write
def prepare_milliquas():
    """
    Main page:
        https://heasarc.gsfc.nasa.gov/w3browse/all/milliquas.html
    Catalog page:
        https://quasars.org/milliquas.htm
    Readme:
        https://quasars.org/Milliquas-ReadMe.txt
    """

    hdul = fits.open('datasets/milliquas_full.fits')
    data = Table(hdul[1].data).to_pandas()
    hdul.close()

    data = data.loc[(data['Z'] != np.nan) & (data['Z'] > 0) & (data['BMAG'] > 0)]

    data['GLAT'] = angles.to_glat(data['RA'], data['DEC'])
    data['GLON'] = angles.to_glon(data['RA'], data['DEC'], data['GLAT'])

    new_data = pd.DataFrame()
    new_data[NewColumnsNames] = data[['RA', 'DEC', 'GLON', 'GLAT', 'Z', 'BMAG', 'RMAG']]

    # redshift to Mpc
    new_data['DIST'] = new_data['Z'].apply(r_comoving)

    return new_data, 'milliquas'


def pdf(x, a, mu, sigma):
    return a * stats.norm.pdf(x, loc=mu, scale=sigma)


def norm_fit(distr):
    n_hist = 25

    hist, bins = np.histogram(distr, n_hist)
    hist = hist.astype('float64')
    bins = (bins[1:] + bins[:-1]) / 2

    a = hist.sum() * (bins[1] - bins[0])
    mu = distr.mean()
    sigma = 1

    output = optimize.curve_fit(pdf, bins, hist, p0=[a, mu, sigma])  # err = np.sqrt(np.diag(output[1]))
    par = output[0]

    return par[1], par[2]


def simulate(name):
    def make_df(ra, dec, glon, glat, redshift, _dist, m, m_abs):
        new = pd.DataFrame()
        new['RA'] = ra
        new['DEC'] = dec
        new['GLON'] = glon
        new['GLAT'] = glat
        new['Z'] = redshift
        new['DIST'] = _dist
        new['MAG'] = m
        new['MAG ABS'] = m_abs
        new['NEU'] = NSunAu / (1 + new['Z']) ** 4 * 10 ** (0.4 * (mSun - new['MAG']))
        hist = normal_pdf_logx_hist(new['NEU'].to_numpy(), new['Z'].to_numpy())
        for i in range(BinsNumber):
            new[f'BIN {i}'] = hist[:, i]

        return new.sort_values('DIST').reset_index(drop=True)

    def make_sim():
        _x, _y, _z, _dist = x, y, z, dist

        _mask = np.random.choice(_x.size, replace=False, size=n)
        _x, _y, _z, _dist = _x[_mask], _y[_mask], _z[_mask], _dist[_mask]

        redshift_dist = pd.read_csv('datasets/z_dist.csv')
        redshift_i = np.searchsorted(redshift_dist['DIST'], _dist) - 1
        redshift = redshift_dist['Z'].iloc[redshift_i].to_numpy()

        mu, sigma = norm_fit(old['MAG ABS'])
        m_abs = rng.normal(mu + mu_offset, sigma, _x.size)
        m = m_abs + 25 + 5 * np.log10(_dist / (1 + redshift))

        glon, glat = angles.to_gal_coords(_x, _y, _z, _dist)
        dec = angles.to_dec(glon, glat)
        ra = angles.to_ra(glon, glat, dec)

        new = make_df(ra, dec, glon, glat, redshift, _dist, m, m_abs)
        new.to_csv(f'datasets/{name}_s.csv.zip', index=False, float_format='%.15f')

    def make_sel():
        _x, _y, _z, _dist = x, y, z, dist

        redshift_dist = pd.read_csv('datasets/z_dist.csv')
        redshift_i = np.searchsorted(redshift_dist['DIST'], _dist) - 1
        redshift = redshift_dist['Z'].iloc[redshift_i].to_numpy()

        mu, sigma = norm_fit(old['MAG ABS'])
        m_abs = rng.normal(mu + mu_offset, sigma, _x.size)
        m = m_abs + 25 + 5 * np.log10(_dist / (1 + redshift))

        _mask = (m < old['MAG'].max() + mag_max_offset)
        _x, _y, _z, _dist, redshift, m_abs, m = \
            _x[_mask], _y[_mask], _z[_mask], _dist[_mask], redshift[_mask], m_abs[_mask], m[_mask]
        _mask = np.random.choice(_x.size, replace=False, size=n)
        _x, _y, _z, _dist, redshift, m_abs, m = \
            _x[_mask], _y[_mask], _z[_mask], _dist[_mask], redshift[_mask], m_abs[_mask], m[_mask]

        glon, glat = angles.to_gal_coords(_x, _y, _z, _dist)
        dec = angles.to_dec(glon, glat)
        ra = angles.to_ra(glon, glat, dec)

        new = make_df(ra, dec, glon, glat, redshift, _dist, m, m_abs)
        new.to_csv(f'datasets/{name}_ss.csv.zip', index=False, float_format='%.15f')

    visuals = {
        '2mrs': {
            'r offset': -400,
            'mu offset': 1.3,
            'mag max offset': -0.1
        },
        '2mrsg': {
            'r offset': 0,
            'mu offset': 1.3,
            'mag max offset': -0.1
        },
        'cf2': {
            'r offset': -140,
            'mu offset': 0,
            'mag max offset': -6.1
        },
        'bzcat': {
            'r offset': -2600,
            'mu offset': 0,
            'mag max offset': -4.4
        }
    }

    r_offset = visuals[name]['r offset']
    mu_offset = visuals[name]['mu offset']
    mag_max_offset = visuals[name]['mag max offset']

    old = read(name)

    n = len(old.index)
    n_tot = max(300 * n, 10 ** 6)
    r = old['DIST'].max() + r_offset

    rng = np.random.default_rng()
    x, y, z = r * (2 * rng.random((3, n_tot)) - 1)

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    mask = (dist < r)
    x, y, z, dist = x[mask], y[mask], z[mask], dist[mask]

    if not os.path.isfile('datasets/z_dist.csv'):
        make_z_dist()

    make_sim()
    make_sel()


def read(name):
    if not os.path.isfile(f'datasets/{name}.csv.zip'):
        if name == '2mrs':
            prepare_2mrs()

        elif name == '2mrsg':
            prepare_2mrsg()

        elif name == 'cf2':
            prepare_cf2()

        elif name == 'bzcat':
            prepare_bzcat()

        elif name == 'milliquas':
            prepare_milliquas()

        elif name[-2:] == '_s':
            simulate(name[:-2])

        elif name[-3:] == '_ss':
            simulate(name[:-3])

    data = pd.read_csv(f'datasets/{name}.csv.zip')

    hist = normal_pdf_logx_hist(data['NEU'].to_numpy(), data['Z'].to_numpy())
    for i in range(BinsNumber):
        data[f'BIN {i}'] = hist[:, i]

    return data
