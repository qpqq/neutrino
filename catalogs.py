import os
import numpy as np
import pandas as pd
from scipy import integrate
from astropy.io import fits
from astropy.table import Table

import angles

c = 299792.458  # speed of light
H = 70  # Hubble constant

omega_m = 0.3  # matter density
omega_v = 0.7  # vacuum density

EMin = 5 * 10 ** 3
EMax = 30 * 10 ** 6
EMean = 1 * 10 ** 6

binsNumber = 50
bestBin = 25
binsEdges = np.linspace(np.log10(EMin), np.log10(EMax), binsNumber + 1)

mu = np.log10(EMean)
sigma = 0.6

mSun = -26.74  # Sun apparent magnitude at 1 a.u.
nSunAu = 7 * 10 ** 10  # neutrino flux from Sun at 1 a.u.

# from https://doi.org/10.3847/1538-4365/aabfdf
mVKs = 1.542  # V-2MASS_Ks color index
mBV = 0.629  # B-V color index
mVR = 0.387  # V-R color index

# identical columns of new tables
new_columns_names = ['RA', 'DEC', 'GLON', 'GLAT', 'Z', 'DIST', 'MAG']

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.show_dimensions', True)
pd.set_option('display.precision', 1)

np.set_printoptions(precision=2)

fullName = {
    '2mrs': '2MASS Redshift Survey',
    '2mrsg': '2MASS Redshift Survey (by groups)',
    'cf2': 'Cosmicflows-2.1',
    'bzcat': 'Roma-BZCAT',
    'milliquas': 'Million Quasars'
}


def r_comoving(z):
    return c / H * integrate.quad(lambda y: 1 / np.sqrt(omega_v + omega_m * (1 + y) ** 3), 0, z)[0]


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
            chunk = 1000
            for i in range(0, size, chunk):
                print(f'{i}/{size}')

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


def prepare_and_write(func):
    def wrapper():
        data, name = func()
        data = data.astype('float64')

        data['MAG ABS'] = data['MAG'] - 25 - 5 * np.log10(data['DIST'])
        data['NEU'] = nSunAu * 10 ** (0.4 * (mSun - data['MAG']))

        hist = normal_pdf_logx_hist(data['NEU'].to_numpy(), data['Z'].to_numpy())
        for i in range(binsNumber):
            data[f'BIN {i}'] = hist[:, i]

        data = data.sort_values('DIST').reset_index(drop=True)
        data.to_csv(f'datasets/{name}_finished.csv.zip', index=False, float_format='%.15f')

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

    hdul = fits.open('datasets/2mrs.fits')
    data = Table(hdul[1].data).to_pandas()
    hdul.close()

    data = data.loc[(data['V'] > 0) & (data['MKTC'] > 0)]

    new_data = pd.DataFrame()
    new_data[new_columns_names] = data[['RA', 'DEC', 'GLON', 'GLAT', 'MKC', 'V', 'MKTC']]

    # m_V - m_ks = mVKs => m_V = m_ks + mVKs
    # m_V - m_R = mVR => m_R = m_V - mVR = m_ks + mVKs - mVR
    new_data['MAG'] = new_data['MAG'] + mVKs - mVR
    # velocity to redshift
    new_data['Z'] = new_data['DIST'] / c
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

    data = pd.read_csv('datasets/2mrsG.csv')

    data = data.loc[(data['Vgp'] > 0) & (data['K_t'] > 0)]

    data['DEC'] = angles.to_dec(data['GLong'], data['GLat'])
    data['RA'] = angles.to_ra(data['GLong'], data['GLat'], data['DEC'])

    new_data = pd.DataFrame()
    new_data[new_columns_names] = data[['RA', 'DEC', 'GLong', 'GLat', 'pgc', 'Vgp', 'K_t']]

    # m_V - m_ks = mVKs => m_V = m_ks + mVKs
    # m_V - m_R = mVR => m_R = m_V - mVR = m_ks + mVKs - mVR
    new_data['MAG'] = new_data['MAG'] + mVKs - mVR
    # velocity to redshift
    new_data['Z'] = new_data['DIST'] / c
    # velocity to Mpc
    new_data['DIST'] /= H

    return new_data, '2mrsG'


@prepare_and_write
def prepare_cf2():
    """
    EDD:
        http://edd.ifa.hawaii.edu/dfirst.php?
    """

    data = pd.read_csv('datasets/cf2.csv')

    data = data.loc[(data['Dist'] > 0) & (data['Btot'] > 0) & (data['Dist'] < 350)]

    new_data = pd.DataFrame()
    new_data[new_columns_names] = data[['RAJ', 'DeJ', 'Glon', 'Glat', 'pgc', 'Dist', 'Btot']]

    new_data['RA'] = angles.from_hms(new_data['RA'])
    new_data['DEC'] = angles.from_dms(new_data['DEC'])

    # m_B - m_V = mBV => m_V = m_B - mBV
    # m_V - m_R = mVR => m_R = m_V - mVR = m_B - mBV - mVR
    new_data['MAG'] = new_data['MAG'] - mBV - mVR
    # Mpc to redshift
    new_data['Z'] = new_data['DIST'] * H / c

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

    hdul = fits.open('datasets/bzcat.fits')
    data = Table(hdul[1].data).to_pandas()
    hdul.close()

    data = data.loc[(data['z'] > 0) & (data['Rmag'] > 0)]

    data['RA'] = 15 * (data['RAh'] + data['RAm'] / 60 + data['RAs'] / 60 / 60)
    sign = np.where(data['DE-'] == '-', -1, 1)
    data['DEC'] = sign * (data['DEd'] + data['DEm'] / 60 + data['DEs'] / 60 / 60)

    new_data = pd.DataFrame()
    new_data[new_columns_names] = data[['RA', 'DEC', 'GLON', 'GLAT', 'z', 'Seq', 'Rmag']]

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

    hdul = fits.open('datasets/milliquas.fits')
    data = Table(hdul[1].data).to_pandas()
    hdul.close()

    data = data.loc[(data['Z'] != np.nan) & (data['Z'] > 0) & (data['BMAG'] > 0)]

    data['GLAT'] = angles.to_glat(data['RA'], data['DEC'])
    data['GLON'] = angles.to_glon(data['RA'], data['DEC'], data['GLAT'])

    new_data = pd.DataFrame()
    new_data[new_columns_names] = data[['RA', 'DEC', 'GLON', 'GLAT', 'Z', 'BMAG', 'RMAG']]

    # redshift to Mpc
    new_data['DIST'] = new_data['Z'].apply(r_comoving)

    return new_data, 'milliquas'


def read(name):
    if not os.path.isfile(f'datasets/{name}_finished.csv.zip'):
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

    return pd.read_csv(f'datasets/{name}_finished.csv.zip')
