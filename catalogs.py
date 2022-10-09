import numpy as np
import pandas as pd
from scipy import integrate
from astropy.io import fits
from astropy.table import Table

import angles

# speed of light
c = 299792.458
# Hubble constant
H = 70

# matter density
omega_m = 0.3
# vacuum density
omega_v = 0.7

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


def print_and_format(func):
    def wrapper():
        data = func()
        data = data.astype('float64')

        data = data.sort_values('DIST').reset_index(drop=True)
        # print(data)
        # print()

        return data

    return wrapper


@print_and_format
def read_2mrs():
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

    # velocity to redshift
    new_data['Z'] = new_data['DIST'] / c
    # velocity to Mpc
    new_data['DIST'] /= H
    # removing nearby galaxies due to inaccuracy in determining the distance
    new_data = new_data.loc[(new_data['DIST'] > 50)]

    return new_data


@print_and_format
def read_2mrsg():
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

    # velocity to redshift
    new_data['Z'] = new_data['DIST'] / c
    # velocity to Mpc
    new_data['DIST'] /= H

    return new_data


@print_and_format
def read_cf2():
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

    # Mpc to redshift
    new_data['Z'] = new_data['DIST'] * H / c

    return new_data


@print_and_format
def read_bzcat():
    """
    TODO насколько отличаются звездные величины в разных фильтрах у АЧТ (хотелось бы нормировать з.в.)

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

    return new_data


@print_and_format
def read_milliquas():
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

    data['GLAT'] = angles.to_dec(data['RA'], data['DEC'])
    data['GLON'] = angles.to_ra(data['RA'], data['DEC'], data['GLAT'])

    new_data = pd.DataFrame()
    new_data[new_columns_names] = data[['RA', 'DEC', 'GLON', 'GLAT', 'Z', 'RMAG', 'BMAG']]

    # redshift to Mpc
    new_data['DIST'] = new_data['Z'].apply(r_comoving)

    return new_data


def read(name):
    if name == '2mrs':
        return read_2mrs()

    elif name == '2mrsg':
        return read_2mrsg()

    elif name == 'cf2':
        return read_cf2()

    elif name == 'bzcat':
        return read_bzcat()

    elif name == 'milliquas':
        return read_milliquas()
