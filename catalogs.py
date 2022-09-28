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
new_columns_names = ['RA', 'DEC', 'GLON', 'GLAT', 'DIST', 'MAG']

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option("display.show_dimensions", True)
pd.set_option("display.precision", 1)

np.set_printoptions(precision=2)


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

    selected_columns = data[['RA', 'DEC', 'GLON', 'GLAT', 'V', 'MKTC']]

    new_data = pd.DataFrame()
    new_data[new_columns_names] = selected_columns.copy()

    # velocity to Mpc
    new_data['DIST'] /= H
    # removing nearby galaxies due to inaccuracy in determining the distance
    new_data = new_data.loc[(new_data['DIST'] > 50)]

    return new_data


@print_and_format
def read_2mrsg():
    """
    TODO посмотреть на совпадение между cf2 и 2mrsG
    TODO пересекаются 4463 объекта { pd.merge(d1, d2, how='inner', on=['pgc']) }
    TODO расстояния сильно варьируются (смотреть на edd)

    EDD:
        http://edd.ifa.hawaii.edu/dfirst.php?
    """

    data = pd.read_csv('datasets/2mrsG.csv')

    data = data.loc[(data['Vgp'] > 0) & (data['K_t'] > 0)]

    data['DEC'] = angles.to_dec(data['GLong'], data['GLat'])
    data['RA'] = angles.to_ra(data['GLong'], data['GLat'], data['DEC'])

    selected_columns = data[['RA', 'DEC', 'GLong', 'GLat', 'Vgp', 'K_t']]

    new_data = pd.DataFrame()
    new_data[new_columns_names] = selected_columns.copy()

    # velocity to Mpc
    new_data['DIST'] /= H

    return new_data


@print_and_format
def read_cf2():
    """
    TODO K_S от d_L

    EDD:
        http://edd.ifa.hawaii.edu/dfirst.php?
    """

    data = pd.read_csv('datasets/cf2.csv')

    data = data.loc[(data['Dist'] > 0) & (data['Btot'] > 0)]

    selected_columns = data[['RAJ', 'DeJ', 'Glon', 'Glat', 'Dist', 'Btot']]

    new_data = pd.DataFrame()
    new_data[new_columns_names] = selected_columns.copy()

    new_data['RA'] = angles.from_hms(new_data['RA'])
    new_data['DEC'] = angles.from_dms(new_data['DEC'])

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

    selected_columns = data[['RA', 'DEC', 'GLON', 'GLAT', 'z', 'Rmag']]

    new_data = pd.DataFrame()
    new_data[new_columns_names] = selected_columns.copy()

    # redshift to Mpc
    new_data['DIST'] = new_data['DIST'].apply(r_comoving)

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

    print(data)

    selected_columns = data[['RA', 'DEC', 'GLON', 'GLAT', 'Z', 'BMAG']]

    new_data = pd.DataFrame()
    new_data[new_columns_names] = selected_columns.copy()

    # redshift to Mpc
    new_data['DIST'] = new_data['DIST'].apply(r_comoving)

    return new_data


# read_milliquas()
