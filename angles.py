import numpy as np


def to_rad(angle):
    return angle / 180 * np.pi


def to_deg(angle):
    return angle / np.pi * 180


def from_dms(angle):
    """
    Conversion from ddmmss.s to degrees
    """

    sign = np.sign(angle)
    angle *= sign

    d = angle // 10000
    angle = np.where(d != 0, angle - d * 10000, angle)

    m = angle // 100
    angle = np.where(m != 0, angle - m * 100, angle)

    s = angle

    return sign * (d + m / 60 + s / 60 / 60)


def from_hms(angle):
    """
    Conversion from hhmmss.s to degrees
    """

    return 15 * from_dms(angle)


# NGP refers to the coordinate values of the north galactic pole and NCP to those of the north celestial pole
# https://en.wikipedia.org/wiki/Astronomical_coordinate_systems#Equatorial_%E2%86%94_galactic
RaNgp = 192.85948
DecNgp = 27.12825
GlonNcp = 122.93192

RaNgpR = to_rad(RaNgp)
DecNgpR = to_rad(DecNgp)
GlonNcpR = to_rad(GlonNcp)


def to_ra(glon, glat, dec):
    """
    Conversion from galactic to equatorial coordinates. All values must be in degrees

    :return: right ascension
    """

    glon_r = to_rad(glon)
    glat_r = to_rad(glat)
    dec_r = to_rad(dec)

    d_ra_sin = np.cos(glat_r) * np.sin(GlonNcpR - glon_r)
    d_ra_sin /= np.cos(dec_r)
    d_ra_cos = np.sin(glat_r) * np.cos(DecNgpR) - np.cos(glat_r) * np.sin(DecNgpR) * np.cos(GlonNcpR - glon_r)
    d_ra_cos /= np.cos(dec_r)

    d_ra_r = np.arctan2(d_ra_sin, d_ra_cos)
    ra = to_deg(RaNgpR + d_ra_r)

    ra = np.where(ra > 360, ra - 360, ra)

    return ra


def to_dec(glon, glat):
    """
    Conversion from galactic to equatorial coordinates. All values must be in degrees

    :return: declination
    """

    glon_r = to_rad(glon)
    glat_r = to_rad(glat)

    dec_r = np.arcsin(
        np.sin(DecNgpR) * np.sin(glat_r) + np.cos(DecNgpR) * np.cos(glat_r) * np.cos(GlonNcpR - glon_r))

    return to_deg(dec_r)


def to_glon(ra, dec, glat):
    """
    Conversion from equatorial to galactic coordinates. All values must be in degrees

    :return: galactic longitude
    """

    ra_r = to_rad(ra)
    dec_r = to_rad(dec)
    glat_r = to_rad(glat)

    d_glon_sin = np.cos(dec_r) * np.sin(ra_r - RaNgpR)
    d_glon_sin /= np.cos(glat_r)
    d_glon_cos = np.sin(dec_r) * np.cos(DecNgpR) - np.cos(dec_r) * np.sin(DecNgpR) * np.cos(ra_r - RaNgpR)
    d_glon_cos /= np.cos(glat_r)

    d_glon_r = np.arctan2(d_glon_sin, d_glon_cos)
    glon = to_deg(GlonNcpR - d_glon_r)

    glon = np.where(glon < 0, 360 + glon, glon)

    return glon


def to_glat(ra, dec):
    """
    Conversion from equatorial to galactic coordinates. All values must be in degrees

    :return: galactic latitude
    """

    ra_r = to_rad(ra)
    dec_r = to_rad(dec)

    glat_r = np.arcsin(
        np.sin(DecNgpR) * np.sin(dec_r) + np.cos(DecNgpR) * np.cos(dec_r) * np.cos(ra_r - RaNgpR))

    return to_deg(glat_r)


def to_gal_coords(x, y, z, dist):
    """
    It is assumed that the xy plane lies in the plane of the galaxy,
    the points on the x-axis have zero longitude,
    the points on the y-axis have a longitude equal to 270 degrees, and
    the points on the z-axis have a latitude equal to 90 degrees.

    dist = np.sqrt(x ** 2, y ** 2, z ** 2)
    """

    lat = np.sign(z) * np.arccos(np.sqrt(x ** 2 + y ** 2) / dist)
    lat = to_deg(lat)

    lon = np.arctan2(y, x)
    lon = to_deg(lon)
    lon = np.where(lon > 0, 360 - lon, -lon)

    return lon, lat
