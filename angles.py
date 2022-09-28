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
ra_ngp = 192.85948
dec_ngp = 27.12825
glon_ncp = 122.93192

ra_ngp_r = to_rad(ra_ngp)
dec_ngp_r = to_rad(dec_ngp)
glon_ncp_r = to_rad(glon_ncp)


def to_ra(glon, glat, dec):
    """
    Conversion from galactic to equatorial coordinates. All values must be in degrees

    :return: declination
    """

    glon_r = to_rad(glon)
    glat_r = to_rad(glat)
    dec_r = to_rad(dec)

    d_ra_sin = np.cos(glat_r) * np.sin(glon_ncp_r - glon_r)
    d_ra_sin /= np.cos(dec_r)
    d_ra_cos = np.sin(glat_r) * np.cos(dec_ngp_r) - np.cos(glat_r) * np.sin(dec_ngp_r) * np.cos(glon_ncp_r - glon_r)
    d_ra_cos /= np.cos(dec_r)

    d_ra_r = np.arctan2(d_ra_sin, d_ra_cos)
    ra = to_deg(ra_ngp_r + d_ra_r)

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
        np.sin(dec_ngp_r) * np.sin(glat_r) + np.cos(dec_ngp_r) * np.cos(glat_r) * np.cos(glon_ncp_r - glon_r))

    return to_deg(dec_r)


def to_glon(ra, dec, glat):
    """
    Conversion from equatorial to galactic coordinates. All values must be in degrees

    :return: galactic longitude
    """

    ra_r = to_rad(ra)
    dec_r = to_rad(dec)
    glat_r = to_rad(glat)

    d_glon_sin = np.cos(dec_r) * np.sin(ra_r - ra_ngp_r)
    d_glon_sin /= np.cos(glat_r)
    d_glon_cos = np.sin(dec_r) * np.cos(dec_ngp_r) - np.cos(dec_r) * np.sin(dec_ngp_r) * np.cos(ra_r - ra_ngp_r)
    d_glon_cos /= np.cos(glat_r)

    d_glon_r = np.arctan2(d_glon_sin, d_glon_cos)
    glon = to_deg(glon_ncp_r - d_glon_r)

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
        np.sin(dec_ngp_r) * np.sin(dec_r) + np.cos(dec_ngp_r) * np.cos(dec_r) * np.cos(ra_r - ra_ngp_r))

    return to_deg(glat_r)
