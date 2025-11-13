import sys
import os
import time
import math

from collections import namedtuple
from typing import List

from copy import copy
from cpredict import PredictException # type: ignore
from cpredict import quick_find as _quick_find # type: ignore
from cpredict import quick_predict as _quick_predict # type: ignore
from dataclasses import dataclass


# Constants used by SGP4/SDP4 code 

km2mi=0.621371		# km to miles
deg2rad=1.745329251994330E-2	# Degrees to radians
pi=3.14159265358979323846	# Pi
pio2=1.57079632679489656	# Pi/2
x3pio2=4.71238898038468967	# 3*Pi/2
twopi=6.28318530717958623	# 2*Pi
e6a=1.0E-6
tothrd=6.6666666666666666E-1	# 2/3
xj2=1.0826158E-3		# J2 Harmonic (WGS '72)
xj3=2.53881E-6		# J3 Harmonic (WGS '72)
xj4=1.65597E-6		# J4 Harmonic (WGS '72)
xke=7.43669161E-2
xkmper=6.378137E3		# WGS 84 Earth radius km
xmnpda=1.44E3			# Minutes per day
ae=1.0
ck2=5.413079E-4
ck4=6.209887E-7
f=3.35281066474748E-3	# Flattening factor
ge=3.986008E5 	# Earth gravitational constant (WGS '72)
s=1.012229
qoms2t=1.880279E-09
secday=8.6400E4	# Seconds per day
omega_E=1.00273790934	# Earth rotations/siderial day
omega_ER=6.3003879	# Earth rotations, rads/siderial day
zns=1.19459E-5
c1ss=2.9864797E-6
zes=1.675E-2
znl=1.5835218E-4
c1l=4.7968065E-7
zel=5.490E-2
zcosis=9.1744867E-1
zsinis=3.9785416E-1
zsings=9.8088458E-1
zcosgs=1.945905E-1
zcoshs=1
zsinhs=0
q22=1.7891679E-6
q31=2.1460748E-6
q33=2.2123015E-7
g22=5.7686396
g32=9.5240898E-1
g44=1.8014998
g52=1.0508330
g54=4.4108898
root22=1.7891679E-6
root32=3.7393792E-7
root44=7.3636953E-9
root52=1.1428639E-7
root54=2.1765803E-9
thdt=4.3752691E-3
rho=1.5696615E-1
mfactor=7.292115E-5
sr=6.96000E5	# Solar radius - km (IAU 76)
AU=1.49597870691E8	# Astronomical unit - km (IAU 76)


# # Global variables
# deg2rad = math.pi / 180.0
# twopi = 2.0 * math.pi
# pio2 = math.pi / 2
# pi = math.pi
# x3pio2 = 3 * pio2



# Entry points of Deep() 

dpinit = 1 # Deep-space initialization code
dpsec = 2 # Deep-space secular code
dpper = 3 # Deep-space periodic code

# Flow control flag definitions

ALL_FLAGS              = -1
SGP_INITIALIZED_FLAG  =  0x000001	# not used
SGP4_INITIALIZED_FLAG =  0x000002
SDP4_INITIALIZED_FLAG =  0x000004
SGP8_INITIALIZED_FLAG =  0x000008	# not used
SDP8_INITIALIZED_FLAG =  0x000010	# not used
SIMPLE_FLAG           =  0x000020
DEEP_SPACE_EPHEM_FLAG =  0x000040
LUNAR_TERMS_DONE_FLAG =  0x000080
NEW_EPHEMERIS_FLAG    =  0x000100	# not used
DO_LOOP_FLAG          =  0x000200
RESONANCE_FLAG        =  0x000400
SYNCHRONOUS_FLAG      =  0x000800
EPOCH_RESTART_FLAG    =  0x001000
VISIBLE_FLAG          =  0x002000
SAT_ECLIPSED_FLAG     =  0x004000



@dataclass
class Observation:
    epoch: float
    orbital_model: str
    norad_id: int
    name: str
    latitude: float
    longitude: float
    altitude: float
    orbital_velocity: float
    footprint: float
    eclipse_depth: float
    orbital_phase: float
    sunlit: str
    orbit: int
    geostationary: str
    azimuth: float
    elevation: float
    slant_range: float
    visibility: str
    has_aos: str
    decayed: str
    doppler: float
    eci_x: float
    eci_y: float
    eci_z: float
    eci_vx: float
    eci_vy: float
    eci_vz: float
    eci_sun_x: float
    eci_sun_y: float
    eci_sun_z: float
    eci_obs_x: float
    eci_obs_y: float
    eci_obs_z: float
    beta_angle: float

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class Satellite:
    line1: str                      # First line of TLE
    line2: str                      # Second line of TLE
    name: str                       # Spacecraft Name
    catnum: int                     # Catalog Number (NORAD id)
    setnum: int                     # Element Set No.
    designator: str                 # Designator
    year: int                       # Reference Epoch (2-digit year)
    refepoch: float                 # Reference Epoch (day of the year + fractional portion)
    incl: float                     # Inclination
    raan: float                     # RAAN
    eccn: float                     # Eccentricity
    argper: float                   # Argument of Perigee
    meanan: float                   # Mean Anomaly
    meanmo: float                   # Mean Motion
    drag: float                     # Decay Rate
    nddot6: float                   # Nddot/6
    bstar: float                    # Bstar Drag Term
    orbitnum: int                   # Orbit Number

    def __getitem__(self, item):
        return getattr(self, item)
    
@dataclass
class qth:
    callsign: str # Observation Position Call Sign
    stnlat: float # Observation Position Latitude
    stnlong: float # Observation Position Longitude
    stnalt: int # Observation Position Altitude


@dataclass
class SatelliteDB:
    name: str                       # Satellite name
    catnum: int                     # Catalog number
    squintflag: str                 # Squint flag (single character)
    alat: float                     # Latitude
    alon: float                     # Longitude
    transponders: int               # Number of transponders
    transponder_name: List[str]     # List of transponder names (10x 80 characters)
    uplink_start: List[float]       # List of uplink start frequencies (10 elements)
    uplink_end: List[float]         # List of uplink end frequencies (10 elements)
    downlink_start: List[float]     # List of downlink start frequencies (10 elements)
    downlink_end: List[float]       # List of downlink end frequencies (10 elements)
    dayofweek: List[int]            # List of day of week (10 elements as unsigned char in C)
    phase_start: List[int]          # List of phase start (10 elements)
    phase_end: List[int]            # List of phase end (10 elements)


@dataclass
class TLE:
    epoch: float                   # Epoch
    xndt2o: float                  # First derivative of mean motion
    xndd6o: float                  # Second derivative of mean motion
    bstar: float                   # BSTAR drag term
    xincl: float                   # Inclination
    xnodeo: float                  # RAAN
    eo: float                      # Eccentricity
    omegao: float                  # Argument of perigee
    xmo: float                     # Mean anomaly
    xno: float                     # Mean motion
    catnr: int                     # Catalog number
    elset: int                     # Element set number
    revnum: int                    # Revolution number
    sat_name: str                  # Satellite name (up to 25 characters)
    idesg: str                     # International designator (up to 9 characters)


@dataclass
class Geodetic:
    lat: float                     # Latitude
    lon: float                     # Longitude
    alt: float                     # Altitude
    theta: float                   # Theta (angle)


@dataclass
class Vector:
    x: float                       # X component
    y: float                       # Y component
    z: float                       # Z component
    w: float = 0.0                 # W component (optional, default 0.0)

@dataclass
class DeepArg:
    eosq: float                    # Eccentricity squared
    sinio: float                   # Sine of inclination
    cosio: float                   # Cosine of inclination
    betao: float                   # Beta angle
    aodp: float                    # Semi-major axis
    theta2: float                  # Theta squared
    sing: float                    # Sine of argument of perigee
    cosg: float                    # Cosine of argument of perigee
    betao2: float                  # Beta squared
    xmdot: float                   # Mean anomaly rate
    omgdot: float                  # Argument of perigee rate
    xnodot: float                  # RAAN rate
    xnodp: float                   # Mean motion at epoch

    xll: float                     # Longitudinal parameter used by dpsec/dpper
    omgadf: float                  # Argument of perigee in the future
    xnode: float                   # RAAN value
    em: float                      # Eccentricity
    xinc: float                    # Inclination
    xn: float                      # Mean motion
    t: float                       # Time since epoch

    ds50: float                    # Julian date since 1950



Flags = 0

def is_flag_set(flag):
    return (Flags & flag)

def is_flag_clear(flag):
    return (~Flags & flag)

def set_flag(flag):
    global Flags
    Flags |= flag

def clear_flag(flag):
    global Flags
    Flags &= ~flag

def sign(arg):
    if arg > 0:
        return 1
    elif arg < 0:
        return -1
    else:
        return 0

def sqr(arg):
    return arg * arg

def cube(arg):
    return arg * arg * arg

def radians(arg):
    return arg * deg2rad

def degrees(arg):
    return arg / deg2rad

def arc_sin(arg):
    if abs(arg) >= 1.0:
        return sign(arg) * pio2
    else:
        return math.atan(arg / math.sqrt(1.0 - arg * arg))

def arc_cos(arg):
    return pio2 - arc_sin(arg)

def magnitude(v):
    v.w = math.sqrt(sqr(v.x) + sqr(v.y) + sqr(v.z))

def vec_add(v1, v2, v3):
    v3.x = v1.x + v2.x
    v3.y = v1.y + v2.y
    v3.z = v1.z + v2.z
    magnitude(v3)

def vec_sub(v1, v2, v3):
    v3.x = v1.x - v2.x
    v3.y = v1.y - v2.y
    v3.z = v1.z - v2.z
    magnitude(v3)

def scalar_multiply(k, v1, v2):
    v2.x = k * v1.x
    v2.y = k * v1.y
    v2.z = k * v1.z
    v2.w = abs(k) * v1.w

def scale_vector(k, v):
    v.x *= k
    v.y *= k
    v.z *= k
    magnitude(v)

def dot(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def angle(v1, v2):
    magnitude(v1)
    magnitude(v2)
    return arc_cos(dot(v1, v2) / (v1.w * v2.w))

def cross(v1, v2, v3):
    v3.x = v1.y * v2.z - v1.z * v2.y
    v3.y = v1.z * v2.x - v1.x * v2.z
    v3.z = v1.x * v2.y - v1.y * v2.x
    magnitude(v3)

def normalize(v):
    v.x /= v.w
    v.y /= v.w
    v.z /= v.w

def ac_tan(sinx, cosx):
    if cosx == 0.0:
        if sinx > 0.0:
            return pio2
        else:
            return x3pio2
    else:
        if cosx > 0.0:
            if sinx > 0.0:
                return math.atan(sinx / cosx)
            else:
                return twopi + math.atan(sinx / cosx)
        else:
            return pi + math.atan(sinx / cosx)

def fmod2p(x):
    ret_val = x % twopi
    if ret_val < 0.0:
        ret_val += twopi
    return ret_val

def modulus(arg1, arg2):
    ret_val = arg1 % arg2
    if ret_val < 0.0:
        ret_val += arg2
    return ret_val

def frac(arg):
    return arg - math.floor(arg)

def round_num(arg):
    return int(math.floor(arg + 0.5))

def int_num(arg):
    return math.floor(arg)


def Convert_Sat_State(pos: Vector, vel: Vector):

	# Converts the satellite's position and velocity
	# vectors from normalized values to km and km/sec

	scale_vector(xkmper, pos)
	scale_vector(xkmper*xmnpda/secday, vel)
