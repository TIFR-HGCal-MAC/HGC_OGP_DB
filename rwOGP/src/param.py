from colorama import Fore
import numpy as np
import logging

baseplates_params = {"vmini": 1.2, "vmaxi": 2.2, "new_angle": 0, "db_table_name": 'bp_inspect', 
                     "mother_table": 'baseplate', "prefix": 'bp'}
hexaboards_params = {"vmini": 1.2, "vmaxi": 2.9, "new_angle": 0, "db_table_name": 'hxb_inspect', 
                     "mother_table": 'hexaboard', 'prefix': 'hxb'}
protomodules_params = {"vmini": 1.37, "vmaxi": 1.79, "new_angle": 270, "db_table_name": 'proto_inspect', 
                       "mother_table": 'proto_assembly', 'prefix': 'proto'}
modules_params = {"vmini": 2.75, "vmaxi": 4.0, "new_angle": 270, "db_table_name": 'module_inspect', 
                  "mother_table": 'module_info', 'prefix': 'module'}

# Map channel numbers to FDs
fd_maps = [1, 8, 111, 197, 191] # FD1, FD2, FD3, FD4, FD5

COMP_PREFIX = {'baseplate': 'bp', 'hexaboard': 'hxb', 'protomodule': 'proto', 'module': 'module'}

default_params = {"PositionID": 1, "Geometry": "Full", "Density": "LD", "TrayNo": 1}

COMPONENT_PARAMS = {'baseplate': baseplates_params, 'hexaboard': hexaboards_params, 'protomodule': protomodules_params, 'module': modules_params}

plot2d_dim = (-100, 100)

colorClassify = {'-1': Fore.MAGENTA + 'NO INFO' + Fore.BLACK, 
                       '0': Fore.GREEN + 'GREEN' + Fore.BLACK,
                       '1': Fore.YELLOW + 'YELLOW' + Fore.BLACK, 
                       '2': Fore.RED + 'RED' + Fore.BLACK}
classify = {'-1': 'NO INFO',
                         '0': 'GREEN',
                         '1': 'YELLOW',
                         '2': 'RED'}

degrees = [0.03, 0.06, 90]
centers = [0.050, 0.100, 10.0]

pin_mapping = {
    'Full': {'LD': {1: ('p1_center_pin', 'p1O'), 2: ('p2_center_pin', 'p2M')}, # hole pin, slot pin
             'HD': {1: ('p1_center_pin', 'p1O'), 2: ('p2_center_pin', 'p2M')}},
    'Left': {'LD': {1: ('p1C', 'p1I'), 2: ('p2A', 'p2K')},
             'HD': {1: ('p1F', 'p1P'), 2: ('p2H', 'p2N')}},
    'Right': {'LD': {1: ('p1A', 'p1K'), 2: ('p2C', 'p2I')},
              'HD': {1: ('p1H', 'p1N'), 2: ('p2F', 'p2P')}},
    'Top': {'LD': {1: ('p1D', 'p1O'), 2: ('p2B', 'p2M')},
            'HD': {1: ('p1E', 'p1O'), 2: ('p2G', 'p2M')}},
    'Bottom': {'LD': {1: ('p1B', 'p1M'), 2: ('p2D', 'p2O')},
               'HD': {1: ('p1_center_pin', 'p1M'), 2: ('p2_center_pin', 'p2O')}},
    'Five': {'LD': {1: ('p1_center_pin', 'p1I'), 2: ('p2_center_pin', 'p2K')},
             'HD': {1: ('p1_center_pin', 'p1I'), 2: ('p2_center_pin', 'p2K')}}
}

ADJUSTMENTS = {
    'protomodule': {
        'Full':   {'LD': {1: (0, -0.1),      2: (0, 0)},      'HD': {1: (0, 0),       2: (0, 0)}},
        'Five':   {'LD': {1: (0, 0),      2: (0, 0)},      'HD': {1: (0, 0),       2: (0, 0)}},
        'Top':    {'LD': {1: (9.72, 0),  2: (-9.72, 0)},   'HD': {1: (8.44, 0),   2: (-8.44, 0)}},
        'Bottom': {'LD': {1: (-9.72, 0),   2: (9.72, 0)},  'HD': {1: (-16, 0),     2: (16, 0)}},
        'Right':  {'LD': {1: (0, -9.72),   2: (0, 9.72)},  'HD': {1: (0, 6.52),    2: (0, -6.52)}},
        'Left':   {'LD': {1: (0, 9.72),  2: (0, -9.72)},   'HD': {1: (0, -6.52),   2: (0, 6.52)}}
    },
    'module': {
        'Full':   {'LD': {1: (0, 0),      2: (0, 0)},      'HD': {1: (0, 0),       2: (0, 0)}},
        'Five':   {'LD': {1: (0, 0),      2: (0, 0)},      'HD': {1: (0, 0),       2: (0, 0)}},
        'Top':    {'LD': {1: (8, 0),     2: (-8, 0)},      'HD': {1: (4, 0),      2: (-4, 0)}},
        'Bottom': {'LD': {1: (-9, 0),      2: (9, 0)},     'HD': {1: (-15, 0),     2: (15, 0)}},
        'Right':  {'LD': {1: (0, 18),     2: (0, -18)},    'HD': {1: (0, 5),       2: (0, -5)}},
        'Left':   {'LD': {1: (0, 8),    2: (0, -8)},     'HD': {1: (0, -5),      2: (0, 5)}}
    }
}


# Define the angle calculation of FD points
ANGLE_CALC_CONFIG = {
    'Bottom':  {
        1: lambda fd3to1, *_: calc_basic_angle(-fd3to1),
        2: lambda fd3to1, *_: calc_basic_angle(fd3to1),
    },
    'Top': {
        'LD': {
            1: lambda fd3to1, *_: calc_basic_angle(-fd3to1),
            2: lambda fd3to1, *_: calc_basic_angle(fd3to1),
        },
        'HD': {
            1: lambda fd3to1, *_: calc_basic_angle(-fd3to1),
            2: lambda fd3to1, *_: calc_basic_angle(fd3to1),
        }
    },
    'Five': {
        1: lambda fd3to1, fdpoints, comp_type: calc_five_angle(fdpoints, fd3to1, comp_type),
        2: lambda fd3to1, fdpoints, comp_type: calc_five_angle(fdpoints, fd3to1, comp_type, True),
    },
    'Left': {
        'LD': {
            1: lambda fd3to1, *_: calc_semi_angle(fd3to1),
            2: lambda fd3to1, *_: calc_semi_angle(fd3to1, True),
        }
    },
    'Right': {
        'LD': {
            1: lambda fd3to1, *_: calc_semi_angle(fd3to1),
            2: lambda fd3to1, *_: calc_semi_angle(-fd3to1),
        }
    },
    'Full': {
        'HD': {
            1: lambda fd3to1, fdpoints, *_: calc_HDfull_angle(fdpoints, None),
            2: lambda fd3to1, fdpoints, *_: calc_HDfull_angle(fdpoints, None, True),
        },
        'LD': {
            # fd3to1 is unused here but kept for a uniform call signature across all entries
            1: lambda fd3to1, fdpoints, comp_type, angle_pin=None: calc_full_angle(fdpoints, comp_type, angle_pin=angle_pin),
            2: lambda fd3to1, fdpoints, comp_type, angle_pin=None: calc_full_angle(fdpoints, comp_type, True, angle_pin=angle_pin),
        }
    }
}

def calc_ref_angle(pinx, pinY, sign):
    if sign == 1:
        logging.debug("Using Angle of Hole(Center) --> Slot(OffCenter) for rotational reference")
    else:
        logging.debug("Using Angle of Slot(OffCenter) --> Hole(Center) for rotational reference")
    return np.degrees(np.arctan2(sign * pinY, sign * pinx))


# Define the hole-to-slot angle calculation functions
angle_lookup = {
    'Full': {
        'HD': {
            1: lambda pinX, pinY: calc_ref_angle(pinX, pinY, -1),
            2: lambda pinX, pinY: calc_ref_angle(pinX, pinY, 1)
        },
        'LD': {
            1: lambda pinX, pinY: calc_ref_angle(pinX, pinY, -1),
            2: lambda pinX, pinY: calc_ref_angle(pinX, pinY, 1)
        }
    },
    'Bottom': {
        'LD': {
            1: lambda pinX, pinY: calc_ref_angle(pinX, pinY, 1), 
            2: lambda pinX, pinY: calc_ref_angle(pinX, pinY, -1)
        }
    },
    'Five': {
        'default': {
            1: lambda pinX, pinY: np.degrees(np.arctan2(-pinX, -pinY)),
            2: lambda pinX, pinY: np.degrees(np.arctan2(pinX, pinY))
        }
    },
    'Left': {
        'LD': {
            1: lambda pinX, pinY: np.degrees(np.arctan2(-pinX, -pinY)),
            2: lambda pinX, pinY: np.degrees(np.arctan2(pinX, pinY))
        },
        'HD': {
            1: lambda pinX, pinY: np.degrees(np.arctan2(pinX, pinY)),
            2: lambda pinX, pinY: np.degrees(np.arctan2(-pinX, -pinY))
        }
    },
    'Right': {
        'LD': {
            1: lambda pinX, pinY: np.degrees(np.arctan2(pinX, pinY)),
            2: lambda pinX, pinY: np.degrees(np.arctan2(-pinX, -pinY))
        },
        'HD': {
            1: lambda pinX, pinY: np.degrees(np.arctan2(pinX, pinY)),
            2: lambda pinX, pinY: np.degrees(np.arctan2(-pinX, -pinY))
        }
    },
    'Top': {
        'default': {
            1: lambda pinX, pinY: np.degrees(np.arctan2(-pinY, -pinX)),
            2: lambda pinX, pinY: np.degrees(np.arctan2(pinY, pinX))
        }
    }
}

def calc_basic_angle(fd3to1) -> float:
    """Calculate the basic angle between two points
    
    Parameters
    ----------
    fd3to1 : array-like
        Vector difference between two fiducial points
    *args, **kwargs : additional arguments (unused but accepted for consistency)
    
    Returns
    -------
    float
        Angle in degrees
    """
    return np.degrees(np.arctan2(fd3to1[0], fd3to1[1]) * -1)

def calc_semi_angle(fd3to1, is_second=False) -> float:
    """Calculate the angle for the LEFT AND RIGHT component
    
    Parameters
    ----------
    fd3to1 : array-like
        Vector difference between two fiducial points
    is_second : bool, optional. Indicate Position (1/2)"""
    sign = 1 if is_second else -1
    return np.degrees(np.arctan2(sign * fd3to1[1], sign * fd3to1[0]))


def calc_five_angle(fdpoints, fd3to1, comp_type, is_second=False) -> float:
    """Calculate the angle deviation for LD FIVE geometry."""
    if comp_type == 'protomodule':
        if fdpoints[0][1] >= 200: points_diff = fdpoints[1] - fdpoints[0]; B = 1;  # vector from FD1 to FD2
        else: points_diff = fdpoints[0] - fdpoints[1]; B = -1; # vector from FD2 to FD1
        angle = B*np.degrees(np.arctan2(
            points_diff[0],
            points_diff[1]))
        logging.debug(f"Angle of FD1 -> FD2: {angle}")
        return angle
    elif comp_type == 'module':
        sign = 1 if is_second else -1
        return np.degrees(np.arctan2(sign * fd3to1[1], sign * fd3to1[0]))


def calc_full_angle(fdpoints, comp_type, is_second=False, angle_pin=None) -> float:
    """Calculate the angle deviation for PM/Modules with Full geometry.

    Parameters
    ----------
    fdpoints : np.ndarray
        Array of fiducial points.
    comp_type : str
        Component type: 'protomodule' or 'module'.
    is_second : bool, optional
        True for position 2 (flips sign).
    angle_pin : float, optional
        Reference pin angle in degrees. When provided the candidate closest to
        angle_pin is chosen (correct for all pin orientations). Falls back to
        the candidate closest to zero when None.
    """
    sign = -1 if is_second else 1
    PERPENDICULAR_CORRECTION_DEG = 90
    if comp_type == 'protomodule':
        # FD1 and FD5 are perpendicular to the pin axis, so subtract 90° to normalise.
        # Try both directions (FD1→FD5 and FD5→FD1) and normalise to [-180, 180] to
        # avoid out-of-range values, then pick the one closest to angle_pin to resolve
        # the 180° ambiguity (falls back to closest-to-zero when angle_pin is absent).
        diff1 = fdpoints[4] - fdpoints[0]  # vector FD1 -> FD5
        diff2 = fdpoints[0] - fdpoints[4]  # vector FD5 -> FD1
        angle1 = (np.degrees(np.arctan2(sign * diff1[1], sign * diff1[0])) - PERPENDICULAR_CORRECTION_DEG + 180) % 360 - 180
        angle2 = (np.degrees(np.arctan2(sign * diff2[1], sign * diff2[0])) - PERPENDICULAR_CORRECTION_DEG + 180) % 360 - 180
        # Prefer the candidate closest to angle_pin (AngleOffset ≈ 0 for a well-placed protomodule).
        # Fall back to closest-to-zero if angle_pin is not available.
        if angle_pin is not None:
            angle = angle1 if abs(angle1 - angle_pin) <= abs(angle2 - angle_pin) else angle2
        else:
            angle = angle1 if abs(angle1) < abs(angle2) else angle2
        direction = "FD1 -> FD5" if angle == angle1 else "FD5 -> FD1"
        logging.debug(f"Using Angle of {direction} (perpendicular, -{PERPENDICULAR_CORRECTION_DEG}° corrected) for rotational offset angle: {angle}")
    elif comp_type == 'module':
        diff1 = fdpoints[2] - fdpoints[5]
        diff2 = fdpoints[5] - fdpoints[2]
        # Normalise to [-180, 180] to avoid out-of-range values after the correction
        angle1 = (np.degrees(np.arctan2(sign * diff1[1], sign * diff1[0])) - PERPENDICULAR_CORRECTION_DEG + 180) % 360 - 180
        angle2 = (np.degrees(np.arctan2(sign * diff2[1], sign * diff2[0])) - PERPENDICULAR_CORRECTION_DEG + 180) % 360 - 180
        # Prefer the candidate closest to angle_pin (AngleOffset ≈ 0 for a well-placed module).
        # Fall back to closest-to-zero if angle_pin is not available.
        if angle_pin is not None:
            chosen = angle1 if abs(angle1 - angle_pin) <= abs(angle2 - angle_pin) else angle2
        else:
            chosen = angle1 if abs(angle1) <= abs(angle2) else angle2
        angle = chosen
        logging.debug(f"Using Module Full angle (FD3/FD6, -{PERPENDICULAR_CORRECTION_DEG}° corrected, normalised): {angle}")
    else:
        logging.error(f"Component type {comp_type} not recognized for angle calculation.")
    return angle

def calc_HDfull_angle(fdpoints, comp_type, is_second=False) -> float:
    """Calculate the angle deviation for HD Full geometry."""
    if fdpoints[0][1] >= 200: points_diff = fdpoints[1] - fdpoints[0]; B = 1;  # vector from FD1 to FD2
    else: points_diff = fdpoints[0] - fdpoints[1]; B = -1; # vector from FD2 to FD1
    angle = B*np.degrees(np.arctan2(
        points_diff[0],
        points_diff[1]))
    logging.debug(f"Angle of FD1 -> FD2: {angle}")
    return angle
