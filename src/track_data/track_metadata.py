"""
Track Metadata Module
=====================

Provides official track information extracted from sector maps including:
- GPS coordinates for start/finish, pit in/out
- Sector boundaries and distances
- Turn locations and characteristics
- Track length and elevation data

Data source: Official track sector maps (PDF)
"""

from typing import Dict, Tuple, Optional
import math

# ============================================================================
# Track Metadata Dictionary
# ============================================================================

TRACK_METADATA = {
    "barber-motorsports-park": {
        "name": "Barber Motorsports Park",
        "location": "Birmingham, Alabama",
        "length_miles": 2.28,
        "length_meters": 3670.56,
        "length_inches": 144672,
        "elevation_ft": 646,
        "elevation_m": 197,
        "gps_finish": {"lat": 33.5326722, "lon": -86.6196083},
        "gps_pit_in": {"lat": 33.531077, "lon": -86.622592},
        "gps_pit_out": {"lat": 33.531111, "lon": -86.622526},
        "sectors": {
            "S1": {
                "name": "Sector 1",
                "start_inches": 0,
                "end_inches": 40512,
                "length_inches": 40512,
                "start_m": 0,
                "end_m": 1029.0,
                "length_m": 1029.0
            },
            "S2": {
                "name": "Sector 2",
                "start_inches": 40512,
                "end_inches": 102732,
                "length_inches": 62220,
                "start_m": 1029.0,
                "end_m": 2609.4,
                "length_m": 1580.4
            },
            "S3": {
                "name": "Sector 3",
                "start_inches": 102732,
                "end_inches": 144672,
                "length_inches": 41940,
                "start_m": 2609.4,
                "end_m": 3674.7,
                "length_m": 1065.3
            }
        },
        "turns": {
            "T1": {"sector": "S1", "name": "Turn 1", "type": "right"},
            "T2": {"sector": "S1", "name": "Turn 2", "type": "left"},
            "T3": {"sector": "S2", "name": "Turn 3", "type": "right"},
            "T4": {"sector": "S3", "name": "Turn 4", "type": "left"},
            "T5": {"sector": "S3", "name": "Turn 5", "type": "hairpin"}
        },
        "pit_lane": {
            "pit_in_from_sf_inches": 131296,
            "pit_out_from_sf_inches": 5352,
            "pit_in_to_out_inches": 18794,
            "time_seconds": 34
        }
    },

    "sebring": {
        "name": "Sebring International Raceway",
        "location": "Sebring, Florida",
        "length_miles": 3.74,
        "length_meters": 6018.9,
        "length_inches": 236966,
        "elevation_ft": 61,
        "elevation_m": 18.6,
        "gps_finish": {"lat": 27.4502340, "lon": -81.3536980},
        "gps_pit_in": {"lat": 27.45012, "lon": -81.35547},
        "gps_pit_out": {"lat": 27.45011, "lon": -81.35051},
        "sectors": {
            "S1": {
                "name": "Sector 1",
                "start_inches": 0,
                "end_inches": 71819,
                "length_inches": 71819,
                "start_m": 0,
                "end_m": 1824.2,
                "length_m": 1824.2,
                "sub_sectors": ["S1-a", "S1-b"]
            },
            "S2": {
                "name": "Sector 2",
                "start_inches": 71819,
                "end_inches": 145193,
                "length_inches": 73374,
                "start_m": 1824.2,
                "end_m": 3687.9,
                "length_m": 1863.7,
                "sub_sectors": ["S2-a", "S2-b"]
            },
            "S3": {
                "name": "Sector 3",
                "start_inches": 145193,
                "end_inches": 236966,
                "length_inches": 91773,
                "start_m": 3687.9,
                "end_m": 6018.9,
                "length_m": 2331.0,
                "sub_sectors": ["S3-a", "S3-b"]
            }
        },
        "turns": {
            "T1": {"sector": "S1", "name": "Turn 1", "number": 1},
            "T2": {"sector": "S1", "name": "Turn 2", "number": 2},
            "T3": {"sector": "S1", "name": "Turn 3", "number": 3},
            "T4": {"sector": "S1", "name": "Turn 4", "number": 4},
            "T5": {"sector": "S1", "name": "Turn 5", "number": 5},
            "T6": {"sector": "S1", "name": "Turn 6", "number": 6},
            "T7": {"sector": "S2", "name": "Turn 7", "number": 7},
            "T8": {"sector": "S2", "name": "Turn 8", "number": 8},
            "T9": {"sector": "S2", "name": "Turn 9", "number": 9},
            "T10": {"sector": "S2", "name": "Turn 10", "number": 10},
            "T11": {"sector": "S2", "name": "Turn 11", "number": 11},
            "T12": {"sector": "S2", "name": "Turn 12", "number": 12},
            "T13": {"sector": "S2", "name": "Turn 13", "number": 13},
            "T14": {"sector": "S3", "name": "Turn 14", "number": 14},
            "T15": {"sector": "S3", "name": "Turn 15", "number": 15},
            "T16": {"sector": "S3", "name": "Turn 16", "number": 16},
            "T17": {"sector": "S3", "name": "Turn 17", "number": 17}
        },
        "pit_lane": {
            "pit_in_from_sf_inches": 230006,
            "pit_out_from_sf_inches": 14378,
            "pit_in_to_out_inches": 21349,
            "time_seconds": 39
        },
        "speed_trap": {
            "location_inches": 1496,
            "location_m": 38
        }
    },

    "circuit-of-the-americas": {
        "name": "Circuit of the Americas",
        "location": "Austin, Texas",
        "length_miles": 3.416,
        "length_meters": 5498.3,
        "length_inches": 216468,
        "elevation_ft": 508,
        "elevation_m": 154.8,
        "gps_finish": {"lat": 30.1335278, "lon": -97.6422583},
        "gps_pit_in": {"lat": 30.1343371, "lon": -97.6340257},
        "gps_pit_out": {"lat": 30.1314446, "lon": -97.6389209},
        "sectors": {
            "S1": {
                "name": "Sector 1",
                "start_inches": 0,
                "end_inches": 51528,
                "length_inches": 51528,
                "start_m": 0,
                "end_m": 1308.8,
                "length_m": 1308.8,
                "sub_sectors": ["S1-a", "S1-b"]
            },
            "S2": {
                "name": "Sector 2",
                "start_inches": 51528,
                "end_inches": 139716,
                "length_inches": 88188,
                "start_m": 1308.8,
                "end_m": 3548.8,
                "length_m": 2240.0,
                "sub_sectors": ["S2-a", "S2-b"]
            },
            "S3": {
                "name": "Sector 3",
                "start_inches": 139716,
                "end_inches": 216468,
                "length_inches": 76752,
                "start_m": 3548.8,
                "end_m": 5498.3,
                "length_m": 1949.5,
                "sub_sectors": ["S3-a", "S3-b"]
            }
        },
        "turns": {
            "T1": {"sector": "S1", "name": "Turn 1", "number": 1},
            "T2": {"sector": "S1", "name": "Turn 2", "number": 2},
            "T3": {"sector": "S1", "name": "Turn 3", "number": 3},
            "T4": {"sector": "S1", "name": "Turn 4", "number": 4},
            "T5": {"sector": "S1", "name": "Turn 5", "number": 5},
            "T6": {"sector": "S2", "name": "Turn 6", "number": 6},
            "T7": {"sector": "S2", "name": "Turn 7", "number": 7},
            "T8": {"sector": "S2", "name": "Turn 8", "number": 8},
            "T9": {"sector": "S2", "name": "Turn 9", "number": 9},
            "T10": {"sector": "S2", "name": "Turn 10", "number": 10},
            "T11": {"sector": "S2", "name": "Turn 11", "number": 11},
            "T12": {"sector": "S3", "name": "Turn 12", "number": 12},
            "T13": {"sector": "S3", "name": "Turn 13", "number": 13},
            "T14": {"sector": "S3", "name": "Turn 14", "number": 14},
            "T15": {"sector": "S3", "name": "Turn 15", "number": 15},
            "T16": {"sector": "S3", "name": "Turn 16", "number": 16},
            "T17": {"sector": "S3", "name": "Turn 17", "number": 17},
            "T18": {"sector": "S3", "name": "Turn 18", "number": 18},
            "T19": {"sector": "S3", "name": "Turn 19", "number": 19},
            "T20": {"sector": "S3", "name": "Turn 20", "number": 20}
        },
        "pit_lane": {
            "pit_in_from_sf_inches": 208068,
            "pit_out_from_sf_inches": 15504,
            "pit_in_to_out_inches": 20052,
            "time_seconds": 36
        },
        "speed_trap": {
            "location_inches": 1176,
            "location_m": 29.9
        }
    },

    "road-america": {
        "name": "Road America",
        "location": "Elkhart Lake, Wisconsin",
        "length_miles": 4.014,
        "length_meters": 6459.6,
        "length_inches": 254316,
        "elevation_ft": 1058,
        "elevation_m": 322,
        "gps_finish": {"lat": 43.7979056, "lon": -87.9896333},
        "gps_pit_in": {"lat": 43.80057, "lon": -87.98992},
        "gps_pit_out": {"lat": 43.7948061, "lon": -87.9897494},
        "sectors": {
            "S1": {
                "name": "Sector 1",
                "start_inches": 0,
                "end_inches": 81048,
                "length_inches": 81048,
                "start_m": 0,
                "end_m": 2058.6,
                "length_m": 2058.6,
                "sub_sectors": ["S1-a", "S1-b"]
            },
            "S2": {
                "name": "Sector 2",
                "start_inches": 81048,
                "end_inches": 167976,
                "length_inches": 86928,
                "start_m": 2058.6,
                "end_m": 4266.6,
                "length_m": 2208.0,
                "sub_sectors": ["S2-a", "S2-b"]
            },
            "S3": {
                "name": "Sector 3",
                "start_inches": 167976,
                "end_inches": 254316,
                "length_inches": 86340,
                "start_m": 4266.6,
                "end_m": 6459.6,
                "length_m": 2193.0,
                "sub_sectors": ["S3-a", "S3-b"]
            }
        },
        "turns": {
            "T1": {"sector": "S1", "name": "Turn 1", "number": 1},
            "T2": {"sector": "S1", "name": "Turn 2", "number": 2},
            "T3": {"sector": "S1", "name": "Turn 3", "number": 3},
            "T4": {"sector": "S1", "name": "Turn 4", "number": 4},
            "T5": {"sector": "S2", "name": "Turn 5", "number": 5},
            "T6": {"sector": "S2", "name": "Turn 6", "number": 6},
            "T7": {"sector": "S2", "name": "Turn 7", "number": 7},
            "T8": {"sector": "S2", "name": "Turn 8", "number": 8},
            "T9": {"sector": "S2", "name": "Turn 9", "number": 9},
            "T10": {"sector": "S2", "name": "Turn 10", "number": 10},
            "T11": {"sector": "S2", "name": "Turn 11", "number": 11},
            "T12": {"sector": "S3", "name": "Turn 12", "number": 12},
            "T13": {"sector": "S3", "name": "Turn 13", "number": 13},
            "T14": {"sector": "S3", "name": "Turn 14", "number": 14}
        },
        "pit_lane": {
            "pit_in_from_sf_inches": 24252,
            "pit_out_from_sf_inches": 13668,
            "pit_in_to_out_inches": 25464,
            "time_seconds": 52
        }
    },

    "sonoma": {
        "name": "Sonoma Raceway",
        "location": "Sonoma, California",
        "length_miles": 2.505,
        "length_meters": 4031.38,
        "length_inches": 158716,
        "elevation_ft": 20,
        "elevation_m": 6.1,
        "gps_finish": {"lat": 38.1615139, "lon": -122.4547166},
        "gps_pit_in": {"lat": 38.1615139, "lon": -122.4547166},  # Note: Pit coordinates not in PDF
        "gps_pit_out": {"lat": 38.1615139, "lon": -122.4547166},  # Using finish as placeholder
        "sectors": {
            "S1": {
                "name": "Sector 1",
                "start_inches": 0,
                "end_inches": 54520,
                "length_inches": 54520,
                "start_m": 0,
                "end_m": 1385.0,
                "length_m": 1385.0,
                "sub_sectors": ["S1-a", "S1-b"]
            },
            "S2": {
                "name": "Sector 2",
                "start_inches": 54520,
                "end_inches": 110496,
                "length_inches": 55976,
                "start_m": 1385.0,
                "end_m": 2807.0,
                "length_m": 1422.0,
                "sub_sectors": ["S2-a", "S2-b"]
            },
            "S3": {
                "name": "Sector 3",
                "start_inches": 110496,
                "end_inches": 158716,
                "length_inches": 48220,
                "start_m": 2807.0,
                "end_m": 4031.38,
                "length_m": 1225.0,
                "sub_sectors": ["S3-a", "S3-b"]
            }
        },
        "turns": {
            "T1": {"sector": "S1", "name": "Turn 1", "number": 1},
            "T2": {"sector": "S1", "name": "Turn 2", "number": 2},
            "T3": {"sector": "S1", "name": "Turn 3", "number": 3},
            "T3A": {"sector": "S1", "name": "Turn 3A", "number": "3A"},
            "T4": {"sector": "S1", "name": "Turn 4", "number": 4},
            "T5": {"sector": "S1", "name": "Turn 5", "number": 5},
            "T6": {"sector": "S2", "name": "Turn 6", "number": 6},
            "T7": {"sector": "S2", "name": "Turn 7", "number": 7},
            "T7A": {"sector": "S2", "name": "Turn 7A", "number": "7A"},
            "T8": {"sector": "S2", "name": "Turn 8", "number": 8},
            "T8A": {"sector": "S2", "name": "Turn 8A", "number": "8A"},
            "T9": {"sector": "S3", "name": "Turn 9", "number": 9},
            "T10": {"sector": "S3", "name": "Turn 10", "number": 10},
            "T11": {"sector": "S3", "name": "Turn 11", "number": 11},
            "T12": {"sector": "S3", "name": "Turn 12", "number": 12}
        },
        "pit_lane": {
            "pit_in_from_sf_inches": -20442,  # Negative: before start/finish
            "pit_out_from_sf_inches": 3775,
            "pit_in_to_out_inches": 24564,
            "time_seconds": 45
        },
        "speed_trap": {
            "location_inches": 1224,
            "location_m": 31
        }
    },

    "virginia-international-raceway": {
        "name": "Virginia International Raceway",
        "location": "Alton, Virginia",
        "length_miles": 3.27,
        "length_meters": 5262.6,
        "length_inches": 207189,
        "elevation_ft": 375,
        "elevation_m": 114.3,
        "gps_finish": {"lat": 36.5688167, "lon": -79.2066639},
        "gps_pit_in": {"lat": 36.567581, "lon": -79.210428},
        "gps_pit_out": {"lat": 36.568667, "lon": -79.206797},
        "sectors": {
            "S1": {
                "name": "Sector 1",
                "start_inches": 0,
                "end_inches": 65064,
                "length_inches": 65064,
                "start_m": 0,
                "end_m": 1652.6,
                "length_m": 1652.6,
                "sub_sectors": ["S1-a", "S1-b"]
            },
            "S2": {
                "name": "Sector 2",
                "start_inches": 65064,
                "end_inches": 150024,
                "length_inches": 84960,
                "start_m": 1652.6,
                "end_m": 3810.6,
                "length_m": 2158.0,
                "sub_sectors": ["S2-a", "S2-b"]
            },
            "S3": {
                "name": "Sector 3",
                "start_inches": 150024,
                "end_inches": 207189,
                "length_inches": 57165,
                "start_m": 3810.6,
                "end_m": 5262.6,
                "length_m": 1452.0,
                "sub_sectors": ["S3-a", "S3-b"]
            }
        },
        "turns": {
            "T1": {"sector": "S1", "name": "Turn 1", "number": 1},
            "T2": {"sector": "S1", "name": "Turn 2", "number": 2},
            "T3": {"sector": "S1", "name": "Turn 3", "number": 3},
            "T4": {"sector": "S1", "name": "Turn 4", "number": 4},
            "T5": {"sector": "S1", "name": "Turn 5", "number": 5},
            "T6": {"sector": "S1", "name": "Turn 6", "number": 6},
            "T7": {"sector": "S2", "name": "Turn 7", "number": 7},
            "T8": {"sector": "S2", "name": "Turn 8", "number": 8},
            "T9": {"sector": "S2", "name": "Turn 9", "number": 9},
            "T10": {"sector": "S2", "name": "Turn 10", "number": 10},
            "T11": {"sector": "S2", "name": "Turn 11", "number": 11},
            "T12": {"sector": "S2", "name": "Turn 12", "number": 12},
            "T12A": {"sector": "S2", "name": "Turn 12a", "number": "12a"},
            "T13": {"sector": "S3", "name": "Turn 13", "number": 13},
            "T14": {"sector": "S3", "name": "Turn 14", "number": 14},
            "T15": {"sector": "S3", "name": "Turn 15", "number": 15},
            "T16": {"sector": "S3", "name": "Turn 16", "number": 16},
            "T17": {"sector": "S3", "name": "Turn 17", "number": 17}
        },
        "pit_lane": {
            "pit_in_from_sf_inches": 192864,
            "pit_out_from_sf_inches": -630,  # Negative: before start/finish
            "pit_in_to_out_inches": 13410,
            "time_seconds": 25
        },
        "speed_trap": {
            "location_inches": 1440,
            "location_m": 36.6
        }
    }
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_track_metadata(track_name: str) -> Dict:
    """
    Get metadata for a specific track

    Args:
        track_name: Track identifier (e.g., 'barber-motorsports-park')

    Returns:
        Dictionary containing track metadata

    Example:
        >>> metadata = get_track_metadata('barber-motorsports-park')
        >>> print(metadata['length_miles'])
        2.28
    """
    return TRACK_METADATA.get(track_name, {})


def get_available_tracks() -> list:
    """
    Get list of tracks with metadata available

    Returns:
        List of track names

    Example:
        >>> tracks = get_available_tracks()
        >>> print(tracks)
        ['barber-motorsports-park', 'sebring']
    """
    return list(TRACK_METADATA.keys())


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two GPS points using Haversine formula

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in meters

    Example:
        >>> # Distance from Barber finish to pit in
        >>> dist = haversine_distance(33.5326722, -86.6196083, 33.531077, -86.622592)
        >>> print(f"{dist:.1f}m")
        1234.5m
    """
    R = 6371000  # Earth radius in meters

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_distance_from_sf(gps_lat: float, gps_lon: float, track_name: str) -> Optional[float]:
    """
    Calculate distance from start/finish line using GPS coordinates

    Args:
        gps_lat: GPS latitude
        gps_lon: GPS longitude
        track_name: Track identifier

    Returns:
        Distance from start/finish in meters, or None if track not found

    Note:
        This is a simple estimation. For accurate results, use track-specific
        GPS-to-distance mapping or interpolation.

    Example:
        >>> dist = calculate_distance_from_sf(33.531077, -86.622592, 'barber-motorsports-park')
        >>> print(f"Distance from SF: {dist:.1f}m")
    """
    metadata = get_track_metadata(track_name)
    if not metadata:
        return None

    finish_gps = metadata['gps_finish']
    distance = haversine_distance(
        finish_gps['lat'], finish_gps['lon'],
        gps_lat, gps_lon
    )

    return distance


def determine_sector(distance_from_sf: float, track_name: str) -> str:
    """
    Determine which sector the car is in based on distance from start/finish

    Args:
        distance_from_sf: Distance from start/finish line in meters
        track_name: Track identifier

    Returns:
        Sector name ('S1', 'S2', 'S3') or 'Unknown'

    Example:
        >>> sector = determine_sector(1500, 'barber-motorsports-park')
        >>> print(sector)
        'S2'
    """
    metadata = get_track_metadata(track_name)
    if not metadata:
        return "Unknown"

    sectors = metadata.get('sectors', {})

    for sector_name, sector_data in sectors.items():
        if sector_data['start_m'] <= distance_from_sf < sector_data['end_m']:
            return sector_name

    return "Unknown"


def get_sector_info(track_name: str, sector_name: str) -> Optional[Dict]:
    """
    Get detailed information about a specific sector

    Args:
        track_name: Track identifier
        sector_name: Sector name ('S1', 'S2', 'S3')

    Returns:
        Dictionary with sector information or None

    Example:
        >>> info = get_sector_info('barber-motorsports-park', 'S1')
        >>> print(f"Sector 1 length: {info['length_m']:.1f}m")
    """
    metadata = get_track_metadata(track_name)
    if not metadata:
        return None

    return metadata.get('sectors', {}).get(sector_name)


def get_turn_info(track_name: str, turn_name: str) -> Optional[Dict]:
    """
    Get information about a specific turn

    Args:
        track_name: Track identifier
        turn_name: Turn identifier (e.g., 'T1', 'T2')

    Returns:
        Dictionary with turn information or None

    Example:
        >>> info = get_turn_info('barber-motorsports-park', 'T5')
        >>> print(f"Turn 5 is in {info['sector']} and is a {info['type']}")
    """
    metadata = get_track_metadata(track_name)
    if not metadata:
        return None

    return metadata.get('turns', {}).get(turn_name)


def get_turns_in_sector(track_name: str, sector_name: str) -> list:
    """
    Get all turns within a specific sector

    Args:
        track_name: Track identifier
        sector_name: Sector name ('S1', 'S2', 'S3')

    Returns:
        List of turn names in that sector

    Example:
        >>> turns = get_turns_in_sector('barber-motorsports-park', 'S1')
        >>> print(turns)
        ['T1', 'T2']
    """
    metadata = get_track_metadata(track_name)
    if not metadata:
        return []

    turns = metadata.get('turns', {})
    return [turn_name for turn_name, turn_info in turns.items()
            if turn_info.get('sector') == sector_name]


# ============================================================================
# Summary Functions
# ============================================================================

def print_track_summary(track_name: str, use_unicode: bool = False):
    """
    Print a formatted summary of track information

    Args:
        track_name: Track identifier
        use_unicode: Use Unicode emojis (may not work on Windows CMD)

    Example:
        >>> print_track_summary('barber-motorsports-park')
    """
    metadata = get_track_metadata(track_name)
    if not metadata:
        print(f"Track '{track_name}' not found")
        return

    # Icons (with fallback for Windows compatibility)
    icons = {
        'length': '📏' if use_unicode else '[LENGTH]',
        'elevation': '🏔️' if use_unicode else '[ELEV]',
        'gps': '📍' if use_unicode else '[GPS]',
        'sectors': '🏁' if use_unicode else '[SECTORS]',
        'turns': '🔄' if use_unicode else '[TURNS]',
        'pit': '🏎️' if use_unicode else '[PIT]'
    }

    print(f"\n{'='*60}")
    print(f"  {metadata['name']}")
    print(f"  {metadata.get('location', 'Unknown location')}")
    print(f"{'='*60}")
    print(f"\n{icons['length']} Track Length: {metadata['length_miles']} miles ({metadata['length_meters']:.1f}m)")
    print(f"{icons['elevation']} Elevation: {metadata['elevation_ft']}ft ({metadata['elevation_m']}m)")
    print(f"\n{icons['gps']} GPS Coordinates:")
    print(f"   Start/Finish: {metadata['gps_finish']['lat']:.6f}, {metadata['gps_finish']['lon']:.6f}")
    print(f"   Pit In:       {metadata['gps_pit_in']['lat']:.6f}, {metadata['gps_pit_in']['lon']:.6f}")
    print(f"   Pit Out:      {metadata['gps_pit_out']['lat']:.6f}, {metadata['gps_pit_out']['lon']:.6f}")

    print(f"\n{icons['sectors']} Sectors:")
    for sector_name, sector_data in metadata['sectors'].items():
        print(f"   {sector_name}: {sector_data['length_m']:.1f}m")

    print(f"\n{icons['turns']} Turns: {len(metadata['turns'])} total")
    for sector in ['S1', 'S2', 'S3']:
        turns = get_turns_in_sector(track_name, sector)
        if turns:
            print(f"   {sector}: {', '.join(turns)}")

    pit_info = metadata.get('pit_lane', {})
    print(f"\n{icons['pit']} Pit Lane: {pit_info.get('time_seconds', 'N/A')} seconds")
    print(f"{'='*60}\n")


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Track Metadata Module")
    print("=" * 60)

    # List available tracks
    tracks = get_available_tracks()
    print(f"\nAvailable tracks: {len(tracks)}")
    for track in tracks:
        print(f"   - {track}")

    # Print summaries (without Unicode for Windows compatibility)
    for track in tracks:
        print_track_summary(track, use_unicode=False)

    # Example distance calculation
    print("\n" + "="*60)
    print("Example: Distance Calculation")
    print("="*60)
    distance = calculate_distance_from_sf(33.531077, -86.622592, 'barber-motorsports-park')
    if distance:
        print(f"Distance from Barber start/finish to pit in: {distance:.1f}m")
        sector = determine_sector(distance, 'barber-motorsports-park')
        print(f"This position is in: {sector}")
