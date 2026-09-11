"""Frozen configuration and latent truth for the NO2 emission inversion task.

This file is authoring-only.  It is never copied into environment/ and never
read by solution/solve.sh.  The verifier receives only the derived values that
tests/truth/ carries.
"""
import numpy as np

MASTER_SEED = 20260911

# ----------------------------------------------------------------- geometry --
LX = 240_000.0            # domain size, metres (projected easting extent)
LY = 180_000.0            # domain size, metres (projected northing extent)
DX_COARSE = 4000.0        # inference grid supplied to the agent
DX_FINE = 2000.0          # truth grid, never disclosed as a grid to the agent
NX_C, NY_C = int(LX // DX_COARSE), int(LY // DX_COARSE)   # 60 x 45
NX_F, NY_F = int(LX // DX_FINE), int(LY // DX_FINE)       # 120 x 90

DX_MET = 12_000.0         # meteorological grid spacing
NX_M, NY_M = int(LX // DX_MET), int(LY // DX_MET)         # 20 x 15

# EPSG:3035 (ETRS89 / LAEA Europe) anchor for the south-west domain corner.
X0_PROJ = 4_612_000.0
Y0_PROJ = 2_874_000.0

# --------------------------------------------------------------- chemistry ---
M_NO2 = 46.0055e-3        # kg mol-1
M_N = 14.0067e-3          # kg mol-1
EMISSION_SCALE = 2.2      # global amplitude of the synthetic inventory
K_DIFF = 400.0            # m2 s-1, prescribed horizontal eddy diffusivity
ZETA0 = 0.35              # shape parameter of the normalised vertical profile

# ------------------------------------------------------------- time layout ---
# 18 episodes across three seasons, three synoptic wind regimes.
EPISODE_HOURS = 10.0      # total integration window per episode
SPINUP_HOURS = 4.0        # first 4 h are spin-up; no graded observation there
DT_FINE = 30.0            # s, truth time step
DT_COARSE = 60.0          # s, reference evaluation time step
N_EPISODES = 18
N_REGIONS = 6
N_STATIONS = 12

REGIME_NAMES = ["SW_advective", "NE_continental", "S_light_variable"]
# regime -> (mean direction the wind blows TOWARD, degrees CCW from east,
#            mean speed m/s, speed spread)
REGIME_WIND = {
    0: (38.0, 9.4, 1.6),
    1: (218.0, 7.1, 1.2),
    2: (95.0, 3.4, 0.9),
}
# episode -> regime, three seasons interleaved so every regime spans the year
EPISODE_REGIME = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
# episodes withheld from the agent (two per regime)
HELD_OUT_EPISODES = [9, 10, 11, 15, 16, 17]

EPISODE_START_ISO = [
    "2023-07-04T04:00:00Z", "2023-07-09T03:00:00Z", "2023-07-17T04:00:00Z",
    "2023-07-22T03:00:00Z", "2023-07-28T04:00:00Z", "2023-08-02T03:00:00Z",
    "2023-10-03T05:00:00Z", "2023-10-08T06:00:00Z", "2023-10-14T05:00:00Z",
    "2023-10-19T06:00:00Z", "2023-10-25T05:00:00Z", "2023-10-30T06:00:00Z",
    "2024-02-05T06:00:00Z", "2024-02-11T07:00:00Z", "2024-02-16T06:00:00Z",
    "2024-02-21T07:00:00Z", "2024-02-27T06:00:00Z", "2024-03-03T07:00:00Z",
]

# ------------------------------------------------------------ latent truth ---
# Chosen before any inversion was run.  Never leaves authoring/ and tests/.
TRUE_SOURCE_SCALE = np.array([1.62, 0.74, 2.08, 0.47, 1.26, 1.91])
TRUE_LIFETIME_S = 16_560.0        # 4.6 h effective NOx loss time
TRUE_WIND_SPEED_SCALE = 1.18
TRUE_WIND_ROTATION_DEG = -13.5
TRUE_BACKGROUND = np.array([2.10e-5, 4.00e-6, -3.20e-6])   # b0, bx, by [mol m-2]

# admissible ranges published to the agent
BOUNDS = {
    "emission_scale": (0.30, 2.50),
    "effective_lifetime_s": (3600.0, 28800.0),
    "wind_speed_scale": (0.70, 1.30),
    "wind_rotation_deg": (-20.0, 20.0),
}

REGION_NAMES = [
    "R1_core_motorway_ring",
    "R2_north_industrial_corridor",
    "R3_east_urban_arterials",
    "R4_south_rural_trunk",
    "R5_west_transit_motorway",
    "R6_southeast_secondary_network",
]
