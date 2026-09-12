"""Frozen configuration and latent truth for the NO2 emission inversion task.

This file is authoring-only.  It is never copied into environment/ and never
read by solution/solve.sh.  The verifier receives only the derived values that
tests/truth/ carries.
"""
import numpy as np

MASTER_SEED = 20260912

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
P_NIGHT = 0.12            # floor of the prescribed photolysis proxy

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
#
# Chemical loss is concentration dependent: the NOx sink saturates as OH is
# suppressed at high NOx, so
#     L(C) = C * P / (tau0 * (1 + C / C_ref))
# with P the prescribed photolysis proxy.  tau0 is the low-NOx loss time at
# P = 1; C_ref is the column at which the effective loss rate is halved.
TRUE_SOURCE_SCALE = np.array([1.62, 0.74, 2.08, 0.47, 1.26, 1.91])
TRUE_TAU0_S = 5_400.0             # 1.5 h low-NOx loss time at unit photolysis
TRUE_C_REF = 6.0e-5               # mol m-2, NOx column that halves the loss rate
TRUE_WIND_SPEED_SCALE = 1.18
TRUE_WIND_ROTATION_DEG = -13.5
TRUE_BACKGROUND = np.array([2.10e-5, 4.00e-6, -3.20e-6])   # b0, bx, by [mol m-2]
TRUE_FIXED_SCALE = 1.34           # non-road sources are also mis-reported
TRUE_ZETA0 = 0.29                 # shape parameter of the vertical profile

# admissible ranges published to the agent
BOUNDS = {
    "emission_scale": (0.30, 2.50),
    "reference_loss_time_s": (1800.0, 28800.0),
    "loss_saturation_column": (1.0e-5, 1.0e-3),
    "wind_speed_scale": (0.70, 1.30),
    "wind_rotation_deg": (-20.0, 20.0),
    "fixed_source_scale": (0.40, 2.20),
    "vertical_shape_zeta0": (0.12, 0.80),
}

# ------------------------------------------------- instrument row anomaly ----
# A contiguous band of across-track positions develops a bias partway through
# the record, as real ultraviolet imagers do.  The quality flag does not catch
# it.  Its existence is disclosed generically in model_spec.md; which rows and
# from when is not, and has to be found in the data.
ANOMALY_GROUND_PIXEL = (13, 26)          # inclusive across-track index band
ANOMALY_FROM_EPISODE_TIME = "2023-10-01T00:00:00Z"
ANOMALY_GAIN = 1.40
ANOMALY_OFFSET = 1.8e-5                  # mol m-2
ANOMALY_NOISE_INFLATION = 1.35

# ------------------------------------------------- observation error model ---
# Reported 1-sigma has a floor and a component proportional to the signal, as
# real retrievals and analysers do.
SAT_SIGMA_FLOOR = (0.55e-5, 1.15e-5)     # uniform draw per footprint, mol m-2
SAT_SIGMA_REL = 0.10
STA_SIGMA_FLOOR = (1.8, 3.6)             # uniform draw per site, ug m-3
STA_SIGMA_REL = 0.07

REGION_NAMES = [
    "R1_core_motorway_ring",
    "R2_north_industrial_corridor",
    "R3_east_urban_arterials",
    "R4_south_rural_trunk",
    "R5_west_transit_motorway",
    "R6_southeast_secondary_network",
]
