"""Design helper, part 1: what the user specifies.

Every knob of the design helper lives here, grouped in dataclasses:

    LaserSpec     the input pulse (front end)
    Target        what "best" means (objective, quality floors)
    Constraints   what is allowed (space, mirrors, Kerr, ionisation, spot pattern, pressure, gas)
    Losses        what is lost on the way in (mode matching)
    SearchSpace   which configurations are tried (gases, RoC, round trips, k, passes, ...)
    Refinement    how the best candidates are fine-tuned with the fast 1+1D code
    Tolerance     alignment tolerances reported for a design (information, not a constraint)

Cell geometry convention
------------------------
A Herriott cell is set by its RoC R, the number of round trips M of one re-entrant cycle and the
index k (coprime with M): d = R (1 + cos(pi k / M)) on the concentric branch (d > R) and
d = R (1 - cos(pi k / M)) on the planar branch (d < R), k < M / 2. One cycle is 2 M passes
(a pass = transit over d + bounce, starting right after a mirror, as in the 4D code) and puts M
spots on each mirror. The number of passes actually used
can be anything from SearchSpace.min_passes to 2 M (odd: the beam leaves on the other mirror).

Conventions
-----------
* SI units unless the field name says otherwise (_fs, _nm, _bar, _mm, _J_cm2, _W_cm2, ...).
* A constraint is a ``Limit``: a value, a bound ("max" | "min"), a mode ("hard" rejects,
  "soft" penalises, "off" ignores), a safety factor and a weight for soft penalties.
  ``value=None`` means "take the value from the catalogue" (e.g. the coating LIDT or the gas
  ionisation threshold), so a knob can stay active without hard-coding a number.
* Defaults are starting points, NOT recommendations: check them against your own setup.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

Mode = Literal["hard", "soft", "off"]

# Peak power of a transform-limited pulse: P = shape_factor * E / FWHM
PEAK_POWER_FACTOR = {"gaussian": 0.9394, "sech2": 0.8814}


# ============================================================ generic constraint
@dataclass
class Limit:
    """One constraint.

    value  : the limit (None = taken from the catalogue when the constraint is evaluated)
    bound  : "max" (x <= allowed) or "min" (x >= allowed)
    mode   : "hard" -> candidate rejected, "soft" -> penalised in the score, "off" -> ignored
    safety : allowed = value * safety for "max", value / safety for "min" (safety <= 1 is stricter)
    weight : weight of the penalty when mode == "soft"
    """
    value: float | None = None
    bound: Literal["max", "min"] = "max"
    mode: Mode = "hard"
    safety: float = 1.0
    weight: float = 1.0

    def allowed(self, default=None):
        v = self.value if self.value is not None else default
        if v is None:
            return None
        return v * self.safety if self.bound == "max" else v / self.safety

    def slack(self, x, default=None):
        """Normalised margin: >= 0 if satisfied, < 0 if violated, None if inactive."""
        a = self.allowed(default)
        if self.mode == "off" or a is None:
            return None
        s = (a - x) if self.bound == "max" else (x - a)
        return s / abs(a) if a != 0 else s

    def penalty(self, x, default=None):
        """0 if satisfied or off, inf if a hard limit is violated, weight * |slack|^2 if soft."""
        s = self.slack(x, default)
        if s is None or s >= 0:
            return 0.0
        return np.inf if self.mode == "hard" else self.weight * s ** 2


def Max(value=None, mode: Mode = "hard", safety=1.0, weight=1.0):
    return Limit(value, "max", mode, safety, weight)


def Min(value=None, mode: Mode = "hard", safety=1.0, weight=1.0):
    return Limit(value, "min", mode, safety, weight)


def Off():
    return Limit(None, "max", "off")


# ============================================================ input laser
@dataclass
class LaserSpec:
    """Front-end pulse injected in the cell.

    duration_fs is the FWHM intensity duration of the injected pulse. If it is chirped, give its
    transform limit as tl_duration_fs: the chirp is then added as a positive GDD.
    """
    energy_J: float = 1e-3
    duration_fs: float = 300.0
    wavelength_nm: float = 1030.0
    tl_duration_fs: float | None = None        # None -> duration_fs (transform limited)
    shape: Literal["gaussian", "sech2"] = "gaussian"
    # spectral phase added before injection (e.g. to explore chirped injection)
    prechirp_gdd_fs2: float = 0.0
    prechirp_tod_fs3: float = 0.0
    rep_rate_Hz: float = 1e3
    polarization: Literal["linear", "circular"] = "linear"
    name: str = "laser"

    @property
    def lambda0(self):
        return self.wavelength_nm * 1e-9

    @property
    def peak_power_W(self):
        """Peak power at the given (possibly chirped) duration."""
        return PEAK_POWER_FACTOR[self.shape] * self.energy_J / (self.duration_fs * 1e-15)

    @property
    def average_power_W(self):
        return self.energy_J * self.rep_rate_Hz

    @property
    def n2_polarization_factor(self):
        """Effective n2 scaling: 1 for linear, 2/3 for circular polarisation (isotropic medium)."""
        return 1.0 if self.polarization == "linear" else 2.0 / 3.0

    @classmethod
    def preset(cls, name, **overrides):
        """LaserSpec from the preset registry, any field overridable."""
        if name not in LASER_PRESETS:
            raise KeyError(f"unknown preset {name!r}; available: {sorted(LASER_PRESETS)}")
        kw = dict(LASER_PRESETS[name])
        kw.update(overrides)
        kw.setdefault("name", name)
        return cls(**kw)


# Manufacturer datasheet values (checked 2026-11): durations are the specified upper bounds
# ("< x fs", gaussian assumed), i.e. worst case. rep_rate_Hz = max power / max energy (the rate
# at which the max energy is specified), not a datasheet line. Override any field at call time:
# LaserSpec.preset("pharos_ph2_sp_2mJ", duration_fs=170.0, rep_rate_Hz=1e3)
LASER_PRESETS: dict[str, dict] = {
    "mpc1_18mJ_350fs": {"energy_J": 18e-3, "duration_fs": 350.0, "wavelength_nm": 1030.0},
    # Light Conversion PHAROS-PH2 (datasheet rev. 260601)
    "pharos_ph2_10w": {"energy_J": 0.2e-3, "duration_fs": 290.0, "rep_rate_Hz": 50e3},
    "pharos_ph2_sp_0.4mJ": {"energy_J": 0.4e-3, "duration_fs": 190.0, "rep_rate_Hz": 50e3},
    "pharos_ph2_sp_1mJ": {"energy_J": 1e-3, "duration_fs": 190.0, "rep_rate_Hz": 20e3},
    "pharos_ph2_sp_2mJ": {"energy_J": 2e-3, "duration_fs": 190.0, "rep_rate_Hz": 10e3},
    "pharos_ph2_5mJ": {"energy_J": 5e-3, "duration_fs": 250.0, "rep_rate_Hz": 4e3},
    "pharos_ph2_up_0.4mJ": {"energy_J": 0.4e-3, "duration_fs": 100.0, "rep_rate_Hz": 50e3},
    "pharos_ph2_up_1mJ": {"energy_J": 1e-3, "duration_fs": 100.0, "rep_rate_Hz": 20e3},
    # Light Conversion CARBIDE (product page 2026-09)
    "carbide_cb3_20w": {"energy_J": 0.4e-3, "duration_fs": 250.0, "rep_rate_Hz": 50e3},
    "carbide_cb3_40w": {"energy_J": 0.4e-3, "duration_fs": 250.0, "rep_rate_Hz": 100e3},
    "carbide_cb3_80w_0.8mJ": {"energy_J": 0.8e-3, "duration_fs": 250.0, "rep_rate_Hz": 100e3},
    "carbide_cb3_80w_2mJ": {"energy_J": 2e-3, "duration_fs": 350.0, "rep_rate_Hz": 40e3},
    "carbide_cb3_120w": {"energy_J": 1e-3, "duration_fs": 250.0, "rep_rate_Hz": 120e3},
    "carbide_cb5_6w": {"energy_J": 100e-6, "duration_fs": 290.0, "rep_rate_Hz": 60e3},
    "carbide_cb5_5w": {"energy_J": 83e-6, "duration_fs": 290.0, "rep_rate_Hz": 60e3},
    "carbide_cb5_sp": {"energy_J": 100e-6, "duration_fs": 190.0, "rep_rate_Hz": 50e3},
    # TRUMPF Scientific Lasers Dira (trumpf-scientific-lasers.com, technical data)
    "dira_200_100": {"energy_J": 2e-3, "duration_fs": 1000.0, "rep_rate_Hz": 100e3},
    "dira_200_1": {"energy_J": 200e-3, "duration_fs": 600.0, "rep_rate_Hz": 1e3},
    "dira_500_10": {"energy_J": 50e-3, "duration_fs": 600.0, "rep_rate_Hz": 10e3},
    "dira_1000_5": {"energy_J": 200e-3, "duration_fs": 600.0, "rep_rate_Hz": 5e3},
    "dira_1000_1": {"energy_J": 500e-3, "duration_fs": 600.0, "rep_rate_Hz": 1e3},
}


def register_laser_preset(name, **fields):
    LASER_PRESETS[name] = fields


# ============================================================ objective
@dataclass
class Target:
    """What the design should achieve.

    objective:
        "shortest"            minimise the compressed duration
        "max_peak_power"      maximise the compressed peak power (duration AND throughput)
        "target_duration"     reach target_duration_fs, not shorter
        "target_compression"  reach target_compression (input duration / output duration)
        "weighted"            minimise sum of weights[key] * metric (normalised), keys below
    """
    objective: Literal["shortest", "max_peak_power", "target_duration",
                       "target_compression", "weighted"] = "max_peak_power"
    target_duration_fs: float | None = None
    target_compression: float | None = None
    # "weighted" objective: duration / input duration, -throughput, separation / max separation,
    # pressure / max pressure
    weights: dict[str, float] = field(default_factory=lambda: {
        "duration": 1.0, "throughput": 1.0, "length": 0.0, "pressure": 0.0})
    # quality floors (constraints on the result, not on the hardware)
    min_throughput: Limit = field(default_factory=lambda: Min(0.80, mode="soft"))
    min_main_peak_fraction: Limit = field(default_factory=lambda: Min(0.70, mode="soft"))


# ============================================================ constraints
@dataclass
class SpaceConstraints:
    max_separation_m: Limit = field(default_factory=lambda: Max(4.0))
    chamber_margin_m: float = 0.2        # chamber length = d + mirror mounts margin
    max_chamber_length_m: Limit = field(default_factory=Off)
    fixed_chamber_length_m: float | None = None   # existing chamber: d + margin must fit inside it


@dataclass
class MirrorConstraints:
    allowed_roc_m: tuple[float, ...] = (0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    allowed_diameters_inch: tuple[float, ...] = (3.0,)
    # cavity coatings allowed (names from the mirror catalogue); None = every cavity coating
    allowed_coatings: tuple[str, ...] | None = None
    # peak fluence on the mirrors; value None -> LIDT of the coating at the local pulse duration
    fluence: Limit = field(default_factory=lambda: Max(None))
    # fraction of the output energy outside the coating band (spectral wings are never exactly 0)
    bandwidth: Limit = field(default_factory=lambda: Max(0.01))
    # thermal: average power on one mirror (high repetition rate)
    max_average_power_W: Limit = field(default_factory=Off)


@dataclass
class KerrConstraints:
    p_over_pcr: Limit = field(default_factory=lambda: Max(0.5))
    b_per_pass_rad: Limit = field(default_factory=lambda: Max(1.0))   # on-axis, per pass
    b_total_rad: Limit = field(default_factory=Off)


@dataclass
class IonizationConstraints:
    # peak intensity at the waist; value None -> ionisation threshold of the gas (catalogue)
    peak_intensity_W_cm2: Limit = field(default_factory=lambda: Max(None, safety=0.5))


@dataclass
class SpotConstraints:
    """Circular spot pattern on each mirror: M spots on a circle of radius r.

    r = pattern_radius_m, or by default the largest radius that keeps edge_clearance_w between
    the spot centres and the mirror edge (mirror radius - edge_clearance_w * w_mirror).
    Adjacent spots are 2 r sin(pi / M) apart."""
    min_spot_separation_w: Limit = field(default_factory=lambda: Min(3.0))   # in units of w_mirror
    edge_clearance_w: Limit = field(default_factory=lambda: Min(2.5))        # spot centre to edge
    pattern_radius_m: float | None = None


@dataclass
class PressureConstraints:
    pressure_bar: tuple[float, float] = (0.0, 3.0)     # chamber / window rating
    resolution_bar: float = 0.01                       # your regulator's resolution


@dataclass
class GasConstraints:
    allowed: tuple[str, ...] = ("He", "Ne", "Ar", "Kr", "Xe", "N2")


@dataclass
class Constraints:
    space: SpaceConstraints = field(default_factory=SpaceConstraints)
    mirrors: MirrorConstraints = field(default_factory=MirrorConstraints)
    kerr: KerrConstraints = field(default_factory=KerrConstraints)
    ionization: IonizationConstraints = field(default_factory=IonizationConstraints)
    spots: SpotConstraints = field(default_factory=SpotConstraints)
    pressure: PressureConstraints = field(default_factory=PressureConstraints)
    gas: GasConstraints = field(default_factory=GasConstraints)


# ============================================================ losses
@dataclass
class Losses:
    """Losses not contained in the catalogues (the coating R(lambda) is)."""
    mode_matching_efficiency: float = 1.0      # fraction of the energy injected in the cell mode


# ============================================================ tolerance (information only)
@dataclass
class Tolerance:
    """Alignment tolerances reported for a design: how much the waist and the mirror spot change
    for these errors. Information for the user, not a constraint."""
    separation_error_mm: float = 1.0
    roc_tolerance_rel: float = 0.005


# ============================================================ search
@dataclass
class SearchSpace:
    """What starting_points() enumerates. Restrict a field to a single value to lock it
    ("fix what I already own": one RoC, one coating, ...)."""
    gases: tuple[str, ...] | None = None              # None -> Constraints.gas.allowed
    roc_m: tuple[float, ...] | None = None            # None -> Constraints.mirrors.allowed_roc_m
    diameters_inch: tuple[float, ...] | None = None   # None -> Constraints.mirrors.allowed_diameters_inch
    coatings: tuple[str, ...] | None = None           # None -> Constraints.mirrors.allowed_coatings
    branches: tuple[Literal["concentric", "planar"], ...] = ("concentric", "planar")
    round_trips: tuple[int, int] = (3, 40)            # inclusive range of M (one re-entrant cycle)
    k_range: tuple[int, int] | None = None            # None -> every k < M/2 coprime with M
    # passes tried per geometry, relative to a full cycle: 2 M + offset (0: full cycle, back on
    # the input mirror; -1: one pass less, out on the other mirror)
    pass_offsets: tuple[int, ...] = (0, -1)
    min_passes: int = 5


# ============================================================ refinement with the 1+1D code
@dataclass
class Refinement:
    """The best candidates are re-run with the fast 1+1D code: their continuous parameters are
    tuned, and the number of passes (same geometry) and the RoC are scanned around them."""
    n_candidates: int = 5
    diversity: bool = True          # spread the candidates over gases, then RoC / branch
    # continuous parameters tuned by the optimiser: "pressure_bar", "prechirp_gdd_fs2"
    tune: tuple[str, ...] = ("pressure_bar",)
    neighbour_n_passes: int = 2     # passes +- this, same geometry, within [min_passes, 2 M]
    neighbour_roc_steps: int = 0    # RoC +- this many steps of the allowed list (same M, k)
    method: Literal["nelder-mead", "powell"] = "nelder-mead"
    max_evaluations: int = 60
    xtol_rel: float = 1e-3
    keep_traces: bool = True        # keep every evaluation for plots


@dataclass
class DesignProblem:
    """Everything in one object (what MPCDesigner takes, and what gets saved with the results)."""
    laser: LaserSpec = field(default_factory=LaserSpec)
    target: Target = field(default_factory=Target)
    constraints: Constraints = field(default_factory=Constraints)
    losses: Losses = field(default_factory=Losses)
    search: SearchSpace = field(default_factory=SearchSpace)
    refinement: Refinement = field(default_factory=Refinement)
    tolerance: Tolerance = field(default_factory=Tolerance)

    def to_dict(self):
        def clean(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            return v
        return clean(asdict(self))
