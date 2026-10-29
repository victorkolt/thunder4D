import thunder4d as m
from helpers import LAM0


def near_concentric(medium, n_passes=2):
    """The cell of benchmark/lens_frame_check.py: d close to 2R, where the lens frame pays off."""
    return m.MPC(R=0.5, d=0.94, medium=medium, n_passes=n_passes)


# ------------------------------------------------------------ suggest_lens_grid
BAND = dict(lambda0=LAM0, lambda_min=900e-9, lambda_max=1200e-9, pulse_fwhm=200e-15)


def test_suggest_lens_grid_spectral_window_covers_the_band():
    g = m.suggest_lens_grid(near_concentric(m.Material("Ar", 1.0)), **BAND, device="cpu")
    lam = g.lam[g.valid]
    assert lam.min() <= 900e-9
    assert lam.max() >= 1200e-9


def test_suggest_lens_grid_time_window_holds_the_pulse():
    g = m.suggest_lens_grid(near_concentric(m.Material("Ar", 1.0)), **BAND, time_window_factor=8,
                            device="cpu")
    assert g.Nt * g.dt >= 8 * 200e-15

