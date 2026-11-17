Bulk and hybrid cells
=====================

A multipass cell can hold solid plates: a **bulk** cell has plates in vacuum, a **hybrid** cell has plates
and a gas. Both run with the lens-frame solver, with the same pulses, diagnostics and files as a gas cell.

Plates and gas regions
----------------------

A plate is a :class:`~thunder4d.materials.Solid` (``FS``, ``CaF2``, ``BaF2``, ``MgF2``, ``sapphire``, ``YAG``)
of a given thickness at a given position, at normal incidence, crossed once per pass. Its two faces carry a
:class:`~thunder4d.plates.Coating`: ``"bare"`` (Fresnel reflection from :math:`n(\lambda)`), ``"ar"``
(no reflection), a flat reflectivity, :math:`R(\lambda)` with a spectral phase, or a tabulated curve.

The position is measured from the cell centre, positive towards mirror B, the mirror hit at the end of
pass 1. The odd passes go from A to B and the even ones from B to A, so along the direction of travel an
off-centre plate is at :math:`+z` on the odd passes and at :math:`-z` on the even ones, and its two faces
swap their roles.

A *gas region* ``(z0, z1, gas)`` fills part of the cell with another gas than the filling one. A vacuum
cell between two window plates in air is the filling gas ``air`` with a vacuum region between the plates.

.. code-block:: python

   import thunder4d as th

   fs = th.Solid("FS")
   plates = [th.Plate(fs, 1e-3, +0.14, coating="ar"), th.Plate(fs, 1e-3, -0.14, coating="ar")]
   cell = th.MPC(0.2, 0.382, th.Material("air", 1.0), 14, plates=plates,
                 regions=[(-0.14, 0.14, th.vacuum())], wavelength=1030e-9)

This is the cell of Westerberg *et al.* (APL Photonics **10**, 096103, 2025). A cell with plates or regions
needs its reference ``wavelength``. With ``medium=None`` the cell is filled with vacuum (a bulk cell).

Diffraction and the equivalent cell
-----------------------------------

For diffraction, a plate of thickness :math:`t` and index :math:`n_p` in a medium of index :math:`n_g` acts
as a stretch of the filling medium of length :math:`t\,n_g/n_p`. A cell with plates is therefore
equivalent to an empty cell of separation

.. math::

   d_\text{eq} = d - \sum_i t_i\left(1 - \frac{n_g}{n_{p,i}}\right),

which sets the eigenmode, the Gouy phase and the lens frame (``MPC.equivalent_separation``).
:math:`0 < d_\text{eq} < 2R` is the stability condition. Dispersion and the Kerr effect act over the physical
thickness, each medium with its own dispersion, group-velocity frame and :math:`n_2`: the solver holds one
solver per medium. The faces multiply the intensity-normalised envelope by :math:`\sqrt{1-R(\lambda)}` and
the phase of the coating.

Kerr-matched injection
----------------------

A plate in the cell can work far above the critical power of the gas, up to :math:`P/P_c \approx 350` in the
plates of the published cells, and the beam must be matched with the Kerr lenses of the plates. With plates,
:meth:`~thunder4d.cell.MPC.nonlinear_mode` follows the second moment of the beam through the segments of
the cell, and :meth:`~thunder4d.pulse.Pulse.cell_matched` builds the beam to give at the cell centre. A
Laguerre-Gauss vortex of order ``l`` is matched with the critical power of that mode (``vortex=l``).

.. code-block:: python

   g = th.suggest_lens_grid(cell, 1030e-9, 880e-9, 1220e-9, 180e-15, box_factor=6,
                            points_per_waist=6, time_window_factor=6)
   pulse = th.Pulse.cell_matched(g, th.gaussian_spectrum(g, fwhm_fs=180), cell, 250e-6, power="weighted")
   sim = th.LensFrameSimulation(pulse, cell, phi_max=0.05)
   sim.run()

``examples/bulk_mpc_westerberg2025.py`` runs this cell at 100 to 250 µJ.

In an input file
----------------

The same cell is described in the input file by the keys ``cell.plates``, ``cell.regions`` and
``cell.wavelength_nm``, and ``medium.gas = "vacuum"`` (which needs no pressure) for a bulk cell. The input
checks, the builder of the graphical interface and :func:`~thunder4d.inputfile.describe_input` see the plates
like the rest of the cell: the summary gives the cell kind (``gas``, ``bulk`` or ``hybrid``) and
:math:`P/P_c` in each plate.

.. code-block:: python

   cfg = th.default_input(
       laser={"energy_J": 250e-6, "duration_fs": 180},
       beam={"injection": "nonlinear"},
       medium={"gas": "air", "pressure_bar": 1.0},
       cell={"roc_m": 0.2, "separation_m": 0.382, "n_passes": 14,
             "plates": [{"material": "FS", "thickness_m": 1e-3, "position_m": 0.14},
                        {"material": "FS", "thickness_m": 1e-3, "position_m": -0.14}],
             "regions": [{"z0_m": -0.14, "z1_m": 0.14, "gas": "vacuum"}]},
       grid={"lambda_min_nm": 880, "lambda_max_nm": 1220},
   )
   b = th.run_input(cfg)

Keys of a plate: ``material``, ``thickness_m``, ``position_m``, ``coating_a`` and ``coating_b`` (``"bare"`` or
``"ar"``, default ``"ar"``: face A looks at mirror A), ``n2`` (None: the table value). Keys of a region:
``z0_m``, ``z1_m``, ``gas`` (default ``"vacuum"``), ``pressure_bar``, ``temperature_K``. With
``beam.injection`` set to ``linear`` or ``nonlinear``, the beam is built by ``Pulse.cell_matched`` (at the
intensity-weighted power for ``nonlinear``). Tabulated coatings and custom solids are available from Python
only.

Limits
------

- Plates are at normal incidence and have no absorption.
- The colour dependence of :math:`n_g/n_p` in the reduced length is neglected: the relative error on the
  diffraction inside a plate is about :math:`(\mathrm{d}n/n)\,t/z_R`.
- The B-integral uses the vacuum wavenumber for the Kerr phase, and the curvature phase of a mirror uses the
  wavenumber of the gas at the mirror. For a gas cell this changes the results by about :math:`10^{-4}`.
- The design helper covers gas-filled cells only.

Benchmarks
----------

Nine benchmarks reproduce published bulk and hybrid cells, from 0.37 µJ to 610 µJ: Westerberg *et al.* 2025
(100 and 250 µJ), Koltalo *et al.* 2025 (Gaussian and LG\ :sub:`0,1` to LG\ :sub:`0,3` vortex beams), Omar *et al.*
2024 (hybrid air and fused silica, 400 µJ) and Hu *et al.* 2025 (oscillator driven, 0.37 and 0.5 µJ). They are in the
:doc:`benchmark report <../benchmarks/index>`, with the calibrations they needed (for example the effective
:math:`n_2` of air, which has no delayed Raman response here).
