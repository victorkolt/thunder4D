Simulation
==========

Model
-----

The field is a complex envelope :math:`A(x, y, t)` with carrier :math:`\exp[i(k_0 z - \omega_0 t)]`,
propagated with the unidirectional pulse propagation equation (UPPE) in the frame moving at the group
velocity at :math:`\lambda_0`:

.. math::

   \frac{\partial \tilde A(k_x, k_y, \Omega)}{\partial z}
   = i\left[k_z - k_0 - k_1\Omega\right]\tilde A + i\,\gamma(\Omega)\,\mathcal{F}_t\!\left[|A|^2 A\right],
   \qquad k_z = \sqrt{k(\omega)^2 - k_x^2 - k_y^2},\qquad
   \gamma = \frac{n_2\,\omega}{c}\,\frac{n(\omega_0)}{n(\omega)} .

- **Linear part:** diffraction and the full dispersion of the gas (Sellmeier,
  Börzsönyi *et al.*, Appl. Opt. **47**, 4856, 2008).
- **Kerr part:** instantaneous electronic response, with self-steepening (``self_steepening=False``
  gives :math:`\gamma = n_2\omega_0/c`). It is tapered to zero at the edges of the spectral window.
- **Not modelled:** ionisation and plasma, Raman response of molecular gases (N₂, air), the folded
  off-axis geometry (incidence astigmatism, real spot pattern), plates, windows and pressure gradients,
  vector effects.

Two solvers share the same pulses, diagnostics and files:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - lens frame (recommended)
     - standard
   * - classes
     - :func:`~thunder4d.lensframe.suggest_lens_grid`, :class:`~thunder4d.lensframe.LensFrameSimulation`
     - :func:`~thunder4d.grid.suggest_grid`, :class:`~thunder4d.simulation.MPCSimulation`
   * - grid
     - follows the beam: a few waists wide, whatever :math:`d`
     - fixed: a few mirror spots wide
   * - diffraction
     - paraxial
     - non-paraxial
   * - use
     - every cell, near-concentric ones above all
     - cross-checks

The lens frame is explained in :doc:`lens_frame`. The standard solver keeps the field in
:math:`(x, y, \Omega)`: the linear step needs only 2D spatial FFTs and the Kerr term only 1D temporal
FFTs. They are combined by Strang splitting, with an RK4 Kerr step of size
:math:`h = \varphi_\text{max}/(k_0 n_2 I_\text{peak})`.

The cell
--------

:class:`~thunder4d.cell.MPC` is a symmetric two-mirror cell, treated **unfolded**: the run starts right
after a mirror and each pass goes to the cell centre, on to the next mirror, and bounces.

.. code-block:: text

   mirror ──(d/2)──▶ cell centre ──(d/2)──▶ next mirror │ bounce

The pulse is defined at the cell centre, and the diagnostics are taken there after every pass. That
plane is the waist of the eigenmode with a flat wavefront, so a Hermite–Gauss decomposition there is
exact. A bounce applies the mirror curvature (focal length :math:`R/2`), its reflectivity, GDD, TOD or
measured phase, and an optional soft clear aperture.

With :math:`g = 1 - d/R`, the Gouy phase per pass is :math:`\theta = \arccos g` and

.. math::

   w_0^2 = \frac{\lambda}{2\pi}\sqrt{d(2R-d)},\qquad
   w_\text{mirror}^2 = \frac{\lambda R}{\pi}\sqrt{\frac{d}{2R-d}} .

``MPC.herriott(R, N, k, medium)`` builds the cell that is re-entrant after :math:`N` passes,
:math:`d = R\,[1\pm\cos(2\pi k/N)]` (``+``: concentric branch). A linear re-entrant cell images any
input onto the output, top-hats included: run it once with :math:`n_2 = 0` and check that it does.

**Nonlinear mode matching.** For a Gaussian at constant power :math:`P`, the Kerr lens changes the
eigenmode as if :math:`\lambda \to \lambda\sqrt{1 - P/P_c}`, with :math:`P_c = \lambda^2/(2\pi n_0 n_2)`:
the matched waist and spot are the linear ones times :math:`(1-P/P_c)^{1/4}`.
``cell.match(pulse)`` returns them with the B-integral per pass. A pulse is matched at one power only:
its wings see a mode closer to the linear one, a mismatch that the 4D simulation resolves.

The pulse
---------

A pulse is a spectrum times a spatial beam, :math:`A(x, y, \Omega) = E(\Omega)\,U(x, y; \Omega)`.

.. code-block:: python

   E = m.gaussian_spectrum(grid, fwhm_fs=350)                    # or fwhm_nm=...
   E = m.measured_spectrum(grid, wl_nm, intensity, phase_rad=ph) # spectrometer (+ retrieved phase)
   E = m.spectrum_from_temporal_field(grid, t_fs, field)         # retrieved pulse
   E = m.add_spectral_phase(grid, E, gdd_fs2=500, tod_fs3=2e3)

   beam = m.Beam(m.top_hat(4e-3),
                 zernike={"astig_0": 0.05, "coma_x": 0.02},     # RMS waves at lambda0
                 spatial_chirp_mm_per_nm=(0.01, 0), pft_fs_per_mm=(2.0, 0))
   beam = m.Beam(m.knife_edge(m.gaussian(w), 0.5 * w, axis="x", side="+"))
   beam = m.Beam(m.measured_profile(camera_image, pixel_size=5.5e-6))

Where the beam is defined matters: a top-hat imaged on the mirror and a top-hat focused at the centre
are different inputs.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Builder
     - Beam defined
   * - :meth:`Pulse.at_focus <thunder4d.pulse.Pulse.at_focus>`
     - at the cell centre (matched Gaussian, tests)
   * - :meth:`Pulse.from_near_field <thunder4d.pulse.Pulse.from_near_field>`
     - collimated, then focused into the cell by an achromatic lens (top-hat → Airy pattern)
   * - :func:`imaged_near_field_lens <thunder4d.lensframe.imaged_near_field_lens>`
     - imaged on a plane of the cell, by default the first mirror, in physical coordinates there

.. code-block:: python

   pulse = m.Pulse.from_near_field(grid, E, m.Beam(m.top_hat(4e-3)), energy,
                                   nf_size=12e-3, nf_points=192, target_waist=mm["w0"])
   pulse = m.imaged_near_field_lens(grid, E, m.Beam(m.top_hat(1.12 * mm["w_mirror"])), energy, cell)

Top-hats, hard edges and Airy patterns need a larger box (``box_factor`` 10–14).

Running
-------

.. code-block:: python

   sim = m.LensFrameSimulation(pulse, cell, phi_max=0.02, gouy_step=0.05, mode_reference=mm["w0"],
                               profile_wavelengths=[1010e-9, 1030e-9, 1050e-9], profile_bandwidth=2e-9)
   sim.run()                     # or sim.run(n_passes=1) to go pass by pass
   sim.save("run.npz")

- ``record_every``, ``near_field=False`` (no collimated-beam diagnostics, a few seconds less per record),
  ``record_mirror``: what is recorded.
- ``profile_wavelengths``: 1D profiles of single colours (bandpass ``profile_bandwidth``) at the centre,
  in the collimated beam and on the mirror, at every pass.
- Mirror records are in the frame of the beam: multiply positions by ``sim.mirror_scale`` and divide
  fluences by its square. The saved file keeps ``mirror_scale``.

**Long runs** stop cleanly and resume:

.. code-block:: python

   if os.path.exists("ckpt.pkl"):
       sim.resume("ckpt.pkl")
   sim.run(max_seconds=3600)     # stops after the pass that exceeds one hour
   sim.checkpoint("ckpt.pkl")

**Between passes**, any element can be applied to ``sim.Aw`` (the field at the centre, physical in both
solvers), for example an aperture:

.. code-block:: python

   aperture = (grid.r2 < (1.5e-3) ** 2)[:, :, None]
   for p in range(cell.n_passes):
       sim.run(n_passes=1, verbose=False)
       sim.Aw *= aperture

**New physics** (ionisation, Raman, ...) goes in the nonlinear term: subclass the solver and override
:meth:`~thunder4d.solver.UPPESolver.nonlinear`, which works in :math:`(x, y, \Omega) \leftrightarrow
(x, y, t)`. :class:`~thunder4d.lensframe.LensFrameSolver` inherits it; there :math:`|B|^2` is the
intensity times :math:`s^2`. Such a solver runs its RK4 in NumPy, not in the Numba kernels
(:ref:`numba-kernels`).

Results
-------

``sim.history`` has one entry per recorded pass: energy, B-integral, peak power and intensity,
spectrum, fluence, spectrograms :math:`S(x,\lambda)` and :math:`S(\theta,\lambda)`, Hermite–Gauss
content, homogeneity, beam radii and :math:`M^2` (:ref:`history-keys`). The functions of
:mod:`thunder4d.diagnostics` analyse any field:

.. code-block:: python

   import numpy as np
   from thunder4d import diagnostics as dg

   x, lam, S = dg.spectrogram_x_lambda(grid, sim.Aw)
   c = dg.compress(grid, sim.Aw)                # best uniform GDD, compressed pulse
   d = np.load("run.npz")                       # the saved history
