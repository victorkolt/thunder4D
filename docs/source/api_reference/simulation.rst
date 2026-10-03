thunder4d.simulation
====================

.. py:module:: thunder4d.simulation

Driver: propagates a pulse through the cell and records diagnostics after every pass.

.. py:class:: MPCSimulation(pulse, cell, *, solver=None, record_every=1, far_field=True, max_order=6, mode_reference="linear", save_fields=(), save_dir=".", record_mirror=True, slit=None, profile_wavelengths=(), profile_bandwidth=0.0, profile_axis="x", **solver_kw)

   :param pulse: input :class:`~thunder4d.pulse.Pulse` (copied), defined at the cell centre
   :param cell: :class:`~thunder4d.cell.MPC`
   :param solver: an existing :class:`~thunder4d.solver.UPPESolver`; otherwise one is
      created with ``**solver_kw`` (e.g. ``phi_max=0.02``)
   :param int record_every: record every n passes (pass 0 and the last pass always)
   :param bool far_field: also record far-field quantities (a few seconds per record)
   :param int max_order: highest Hermite–Gauss order in the modal decomposition
   :param mode_reference: ``"linear"`` (linear eigenmode :math:`w_0(\lambda)`) or a float
      waist at :math:`\lambda_0` (e.g. the matched ``w0``), scaled as
      :math:`\sqrt{\lambda/\lambda_0}`
   :param save_fields: pass numbers whose full 3D field is written to
      ``save_dir/field_passNNN.npy``
   :param bool record_mirror: record fluence and profiles at the mirror before each bounce
   :param float slit: slit width [m] integrated around :math:`y = 0` for
      :math:`S(x,\lambda)` (``None`` = one row)
   :param profile_wavelengths: wavelengths [m] whose 1D profiles are recorded every
      pass, with the global profile
   :param float profile_bandwidth: bandpass width [m] (0 = nearest spectral bin, of
      width :math:`\lambda^2/cT`)
   :param str profile_axis: ``"x"`` or ``"y"``

   .. py:method:: run(n_passes=None, verbose=True, max_seconds=None)

      Propagate ``n_passes`` passes (default: up to ``cell.n_passes``). ``max_seconds``
      stops cleanly after the pass that exceeds the wall-clock budget.

   .. py:method:: record(p, mirror_fluence=None, mirror_profiles=None)

      Record diagnostics for pass ``p`` (called automatically).

   .. py:method:: checkpoint(path)

      Pickle the full state (field, history, solver counters).

   .. py:method:: resume(path)

      Restore a checkpoint into a simulation built with the same grid and cell.

   .. py:method:: save(path, include_field=True)

      Write history, axes, metadata (and the final field) to a compressed ``.npz``.

   .. py:property:: done

      ``True`` once ``cell.n_passes`` passes have been made.

   .. py:property:: pulse

      Current field as a :class:`~thunder4d.pulse.Pulse`.

   **Other attributes:** ``Aw`` (current field), ``history``, ``passes_done``,
   ``lam``, ``x``, ``theta``, ``prof_coord`` (sorted axes of the recorded quantities),
   ``solver``.

.. _history-keys:

Recorded history
----------------

``sim.history`` is a dict of lists with one entry per recorded pass. Wavelength-resolved
quantities are interpolated on ``sim.lam``.

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Key
     - Shape per entry
     - Content
   * - ``pass``
     - scalar
     - pass number (0 = input)
   * - ``energy``
     - scalar
     - energy [J]
   * - ``B``
     - scalar
     - accumulated on-axis peak B-integral [rad]
   * - ``peak_power``
     - scalar
     - peak of :math:`P(t)` [W]
   * - ``spectrum``
     - (Nλ,)
     - spatially integrated spectrum
   * - ``fluence``
     - (Nx, Ny)
     - fluence at the cell centre [J/m²]
   * - ``fluence_mirror``
     - (Nx, Ny) or None
     - fluence at the mirror before the bounce
   * - ``sgram_x``
     - (Nx, Nλ)
     - :math:`S(x,\lambda)` at :math:`y = 0` (or over ``slit``)
   * - ``sgram_theta``
     - (Nθ, Nλ) or None
     - :math:`S(\theta,\lambda)` at :math:`\theta_y = 0`
   * - ``eta00``
     - (Nλ,)
     - HG₀₀ fraction vs wavelength
   * - ``eta00_total``
     - scalar
     - spectrally integrated HG₀₀ fraction
   * - ``order_frac``
     - (max_order+2,)
     - energy fraction per mode order
   * - ``V_near``, ``V_far``
     - scalar
     - mean spectral homogeneity at the centre / in the far field
   * - ``wx``, ``wy``, ``M2x``, ``M2y``
     - (Nλ,)
     - second-moment radii and :math:`M^2` vs wavelength
   * - ``w_peak_x``, ``w_peak_y``
     - scalar
     - second-moment radius [m] at the spectral peak of the input pulse (same colour every pass)
   * - ``M2x_mean``, ``M2y_mean``
     - scalar
     - spectrally weighted :math:`M^2`
   * - ``prof_near``
     - (nλ+1, Nx)
     - 1D cuts at the centre (bands + global)
   * - ``prof_near_proj``
     - (nλ+1, Nx)
     - 1D projections at the centre
   * - ``prof_far``
     - (nλ+1, Nθ)
     - far-field 1D cuts
   * - ``prof_mirror``
     - (nλ+1, Nx) or None
     - mirror-plane 1D cuts (None at pass 0)
   * - ``band_energy``
     - (nλ+1,)
     - energy in each band, last = total [J]
   * - ``wall_s``
     - scalar
     - elapsed wall-clock time [s]

Saved file
----------

``sim.save(path)`` writes a compressed ``.npz`` with:

- every non-empty history key, stacked over passes;
- ``prof_mirror`` and ``fluence_mirror`` for passes 1 … N only (before each bounce);
- the axes ``lam``, ``x``, ``t``, ``theta`` and, if profiles were recorded,
  ``prof_coord``, ``prof_wavelengths``, ``prof_bandwidth``;
- metadata ``R``, ``d``, ``n_passes``, ``gas``, ``pressure_bar``, ``n2``, ``lambda0`` and
  ``grid`` = ``[Nx, Ny, Nt, dx, dy, dt]``;
- ``Aw_final`` if ``include_field=True``.
