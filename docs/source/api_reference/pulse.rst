thunder4d.pulse
===============

.. py:module:: thunder4d.pulse

Pulse initialisation. A pulse is built as

.. math::

   A(x, y, \Omega) = E(\Omega)\; U(x, y; \Omega),

where :math:`E(\Omega)` is a 1D complex spectral amplitude and :math:`U` a spatial field
evaluated separately at every frequency (so it can be chromatic) and normalised to unit
power at each frequency. The spatially integrated spectrum is therefore exactly
:math:`|E(\Omega)|^2`.

Spectral amplitude
------------------

All functions return a complex 1D array on ``grid.Omega`` (FFT order).

.. py:function:: gaussian_spectrum(grid, fwhm_fs=None, fwhm_nm=None, center_nm=None)

   Transform-limited Gaussian. Give either the intensity FWHM duration [fs] or the
   spectral FWHM [nm]. ``center_nm`` defaults to :math:`\lambda_0`.

.. py:function:: measured_spectrum(grid, wavelength_nm, intensity, phase_rad=None, background=0.0)

   Spectrum measured against wavelength. Applies the Jacobian
   :math:`S_\omega = S_\lambda\,\lambda^2/2\pi c`, subtracts ``background``, clips
   negative values and interpolates onto the grid. An optional spectral phase (e.g.
   from a d-scan or FROG retrieval) can be given on the same wavelength axis; it is
   unwrapped before interpolation.

.. py:function:: spectrum_from_temporal_field(grid, t_fs, field)

   Complex temporal envelope (e.g. a retrieved pulse :math:`|E|e^{i\varphi}`), relative to
   ``grid.lambda0``, interpolated on the time grid and transformed.

.. py:function:: add_spectral_phase(grid, E, gdd_fs2=0.0, tod_fs3=0.0, fod_fs4=0.0, delay_fs=0.0, phase=None)

   Returns :math:`E\,e^{i\varphi}` with
   :math:`\varphi = \frac{\text{GDD}}{2}\Omega^2 + \frac{\text{TOD}}{6}\Omega^3 +
   \frac{\text{FOD}}{24}\Omega^4 + \text{delay}\cdot\Omega` (:math:`\Omega` in rad/fs).
   ``phase`` adds an arbitrary term: a callable of :math:`\Omega` [rad/s] or an array on
   ``grid.Omega``.

Spatial profiles
----------------

.. py:class:: Profile(func, radius, name)

   Callable ``f(X, Y)`` returning a complex amplitude. ``radius`` is used as the default
   Zernike pupil.

.. py:function:: gaussian(w)

   :math:`\exp(-r^2/w^2)` (1/e² intensity radius :math:`w`).

.. py:function:: super_gaussian(w, order=10)

   :math:`\exp[-(r/w)^\text{order}]`; order 2 is a Gaussian.

.. py:function:: top_hat(radius, order=24)

   Super-Gaussian with soft edges. Higher orders give harder edges and need a finer
   near-field sampling (e.g. order 100 with 384 points).

.. py:function:: knife_edge(profile, edge, axis="x", side="+", width=None)

   Clip any profile with a hard edge at ``edge`` [m] along ``axis``.

   :param str side: ``"+"`` blocks :math:`x > \text{edge}`, ``"-"`` blocks
      :math:`x < \text{edge}`, ``"both"`` keeps :math:`|x| < |\text{edge}|` (slit)
   :param float width: 10–90 % width of a tanh edge [m]; ``None`` gives a step

.. py:function:: measured_profile(image, pixel_size, center="centroid", background=0.0)

   Camera image (rows = y, columns = x) → amplitude :math:`\sqrt{I}`, bilinear
   interpolation, zero outside the image. ``center`` is ``"centroid"`` or
   ``(cx, cy)`` [m].

Aberrations
-----------

.. py:function:: zernike_wavefront(X, Y, coeffs_waves, pupil_radius, lambda0)

   Wavefront error :math:`W` [m] from RMS-normalised Zernike terms with coefficients in
   waves at :math:`\lambda_0`. In :class:`Beam` it is applied as the phase
   :math:`(\omega/c)\,W`, an achromatic path error correct for every colour.

   Available terms: ``piston``, ``tilt_x``, ``tilt_y``, ``defocus``, ``astig_0``,
   ``astig_45``, ``coma_x``, ``coma_y``, ``trefoil_0``, ``trefoil_30``, ``spherical``.

Beam
----

.. py:class:: Beam(profile, zernike=None, pupil_radius=None, spatial_chirp_mm_per_nm=(0, 0), angular_dispersion_urad_per_nm=(0, 0), pft_fs_per_mm=(0, 0), pfc_fs_per_mm2=0.0, radial_gdd_fs2_per_mm2=0.0, tilt_urad=(0, 0), offset_m=(0, 0), custom_phase=None)

   Spatial part of the pulse: profile + aberrations + space-time couplings. With
   :math:`\Delta\lambda = \lambda - \lambda_0` and :math:`\Omega` in rad/fs:

   .. list-table::
      :widths: 35 65

      * - ``profile``
        - spatial amplitude (:class:`Profile`)
      * - ``zernike``
        - dict, e.g. ``{"astig_0": 0.1, "coma_x": 0.05}`` (RMS waves at :math:`\lambda_0`)
      * - ``pupil_radius``
        - Zernike normalisation radius [m] (default ``profile.radius``)
      * - ``spatial_chirp_mm_per_nm``
        - ``(sx, sy)``: each colour shifted by :math:`s\,\Delta\lambda`
      * - ``angular_dispersion_urad_per_nm``
        - ``(ax, ay)``: propagation angle :math:`a\,\Delta\lambda`
      * - ``pft_fs_per_mm``
        - ``(px, py)``: pulse-front tilt, delay :math:`p\,x`
      * - ``pfc_fs_per_mm2``
        - pulse-front curvature, delay :math:`a\,r^2`
      * - ``radial_gdd_fs2_per_mm2``
        - GDD varying as :math:`r^2` (radial chirp)
      * - ``tilt_urad``
        - pointing, same for all colours
      * - ``offset_m``
        - transverse position
      * - ``custom_phase``
        - callable ``(X, Y, Omega) -> phase [rad]``

   .. py:method:: field(X, Y, omega, omega0)

      Unnormalised complex field at one angular frequency.

Pulse
-----

.. py:class:: Pulse(grid, Aw)

   Field :math:`A(x, y, \Omega)` of shape ``grid.shape``.

   **Builders**

   .. py:classmethod:: at_focus(grid, E, beam, energy)

      Beam defined directly at the cell centre.

   .. py:classmethod:: from_near_field(grid, E, beam, energy, nf_size, nf_points=256, focal_length=None, target_waist=None, verbose=True)

      Beam defined in the collimated near field, in the front focal plane of an
      achromatic lens; the cell centre is the focal plane. Exact per-colour Fraunhofer
      transform (spot size :math:`\propto\lambda`): a top-hat gives an Airy pattern.

      :param float nf_size: full width of the near-field window [m]
      :param int nf_points: near-field sampling
      :param float focal_length: lens focal length [m]. If ``None``, it is chosen by
         golden-section search to maximise the overlap at :math:`\lambda_0` with a
         Gaussian of waist ``target_waist`` and stored in ``pulse.focal_length``.

      A warning is printed if less than 99 % of the focused power fits in the box.

   .. py:classmethod:: imaged_near_field(grid, E, beam, energy, cell, image_z=None, curvature_radius=None, verbose=True)

      Near field relay-imaged (same size for every colour) onto the plane at distance
      ``image_z`` from the centre, on the input side (default: the mirror, :math:`d/2`),
      with a spherical wavefront of radius ``curvature_radius`` (default
      :math:`R(z) = z + z_R^2/z` of the linear mode, equal to the mirror ROC at the
      mirror plane). The field is then propagated back (linear, exact) to the cell
      centre.

      .. tip::

         For a top-hat, a radius of 1.12 × ``w_mirror`` maximises the overlap with the
         Gaussian mode (:math:`\eta = 0.815`).

   All builders normalise each colour to unit power before multiplying by
   :math:`E(\Omega)`, then scale the total to ``energy`` [J].

   **Methods and properties**

   .. py:method:: set_energy(energy)

      Rescale to ``energy`` [J]; returns self.

   .. py:property:: energy

      Energy [J].

   .. py:method:: At()

      Field in :math:`(x, y, t)`.

   .. py:method:: power()

      Spatially integrated power :math:`P(t)` [W] (FFT order).

   .. py:method:: peak_power()

      :math:`\max P(t)` [W].

   .. py:method:: weighted_power()

      :math:`\int P^2 dt / \int P\,dt` (= :math:`P_\text{peak}/\sqrt 2` for a Gaussian).

   .. py:method:: add_spectral_phase(**kw)

      Same arguments as :func:`add_spectral_phase`, applied to every pixel.

   .. py:method:: copy()

      Independent copy.
