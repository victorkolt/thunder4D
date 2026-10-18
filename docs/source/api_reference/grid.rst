thunder4d.grid
==============

.. py:module:: thunder4d.grid

Simulation grid and FFT conventions (see :doc:`../overview/conventions`).

.. py:data:: C0

   Speed of light in vacuum [m/s].

Grid
----

.. py:class:: Grid(Nx, dx, Nt, lambda0, *, dt=None, lambda_min=None, lambda_max=None, Ny=None, dy=None, dtype=np.complex64, fft_backend="auto", workers=None, device="auto")

   :math:`(x, y, t)` grid centred on :math:`\lambda_0`.

   :param int Nx: number of transverse points (``Ny`` defaults to the same)
   :param float dx: transverse step [m] (``dy`` defaults to the same)
   :param int Nt: number of temporal points
   :param float lambda0: carrier wavelength [m]; defines :math:`\Omega = 0` and the
      moving frame
   :param float dt: time step [s]. Alternatively give ``lambda_min`` (and optionally
      ``lambda_max``) and ``dt`` is chosen so the spectral window covers them.
   :param dtype: ``np.complex64`` (default, half the memory) or ``np.complex128``
   :param str device: ``"auto"`` (GPU if CuPy and a CUDA device work, else CPU), ``"cpu"`` or
      ``"gpu"`` (raises if no GPU is usable). The field, propagators and mirror transfer function live
      on that device (``grid.device``, ``grid.xp``); axes such as ``grid.x`` and ``grid.lam`` stay NumPy.
   :param str fft_backend: ``"auto"`` (SciPy if installed, else NumPy), ``"scipy"`` or
      ``"numpy"``
   :param int workers: threads for the SciPy backend (default: all cores)

   **Attributes**

   .. list-table::
      :widths: 30 70

      * - ``x, y``
        - coordinates [m], FFT order
      * - ``kx, ky``
        - transverse wavenumbers [rad/m], FFT order
      * - ``t``
        - time [s], FFT order
      * - ``Omega``
        - angular frequency offset :math:`\Omega` [rad/s], FFT order
      * - ``omega``
        - absolute angular frequency :math:`\omega_0 + \Omega`
      * - ``lam``
        - wavelength [m] for each :math:`\Omega` (``inf`` where :math:`\omega \le 0`)
      * - ``valid``
        - boolean mask, :math:`\omega > 0`
      * - ``omega0, lambda0``
        - carrier
      * - ``dtype, rdtype``
        - complex and matching real dtype
      * - ``fft``
        - :class:`FFT` object
      * - ``shape``
        - ``(Nx, Ny, Nt)``
      * - ``X, Y``
        - broadcastable coordinates, shapes ``(Nx, 1)`` and ``(1, Ny)``
      * - ``r2``
        - :math:`x^2 + y^2`, shape ``(Nx, Ny)``
      * - ``dA``
        - :math:`dx\,dy`
      * - ``xs, ys, ts, kxs, lams, Omegas``
        - sorted versions, for plotting

   .. py:method:: energy(Aw)

      Energy [J] of a field in :math:`(x, y, \Omega)`.

   .. py:method:: energy_t(At)

      Energy [J] of a field in :math:`(x, y, t)`.

   .. py:staticmethod:: shift_xy(a)

      ``fftshift`` over the two spatial axes.

   .. py:staticmethod:: shift_w(a, axis=-1)

      ``fftshift`` over the spectral or temporal axis.

   .. py:method:: describe()

      One-line summary: sizes, memory per field, box, time window, wavelength window.

FFT
---

.. py:class:: FFT(backend="auto", workers=None)

   Thin wrapper applying the conventions of :doc:`../overview/conventions`.

   .. py:method:: fft2(a)

      :math:`(x, y) \to (k_x, k_y)`, axes 0 and 1.

   .. py:method:: ifft2(a)

      :math:`(k_x, k_y) \to (x, y)`.

   .. py:method:: ft_t(a)

      :math:`t \to \Omega` (last axis).

   .. py:method:: ift_t(a)

      :math:`\Omega \to t`.

suggest_grid
------------

.. py:function:: suggest_grid(cell, lambda0, lambda_min, lambda_max, pulse_fwhm, *, box_factor=8.0, points_per_waist=8.0, time_window_factor=8.0, **grid_kw)

   Choose a :class:`Grid` adapted to a cell.

   - **Box width:** ``box_factor`` × the largest mirror spot radius (at ``lambda_max``).
   - **Step dx:** the smaller of the smallest waist (at ``lambda_min``) divided by
     ``points_per_waist``, and the Nyquist limit of the mirror curvature phase at the
     corner of the box, :math:`0.8\,\lambda_\text{min} R / (\sqrt 2 L)`.
   - **Time step:** covers [``lambda_min``, ``lambda_max``] around ``lambda0``.
   - **Point counts:** ``Nt`` covers ``time_window_factor`` × the input FWHM; ``Nx`` and
     ``Nt`` are rounded up to fast FFT sizes (:math:`2^a 3^b 5^c`).

   :param cell: :class:`~thunder4d.cell.MPC`
   :param float pulse_fwhm: input duration [s]
   :param grid_kw: passed to :class:`Grid` (e.g. ``dtype``, ``fft_backend``, ``device``)
   :returns: :class:`Grid`

   .. tip::

      Use ``box_factor`` ≥ 12–14 for top-hat, clipped or aberrated inputs: their
      higher-order content is much wider than the fundamental mode.


GPU helpers
-----------

.. py:function:: gpu_available()

   ``True`` if CuPy is installed and a CUDA device can run a kernel and an FFT (checked once).

.. py:function:: to_host(a)

   NumPy array from an array that may live on the GPU (no copy for NumPy input). The field
   ``sim.Aw`` and ``Pulse.Aw`` are device arrays on a GPU grid; the diagnostics accept either and
   always return NumPy arrays (they copy the field to the host once per call).
