Conventions and units
=====================

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Quantity
     - Convention
   * - Field
     - complex envelope, :math:`|A|^2` = intensity [W/m²]
   * - Temporal transform
     - :math:`\tilde A(\Omega) = \sum_t A(t)\,e^{+i\Omega t}` (``grid.fft.ft_t``);
       :math:`A(t) = \frac{1}{N_t}\sum_\Omega \tilde A\,e^{-i\Omega t}` (``grid.fft.ift_t``)
   * - Spatial transform
     - standard ``fft2``; plane waves :math:`e^{+ik_x x}`, propagation :math:`e^{+ik_z z}`
   * - Spectral phase
     - :math:`\varphi = +\frac{\text{GDD}}{2}\Omega^2`: **positive GDD is an up-chirp**;
       :math:`e^{+i\Omega t_0}` delays the pulse by :math:`t_0`
   * - Field array
     - shape ``(Nx, Ny, Nt)``, home domain :math:`(x, y, \Omega)`
   * - Array ordering
     - **FFT order** everywhere: index 0 is :math:`x = 0`, :math:`t = 0`, :math:`\Omega = 0`
   * - Units
     - SI (m, s, J, W) unless the argument name says otherwise:
       ``_fs``, ``_fs2``, ``_nm``, ``_mm``, ``_urad``, ``_bar``
   * - Energy
     - ``grid.energy(Aw)`` in J (Parseval with :math:`dx\,dy\,dt/N_t`)

FFT ordering
------------

Arrays are kept in FFT order because every operation of the solver is local: ordering
doesn't matter there, and no shifts are needed at each step. For plotting, use the
sorted axes ``grid.xs``, ``grid.ys``, ``grid.ts``, ``grid.lams``, ``grid.Omegas``,
``grid.kxs`` and the helpers ``grid.shift_xy(a)`` and ``grid.shift_w(a)``.

.. note::

   Every function of :mod:`thunder4d.diagnostics` returns **sorted** axes, and
   wavelength-resolved densities are per unit wavelength unless ``per_lambda=False``.
