thunder4d.materials
===================

.. py:module:: thunder4d.materials

Gas filling the cell.

.. py:class:: Material(name, pressure_bar=1.0, temperature_K=293.15, *, n2_1bar=None, sellmeier=None, n_func=None, n2=None)

   :param str name: ``"He"``, ``"Ne"``, ``"Ar"``, ``"Kr"``, ``"Xe"``, ``"N2"``, ``"air"``,
      or any label if ``n_func`` is given
   :param float pressure_bar: pressure [bar]
   :param float temperature_K: temperature [K]; :math:`n^2-1` scales as :math:`p/T` and
      :math:`n_2` as :math:`p \cdot 293.15/T`
   :param float n2_1bar: override the tabulated :math:`n_2` at 1 bar [m²/W]
   :param tuple sellmeier: override the coefficients ``(B1, C1, B2, C2)`` (λ in µm)
   :param callable n_func: ``n(omega)`` replacing the Sellmeier formula entirely
   :param float n2: absolute :math:`n_2` [m²/W] at the given conditions (no scaling)

   Dispersion (Börzsönyi *et al.*, Appl. Opt. **47**, 4856, 2008):

   .. math::

      n^2 - 1 = \frac{p}{p_0}\frac{T_0}{T}\left[\frac{B_1\lambda^2}{\lambda^2 - C_1}
      + \frac{B_2\lambda^2}{\lambda^2 - C_2}\right],\qquad p_0 = 1\ \text{bar},\ T_0 = 273.15\ \text{K}.

   .. warning::

      The tabulated :math:`n_2` values (``N2_1BAR``) are approximate literature values
      (electronic Kerr only, no Raman). Replace them with your own.

   .. py:method:: n(omega)

      Refractive index (1 for :math:`\omega \le 0`).

   .. py:method:: k(omega)

      Wavenumber :math:`n\omega/c` [rad/m].

   .. py:method:: gvd(lam)

      Group-velocity dispersion [fs²/m] at wavelength ``lam``.

   .. py:method:: critical_power(lam)

      :math:`P_c = \lambda^2/(2\pi n_0 n_2)` [W] (Gaussian beam, aberration-free).

   .. py:attribute:: n2

      :math:`n_2` at the current pressure and temperature [m²/W].

.. py:data:: SELLMEIER

   Dictionary of Sellmeier coefficients ``(B1, C1, B2, C2)`` per gas.

.. py:data:: N2_1BAR

   Dictionary of :math:`n_2` at 1 bar [m²/W] per gas (approximate).
