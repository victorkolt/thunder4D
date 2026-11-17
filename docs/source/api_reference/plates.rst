thunder4d.plates
================

.. py:module:: thunder4d.plates

Solid plates inside a multipass cell (bulk and hybrid cells), see :doc:`../user_guide/bulk_hybrid`.

.. py:class:: Plate(material, thickness, position=0.0, *, coating="bare", name=None)

   A plate at normal incidence, crossed once per pass.

   :param material: :class:`~thunder4d.materials.Solid`, or any medium with ``n(omega)`` and ``n2``
   :param float thickness: [m], > 0
   :param float position: signed distance [m] from the cell centre to the plate centre, positive
      towards mirror B (the mirror hit at the end of pass 1)
   :param coating: ``"bare"``, ``"ar"`` or a :class:`Coating`, or a pair ``(face A, face B)``; face A
      looks at mirror A
   :param str name: label (default: material and thickness)

   .. py:method:: reduced_length(ambient, lam)

      Length of the filling medium with the same diffraction: :math:`t\,n_g/n_p`.

   .. py:method:: face_R(face, ambient, lam)

      Power reflectivity of face ``"A"`` or ``"B"`` (Fresnel for a bare face).

   .. py:method:: transmission(ambient, lam)

      Power transmission of the plate, both faces, no absorption.

   .. py:method:: describe(ambient, lam)

      One-line summary: material, position, coatings, transmission, reduced length.

.. py:class:: Coating(reflectivity=0.0, *, gdd_fs2=0.0, tod_fs3=0.0, phase=None, name="coating")

   Coating of one face, seen in transmission. ``reflectivity`` is a number or a function
   ``R(lambda_m)``; ``gdd_fs2`` and ``tod_fs3`` and the optional ``phase(lambda_m)`` [rad] are added in
   transmission.

   .. py:classmethod:: bare()

      Uncoated face: Fresnel reflection from :math:`n(\lambda)`.

   .. py:classmethod:: ideal_ar()

      :math:`R = 0` at every wavelength.

   .. py:classmethod:: from_table(lam, R, phase_rad=None, name="table")

      A measured or vendor curve (``lam`` [m], ``R`` in 0 to 1, optional phase [rad]), linearly
      interpolated, constant outside.
