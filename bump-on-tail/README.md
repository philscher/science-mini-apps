Bump-On-Tail Instability
------------------------

Note: *Not yet validated*

Implementation of a 1D-1V Vlasov-Poisson solver to model the two-stream plasma instability as discussed by
_M. M. Shoucri: Nonlinear evolution of the bump-on-tail instability (1979)_.

The 1D-1V Vlasov-Poisson equation is an integro-differential equations which predicts the evolution of
the electron phase space under the influence of an electrical potential assuming a fixed ion background.


The code is implemented using the kokkos framework (https://github.com/kokkos/kokkos) to run on either
host or devices.


License
-------

GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html)

Copyright (C) 2020 Paul Hilscher

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
