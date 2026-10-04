# Third-party notices

## XTC/XDR reader (`src/gmxtraj`)

`xdrfile.cpp`, `xdrfile_xtc.cpp`, `xdrfile_trr.cpp` and the headers in `src/gmxtraj/include` come from the standalone
GROMACS xdrfile library, Copyright (c) 2009-2014, Erik Lindahl & David van der Spoel, under the 2-clause BSD license
reproduced at the top of each file.

TITAN's updates to that reader, which add XTC magic 2023, 64-bit compressed payload lengths and the float-coordinate
decoder, were derived from the GROMACS 2026.3 sources `src/gromacs/fileio/xdrf.h`, `src/gromacs/fileio/libxdrf.cpp`
and `src/gromacs/fileio/xtcio.cpp`. Those GROMACS files are licensed under the GNU Lesser General Public License,
version 2.1 or later, by the GROMACS authors (https://www.gromacs.org). The modified reader is distributed here as
source, and the same terms apply to it. See also `src/gmxtraj/TITAN_XDR_NOTICE.license`.

## Eigen (`src/eigen`)

The Eigen C++ template library for linear algebra (https://eigen.tuxfamily.org) is included as headers. Eigen is
licensed primarily under the Mozilla Public License, version 2.0 (https://mozilla.org/MPL/2.0/), as stated at the top
of each file. A few files carry other permissive terms stated in them: `Eigen/src/Core/arch/Default/BFloat16.h`
(Apache License 2.0) and part of `Eigen/src/Geometry/AlignedBox.h` (BSD license, Willow Garage).

## GBn2 tables (`src/titan_gbn2_tables.h`)

The GBn2 neck lookup data are taken from OpenMM (`customgbforces.py`), portions copyright (c) 2012-2022 University of
Virginia and the Authors, under the MIT-style license reproduced at the top of the file.

The MIT license in `LICENSE` covers the rest of this repository.
