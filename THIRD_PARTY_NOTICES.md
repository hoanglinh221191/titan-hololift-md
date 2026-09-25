# Third-party notices

## XTC/XDR reader (`src/gmxtraj`)

`xdrfile.cpp`, `xdrfile_xtc.cpp` and the headers in `src/gmxtraj/include` come from the standalone GROMACS xdrfile
library, Copyright (c) 2009-2014, Erik Lindahl & David van der Spoel, under the 2-clause BSD license reproduced at the
top of each file.

TITAN's updates to that reader, which add XTC magic 2023, 64-bit compressed payload lengths and the float-coordinate
decoder, were derived from the GROMACS 2026.3 sources `src/gromacs/fileio/xdrf.h`, `src/gromacs/fileio/libxdrf.cpp`
and `src/gromacs/fileio/xtcio.cpp`. Those GROMACS files are licensed under the GNU Lesser General Public License,
version 2.1 or later, by the GROMACS authors (https://www.gromacs.org). The modified reader is distributed here as
source, and the same terms apply to it. See also `src/gmxtraj/TITAN_XDR_NOTICE.license`.

The MIT license in `LICENSE` covers the rest of this repository.
