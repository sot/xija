# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function

import sys
import os
import argparse

import xija

def main():
    parser = argparse.ArgumentParser(description='Convert Xija JSON model spec'
                                     ' to Python')
    parser.add_argument('model_spec',
                        type=str,
                        help='Input Xija model spec file name')
    parser.add_argument('--force',
                        action='store_true',
                        help='Overwrite existing outfile')
    parser.add_argument('--outfile',
                        type=str,
                        help='Output Python file (default=<model_spec>.py)')
    args = parser.parse_args()

    infile = args.model_spec
    outfile = args.outfile

    if outfile is None:
        if infile.endswith('.json'):
            outfile = infile[:-5] + '.py'
        else:
            outfile = infile + '.py'

    if os.path.exists(outfile) and not args.force:
        print('Error: {} exists.  Use --force to overwrite.'.format(outfile))
        sys.exit(1)

    model = xija.XijaModel(model_spec=infile)
    model.write(outfile)
    print('Wrote', outfile)

if __name__ == '__main__':
    main()
