#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Run the gui_fit tool
"""

import pygtk
pygtk.require('2.0')
import gtk
import gobject

if __name__ == '__main__':
    from xija import gui_fit
    gui_fit.main()
