#!/usr/bin/env python
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
