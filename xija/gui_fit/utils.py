# must import these first, to set up Qt properly
from PyQt5 import QtWidgets

from IPython.core.interactiveshell import InteractiveShell
from IPython.core.completer import IPCompleter

from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget

import os
import pkg_resources

# We need to do the following to make sure that the outputs in the IPython
# terminal don't get cached. This is because if a user does e.g.
#
# In  [1]: viewer
# Out [1]: <HistogramViewer...>
#
# then there will be a remaining reference to the viewer in the IPython
# namespace.
InteractiveShell.cache_size.default_value = 0

# Make sure that tab competion only shows items returned by
# _ipython_key_completions_
if hasattr(IPCompleter, 'dict_keys_only'):
    IPCompleter.dict_keys_only.default_value = True

kernel_manager = None
kernel_client = None


def start_in_process_kernel():

    global kernel_manager, kernel_client

    kernel_manager = QtInProcessKernelManager()
    kernel_manager.start_kernel()

    kernel_client = kernel_manager.client()
    kernel_client.start_channels()


def in_process_console(console_class=RichJupyterWidget, **kwargs):
    """
    Create a console widget, connected to an in-process Kernel

    Keyword arguments will be added to the namespace of the shell.

    Parameters
    ----------
    console_class : `type`
        The class of the console widget to create
    """

    global kernel_manager, kernel_client

    if kernel_manager is None:
        start_in_process_kernel()

    def stop():
        kernel_client.stop_channels()
        kernel_manager.shutdown_kernel()

    control = console_class()
    control._display_banner = False
    control.kernel_manager = kernel_manager
    control.kernel_client = kernel_client
    control.exit_requested.connect(stop)
    control.shell = kernel_manager.kernel.shell
    control.shell.user_ns.update(**kwargs)
    control.setWindowTitle('xija_gui_fit IPython Terminal')

    return control


def icon_path(icon_name, icon_format='png'):
    """
    Return the absolute path to an icon

    Parameters
    ----------
    icon_name : str
       Name of icon, without extension or directory prefix
    icon_format : str, optional
        Can be either 'png' or 'svg'

    Returns
    -------
    path : str
      Full path to icon
    """

    icon_name += '.{0}'.format(icon_format)

    try:
        if pkg_resources.resource_exists('xija.gui_fit', icon_name):
            return pkg_resources.resource_filename('xija.gui_fit', icon_name)
        else:
            raise RuntimeError("Icon does not exist: %s" % icon_name)
    except NotImplementedError:  # workaround for mac app
        result = os.path.dirname(__file__)
        return os.path.join(result.replace('site-packages.zip', 'xija'),
                            icon_name)
