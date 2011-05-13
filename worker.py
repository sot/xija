#!/usr/bin/env python
"""
Worker
"""

print 'xija_worker: starting'
import sys
import os
import signal
import json

import numpy as np
from Chandra.Time import DateTime
from mpi4py import MPI
import pyyaks.context

import xija

src = pyyaks.context.ContextDict('src')
files = pyyaks.context.ContextDict('files', basedir=os.getcwd())
files.update(xija.files)

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
procname = MPI.Get_processor_name()

def timeout_handler(signum, frame):
    raise RuntimeError('Timeout')

signal.signal(signal.SIGALRM, timeout_handler)

while True:
    signal.alarm(60)
    msg = comm.bcast(None, root=0)
    cmd = msg['cmd']
    # print 'xija_worker {1} of {2}: got cmd "{0}"'.format(cmd, rank, size)

    if cmd == 'stop':
        semaphore = np.array(0, 'd')
        comm.Reduce([semaphore, MPI.DOUBLE], None, op=MPI.SUM, root=0)
        break

    elif cmd == 'init':
        fit_start = DateTime(msg['tstart']).secs
        fit_stop = DateTime(msg['tstop']).secs
        dt = (fit_stop - fit_start) / size
        tstart = fit_start + dt * rank
        tstop = tstart + dt
        
        print 'xija_worker {4} of {5}: Working init {0}:{1} fetching data {2} {3}'.format(
            rank, procname, DateTime(tstart).date, DateTime(tstop).date, rank, size)

        src.update((x, msg[x]) for x in ('model', 'outdir', 'pardir'))
        model_spec = json.load(open(files['model_spec.json'].abs, 'r'))
        model = xija.ThermalModel(name=msg['model'], start=tstart, stop=tstop, model_spec=model_spec)

    elif cmd == 'calc_model':
        model.parvals[:] = msg['parvals']

    elif cmd == 'calc_stat':
        fit_stat = model.calc_stat()
        comm.Reduce([fit_stat, MPI.DOUBLE], None, op=MPI.SUM, root=0)

    elif cmd == 'model':
        model_func = getattr(model, msg['func'])
        args = msg.get('args', [])
        kwargs = msg.get('kwargs', {})
        model_func(*args, **kwargs)

print 'Exiting process {0} on host {1}'.format(rank, procname)
comm.Disconnect()

signal.alarm(0)
