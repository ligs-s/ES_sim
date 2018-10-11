#!/usr/bin/env python

"""
    source types: area_iso, area_norm, point_iso
    source positions: 
    sphere options: 0, without; 1, with
"""

import os, sys

source = 'source_area'
sphere = 'wi_sphere'

filename = source + '_' + sphere + '.root'

# argument for ES_sim.py
# [1]: source type
# [2]: source position
# [3]: sphere options

outdir = './data/teflon_spec_0.05_diff_0.65'


#os.system('python ES_sim.py area_iso -95 1 1000000 '  + outdir)
#os.system('python ES_sim.py area_iso -95 0 10000000 ' + outdir)
#
#os.system('python ES_sim.py area_iso -116 1 1000000 ' + outdir)
#os.system('python ES_sim.py area_iso -116 0 2000000 ' + outdir)

os.system('python ES_sim.py point_iso -116 1 1000000 ' + outdir)
os.system('python ES_sim.py point_iso -116 0 2000000 ' + outdir)

#os.system('python ES_sim.py area_norm -100 1 50000')
#os.system('python ES_sim.py area_norm -100 0 50000')

#os.system('python ES_sim.py area_norm -100 1')
#os.system('python ES_sim.py point_iso -100 1')
#os.system('python ES_sim.py area_iso -50 1')
#os.system('python ES_sim.py area_norm -50 1')
#os.system('python ES_sim.py point_iso -50 1')
