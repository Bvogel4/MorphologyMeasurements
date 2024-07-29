#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:05:07 2024

@author: blake
"""

#MaskALL


import argparse,os,pickle,sys


script = 'IsophoteMaskingNew.py'

for feedback in ['BW','SB','RDZ']:
    try:
        sims = pickle.load(open(f'../SimulationInfo.{feedback}.pickle','rb'))
        #os.chdir(subdir)
        for s in sims:
            #check for manual flag for already done sims
            
            os.system(f"{sys.executable} {script} -f {feedback} -s {s}")
            #os.system(f"/usr/local/anaconda/bin/python {script} -f {feedback} -s {s} -n {args.numproc}")
    except FileNotFoundError:
        print(f'No file found for feedback type {feedback}')
    except Exception as e:
        print(f"An error occurred: {e}")
