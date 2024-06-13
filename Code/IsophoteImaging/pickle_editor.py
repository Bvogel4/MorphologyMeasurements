#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:06:09 2024

@author: blake
"""


import argparse
import pickle
import pprint


parser = argparse.ArgumentParser(description='Edit pickle files')
parser.add_argument('-f', '--feedback', choices=['BW', 'SB', 'RDZ'], default='RDZ', help='Feedback Model')
parser.add_argument('-s', '--simulation', required=True, help='Simulation to analyze')
args = parser.parse_args()

shapefile_path = f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle'
maskfile_path = f'../../Data/{args.simulation}.{args.feedback}.Masking.pickle'

Images = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Images.pickle','rb'))
ShapeData = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.ShapeData.pickle','rb'))
Masking = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Masking.pickle','rb'))
Profiles = pickle.load(open(f'../../Data/{args.simulation}.{args.feedback}.Profiles.pickle','rb'))





#pprint.pprint(Masking)
#pprint.pprint(ShapeData)
#pprint.pprint(Profiles)
pprint.pprint(Images)
