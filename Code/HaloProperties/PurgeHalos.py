import argparse,os,pickle

parser = argparse.ArgumentParser(description='Collect images of all resolved halos from a given simulation. Images will be generated across all orientations.')
parser.add_argument('-f','--feedback',choices=['BW','SB'],default='BW',help='Feedback Model')
parser.add_argument('-s','--simulation',required=True,help='Simulation to clean')
args = parser.parse_args()

SimInfo = pickle.load(open(f'SimulationInfo.{args.feedback}.pickle','rb'))
datadir = os.chdir('../Data')
for f in os.listdir():
    if f.split('.')[0]==args.simulation:
        if f.split('.')[1]==args.feedback:
            s = pickle.load(open(f'{f}','rb'))
            for halo in s:
                if int(halo) not in SimInfo[args.simulation]['goodhalos']: del s[halo]
            pickle.dump(s,open(f'{f}','wb'))