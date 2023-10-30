import argparse,os,pickle,sys

parser = argparse.ArgumentParser(description='Collect data from all simulations')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-v','--verbose',action='store_true',help='Print halo IDs being analyzed')
parser.add_argument('-o','--overwrite',action='store_true',help='Overwrite existing images')
args = parser.parse_args()

overwrite = '-o' if args.overwrite else ''
verbose = '-v' if args.verbose else ''

loop = True
while loop:
    type = input('Collect Images, Shapes, Gala, or Mdyn (I/S/G/M): ')
    if type in ['I','S','G','M']:
        loop = False 
if type=='I':
    subdir = 'IsophoteImaging'
    loop = True
    while loop:
        im = input('Generate images in addition to Profiles? (y/n): ')
        if im in ['y','n']: loop = False
    gen_im = '-i' if im=='y' else ''
elif type=='S':
    subdir = 'IntrinsicShapes'
    loop = True
    while loop:
        im = input('Stellar Shapes or DM Shapes (S/D): ')
        if im in ['S','D']: loop = False
    stype = 'Stars' if im=='S' else 'Dark'
elif type=='G':
    subdir = 'Gala'
    loop = True
    while loop:
        im = input('Plot Density Profiles? (y/n): ')
        if im in ['y','n']: loop = False
    gen_im = '-i' if im=='y' else ''
elif type=='M':
    subdir='XuCorrelation'

for feedback in ['BW','SB']:
    sims = pickle.load(open(f'SimulationInfo.{feedback}.pickle','rb'))
    os.chdir(subdir)
    for s in sims:
        if type=='I':
            os.system(f"{sys.executable} ImageCollection.py -f {feedback} -s {s} {gen_im} -n {args.numproc} {verbose} {overwrite}")
        elif type=='S':
            os.system(f"{sys.executable} 3DShapeCollection.{stype}.py -f {feedback} -s {s} -n {args.numproc} {verbose}")
        elif type=='G':
            os.system(f"{sys.executable} GalaCollector.py -f {feedback} -s {s} -n {args.numproc} {gen_im} {verbose}")
        elif type=='M':
            os.system(f"{sys.executable} DynamicalMass.py -f {feedback} -s {s} -n {args.numproc}")
