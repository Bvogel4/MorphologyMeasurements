import argparse,os,pickle,sys
import traceback
parser = argparse.ArgumentParser(description='Fit Isophotes to all generated images.')
parser.add_argument('-n','--numproc',type=int,required=True,help='Number of processors to use')
parser.add_argument('-o','--overwrite',action='store_true',help='Overwrite existing data')
args = parser.parse_args()

loop = True
while loop:
    method = input('Fit Isophotes, Project 3D Ellipsoids, or Run MCMC (I/3/M): ')
    if method in ['I','3','M']:
        loop = False
        if method=='I':
            subdir = 'IsophoteImaging'
            script = 'IsophoteFitting.py'
        elif method=='3':
            subdir = 'IntrinsicShapes'
            script = '2DShapeProjection.py -p'
        else:
            subdir = 'MCMC'
            script = 'MCMC.py'

over = '-o' if args.overwrite else ''
feedbacks = ['RDZ'] #['BW','SB','RDZ']
print('feedbacks',feedbacks)
for feedback in feedbacks:
    try:
        sims = pickle.load(open(f'SimulationInfo.{feedback}.pickle','rb'))
        os.chdir(subdir)
        for s in sims:
            os.system(f"{sys.executable} {script} -f {feedback} -s {s} -n {args.numproc} {over}")
            #os.system(f"/usr/local/anaconda/bin/python {script} -f {feedback} -s {s} -n {args.numproc}")
    except FileNotFoundError:
        print(traceback.format_exc())
        print(f'No file found for feedback type {feedback}')
    except Exception as e:
        print(f"An error occurred: {e}")
