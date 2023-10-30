import os,pickle

gitdir = os.getcwd().rstrip('Code')
loop,erase = True,False
while loop:
    query = input('Delete .tar.gz files after extracting? (y/n): ')
    if query in ['y','Y']:
        erase = True
        loop = False
    elif query in ['n','N']:
        erase = False
        loop = False

for feedback in ['BW','SB']:
    os.chdir(f'{gitdir}Code')
    SimInfo = pickle.load(open(f'SimulationInfo.{feedback}.pickle','rb'))
    for sim in SimInfo:
        for halo in SimInfo[sim]['halos']:
            os.chdir(f'{gitdir}Figures/Images/{sim}.{feedback}/{halo}/')
            os.system(f'tar -xzf {halo}.Images.tar.gz')
            if erase: os.system(f'rm {halo}.Images.tar.gz')
            os.system(f'tar -xzf {halo}.Isophotes.tar.gz')
            if erase: os.system(f'rm {halo}.Isophotes.tar.gz')
            os.system(f'tar -xzf {halo}.Intrinsic.tar.gz')
            if erase: os.system(f'rm {halo}.Intrinsic.tar.gz')
os.chdir(f'{gitdir}Figures/EllipticityComparison/')
os.system(f'tar -xzf EllipticityComparison.tar.gz Individual/')
if erase: os.system(f'rm EllipticityComparison.tar.gz')
print('Done')