import argparse
import os
import pickle
import sys
import multiprocessing
from SimInfoDicts.sim_type_name import sim_type_name


def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect data from all simulations')
    parser.add_argument('-n', '--numproc', type=int, required=True, help='Number of processors to use')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print halo IDs being analyzed')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing images')
    return parser.parse_args()


def run_command(command):
    os.system(command)


def process_simulation(simulation, feedback, args):
    commands = [
        f"{sys.executable} IsophoteImaging/ImageCollection.py -f {feedback} -s {simulation} -i -n {args.numproc} {'-v' if args.verbose else ''} {'-o' if args.overwrite else ''}",
        f"{sys.executable} IsophoteImaging/ImageCollection.py -f {feedback} -s {simulation} -n {args.numproc} {'-v' if args.verbose else ''} {'-o' if args.overwrite else ''}",
        f"{sys.executable} IntrinsicShapes/3DShapeCollection.Stars.py -f {feedback} -s {simulation} -n {args.numproc} {'-v' if args.verbose else ''} {'-o' if args.overwrite else ''}",
        f"{sys.executable} IntrinsicShapes/3DShapeCollection.Dark.py -f {feedback} -s {simulation} -n {args.numproc} {'-v' if args.verbose else ''} {'-o' if args.overwrite else ''}",
        f"{sys.executable} IntrinsicShapes/3DShapeSmoothing.py -f {feedback} -s {simulation} -n {args.numproc} {'-v' if args.verbose else ''} {'-o' if args.overwrite else ''}"
    ]

    for command in commands:
        run_command(command)


def main():
    args = parse_arguments()

    for feedback, use_sim in sim_type_name.items():
        if use_sim:
            pickle_path = f'PickleFiles/SimulationInfo.{feedback}.pickle'
            print(f'Processing {feedback} feedback type')
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    sims = pickle.load(f)
                with multiprocessing.Pool(args.numproc) as p:
                    p.starmap(process_simulation, [(s, feedback, args) for s in sims])
            else:
                print(f"No pickle file found for {feedback} feedback type.")


if __name__ == "__main__":
    main()