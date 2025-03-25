import subprocess

sims = ['h148.cosmo50PLK.3072g3HbwK1BH', 'h148.cosmo50PLK.3072g3HbwK1BH', 'h329.cosmo50PLK.3072gst5HbwK1BH', 'storm.cosmo25cmb.4096g5HbwK1BH', 'storm.cosmo25cmb.4096g5HbwK1BH', 'storm.cosmo25cmb.4096g5HbwK1BH', 'cptmarvel.cosmo25cmb.4096g5HbwK1BH']
merger_halos = [2, 11, 7, 2, 6, 7, 7]

def run_command(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(f"Error message: {e}")

for sim, halo in zip(sims, merger_halos):
    # Command 1: Write Mvir
    #cmd1 = f"tangos write Mvir --sim {sim} --include-only=\"latest().halo_number()=={halo}\" --include-only=\"t()>10\" --backwards"
    #run_command(cmd1)



    # Command 2: Write reff and ba_s
    #example command 
    cmd2 = f"tangos write reff ba_s --sim {sim} --include-only=\"latest().halo_number()=={halo}\" --include-only=\"t()>10\" --backwards --with-prerequisites"
    run_command(cmd2)

print("All commands executed.")