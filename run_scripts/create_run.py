#!/usr/bin/env python3
import os
import subprocess as sp
from pathlib import Path
import datetime as dt
import sys
import json

COMPONENT_NAMES = ["ATM", "CPL", "OCN", "WAV", "GLC", "ICE", "ROF", "LND", "ESP", "IAC"]

def set_tasks(ninst, case_dir):
    os.chdir(case_dir)
    for comp in COMPONENT_NAMES:
        _out = sp.run(["./xmlquery", f"NTASKS_{comp}"], check=True, stdout=sp.PIPE)
        ntasks = int(_out.stdout.strip().split()[1])
        os.system(f"./xmlchange NTASKS_{comp}={ntasks * ninst / 3}")
        if comp != "CPL":
            os.system(f"./xmlchange NINST_{comp}={ninst}")


def compute_run_wc(nmonths, spmd=8, wiggle=1.1):
    run_days = nmonths * 30
    total_time = dt.timedelta(seconds=(run_days * spmd) * wiggle)
    days = total_time.days
    hrs, rem = divmod(total_time.seconds, 3600)
    mins, secs = divmod(rem, 60)

    if secs > 0:
        mins += 1
    if days > 0:
        wall_clock_str = f"{days:02d}-{hrs:02d}:{mins:02d}:00"
    else:
        wall_clock_str = f"{hrs:02d}:{mins:02d}:00"

    return wall_clock_str


def main(build_case=False, run_case=False):

    model_component = "eam"
    stop_option = "nmonths"
    total_sim = 1 # stop_options
    n_resub = 0

    if n_resub != 0:
        sim_length = total_sim // n_resub
    else:
        sim_length = total_sim

    # Load a shorted list of output variables
    output_vars = json.load(open("new_vars.json", "r"))
    # Surround variable names in single quotes (e.g. T -> 'T')
    output_vars = [f"'{_var}'" for _var in output_vars["default"]]

    ninst = 50

    compset = "F2010"
    grid = "ne4_oQU240"
    mach = "chrysalis"
    nhtfrq = 1
    compiler="intel"
    today = dt.datetime.now().strftime("%Y%m%d")
    branch = "maint-2.0"

    zmconv_c0 = 0.0020
    zmconv_str = f"{zmconv_c0:.04f}".replace('.', 'p')

    case = f"{today}.{compset}.{grid}.dtcl_zmconv_c0_{zmconv_str}_n{ninst:04d}"

    case_dir = Path(os.environ["HOME"], "e3sm_scripts", case)
    case_dir = Path(os.getcwd(), case)
    cime_scripts_dir = Path(
        os.environ["HOME"], "baseline", branch, "E3SM", "cime", "scripts"
    )
    # wall_clock_request = compute_run_wc(total_sim)

    create_script = [
        str(Path(cime_scripts_dir, "create_newcase")),
        f"--compset {compset}",
        f"--res {grid}",
        f"--walltime 01:00:00",
        f"--case {case}",
        f"--machine {mach}",
        f"--ninst {ninst}",
        f"--compiler {compiler}",
    ]

    print(f"{'*'*20}CREATING CASE{'*'*20}")
    print(" ".join(create_script))
    os.system(" ".join(create_script))
    os.chdir(case_dir)

    # Generate user namelists to modify parameters for each ensemble member
    for iinst in range(1, ninst + 1):
        with open(
            "user_nl_{}_{:04d}".format(model_component, iinst), "w"
        ) as nl_atm_file:
            nl_atm_file.write("new_random = .true.\n")
            nl_atm_file.write("pertlim = 1.0e-14\n")
            nl_atm_file.write("seed_custom = {}\n".format(iinst))
            nl_atm_file.write("seed_clock = .true.\n")
            nl_atm_file.write(f"nhtfrq = {nhtfrq}\n")
            nl_atm_file.write("avgflag_pertape = 'I'\n")
            nl_atm_file.write("mfilt = 400\n")
            nl_atm_file.write(f"fincl1 = {', '.join(output_vars)}\n")
            nl_atm_file.write("empty_htapes = .true.\n")
            nl_atm_file.write(f"zmconv_c0_ocn = {zmconv_c0}\n")
            nl_atm_file.write(f"zmconv_c0_lnd = {zmconv_c0}")

    print(f"{'*' * 20} PREVIEW {'*' * 20}")
    os.system("./preview_run")
    print("*" * 49)

    set_tasks(ninst, case_dir)

    os.system("./case.setup")

    if n_resub > 0:
        os.system(f"./xmlchange RESUBMIT={n_resub}")

    os.system(f"./xmlchange STOP_N={sim_length}")
    os.system(f"./xmlchange STOP_OPTION={stop_option}")
    os.system(f"./xmlchange REST_N=1")
    os.system(f"./xmlchange REST_OPTION=nmonths")
    os.system("./preview_namelists")

    try:
        sp.run("./check_input_data", check=True)
    except sp.CalledProcessError:
        print("MISSING INPUT DATA")
        sys.exit(1)

    if build_case:
        sp.run("./case.build", check=True)

    if build_case and run_case:
        sp.run("./case.submit", check=True)


if __name__ == "__main__":
    main(True, True)