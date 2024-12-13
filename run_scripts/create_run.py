#!/usr/bin/env python3
import datetime as dt
import json
import os
import subprocess as sp
import sys
from pathlib import Path

COMPONENT_NAMES = ["ATM", "CPL", "OCN", "WAV", "GLC", "ICE", "ROF", "LND", "ESP", "IAC"]
"ATM" "CPL" "OCN" "WAV" "GLC" "ICE" "ROF" "LND" "ESP" "IAC"
# INIT_COND_FILE_TEMPLATE = "20210915.v2.ne4_oQU240.F2010.{}.{}.0003-{:02d}-01-00000.nc"
# 20231002.F2010.ne4_oQU240_init.elm.h0.0002-07.nc
INIT_COND_FILE_TEMPLATE = "20231002.F2010.ne4_oQU240_init.{}.{}.0003-{:02d}-01-00000.nc"


def set_tasks(ninst, case_dir):
    os.chdir(case_dir)
    for comp in COMPONENT_NAMES:
        _out = sp.run(["./xmlquery", f"NTASKS_{comp}"], check=True, stdout=sp.PIPE)
        ntasks = int(_out.stdout.strip().split()[1])
        if ntasks == 1:
            os.system(f"./xmlchange NTASKS_{comp}={ntasks * ninst}")
        else:
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
    total_sim = 14  # stop_options
    n_resub = 0

    if n_resub != 0:
        sim_length = total_sim // n_resub
    else:
        sim_length = total_sim

    # Load a shorted list of output variables
    output_vars = json.load(open("new_vars.json", "r"))
    # Surround variable names in single quotes (e.g. T -> 'T')
    output_vars = [f"'{_var}'" for _var in output_vars["default"]]

    ninst = 120
    compset = "F2010"
    grid = "ne4_oQU240"
    mach = "chrysalis"
    # mach = "anvil"
    nhtfrq = None
    compiler = "intel"
    today = dt.datetime.now().strftime("%Y%m%d")
    # branch = "maint-2.0"
    branch = "master"
    plim = 1e-10

    # zmconv_c0 = 0.0022
    # zmconv_str = f"{zmconv_c0:.05f}".replace('.', 'p')
    # clubb_c1 = 2.424
    # clubb_c1_str = f"{clubb_c1:.04f}".replace('.', 'p')
    # param_name = "effgw_oro"
    param_name = "ctl"
    param_val = 0.37875
    # param_str = f"{param_val:.04f}".replace(".", "p")
    # case = f"{today}.{compset}.{grid}.dtcl_{param_name}_{param_str}_n{ninst:04d}"
    case = f"{today}.{compset}.{grid}.dtcl_{param_name}_{branch}_n{ninst:04d}"
    # case = f"{today}.{compset}.{grid}.dtcl_zmconv_c0_{zmconv_str}_n{ninst:04d}"
    # case = f"{today}.{compset}.{grid}.dtcl_pertlim_{plim}_n{ninst:04d}"
    print(f"{'#' * 40}\nCREATING CASE:\n{case}\n{'#' * 40}")

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
        "--walltime 03:00:00",
        f"--case {case}",
        f"--machine {mach}",
        f"--ninst {ninst}",
        f"--compiler {compiler}",
    ]

    print(f"{'*'*20}CREATING CASE{'*'*20}")
    print(" ".join(create_script))
    os.system(" ".join(create_script))
    os.chdir(case_dir)

    # data_root = Path("/lcrc/group/e3sm/data/inputdata")
    # csmdata_atm = Path(data_root, "atm/cam/inic/homme/ne4_v2_init")
    # csmdata_lnd = Path(data_root, "lnd/clm2/initdata/ne4_oQU240_v2_init")
    data_root = Path(
        "/lcrc/group/e3sm/ac.mkelleher/scratch/chrys/20231002.F2010.ne4_oQU240_init"
    )
    csmdata_atm = Path(data_root, "run")
    csmdata_lnd = Path(data_root, "run")

    # Generate user namelists to modify parameters for each ensemble member
    for iinst in range(1, ninst + 1):
        with open(
            "user_nl_{}_{:04d}".format(model_component, iinst), "w"
        ) as nl_atm_file, open(
            "user_nl_{}_{:04d}".format("elm", iinst), "w"
        ) as nl_lnd_file:

            fatm_in = Path(
                csmdata_atm,
                INIT_COND_FILE_TEMPLATE.format("eam", "i", 1),
            )
            flnd_in = Path(
                csmdata_lnd,
                INIT_COND_FILE_TEMPLATE.format("elm", "r", 1),
            )

            nl_atm_file.write("ncdata  = '{}' \n".format(fatm_in))
            nl_lnd_file.write("finidat = '{}' \n".format(flnd_in))

            nl_atm_file.write("new_random = .true.\n")
            nl_atm_file.write(f"pertlim = {plim}\n")
            nl_atm_file.write("seed_custom = {}\n".format(iinst))
            nl_atm_file.write("seed_clock = .false.\n")

            if nhtfrq is not None:
                nl_atm_file.write(f"nhtfrq = {nhtfrq}\n")
                nl_atm_file.write("avgflag_pertape = 'I'\n")
                nl_atm_file.write("mfilt = 400\n")

            # nl_atm_file.write(f"fincl1 = {', '.join(output_vars)}\n")
            # nl_atm_file.write("empty_htapes = .true.\n")
            # if param_name != "ctl":
            if "ctl" not in param_name:
                nl_atm_file.write(f"{param_name} = {param_val}\n")
            # nl_atm_file.write(f"clubb_c1 = {clubb_c1}\n")
            # nl_atm_file.write(f"zmconv_c0_ocn = {zmconv_c0}\n")
            # nl_atm_file.write(f"zmconv_c0_lnd = {zmconv_c0}")

    print(f"{'*' * 20} PREVIEW {'*' * 20}")
    os.system("./preview_run")
    print("*" * 49)

    set_tasks(ninst, case_dir)

    os.system("./case.setup")

    if n_resub > 0:
        os.system(f"./xmlchange RESUBMIT={n_resub}")

    os.system(f"./xmlchange STOP_N={sim_length}")
    os.system(f"./xmlchange STOP_OPTION={stop_option}")
    os.system("./xmlchange REST_N=7")
    os.system("./xmlchange REST_OPTION=nmonths")
    os.system("./xmlquery GMAKE_J")
    os.system("./xmlchange GMAKE_J=64")
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
