#!/usr/bin/env python3


import subprocess
import os
import csv

def print_green(text):
    print("\033[92m{}\033[00m" .format(text))

def print_red(text):
    print("\033[91m{}\033[00m" .format(text))

def print_magenta(text):
    print("\033[95m{}\033[00m" .format(text))

def print_message_in_box(message):
    print_red("=" * (len(message) + 4))
    print_red(f"| {message} |")
    print_red("=" * (len(message) + 4))


def compile_with_afl_clang_fast(input_file, output_file, delim=0.001):
        
    # delete old output binary
    if os.path.exists(output_file):
        os.remove(output_file)    

    try:
        # Example command: afl-clang-fast -o output_file input_file.c
        afldir = os.environ.get('AFLDIR')        

        command = [afldir + '/afl-clang-fast', '-o', output_file, input_file]
        subprocess.run(command, check=True)
        print(f"Compilation successful. Output binary: {output_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed. Error: {e}")
        return False

    return True


def get_crash_info():
    with open('crashing_data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        # read each row into a dictionary
        data = {row[0]: row[1] for row in reader}
        print_green(f"Crash info: {data}")
        return data


def get_diff_value(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        diff = None
        checkid = None

        # Create a dictionary to store key-value pairs
        data = {row[0]: row[1] for row in reader}
        print(f"Data: {data}")

        # Check if the 'diff' key exists
        if 'diff' in data:
            diff = float(data['diff'])
        elif 'checkid' in data:
            checkid = int(data['checkid'])

        return diff, checkid
        

def fuzz_check(executable, input_directory, output_directory):
    # run afl-fuzz
    aflbin = os.environ['AFLDIR'] + "/" + "afl-fuzz"
    command = [aflbin, "-i", "indir", "-o", "outdir", "-D", "--", "./target.exe"]
    # set environment variable 
    env = os.environ.copy()
    env["AFL_BENCH_UNTIL_CRASH"] = "1"
    env["AFL_NO_UI"] = "1"
    env["AFL_SKIP_CPUFREQ"] = "1"
    env["AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES"] = "1"
    env["AFL_TRY_AFFINITY"] = "1"

    print("Executing command: ", ' '.join(command))
    fuzzlog = open("fuzzer.log", "w")
    afltout = False
    # execute command for 20 seconds
    try:
        fuzzproc = subprocess.run(command, timeout=45, env=env, stdout=fuzzlog, stderr=subprocess.PIPE)
        print('fuzzer exit code: ', fuzzproc.returncode)
    except subprocess.TimeoutExpired:
        print("Timeout occurred, no crash found")
        afltout = True

    # check if crash found
    if not afltout:
        print_message_in_box("May be a crash! TODO: validate crash!")
        return "crash"
    else:
        print_red("Crash not found")
        return "timeout"



def validate_invariant(candiate_inv, target_template, ib_list):
    
    # Replace 'input_file.c' and 'output_binary' with your actual C source file and desired output binary name
    delim = 0.001
    if compile_with_afl_clang_fast('target.exe', delim):
        print_green("Compilation successful")
        print_green("Starting fuzzing...")
        fuzzstat = fuzz_check('target.exe', 'indir', 'outdir') 

        if fuzzstat == "timeout":
            print_red("Timeout occurred. Exiting...")
            return True, None
        
        delim = 0.0
        cex_list = []
        while fuzzstat == "crash":
            compile_with_afl_clang_fast('target.exe', delim)
            fuzzstat = fuzz_check('target.exe', 'indir', 'outdir')
            if fuzzstat == "crash":
                # from the list of crashing inputs, find the first input that caused a crash
                list_of_crashes = os.listdir('outdir/default/crashes/')
                # remove README.txt from the list
                list_of_crashes.remove('README.txt')
                crash_file = list_of_crashes[0]
                print_green(f"Crashing input: {crash_file}")

                # run target.exe against the crashing input from stdin
                command = ['./target.exe', '<', 'outdir/default/crashes/' + crash_file]
                curr_crash_info = get_crash_info()
                delim = round(float(curr_crash_info['diff']), 0) + 1
                print_magenta(f"New delim: {delim}")
                cex_list.append(curr_crash_info)

                # save crashing input to crashing_inputs directory
                # command = ['cp', 'outdir/default/crashes/' + crash_file, 'indir/']
                # subprocess.run(command, check=True)

            else:
                print_red("Could not find a crash. Exiting...")
                break


    print(f"CEX list: ")
    cex_result = []
    for cex in cex_list:
        # print vars from ib_list
        cex_p = {}
        for ib in ib_list:
            cex_p[ib] = cex[ib]
        
        cex_result.append(cex_p)

    return False, cex_result