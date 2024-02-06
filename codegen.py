import json
import ast

import os
import csv
import subprocess


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

class Codegen:

    def __init__(self, cexpr, json_config, cfile='target.c'):
        self.candidate = cexpr
        print_green(f'Candidate expression: {self.candidate}')
        self.json_config = json_config
        self.cfile = cfile
        self.target_binary = 'target.exe'

    def clang_format(self):
            os.system(f'clang-format -i {self.cfile}')


    def gen_c_code(self, delim=0.001):
        # Read the JSON file

        # Extract the input variable from the JSON data
        # convert string to dictionary        
        # input_variable = ast.literal_eval(self.json_config['cgen']['vars'])

        # open file target.c.in
        target_c_in = self.json_config['cgen']['template']
        print_green(f"Target C template: {target_c_in}")

        # delete old target.c file
        if os.path.exists(self.cfile):
            os.remove(self.cfile)

        
        with open(target_c_in) as file:
            c_code = file.read()
            c_code = c_code.replace('@EXIST_EXPRESSION@', self.candidate)
            c_code = c_code.replace('@DELIM_VALUE@', str(delim))
            with open(self.cfile, 'w') as file:
                file.write(c_code)



    def compile_with_afl_clang_fast(self):
        # delete old output binary
        if os.path.exists(self.target_binary):
            os.remove(self.target_binary)    

        try:
            # Example command: afl-clang-fast -o output_file input_file.c
            afldir = os.environ.get('AFLDIR')        

            command = [afldir + '/afl-clang-fast', '-o', self.target_binary, self.cfile, '-lm']
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Compilation successful. Output binary: {self.target_binary}")
            
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed. Error: {e}")
            return False

        return True


    def get_crash_info(self):
        with open('crashing_data.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            # read each row into a dictionary
            data = {row[0]: row[1] for row in reader}
            print_green(f"Crash info: {data}")
            return data


    def get_diff_value(self, file_path):
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
            

    def fuzz_check(self, executable, input_directory, output_directory):
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
            fuzzproc = subprocess.run(command, timeout=60, env=env, stdout=fuzzlog, stderr=subprocess.PIPE)
            print('fuzzer exit code: ', fuzzproc.returncode)
        except subprocess.TimeoutExpired:
            print("Timeout occurred, no crash found")
            afltout = True

        # check if crash found
        if not afltout:
            print_red("Found a crash!")
            return "crash"
        else:
            print_red("Crash not found")
            return "timeout"



    def validate_invariant(self, candiate_inv, target_template, var_types):
        
        # FIXME: should put a timeout for the loop

        # Replace 'input_file.c' and 'output_binary' with your actual C source file and desired output binary name
        delim = 0.001
        self.gen_c_code(delim)
        cex_list = []

        if self.compile_with_afl_clang_fast():
            print_green("Compilation successful")
            print_green("Starting fuzzing...")
            fuzzstat = self.fuzz_check(self.target_binary, 'indir', 'outdir') 

            if fuzzstat == "timeout":
                print_red("Timeout occurred. Exiting...")
                return True, None
            
            delim = 0.0
            iters = 0
            while fuzzstat == "crash":  # TODO: could simply use while True
                self.compile_with_afl_clang_fast()
                fuzzstat = self.fuzz_check(self.target_binary, 'indir', 'outdir')
                if fuzzstat == "crash":
                    # from the list of crashing inputs, find the first input that caused a crash
                    list_of_crashes = os.listdir('outdir/default/crashes/')
                    # remove README.txt from the list
                    if 'README.txt' in list_of_crashes:
                        list_of_crashes.remove('README.txt')

                    crash_file = list_of_crashes[0]
                    print_green(f"Crashing input: {crash_file}")

                    # run target.exe against the crashing input from stdin
                    input = open('outdir/default/crashes/' + crash_file, 'r')
                    command = ['./target.exe']
                    # run command and get status
                    print_green("Validating crashing input...")
                    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=input)
                    stdout, stderr = proc.communicate('\n')
                    input.close()

                    curr_crash_info = self.get_crash_info()
                    delim = round(float(curr_crash_info['diff']), 0) + 1
                    print_magenta(f"New delim: {delim}")
                    cex_list.append(curr_crash_info)

                    self.gen_c_code(delim)
                    iters += 1
                    if iters > 50:
                        break

                else:
                    print_red("Could not find a crash. Exiting...")
                    break


        # print(f"CEX list: ")
        cex_result = []
        # sort cex_list by diff value
        # cex_list.sort(key=lambda x: float(x['diff']))
        for cex in cex_list:
            # print vars from ib_list
            cex_p = {}

            for var in var_types:
                typ = var_types[var]
                if typ == 'int':
                    cex_p[var] = int(cex[var])
                elif typ == 'double':
                    cex_p[var] = float(cex[var])
                elif typ == 'bool':
                    cex_p[var] = int(cex[var])
                      
            cex_result.append(cex_p)

        print_red(f"CEX found: {cex_result}")
        return False, cex_result

    
