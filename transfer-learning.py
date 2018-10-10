import subprocess

# Build
with open('results/output.log', 'a+') as out:
    out.write("Beginning building.")
    p = subprocess.Popen('python3 build_data.py', shell=True, stdout=subprocess.PIPE, universal_newlines = True,
                         stderr=subprocess.STDOUT)
    while True:
        output = p.stdout.readline()
        if output == '' and p.poll() is not None:
            break
        if output:
            out.write(output.strip() + '\n')
            out.flush()
    retval = p.poll()
    out.write("Finished building. exit code:{}\n".format(str(retval)))
    out.flush()
print("Built model.")

with open('results/output.log', 'a+') as out:
    out.write("Beginning training.")
    p = subprocess.Popen('python3 train.py', shell=True, stdout=subprocess.PIPE, universal_newlines = True,
                         stderr=subprocess.STDOUT)
    while True:
        output = p.stdout.readline()
        if output == '' and p.poll() is not None:
            break
        if output:
            out.write(output.strip() + '\n')
            out.flush()
    retval = p.poll()
    out.write("Finished training. exit code:{}\n".format(str(retval)))
    out.flush()
print("Trained model.")

# Evaluate
with open('results/output.log', 'a+') as out:
    out.write("Beginning eval.")
    p = subprocess.Popen('python3 evaluate.py', shell=True, stdout=subprocess.PIPE, universal_newlines = True,
                         stderr=subprocess.STDOUT)
    while True:
        output = p.stdout.readline()
        if output == '' and p.poll() is not None:
            break
        if output:
            out.write(output.strip() + '\n')
            out.flush()
    retval = p.poll()
    out.write("Finished eval. exit code:{}\n".format(str(retval)))
    out.flush()
print("Evaluated model.")