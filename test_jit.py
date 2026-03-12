import subprocess
for x in ["n", "s", "m", "l"]: subprocess.run(["python", "jit_correct.py", x])