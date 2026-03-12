import subprocess
for x in ["n", "s", "m", "l"]: subprocess.run(["python", "test/run_jit.py", x])