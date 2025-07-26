import os, sys
from subprocess import Popen, PIPE
import json
import re

Popen(["cmake", ".", "-B", "build"]).wait()
with open("build/compile_commands.json", "r") as f:
    compile_commands = json.load(f)

# --options-file CMakeFiles/flash_attn_2_cuda.dir/includes_CUDA.rsp
re_options = re.compile(r"--options-file ([A-Za-z0-9/\._]*)")

for command in compile_commands:
    if re_options.search(command["command"]):
        m = re_options.search(command["command"])
        options_file = m.group(1)
        with open(os.path.join('build', options_file), "r") as f:
            options = f.read()
        command["command"] = command["command"].replace(m.group(0), options)

with open("compile_commands.json", "w") as f:
    json.dump(compile_commands, f, indent=2)