import sys
import time

# Load the generated python code from output.txt
with open('output.txt', 'r', encoding='utf-16') as f:
    content = f.read()

# Find the start of the Python code (after "Generated Python Code:\n--------------------")
start_idx = content.find("import sys\nimport builtins")
if start_idx == -1:
    # Try with \r\n
    start_idx = content.find("import sys\r\nimport builtins")

# Find the start of execution output (before "--------------------\nExecuting Code:\n--------------------")
end_idx = content.find("--------------------\nExecuting Code:")
if end_idx == -1:
    end_idx = content.find("--------------------\r\nExecuting Code:")

if start_idx == -1 or end_idx == -1:
    print(f"Error: Could not locate Python code boundaries in output.txt. start={start_idx}, end={end_idx}")
    sys.exit(1)

code_to_exec = content[start_idx:end_idx]

# Modify the code to print timing of the script execution or wrap it in a function
# Let's execute the code and measure the time!
# To benchmark just the execution part, we will execute it 20 times.
# We'll use exec() but we'll print the time it takes.

print("Benchmarking Dedekind-compiled python code...")
t0 = time.perf_counter()
# We execute it once to warm up and see the outputs
global_dict = {}
exec(code_to_exec, global_dict)
t1 = time.perf_counter()
print(f"First run (with initialization and print statements): {(t1-t0)*1000.0:.2f} ms")

# Let's measure subsequent executions
runs = []
for i in range(20):
    t_start = time.perf_counter()
    # We can execute the compiled code again, but since it has global script execution,
    # let's run it.
    exec(code_to_exec, global_dict)
    t_end = time.perf_counter()
    runs.append((t_end - t_start) * 1000.0)

avg_time = sum(runs) / len(runs)
print(f"Average execution time of Dedekind-compiled code over 20 runs: {avg_time:.2f} ms")
