import sys
import os
import time
import csv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def timeit(func):
    """
    Decorator to time the execution of a function.
    """
    def wrapper(*args, **kwargs):
        #print(f"Starting execution of {func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run.")
        return result
    return wrapper

class SentinelProfiler:
    def __init__(self, output_file="performance_log.csv"):
        self.output_file = output_file
        self.call_data = []  # Store function call info
        self.start_times = {}  # Track function start times

        # Dynamically detect the project root from the script being executed (main.py)
        main_script = sys.argv[0]  # Path to the script being run
        self.project_root = os.path.dirname(os.path.abspath(main_script))

        # Normalize project root for consistent comparison (case insensitive and format consistent)
        self.project_root = os.path.normpath(self.project_root).lower()

        print(f"Profiler initialized. Project root: {self.project_root}")  # Debugging

    def _trace_calls(self, frame, event, arg):
        """Traces function calls but only within the project's modules."""
        # Debugging print to see if _trace_calls is triggered



        module_name = frame.f_globals.get("__name__", "")
        file_path = frame.f_globals.get("__file__", "")

        # Normalize the file path
        real_file_path = os.path.realpath(file_path)
        real_file_path = os.path.normpath(real_file_path).lower()  # Normalize path format and case
          # Debugging output
        # Check if the file path is within the project root (any module)

        if "lib\site-packages" in real_file_path:
            return None
        
        if real_file_path.startswith(self.project_root):
            #print(real_file_path)
            func_name = frame.f_code.co_name
            #print(f"DEBUG: Trace call event: {event}, Module: {module_name}, File: {real_file_path}, Function: {func_name}")  # Debugging output
            # Detect instance methods
            if "self" in frame.f_locals:
                class_name = frame.f_locals["self"].__class__.__name__
                func_name = f"{class_name}.{func_name}"

            qualified_name = f"{module_name}.{func_name}"

            # **Calculate Nesting Depth** (Count stack frames)
            depth = 0
            current_frame = frame
            while current_frame:
                depth += 1
                current_frame = current_frame.f_back 

            if event == "call":
                self.start_times[qualified_name] = time.time()
                #print(f"DEBUG: Function called: {qualified_name}")

            # Calculate elapsed time on any event
            elif event in ["return", "exception"]:
                start_time = self.start_times.pop(qualified_name, time.time())  # Get start time (or current time if not found)
                elapsed_time = time.time() - start_time
                #print(f"DEBUG: Function: {qualified_name} | Event: {event} | Elapsed Time: {elapsed_time:.6f} seconds | Depth: {depth}")  # Debugging output

                # Save the time data even for non-returning functions
                self.call_data.append((module_name, func_name, event, elapsed_time, depth))

        return None

    def start(self):
        sys.setprofile(self._trace_calls)

    def stop(self):
        sys.setprofile(None)
        self._save_results()
        #self._visualize_results()

    # def _save_results(self):
    #     """Saves collected data to a CSV file."""
    #     with open(self.output_file, mode="w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["Module", "Function", "Event", "Execution Time (s)", "Depth"])
    #         writer.writerows(self.call_data)

    #     print(f"Profiling data saved to {self.output_file}")

    def _save_results(self):
        """Processes and saves profiling data with total call counts."""
        
        # Convert call_data to a DataFrame for aggregation
        df = pd.DataFrame(self.call_data, columns=["Module", "Function", "Event", "Execution Time (s)", "Depth"])
        
        # Count total calls per function
        total_counts = df.groupby(["Module", "Function"]).size().reset_index(name="Total Calls")
        
        # Merge back into the main DataFrame
        df = df.merge(total_counts, on=["Module", "Function"], how="left")
        
        # Write to CSV using a context manager
        with open(self.output_file, mode="w", newline="") as f:
            df.to_csv(f, index=False)

        #print(f"Profiling data saved to {self.output_file}")


    def _visualize_results(self):
        """Visualizes the profiling results using Seaborn with circles for data points."""
        
        # Step 1: Load the CSV data into a pandas DataFrame
        df = pd.read_csv(self.output_file)

        plt.figure(figsize=(12, 6))

        # Step 2: Set the Seaborn style for a cleaner look
        sns.scatterplot(y="Execution Time (s)", x="Function", data=df, size="Total Calls", hue="Depth")

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")  

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        plt.show()