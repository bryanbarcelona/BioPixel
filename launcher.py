import sys
import os
import subprocess
import time
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import threading
from PIL import Image, ImageTk  # Import Pillow for image handling

class LauncherWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.status_file_path = os.path.abspath('startup_status')
        self.update_status()  # Start the status update loop

    def initUI(self):
        self.overrideredirect(True)  # Remove window decorations
        self.geometry("661x377")
        
        # Load the background image
        bg_image_path = os.path.abspath('Designer__1_-removebg-preview.png')
        bg_image = Image.open(bg_image_path)
        self.bg_image_tk = ImageTk.PhotoImage(bg_image)

        # Create a canvas to display the background image
        canvas = tk.Canvas(self, width=800, height=600)
        canvas.pack(fill='both', expand=True)
        canvas.create_image(0, 0, image=self.bg_image_tk, anchor='nw')

        # Create a ScrolledText widget for displaying status
        self.textEdit = ScrolledText(self, state='disabled', bg='#D3D3D3')  # Adjust background to match transparency
        self.textEdit.place(x=10, y=550, width=780, height=40)  # Position at bottom left

    def read_status(self):
        try:
            with open(self.status_file_path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return "Status file not found."

    def update_status(self):
        current_status = self.read_status()
        self.textEdit.configure(state='normal')
        self.textEdit.delete(1.0, tk.END)
        self.textEdit.insert(tk.END, current_status)
        self.textEdit.configure(state='disabled')
        self.after(100, self.update_status)  # Schedule the next update in 100 ms

def launch_main_app():
    try:
        print("Starting Splash Screen...")
        print("Initializing BioPixel Main...")
        print("Version 1.0.0")

        exe_path = os.path.abspath('BioPixel.exe')
        print(f"Attempting to launch BioPixel.exe from: {exe_path}")

        # Launch the main application in a new console window
        subprocess.Popen(
            ['cmd', '/c', 'start', 'BioPixel.exe'],
            cwd=os.path.dirname(exe_path),
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )

        # Start the tkinter application in a separate thread
        def start_gui():
            app = LauncherWindow()
            app.mainloop()

        gui_thread = threading.Thread(target=start_gui)
        gui_thread.start()

    except Exception as e:
        print(f"Failed to launch the main application: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    launch_main_app()


# import sys
# import os
# import subprocess
# import time
# import tkinter as tk
# from tkinter.scrolledtext import ScrolledText
# import threading

# class LauncherWindow(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
#         self.status_file_path = os.path.abspath('startup_status')
#         self.update_status()  # Start the status update loop

#     def initUI(self):
#         self.title("BioPixel Launcher")
#         self.geometry("800x600")

#         self.textEdit = ScrolledText(self, state='disabled')
#         self.textEdit.pack(expand=True, fill='both')

#     def read_status(self):
#         try:
#             with open(self.status_file_path, "r") as f:
#                 return f.read().strip()
#         except FileNotFoundError:
#             return "Status file not found."

#     def update_status(self):
#         current_status = self.read_status()
#         self.textEdit.configure(state='normal')
#         self.textEdit.delete(1.0, tk.END)
#         self.textEdit.insert(tk.END, current_status)
#         self.textEdit.configure(state='disabled')
#         self.after(100, self.update_status)  # Schedule the next update in 100 ms

# def launch_main_app():
#     try:
#         print("Starting Splash Screen...")
#         print("Initializing BioPixel Main...")
#         print("Version 1.0.0")

#         exe_path = os.path.abspath('BioPixel.exe')
#         print(f"Attempting to launch BioPixel.exe from: {exe_path}")

#         # Launch the main application in a new console window
#         subprocess.Popen(
#             ['cmd', '/c', 'start', 'BioPixel.exe'],
#             cwd=os.path.dirname(exe_path),
#             creationflags=subprocess.CREATE_NEW_CONSOLE
#         )

#         # Start the tkinter application in a separate thread
#         def start_gui():
#             app = LauncherWindow()
#             app.mainloop()

#         gui_thread = threading.Thread(target=start_gui)
#         gui_thread.start()

#     except Exception as e:
#         print(f"Failed to launch the main application: {e}", file=sys.stderr)
#         sys.exit(1)

# if __name__ == "__main__":
#     launch_main_app()




# import subprocess
# import sys
# import time
# import os

# def read_status():
#     #status_file_path = os.path.join(os.path.dirname(__file__), "startup_status")
#     status_file_path = os.path.abspath('startup_status')
#     try:
#         with open(status_file_path, "r") as f:
#             return f.read().strip()  # No need to handle newline
#     except FileNotFoundError:
#         return "Status file not found."
    
# def launch_main_app():
#     try:
#         print("Starting Splash Screen...")
#         print("Initializing BioPixel Main...")
#         print("Version 1.0.0")
#         # Full path to the BioPixel executable
#         exe_path = os.path.abspath('BioPixel.exe')
#         status_file_path = os.path.abspath('startup_status')
#         print(f"Status file path: {status_file_path}")
#         print(f"Attempting to launch BioPixel.exe from: {exe_path}")

#         # Launch the main application in a new console window
#         process = subprocess.Popen(
#             ['BioPixel.exe'],
#             cwd=os.path.dirname(exe_path),  # Set working directory to the location of BioPixel.exe
#             creationflags=subprocess.CREATE_NEW_CONSOLE
#         )

#         status = read_status()
#         print(f"Initial status: {status}")

#         while True:
#             current_status = read_status()
            
#             if "END" in status:
#                 print("Initialization complete. Exiting launcher...")
#                 break

#             if current_status != status:
#                 print(f"Launcher status: {current_status}")
#                 status = current_status

#             time.sleep(0.01)

#         sys.exit(0)

#     except Exception as e:
#         print(f"Failed to launch the main application: {e}", file=sys.stderr)
#         sys.exit(1)

# if __name__ == "__main__":
#     launch_main_app()
