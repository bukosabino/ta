import subprocess
import shutil
from pathlib import Path

# Clone TA-Lib repository
ta_lib_url = "https://github.com/mrjbq7/ta-lib.git"
clone_dir = "ta-lib"
subprocess.run(["git", "clone", ta_lib_url, clone_dir])

# Change directory to the cloned repository
os.chdir(clone_dir)

# Build TA-Lib
subprocess.run(["python", "setup.py", "build"])

# Build the wheel
subprocess.run(["python", "setup.py", "bdist_wheel"])

# Find the generated wheel file
dist_dir = Path("dist")
wheel_files = list(dist_dir.glob("*.whl"))

if not wheel_files:
    print("Error: No wheel file found.")
else:
    # Move the wheel file to the current directory
    shutil.move(wheel_files[0], wheel_files[0].name)
    print(f"Wheel file created: {wheel_files[0].name}")
