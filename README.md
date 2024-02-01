# Create your own virtual environment from your terminal (CMD, Git Bash, Powershell, etc.)
$  python3 -m venv yourVirtualEnvironmentName

# Activate the virtual environment
# For Command (CMD) Prompt
$  .\yourVirtualEnvironmentName\Scripts\activate

# For Powershell
$  .\yourVirtualEnvironmentName\Scripts\Activate.ps1

# For Unix-like Shells on Windows OS (e.g., Git Bash), or a Linux OS
$  source yourVirtualEnvironmentName/bin/activate

# Make sure `install_reqs.[bat, sh]` and `requirements.txt` are in your current directory. Then install the requirements in your virtual environment.
# For Windows
$  dir install_reqs.bat  
$  dir requirements.txt  
$  install_reqs.bat  

# For Linux
$  ls -l install_reqs.sh  
$  ls -l requirements.txt  
$  chmod +x install_reqs.sh  
$  ./install_reqs.sh  

# Now you should be set up to train transformer models.

