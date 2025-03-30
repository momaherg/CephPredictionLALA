#!/usr/bin/env python
"""
Simple script to install the timm package for pretrained models
"""
import subprocess
import sys

def install_timm():
    """Install the timm package using pip"""
    print("Installing timm package for pretrained model support...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "timm>=0.6.0"])
        print("Successfully installed timm!")
        print("You can now run your training script and it will use the pretrained HRNet-W32 model.")
    except Exception as e:
        print(f"Error installing timm: {str(e)}")
        print("Please install manually with: pip install timm>=0.6.0")

if __name__ == "__main__":
    install_timm() 