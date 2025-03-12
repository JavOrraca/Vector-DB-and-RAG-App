#!/usr/bin/env python3
"""
Fast dependency installer for Vector-DB-and-RAG-App using uv.

This script uses the uv package manager to quickly install all required dependencies.
uv is significantly faster than pip for Python package installation.
"""

import os
import sys
import subprocess
import tempfile
import platform
from pathlib import Path

def install_uv():
    """Install uv if not already installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv is already installed.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing uv...")
        
        # Determine install script based on platform
        if platform.system() == "Windows":
            install_script = "powershell -c \"iwr -useb https://astral.sh/uv/install.ps1 | iex\""
        else:
            install_script = "curl -fsSL https://astral.sh/uv/install.sh | sh"
        
        try:
            if platform.system() == "Windows":
                subprocess.run(install_script, shell=True, check=True)
            else:
                subprocess.run(install_script, shell=True, check=True)
            
            # Add to PATH for the current session if needed
            if platform.system() != "Windows":
                uv_path = os.path.expanduser("~/.cargo/bin")
                if uv_path not in os.environ["PATH"]:
                    os.environ["PATH"] = f"{uv_path}:{os.environ['PATH']}"
            
            # Check installation
            subprocess.run(["uv", "--version"], check=True)
            print("✅ uv installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install uv. Please install it manually from https://github.com/astral-sh/uv")
            return False

def install_dependencies():
    """Install the project dependencies using uv."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"❌ Could not find requirements.txt at {requirements_file}")
        return False
    
    print(f"📦 Installing dependencies from {requirements_file}")
    
    try:
        # Create and activate a virtual environment with uv if needed
        venv_path = Path(__file__).parent / ".venv"
        
        if not venv_path.exists():
            print(f"🔨 Creating virtual environment at {venv_path}")
            subprocess.run(["uv", "venv", str(venv_path)], check=True)
        
        # Install dependencies with uv pip
        print("📦 Installing packages with uv (much faster than pip)...")
        subprocess.run(["uv", "pip", "install", "-r", str(requirements_file)], check=True)
        
        print("✅ Dependencies installed successfully!")
        
        # Print activation instructions
        if platform.system() == "Windows":
            activate_cmd = f".venv\\Scripts\\activate"
        else:
            activate_cmd = "source .venv/bin/activate"
        
        print(f"\n🚀 To activate the virtual environment, run:")
        print(f"    {activate_cmd}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Fast dependency installer for Vector-DB-and-RAG-App")
    if install_uv():
        install_dependencies()
    else:
        print("❌ Installation failed")
        sys.exit(1)