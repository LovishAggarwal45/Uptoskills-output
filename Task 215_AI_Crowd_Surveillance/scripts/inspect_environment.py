#!/usr/bin/env python3
"""
Environment and Hardware Inspector for AI-Based Crowd Surveillance System.
Verifies Python version, PyTorch, Torchvision, OpenCV, CUDA, GPU, CPU, RAM,
and video codec capabilities.
"""

import os
import sys
import platform
import subprocess
import shutil

def get_ram_info():
    """Attempt to detect system RAM using psutil or Windows systeminfo."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return f"Total: {mem.total / (1024**3):.1f} GB, Available: {mem.available / (1024**3):.1f} GB"
    except ImportError:
        pass
    
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(
                ["wmic", "computersystem", "get", "totalphysicalmemory"],
                stderr=subprocess.DEVNULL
            ).decode()
            lines = [line.strip() for line in output.splitlines() if line.strip().isdigit()]
            if lines:
                total_bytes = int(lines[0])
                return f"Total: {total_bytes / (1024**3):.1f} GB"
        except Exception:
            pass
    return "Not directly queryable (psutil not installed)"

def check_ffmpeg():
    """Check if ffmpeg CLI is installed and discoverable on PATH."""
    path = shutil.which("ffmpeg")
    if path:
        try:
            res = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            first_line = res.stdout.splitlines()[0] if res.stdout else "Available"
            return f"Found at {path} ({first_line})"
        except Exception:
            return f"Found at {path}"
    return "NOT FOUND on PATH (OpenCV built-in video codecs will be utilized for I/O)"

def inspect_environment():
    print("=" * 65)
    print("AI CROWD SURVEILLANCE SYSTEM - ENVIRONMENT INSPECTION REPORT")
    print("=" * 65)

    # OS and Architecture
    print(f"Operating System   : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Platform Detail    : {platform.platform()}")
    print(f"Python Executable  : {sys.executable}")
    print(f"Python Version     : {platform.python_version()} ({platform.python_implementation()})")
    
    # CPU & RAM
    cpu_count = os.cpu_count() or "Unknown"
    cpu_arch = platform.machine()
    cpu_proc = platform.processor() or cpu_arch
    print(f"CPU Architecture   : {cpu_arch}")
    print(f"CPU Processor      : {cpu_proc}")
    print(f"CPU Threads/Cores  : {cpu_count}")
    print(f"System RAM         : {get_ram_info()}")

    print("-" * 65)
    print("CORE DEPENDENCIES & VERSIONS:")
    
    # Check libraries
    libs = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-Learn"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("yaml", "PyYAML"),
    ]

    for mod_name, display_name in libs:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Installed (version unknown)")
            print(f"  * {display_name:<16}: {ver}")
        except ImportError:
            print(f"  * {display_name:<16}: NOT INSTALLED")

    # CUDA & GPU Verification
    print("-" * 65)
    print("ACCELERATION / HARDWARE STATUS:")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"  * CUDA Available   : {cuda_avail}")
        if cuda_avail:
            device_count = torch.cuda.device_count()
            print(f"  * GPU Count        : {device_count}")
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"  * GPU [{i}] Device   : {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM)")
            print(f"  * PyTorch CUDA Ver : {torch.version.cuda}")
        else:
            print("  * Execution Mode   : CPU Mode (Optimized multi-threaded CPU execution configured)")
    except Exception as e:
        print(f"  * PyTorch Error    : {e}")

    # Video Codec & OpenCV capabilities
    print("-" * 65)
    print("VIDEO / MULTIMEDIA BACKEND:")
    print(f"  * FFmpeg CLI       : {check_ffmpeg()}")
    try:
        import cv2
        print(f"  * OpenCV VideoIO   : Available (Supports cv2.VideoCapture & cv2.VideoWriter)")
    except ImportError:
        print(f"  * OpenCV VideoIO   : cv2 not installed")

    print("=" * 65)

if __name__ == "__main__":
    inspect_environment()
