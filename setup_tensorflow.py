#!/usr/bin/env python3
"""
TensorFlow Setup and Compatibility Checker
This script helps diagnose and fix TensorFlow installation issues.
"""

import sys
import subprocess
import platform

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ TensorFlow requires Python 3.8 or higher")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_system_info():
    """Display system information"""
    print(f"Operating System: {platform.system()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Machine: {platform.machine()}")

def test_tensorflow_import():
    """Test if TensorFlow can be imported"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        return True
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"❌ TensorFlow import error: {e}")
        return False

def install_tensorflow_cpu():
    """Install TensorFlow CPU version"""
    print("Installing TensorFlow CPU...")
    
    commands = [
        "pip uninstall tensorflow tensorflow-gpu -y",
        "pip install tensorflow-cpu>=2.10.0,<2.13.0"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        success, stdout, stderr = run_command(cmd)
        if not success:
            print(f"Error: {stderr}")
            return False
    
    return True

def install_alternative_versions():
    """Try installing different TensorFlow versions"""
    versions = [
        "tensorflow-cpu==2.12.0",
        "tensorflow-cpu==2.11.0", 
        "tensorflow-cpu==2.10.0"
    ]
    
    for version in versions:
        print(f"\nTrying to install {version}...")
        success, _, _ = run_command(f"pip install {version}")
        if success and test_tensorflow_import():
            print(f"✅ Successfully installed {version}")
            return True
        else:
            run_command(f"pip uninstall {version.split('==')[0]} -y")
    
    return False

def main():
    print("🔍 TensorFlow Setup and Compatibility Checker")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n💡 Please upgrade to Python 3.8 or higher")
        return
    
    # Display system info
    print("\n📋 System Information:")
    check_system_info()
    
    # Test current TensorFlow installation
    print("\n🧪 Testing TensorFlow import...")
    if test_tensorflow_import():
        print("\n🎉 TensorFlow is working correctly!")
        return
    
    # Try to fix installation
    print("\n🔧 Attempting to fix TensorFlow installation...")
    
    # Method 1: Install TensorFlow CPU
    if install_tensorflow_cpu():
        if test_tensorflow_import():
            print("\n🎉 TensorFlow CPU installed successfully!")
            return
    
    # Method 2: Try alternative versions
    print("\n🔄 Trying alternative TensorFlow versions...")
    if install_alternative_versions():
        print("\n🎉 Alternative TensorFlow version installed successfully!")
        return
    
    # If all fails
    print("\n❌ Could not install TensorFlow successfully")
    print("\n💡 Recommendations:")
    print("1. Use the sentiment analysis app with traditional ML models only")
    print("2. Try installing in a fresh virtual environment:")
    print("   python -m venv new_env")
    print("   source new_env/bin/activate  # On Windows: new_env\\Scripts\\activate")
    print("   pip install tensorflow-cpu==2.10.0")
    print("3. Consider using Google Colab for deep learning models")
    print("4. Check TensorFlow installation guide: https://www.tensorflow.org/install")

if __name__ == "__main__":
    main()
