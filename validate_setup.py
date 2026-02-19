#!/usr/bin/env python3
"""
Validation script to check Orpheus-FastAPI setup and configuration.
This script verifies that all dependencies are properly installed and
device detection is working correctly.
"""

import sys
import os

# Python version requirements
MIN_PYTHON_MAJOR = 3
MIN_PYTHON_MINOR = 10

def check_python_version():
    """Check if Python version meets requirements."""
    print("=" * 60)
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < MIN_PYTHON_MAJOR or (version.major == MIN_PYTHON_MAJOR and version.minor < MIN_PYTHON_MINOR):
        print(f"❌ ERROR: Python {MIN_PYTHON_MAJOR}.{MIN_PYTHON_MINOR}+ is required")
        return False
    print("✓ Python version is compatible")
    return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n" + "=" * 60)
    print("Checking dependencies...")
    
    dependencies = {
        'torch': 'PyTorch',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'pydantic': 'Pydantic',
        'requests': 'Requests',
        'numpy': 'NumPy',
        'sounddevice': 'SoundDevice',
        'snac': 'SNAC',
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"❌ {name} is NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    return True

def check_torch_device():
    """Check PyTorch device availability and configuration."""
    print("\n" + "=" * 60)
    print("Checking PyTorch device configuration...")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  - Device: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  - GPU Memory: {gpu_memory:.1f} GB")
            
            # Classify GPU
            gpu_name = torch.cuda.get_device_name(0).lower()
            if any(x in gpu_name for x in ['4090', '3090', 'a100', 'h100', 'v100']):
                print(f"  - Classification: High-end GPU (optimal performance)")
            elif any(x in gpu_name for x in ['4080', '3080', '3070', '2080', '2070', 'rtx']):
                print(f"  - Classification: Mid-range GPU (good performance)")
            else:
                print(f"  - Classification: Entry-level GPU (acceptable performance)")
        
        # Check MPS (Apple Silicon)
        elif torch.backends.mps.is_available():
            print(f"✓ Apple Metal Performance Shaders (MPS) is available")
            print(f"  - Device: Apple Silicon (M1/M2/M3)")
            print(f"  - Classification: Good performance on macOS")
        
        # CPU only
        else:
            print(f"⚠️  No GPU detected - using CPU")
            print(f"  - CPU threads: {torch.get_num_threads()}")
            print(f"  - Classification: CPU mode (slower performance)")
            print(f"\n  Note: For better performance, install PyTorch with CUDA or run on Apple Silicon")
        
        return True
        
    except ImportError:
        print("❌ PyTorch is not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False

def check_directories():
    """Check if required directories exist."""
    print("\n" + "=" * 60)
    print("Checking directories...")
    
    directories = ['outputs', 'static', 'templates', 'tts_engine']
    all_exist = True
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory}/ exists")
        else:
            print(f"⚠️  {directory}/ does NOT exist")
            if directory in ['outputs', 'static']:
                print(f"  Creating {directory}/...")
                try:
                    os.makedirs(directory, exist_ok=True)
                    print(f"  ✓ Created {directory}/")
                except Exception as e:
                    print(f"  ❌ Failed to create {directory}/: {e}")
                    all_exist = False
            else:
                all_exist = False
    
    return all_exist

def check_tts_engine():
    """Check if TTS engine can be imported."""
    print("\n" + "=" * 60)
    print("Checking TTS engine...")
    
    try:
        from tts_engine import AVAILABLE_VOICES, DEFAULT_VOICE
        print(f"✓ TTS engine imported successfully")
        print(f"  - Available voices: {', '.join(AVAILABLE_VOICES)}")
        print(f"  - Default voice: {DEFAULT_VOICE}")
        return True
    except Exception as e:
        print(f"❌ Failed to import TTS engine: {e}")
        return False

def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("ORPHEUS-FASTAPI SETUP VALIDATION")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("PyTorch Device", check_torch_device),
        ("Directories", check_directories),
        ("TTS Engine", check_tts_engine),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Unexpected error during {name} check: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("=" * 60)
    
    if all_passed:
        print("✓ All checks passed! Your setup is ready.")
        print("\nYou can start the server with:")
        print("  python app.py")
        print("  or")
        print("  uvicorn app:app --host 0.0.0.0 --port 5005")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
