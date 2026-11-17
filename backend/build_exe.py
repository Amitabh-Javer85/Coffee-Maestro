import os
import subprocess
import sys

# Fix Windows console encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def main():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dist = os.path.join(backend_dir, "..", "frontend", "dist")
    
    # Check if frontend dist exists
    if not os.path.exists(frontend_dist):
        print(f"Error: Frontend build not found at {frontend_dist}")
        sys.exit(1)
        
    print(f"Installing PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        
    command = [
        sys.executable, "-m", "PyInstaller",
        "--name", "Coffee_Maestro",
        "--add-data", f"{frontend_dist}{os.pathsep}dist",
        "--hidden-import", "uvicorn",
        "--hidden-import", "fastapi",
        "--hidden-import", "pinecone",
        "--hidden-import", "pydantic",
        "--hidden-import", "google.genai",
        "--collect-submodules", "google.cloud.dialogflowcx_v3",
        "--hidden-import", "vertexai",
        # Exclude heavy modules that cause crashes or bloat
        "--exclude-module", "tensorflow",
        "--exclude-module", "tensorboard",
        # To make sure uvicorn standard logging and others are included
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--clean",
        "--noconfirm",
        "main.py"
    ]
    
    print("\nRunning PyInstaller:", " ".join(command))
    subprocess.run(command, check=True, cwd=backend_dir)
    print("\n[OK] Build complete! Executable is located in backend/dist/")

if __name__ == "__main__":
    main()

