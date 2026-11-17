import os
import subprocess
import time
import webbrowser
import sys
import socket
import logging
import json
from backend.error_logging import configure_error_logging

logger = logging.getLogger("coffee.launcher")


def configured_backend_python(base_dir):
    """Prefer a shell override, then this machine's local runtime selection."""
    override = os.getenv("COFFEE_MAESTRO_PYTHON")
    if override:
        return override
    try:
        with open(os.path.join(base_dir, "launcher.runtime.json"), encoding="utf-8") as stream:
            runtime = json.load(stream)
    except FileNotFoundError:
        return None
    if not isinstance(runtime, dict) or not isinstance(runtime.get("python_executable"), str) or not runtime["python_executable"].strip():
        raise ValueError("launcher.runtime.json must contain a nonempty python_executable string.")
    return runtime["python_executable"]

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def kill_port_owner(port):
    """Finds and kills the process currently using the specified port on Windows."""
    try:
        # Get netstat output for the port
        output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        pids = set()
        for line in output.strip().split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.strip().split()
                if len(parts) >= 5:
                    pids.add(parts[-1])
        
        for pid in pids:
            print(f"[!] Port {port} is busy. Stopping existing process (PID {pid})...")
            subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
            time.sleep(1) # Give it a moment to release the port
    except Exception:
        # Likely port is not in use or netstat failed to find anything
        pass

def wait_for_server(port, timeout=30):
    """Wait for the server to start responding on the given port."""
    print(f"[-] Waiting for server to initialize on port {port}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                print("[OK] Server reached!")
                return True
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(1)
    return False

def run_app():
    log_path = configure_error_logging()
    print(f"[*] Error log: {log_path}")
    print("[*] Coffee Maestro: Initializing Application...")
    
    # 1. Verify Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(base_dir, "frontend")
    backend_dir = os.path.join(base_dir, "backend")

    # 2. Build check
    dist_path = os.path.join(frontend_dir, "dist")
    if not os.path.exists(dist_path):
        print("[-] Frontend build missing. Building now...")
        try:
            subprocess.run("npm run build", cwd=frontend_dir, shell=True, check=True)
            print("[OK] Frontend build complete.")
        except subprocess.CalledProcessError:
            logger.exception("Frontend build failed")
            print("[ERROR] Error building frontend. Please run 'npm install' in the frontend directory first.")
            return

    # 3. Environment detection
    # An explicit interpreter lets deployments use a separately managed runtime.
    venv_python = os.path.join(backend_dir, "venv", "Scripts", "python.exe")
    try:
        configured_python = configured_backend_python(base_dir)
    except (OSError, ValueError):
        logger.exception("Invalid local launcher runtime configuration")
        print("[ERROR] Check launcher.runtime.json; it must select an existing Python executable.")
        return
    if configured_python:
        python_exe = os.path.abspath(os.path.expanduser(configured_python))
        if not os.path.isfile(python_exe):
            logger.error("Configured backend interpreter does not exist")
            print("[ERROR] COFFEE_MAESTRO_PYTHON must point to an existing Python executable.")
            return
        print(f"[PYTHON] Using configured interpreter: {python_exe}")
    elif os.path.exists(venv_python):
        python_exe = venv_python
        print(f"[VENV] Using virtual environment: {venv_python}")
    else:
        python_exe = sys.executable
        print(f"[WARNING] Virtual environment not found. Using default Python: {python_exe}")

    # 4. Port clearing
    PORT = 8000
    kill_port_owner(PORT)

    # 5. Launch Backend
    print(f"[*] Launching Production Server at http://127.0.0.1:{PORT}...")
    backend_script = os.path.join(backend_dir, "main.py")
    
    # Start the backend in a new process
    # Note: We use CREATE_NEW_CONSOLE so the user can see the logs/errors
    try:
        backend_env = os.environ.copy()
        backend_env["COFFEE_MAESTRO_OPEN_BROWSER"] = "0"
        backend_process = subprocess.Popen(
            [python_exe, backend_script],
            cwd=backend_dir,
            env=backend_env,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    except Exception as e:
        logger.exception("Backend launch failed")
        print(f"[ERROR] Failed to launch backend: {e}")
        return

    # 6. Wait for Ready and Open Browser
    if wait_for_server(PORT):
        webbrowser.open(f"http://127.0.0.1:{PORT}")
        print("\n[SUCCESS] Coffee Maestro is now running!")
        print("   Keep the backend console open while using the app.")
    else:
        logger.error("Backend did not become ready within 30 seconds")
        print(f"\n[ERROR] Server failed to start within 30 seconds.")
        print("   Please check the backend console for errors.")
    
if __name__ == "__main__":
    try:
        run_app()
    except KeyboardInterrupt:
        print("\n[EXIT] Shutdown requested.")
    except Exception as e:
        logger.exception("Unexpected launcher failure")
        print(f"\n[ERROR] An unexpected error occurred: {e}")


