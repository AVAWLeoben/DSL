# -*- coding: utf-8 -*-
"""Configure a YOLO training run, split a dataset, and start training."""

import os
import re
import subprocess
import sys

# Workaround for possible OpenMP library conflicts (Spyder / Windows).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# --- OPTIONAL DEPENDENCIES ---
def _load_dependency(import_name, pip_name):
    """Import a dependency; install it into the running Python environment if needed."""
    try:
        return __import__(import_name)
    except ImportError:
        print(f"Installing missing dependency: {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        return __import__(import_name)


def _build_model_filename(version, size):
    """Return a model name such as yolov8n.pt, yolo11s.pt, or yolo26m.pt."""
    version_key = str(version).lower().strip()
    size_key = str(size).lower().strip()

    if size_key not in ("n", "s", "m", "l", "x"):
        raise ValueError(f"Invalid YOLO model size: {size!r}")

    versions = {
        "8": "yolov8", "v8": "yolov8", "yolov8": "yolov8",
        "11": "yolo11", "v11": "yolo11", "yolo11": "yolo11",
        "26": "yolo26", "v26": "yolo26", "yolo26": "yolo26",
    }
    if version_key not in versions:
        raise ValueError(f"Unsupported YOLO version: {version!r}")
    return f"{versions[version_key]}{size_key}.pt"


def _normalize_device_choice(device):
    """Normalize supported device settings without probing hardware."""
    if device is None:
        return "auto"
    value = str(device).strip().lower()
    if value in ("auto", "automatic", "auto (cpu default)", "auto (prefer gpu)"):
        return "auto"
    if value in ("0", "gpu", "gpu 0", "cuda:0", "gpu 0 (cuda/rocm)"):
        return "gpu"
    if value == "cpu":
        return "cpu"
    raise ValueError(f"Invalid training device: {device!r}. Choose 'auto', 0, or 'cpu'.")


def _show_configuration(version, size, epochs, time, batch, dataset_path, device, auto_install_cuda):
    """Show one Tkinter setup window with dropdowns and a dataset path field."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("YOLO Training Setup")
    root.geometry("705x510")
    root.minsize(660, 480)

    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass

    content = ttk.Frame(root, padding=22)
    content.pack(fill="both", expand=True)
    content.columnconfigure(1, weight=1)

    ttk.Label(
        content, text="Model and Dataset Configuration",
        font=("Segoe UI", 15, "bold"),
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 16))

    version_aliases = {
        "8": "8", "v8": "8", "yolov8": "8",
        "11": "11", "v11": "11", "yolo11": "11",
        "26": "26", "v26": "26", "yolo26": "26",
    }
    default_version = version_aliases.get(str(version).lower().strip(), "11")
    default_size = str(size).lower().strip() if size is not None else "n"
    if default_size not in ("n", "s", "m", "l", "x"):
        default_size = "n"

    version_var = tk.StringVar(value=default_version)
    size_var = tk.StringVar(value=default_size)
    duration_mode_var = tk.StringVar(value="Hours" if time is not None else "Epochs")
    duration_value_var = tk.StringVar(
        value=str(time) if time is not None else str(epochs if epochs is not None else 10)
    )
    batch_var = tk.StringVar(value=str(batch))
    dataset_var = tk.StringVar(value=os.fspath(dataset_path) if dataset_path else "")
    device_labels = {
        "auto": "Auto (CPU default)",
        "cpu": "CPU (always)",
        "gpu": "GPU 0 (CUDA/ROCm)",
    }
    device_var = tk.StringVar(value=device_labels[_normalize_device_choice(device)])
    install_cuda_var = tk.BooleanVar(value=bool(auto_install_cuda))

    def add_label(label, row):
        ttk.Label(content, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 16), pady=7
        )

    add_label("YOLO version", 1)
    ttk.Combobox(
        content, textvariable=version_var, values=("8", "11", "26"),
        state="readonly", width=19,
    ).grid(row=1, column=1, sticky="ew", pady=7)

    add_label("Model size", 2)
    ttk.Combobox(
        content, textvariable=size_var, values=("n", "s", "m", "l", "x"),
        state="readonly", width=19,
    ).grid(row=2, column=1, sticky="ew", pady=7)

    add_label("Training limit", 3)
    duration_frame = ttk.Frame(content)
    duration_frame.grid(row=3, column=1, sticky="ew", pady=7)
    duration_frame.columnconfigure(1, weight=1)

    duration_combo = ttk.Combobox(
        duration_frame, textvariable=duration_mode_var, values=("Epochs", "Hours"),
        state="readonly", width=13,
    )
    duration_combo.grid(row=0, column=0, sticky="w", padx=(0, 12))
    ttk.Entry(duration_frame, textvariable=duration_value_var).grid(
        row=0, column=1, sticky="ew"
    )

    def update_duration(_event=None):
        # Suggest a default when the user changes the duration unit.
        duration_value_var.set("10" if duration_mode_var.get() == "Epochs" else "1")

    duration_combo.bind("<<ComboboxSelected>>", update_duration)

    add_label("Batch size", 4)
    ttk.Entry(content, textvariable=batch_var).grid(
        row=4, column=1, sticky="ew", pady=7
    )

    add_label("Training device", 5)
    ttk.Combobox(
        content, textvariable=device_var, values=tuple(device_labels.values()),
        state="readonly", width=22,
    ).grid(row=5, column=1, sticky="ew", pady=7)

    cuda_checkbox = ttk.Checkbutton(
        content,
        text="Install CUDA-enabled PyTorch when needed (NVIDIA GPU only)",
        variable=install_cuda_var,
    )
    cuda_checkbox.grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 8))

    def update_cuda_checkbox(*_args):
        # Avoid accidental installations during a beginner's CPU training run.
        if device_var.get() == device_labels["gpu"]:
            cuda_checkbox.state(["!disabled"])
        else:
            install_cuda_var.set(False)
            cuda_checkbox.state(["disabled"])

    device_var.trace_add("write", update_cuda_checkbox)
    update_cuda_checkbox()

    add_label("Dataset directory", 7)
    ttk.Entry(content, textvariable=dataset_var).grid(
        row=7, column=1, sticky="ew", pady=7
    )

    def browse_dataset():
        current_path = dataset_var.get().strip().strip('"')
        initial_directory = (
            current_path if os.path.isdir(current_path) else os.path.expanduser("~")
        )
        selected_path = filedialog.askdirectory(
            parent=root, title="Choose Dataset Directory",
            initialdir=initial_directory, mustexist=True,
        )
        if selected_path:
            dataset_var.set(selected_path)

    ttk.Button(content, text="Browse...", command=browse_dataset).grid(
        row=7, column=2, sticky="w", padx=(10, 0), pady=7
    )

    ttk.Label(
        content,
        text="Select a folder containing images and their YOLO annotation files (.txt).",
        foreground="#555555",
    ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(2, 16))

    result = {}

    def start_training():
        directory_input = dataset_var.get().strip().strip('"')
        directory = os.path.abspath(os.path.expanduser(directory_input))
        if not directory_input or not os.path.isdir(directory):
            messagebox.showerror(
                "Invalid Dataset", "Please choose an existing dataset directory.", parent=root
            )
            return

        try:
            batch_count = int(batch_var.get().strip())
            if batch_count <= 0:
                raise ValueError("Batch size must be a positive integer.")
            if duration_mode_var.get() == "Hours":
                hours = float(duration_value_var.get().strip().replace(",", "."))
                if not 0 < hours < float("inf"):
                    raise ValueError("Training hours must be a finite positive number.")
                epoch_count = None
            else:
                epoch_count = int(duration_value_var.get().strip())
                if epoch_count <= 0:
                    raise ValueError("Epochs must be a positive integer.")
                hours = None
        except ValueError as error:
            messagebox.showerror(
                "Invalid Settings", f"Please check the entered values.\n{error}",
                parent=root,
            )
            return

        chosen_device = next(
            key for key, label in device_labels.items() if label == device_var.get()
        )
        result.update({
            "version": version_var.get(),
            "size": size_var.get(),
            "epochs": epoch_count,
            "time": hours,
            "batch": batch_count,
            "dataset_path": directory,
            "device": chosen_device,
            "auto_install_cuda": install_cuda_var.get(),
        })
        root.destroy()

    buttons = ttk.Frame(content)
    buttons.grid(row=9, column=0, columnspan=3, sticky="e", pady=(5, 0))
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(
        side="left", padx=(0, 10)
    )
    ttk.Button(buttons, text="Start Training", command=start_training).pack(side="left")

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return result or None


# Verified wheel families: CUDA 12.6, 13.0, and 13.2.
# Pair torch 2.14.1 with its matching torchvision 0.29.1 release.
_TORCH_VERSION = "2.14.1"
_TORCHVISION_VERSION = "0.29.1"


def _nvidia_gpu_info():
    """Return (GPU model, maximum supported CUDA version) reported by nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True,
            timeout=15, check=False, errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout
    version_match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    # Use a second, machine-readable query for the GPU name.
    try:
        name_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
            check=False, errors="replace",
        )
        name = name_result.stdout.strip().splitlines()[0] if name_result.returncode == 0 else "NVIDIA GPU"
    except (OSError, subprocess.TimeoutExpired, IndexError):
        name = "NVIDIA GPU"
    version = (int(version_match[1]), int(version_match[2])) if version_match else None
    return name, version


def _cuda_index_for_driver(max_cuda):
    """Choose a documented CUDA wheel that the installed driver can support."""
    if max_cuda is None:
        raise RuntimeError(
            "nvidia-smi did not report the driver's CUDA version. "
            "Check the NVIDIA driver and install PyTorch manually from "
            "https://pytorch.org/get-started/locally/."
        )
    if max_cuda >= (13, 2):
        return "cu132"
    if max_cuda >= (13, 0):
        return "cu130"
    if max_cuda >= (12, 6):
        return "cu126"
    raise RuntimeError(
        f"NVIDIA driver reports CUDA {max_cuda[0]}.{max_cuda[1]}, "
        "but this script's supported PyTorch wheels require CUDA 12.6 or later. "
        "Update the NVIDIA driver, then try again."
    )


def _probe_pytorch():
    """Inspect PyTorch in a fresh process (important when replacing installed wheels)."""
    probe_code = (
        "import torch; "
        "print('TORCH_VERSION=' + str(torch.__version__)); "
        "print('CUDA_BUILD=' + str(torch.version.cuda)); "
        "print('HIP_BUILD=' + str(getattr(torch.version, 'hip', None))); "
        "print('GPU_AVAILABLE=' + str(torch.cuda.is_available())); "
        "print('GPU_NAME=' + (torch.cuda.get_device_name(0) "
        "if torch.cuda.is_available() else 'none'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe_code],
        capture_output=True, text=True, check=False, timeout=90,
        errors="replace",
    )
    if result.returncode != 0:
        return {"ok": False, "error": (result.stderr or result.stdout).strip()}
    details = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            details[key] = value.strip()
    return {"ok": "GPU_AVAILABLE" in details, **details}


def _ensure_gpu_pytorch(device, auto_install_cuda):
    """Optionally repair a CPU-only PyTorch install, before importing Ultralytics."""
    preference = _normalize_device_choice(device)
    print(f"Python executable: {sys.executable}")
    existing = _probe_pytorch()
    if existing["ok"]:
        print(f"PyTorch version: {existing.get('TORCH_VERSION', 'unknown')}")
        print(f"PyTorch CUDA build: {existing.get('CUDA_BUILD', 'unknown')}")
        print(f"GPU visible to PyTorch: {existing.get('GPU_AVAILABLE', 'False')}")
    else:
        print("PyTorch is missing or could not be loaded: " + existing.get("error", "unknown"))

    # Auto is deliberately CPU-first for students. Only an explicit GPU
    # selection can trigger a CUDA installation or a GPU requirement.
    if preference in ("auto", "cpu") or existing.get("GPU_AVAILABLE") == "True":
        return

    # A CUDA build is already installed. Reinstalling cannot fix a missing
    # driver or an unsupported GPU, so do not repeat the download every run.
    if existing["ok"] and existing.get("CUDA_BUILD") not in (None, "None", ""):
        message = (
            "CUDA-enabled PyTorch is installed, but it cannot access a GPU. "
            "Check nvidia-smi, NVIDIA driver compatibility, and GPU support."
        )
        if preference == "gpu":
            raise RuntimeError(message)
        print("WARNING: " + message + " Using CPU.")
        return

    gpu = _nvidia_gpu_info()
    if gpu is None:
        if preference == "gpu":
            raise RuntimeError(
                "GPU 0 was requested, but PyTorch cannot access it and nvidia-smi "
                "cannot detect an NVIDIA GPU. Install/check your NVIDIA driver "
                "and run nvidia-smi from this terminal."
            )
        print("WARNING: NVIDIA GPU/driver not detected; using CPU. "
              "No CUDA packages were installed.")
        return

    name, maximum_cuda = gpu
    print(f"NVIDIA GPU detected: {name}")
    print("Driver maximum CUDA version: " + (
        f"{maximum_cuda[0]}.{maximum_cuda[1]}" if maximum_cuda else "unknown"
    ))
    if not auto_install_cuda:
        if preference == "gpu":
            raise RuntimeError(
                "GPU requested but PyTorch cannot access it. Enable 'Install "
                "CUDA-enabled PyTorch' in the setup window, or install a CUDA "
                "build manually using https://pytorch.org/get-started/locally/."
            )
        print("WARNING: GPU is detected but CUDA PyTorch installation was disabled; "
              "using CPU.")
        return

    cuda_variant = _cuda_index_for_driver(maximum_cuda)
    index_url = f"https://download.pytorch.org/whl/{cuda_variant}"
    # Explicit opt-in from the setup form; force replacement of +cpu wheels.
    command = [
        sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
        f"torch=={_TORCH_VERSION}", f"torchvision=={_TORCHVISION_VERSION}",
        "--index-url", index_url,
    ]
    print(f"Installing CUDA-enabled PyTorch ({cuda_variant}) in this Python environment...")
    print("This may download several GB and change packages in the active environment.")
    print("Command: " + subprocess.list2cmdline(command))
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"CUDA PyTorch installation failed (exit code {exc.returncode}). "
            f"Check the pip messages above and use {index_url} for troubleshooting."
        ) from exc

    updated = _probe_pytorch()
    if not updated["ok"] or updated.get("GPU_AVAILABLE") != "True":
        raise RuntimeError(
            "CUDA PyTorch was installed, but a fresh Python process still cannot "
            "access the GPU. Verify your driver/GPU compatibility and check "
            f"the installation output. Probe: {updated}"
        )
    print(f"CUDA PyTorch ready: {updated.get('TORCH_VERSION')} | "
          f"GPU: {updated.get('GPU_NAME')}")
    if "torch" in sys.modules:
        raise RuntimeError(
            "PyTorch was already imported in this process before the CUDA "
            "installation. Restart the Python process or Spyder kernel, then "
            "run DSL_train.py again."
        )


def _select_training_device(device, torch):
    """Use CPU by default. Require an explicit selection to train on GPU 0."""
    preference = _normalize_device_choice(device)
    available = bool(torch.cuda.is_available())
    print(f"Active PyTorch version: {torch.__version__}")
    print(f"Active CUDA build: {torch.version.cuda or 'not installed'}")
    print(f"GPU visible to PyTorch: {'Yes' if available else 'No'}")
    if preference == "auto":
        print("Selected training device: CPU (student-friendly automatic default).")
        return "cpu"
    if preference == "cpu":
        print("Selected training device: CPU (requested explicitly).")
        return "cpu"
    if not available:
        raise RuntimeError(
            "GPU 0 was selected but CUDA-enabled PyTorch is not available in this "
            "Python process. Check the NVIDIA driver and the PyTorch installation."
        )
    print(f"Selected training device: GPU 0 ({torch.cuda.get_device_name(0)}).")
    return 0


def _select_split_output(dataset_path):
    """Reuse a completed split, or choose a new path without deleting any files."""
    base = os.path.join(dataset_path, "splitted_dataset")
    if not os.path.exists(base):
        return base, False
    yaml_path = os.path.join(base, "data.yaml")
    if os.path.isfile(yaml_path):
        return base, True
    index = 2
    while os.path.exists(f"{base}_{index}"):
        candidate = f"{base}_{index}"
        if os.path.isfile(os.path.join(candidate, "data.yaml")):
            return candidate, True
        index += 1
    return f"{base}_{index}", False


def split_data_and_train_model(version=None, size=None, epochs=None, time=None,
                               batch=4, dataset_path=None, device="auto",
                               auto_install_cuda=False):
    """
    Configure YOLO, split a mixed dataset 80% train / 20% validation, and train.

    Display one setup window if any required option is missing; otherwise run
    without prompting. 'device' accepts 'auto', 0 (GPU 0), or 'cpu'.
    Auto chooses CPU (recommended for first-time users). GPU use is explicit.
    'auto_install_cuda' is disabled by default and only applies in GPU mode.
    """
    if (version is None or size is None or (epochs is None and time is None)
            or dataset_path is None):
        options = _show_configuration(
            version, size, epochs, time, batch, dataset_path, device, auto_install_cuda
        )
        if options is None:
            print("Configuration cancelled by the user.")
            return None
        version = options["version"]
        size = options["size"]
        epochs = options["epochs"]
        time = options["time"]
        batch = options["batch"]
        dataset_path = options["dataset_path"]
        device = options["device"]
        auto_install_cuda = options["auto_install_cuda"]

    dataset_path = os.path.abspath(
        os.path.expanduser(os.fspath(dataset_path).strip().strip('"'))
    )
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(f"Dataset directory not found: {dataset_path}")
    batch = int(batch)
    if batch <= 0:
        raise ValueError("Batch size must be positive.")
    if time is not None:
        time = float(time)
        if not 0 < time < float("inf"):
            raise ValueError("Training time must be a finite positive number.")
    else:
        if epochs is None:
            raise ValueError("Epoch count is required when no training time is set.")
        epochs = int(epochs)
        if epochs <= 0:
            raise ValueError("Epoch count must be positive.")

    model_filename = _build_model_filename(version, size)
    device = _normalize_device_choice(device)

    # Install/verify CUDA PyTorch BEFORE importing Ultralytics (which imports torch).
    _ensure_gpu_pytorch(device, auto_install_cuda)
    _load_dependency("ultralytics", "ultralytics")
    _load_dependency("yolosplitter", "yolosplitter")
    from ultralytics import YOLO
    from yolosplitter import YoloSplitter
    import torch

    selected_device = _select_training_device(device, torch)

    print(f"Dataset directory: {dataset_path}")
    output_directory, reusable_split = _select_split_output(dataset_path)
    if reusable_split:
        print(f"Reusing existing dataset split: {output_directory}")
    else:
        print("Splitting the dataset with YoloSplitter...")
        splitter = YoloSplitter(
            imgFormat=[".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
            labelFormat=[".txt", ".TXT"],
        )
        splitter.from_mixed_dir(
            input_dir=dataset_path,
            ratio=(0.8, 0.2, 0.0),  # training / validation / test
        )
        print(f"Writing the dataset split to: {output_directory}")
        splitter.save_split(output_dir=output_directory)
    # save_split writes data.yaml, but does not necessarily return its location.
    yaml_path = os.path.join(output_directory, "data.yaml")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(
            f"YoloSplitter did not create the expected YAML file: {yaml_path}"
        )
    print(f"Dataset split completed. YAML file: {yaml_path}")

    print(f"Starting YOLO training: {model_filename} (batch={batch})")
    model = YOLO(model_filename)
    training_args = {
        "data": yaml_path,
        "imgsz": 640,
        "batch": batch,
        "device": selected_device,
    }
    if time is not None:
        training_args["time"] = time
        print(f"Training limit: {time} hour(s)")
    else:
        training_args["epochs"] = epochs
        print(f"Training limit: {epochs} epoch(s)")

    results = model.train(**training_args)
    print("Training completed successfully.")
    return results


if __name__ == "__main__":
    split_data_and_train_model()
