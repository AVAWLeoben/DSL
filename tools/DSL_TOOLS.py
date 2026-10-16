# -*- coding: utf-8 -*-
"""Train YOLO with YoloSplitter or create no-copy Ultralytics AutoSplit lists."""

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



# --- NO-COPY ULTRALYTICS AUTOSPLIT TOOL ---
def _autosplit_source_paths(dataset_paths):
    """Normalize one directory or a sequence, retaining selection order."""
    from pathlib import Path

    if dataset_paths is None:
        return []
    if isinstance(dataset_paths, (str, os.PathLike)):
        dataset_paths = [dataset_paths]
    seen = set()
    result = []
    for item in dataset_paths:
        text = os.fspath(item).strip().strip('"')
        if not text:
            continue
        path = Path(os.path.expanduser(text)).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Dataset folder not found: {path}")
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result


def _autosplit_layout(selected):
    """Return (images directory, dataset root) for a source selection."""
    if selected.name.lower() == "images":
        return selected, selected.parent
    if (selected / "images").is_dir():
        return selected / "images", selected
    return selected, selected


def _show_autosplit_configuration(dataset_paths, weights, ignore_empty_images,
                                  output_dir):
    """GUI: add/remove source folders, choose output folder and split settings."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from pathlib import Path

    root = tk.Tk()
    root.title("Ultralytics AutoSplit - Multiple Datasets, No Copies")
    root.geometry("790x535")
    root.minsize(680, 475)
    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass

    content = ttk.Frame(root, padding=18)
    content.pack(fill="both", expand=True)
    content.columnconfigure(1, weight=1)
    content.rowconfigure(1, weight=1)

    ttk.Label(content, text="Combine Dataset Folders into Split Lists",
              font=("Segoe UI", 15, "bold")).grid(
        row=0, column=0, columnspan=4, sticky="w", pady=(0, 12)
    )
    ttk.Label(content, text="Source folders").grid(
        row=1, column=0, sticky="nw", padx=(0, 12), pady=6
    )
    source_frame = ttk.Frame(content)
    source_frame.grid(row=1, column=1, columnspan=2, sticky="nsew", pady=6)
    source_frame.columnconfigure(0, weight=1)
    source_frame.rowconfigure(0, weight=1)
    source_list = tk.Listbox(source_frame, height=7, selectmode="extended",
                             exportselection=False)
    source_list.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(source_frame, command=source_list.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    source_list.configure(yscrollcommand=scrollbar.set)

    sources = [str(p) for p in dataset_paths]
    for path in sources:
        source_list.insert(tk.END, path)

    output_var = tk.StringVar(value=str(output_dir) if output_dir else "")
    output_auto = not bool(output_dir)

    def update_default_output():
        if not output_auto or not sources:
            return
        first = Path(sources[0])
        images_dir, root_dir = _autosplit_layout(first)
        if len(sources) > 1:
            output_var.set(str(root_dir / "combined_split_lists"))
        elif images_dir == root_dir:
            output_var.set(str(root_dir / "split_lists"))
        else:
            output_var.set(str(root_dir))

    update_default_output()

    actions = ttk.Frame(content)
    actions.grid(row=1, column=3, sticky="n", padx=(12, 0), pady=6)

    def add_folder():
        initial = sources[-1] if sources else os.path.expanduser("~")
        chosen = filedialog.askdirectory(parent=root, title="Add Dataset Folder",
                                         initialdir=initial, mustexist=True)
        if chosen:
            path = str(Path(chosen).resolve())
            if path not in sources:
                sources.append(path)
                source_list.insert(tk.END, path)
                update_default_output()

    def remove_selected():
        for idx in reversed(source_list.curselection()):
            del sources[idx]
            source_list.delete(idx)
        update_default_output()

    ttk.Button(actions, text="Add Folder...", command=add_folder).pack(
        fill="x", pady=(0, 8))
    ttk.Button(actions, text="Remove Selected", command=remove_selected).pack(fill="x")

    ttk.Label(
        content,
        text="Each folder can contain images/ + labels/ or image files beside their .txt labels.",
        foreground="#555555",
    ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(2, 12))

    ttk.Label(content, text="Output folder").grid(
        row=3, column=0, sticky="w", padx=(0, 12), pady=6)
    ttk.Entry(content, textvariable=output_var).grid(
        row=3, column=1, columnspan=2, sticky="ew", pady=6)

    def browse_output():
        nonlocal output_auto
        initial = output_var.get().strip()
        if not os.path.isdir(initial):
            initial = str(Path(initial).parent) if initial else os.path.expanduser("~")
        if not os.path.isdir(initial):
            initial = os.path.expanduser("~")
        chosen = filedialog.askdirectory(parent=root, title="Choose Output Folder",
                                         initialdir=initial, mustexist=True)
        if chosen:
            output_auto = False
            output_var.set(chosen)

    ttk.Button(content, text="Browse...", command=browse_output).grid(
        row=3, column=3, padx=(12, 0), pady=6)

    normalized = [100 * float(w) / sum(weights) for w in weights]
    split_vars = [tk.StringVar(value=f"{v:g}") for v in normalized]
    ttk.Label(content, text="Split percentages").grid(
        row=4, column=0, sticky="w", padx=(0, 12), pady=8)
    ratios = ttk.Frame(content)
    ratios.grid(row=4, column=1, columnspan=3, sticky="w", pady=8)
    for i, label in enumerate(("Train %", "Val %", "Test %")):
        ttk.Label(ratios, text=label).grid(row=0, column=2 * i, padx=(0, 5))
        ttk.Entry(ratios, textvariable=split_vars[i], width=7).grid(
            row=0, column=2 * i + 1, padx=(0, 18))

    ignore_var = tk.BooleanVar(value=bool(ignore_empty_images))
    ttk.Checkbutton(
        content,
        text="IGNORE EMPTY IMAGES (missing, empty or whitespace-only .txt labels)",
        variable=ignore_var,
    ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(12, 4))
    ttk.Label(
        content,
        text="Combines all selected folders into train.txt / val.txt / test.txt + autosplit_data.yaml.",
        foreground="#555555",
    ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(4, 3))
    ttk.Label(
        content, text="Only file paths are written. Source images and annotations stay in place.",
        foreground="#555555",
    ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(0, 8))

    result = {}

    def confirm():
        if not sources:
            messagebox.showerror("Missing Dataset", "Add at least one source folder.",
                                 parent=root)
            return
        try:
            paths = _autosplit_source_paths(sources)
            values = tuple(float(v.get().strip().replace(",", ".")) for v in split_vars)
            if (not all(0 <= v <= 100 for v in values)
                    or abs(sum(values) - 100) > 0.01):
                raise ValueError("Train, validation and test percentages must add up to 100.")
            if values[0] == 0:
                raise ValueError("Training percentage must be above zero.")
            dest = output_var.get().strip().strip('"')
            if not dest:
                raise ValueError("Select an output folder.")
            dest = Path(os.path.expanduser(dest)).resolve()
            if dest.exists() and not dest.is_dir():
                raise ValueError("Output path exists but is not a folder.")
        except (ValueError, NotADirectoryError) as exc:
            messagebox.showerror("Invalid Settings", str(exc), parent=root)
            return
        existing = [str(dest / filename) for filename in
                    ("train.txt", "val.txt", "test.txt", "autosplit_data.yaml")
                    if (dest / filename).exists()]
        if existing and not messagebox.askyesno(
            "Replace Existing Split Lists",
            "These output files already exist and will be replaced:\n\n"
            + "\n".join(existing) + "\n\nContinue?", parent=root,
        ):
            return
        result.update({
            "dataset_paths": paths,
            "output_dir": dest,
            "weights": tuple(v / 100 for v in values),
            "ignore_empty_images": ignore_var.get(),
        })
        root.destroy()

    controls = ttk.Frame(content)
    controls.grid(row=8, column=0, columnspan=4, sticky="e", pady=(6, 0))
    ttk.Button(controls, text="Cancel", command=root.destroy).pack(
        side="left", padx=(0, 10))
    ttk.Button(controls, text="Create Combined Split Lists", command=confirm).pack(side="left")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return result or None


def _autosplit_read_names(dataset_root):
    """Get explicit class names from an original dataset YAML, if supplied."""
    import yaml

    for filename in ("data.yaml", "dataset.yaml"):
        source = dataset_root / filename
        if source.is_file():
            data = yaml.safe_load(source.read_text(encoding="utf-8-sig")) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Invalid class YAML: {source}")
            names = data.get("names")
            if isinstance(names, list):
                return {i: str(name) for i, name in enumerate(names)}
            if isinstance(names, dict) and names:
                try:
                    return {int(key): str(value) for key, value in names.items()}
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid class ID in {source}") from exc
    return None


def _autosplit_combined_names(sources, eligible_images, label_for):
    """Validate that source datasets use compatible class IDs and names."""
    names = None
    names_origin = None
    for _, root_dir in sources:
        source_names = _autosplit_read_names(root_dir)
        if source_names is None:
            continue
        if names is None:
            names, names_origin = source_names, root_dir
        elif source_names != names:
            raise ValueError(
                f"Class names do not match across dataset folders:\n"
                f"  {names_origin}: {names}\n  {root_dir}: {source_names}\n"
                "Class ID mappings must be identical before combining datasets."
            )

    max_class_id = -1
    for image in eligible_images:
        label = label_for(image)
        if not label.is_file():
            continue
        with label.open("r", encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                columns = line.split()
                if not columns:
                    continue
                try:
                    class_id = int(columns[0])
                except ValueError:
                    continue
                if class_id >= 0:
                    max_class_id = max(max_class_id, class_id)

    if names is None:
        if max_class_id > 1000:
            raise ValueError("Unexpected class ID > 1000; check annotation files.")
        return {i: f"class_{i}" for i in range(max(1, max_class_id + 1))}, True

    if max_class_id >= 0 and max_class_id not in names:
        raise ValueError(
            f"Annotation uses class ID {max_class_id}, but the supplied data.yaml "
            "does not define that ID. Check class mappings before training."
        )
    return names, False


def autosplit_dataset(dataset_path=None, weights=(0.8, 0.2, 0.0),
                      ignore_empty_images=True, dataset_paths=None, output_dir=None):
    """Combine one or more datasets into YOLO split lists, without copying images.

    ``dataset_path`` accepts one folder or a list of folders for compatibility.
    ``dataset_paths`` is the clearer multi-source keyword argument.
    With no paths provided, opens a Tkinter window allowing the user to add
    multiple source folders, remove selections, choose an output directory,
    set split ratios and toggle IGNORE EMPTY IMAGES (on by default).

    Each source may contain ``images/`` and ``labels/`` subfolders, or mixed
    images and labels together. The function invokes Ultralytics autosplit
    individually on each images folder, then combines the split assignments
    into three absolute-path manifests with duplicate paths removed. Existing
    autosplit_*.txt files in source folders are restored, not overwritten.

    IMPORTANT: Class IDs must have the same meaning in every source dataset.
    When data.yaml class names exist, mismatches raise an error. If no class
    YAML exists, placeholder names are generated and need user verification.
    """
    import math
    from pathlib import Path

    weights = tuple(float(w) for w in weights)
    if (len(weights) != 3 or not all(math.isfinite(w) and w >= 0 for w in weights)
            or sum(weights) <= 0 or weights[0] <= 0):
        raise ValueError("weights must be (train, val, test), nonnegative, with train > 0.")
    weights = tuple(w / sum(weights) for w in weights)

    if dataset_path is not None and dataset_paths is not None:
        raise ValueError("Use dataset_path or dataset_paths, not both.")
    supplied = dataset_paths if dataset_paths is not None else dataset_path
    selected_paths = _autosplit_source_paths(supplied)

    if not selected_paths:
        options = _show_autosplit_configuration([], weights, ignore_empty_images, output_dir)
        if options is None:
            print("AutoSplit cancelled by the user.")
            return None
        selected_paths = options["dataset_paths"]
        weights = options["weights"]
        ignore_empty_images = options["ignore_empty_images"]
        output_dir = options["output_dir"]

    sources = [_autosplit_layout(p) for p in selected_paths]
    if output_dir is None:
        first_images, first_root = sources[0]
        if len(selected_paths) > 1:
            output_dir = first_root / "combined_split_lists"
        else:
            output_dir = first_root / "split_lists" if first_images == first_root else first_root
    output_dir = Path(os.path.expanduser(os.fspath(output_dir).strip().strip('"'))).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output folder is not a directory: {output_dir}")

    _load_dependency("ultralytics", "ultralytics")
    from ultralytics.data.split import autosplit
    from ultralytics.data.utils import IMG_FORMATS, img2label_paths
    import yaml

    def label_for(image):
        return Path(img2label_paths([str(image)])[0])

    def nonempty_label(label):
        if not label.is_file():
            return False
        with label.open("r", encoding="utf-8-sig", errors="replace") as stream:
            return any(line.strip() for line in stream)

    raw_names = ("train", "val", "test")
    combined = {split: [] for split in raw_names}
    seen_images = set()
    stats = {
        "excluded_empty_images": 0,
        "excluded_duplicates": 0,
        "excluded_old_split_images": 0,
    }
    total_seen = 0
    print(f"Combining {len(sources)} source dataset(s), no image/label copies.")
    print(f"Ignore empty images: {'YES' if ignore_empty_images else 'NO'}")
    print(f"Output folder: {output_dir}")
    for idx, (images_dir, root_dir) in enumerate(sources, start=1):
        print(f"[{idx}/{len(sources)}] Ultralytics autosplit: {images_dir}")
        if not any(p.is_file() and p.suffix[1:].lower() in IMG_FORMATS
                   for p in images_dir.rglob("*")):
            raise ValueError(f"No supported images found in {images_dir}")

        intermediates = {
            kind: images_dir.parent / f"autosplit_{kind}.txt"
            for kind in raw_names
        }
        # Ultralytics deletes/recreates intermediates in images_dir.parent.
        # Preserve any user files already present there, even after a failure.
        original = {path: path.read_bytes() if path.exists() else None
                    for path in intermediates.values()}
        try:
            autosplit(path=images_dir, weights=weights, annotated_only=False)
            for kind, raw in intermediates.items():
                if not raw.is_file():
                    continue
                for entry in raw.read_text(encoding="utf-8-sig").splitlines():
                    if not entry.strip():
                        continue
                    image = (raw.parent / entry.strip()).resolve()
                    if not image.is_file():
                        raise FileNotFoundError(f"Ultralytics AutoSplit returned missing image: {image}")
                    total_seen += 1
                    if image in seen_images:
                        stats["excluded_duplicates"] += 1
                        continue
                    seen_images.add(image)
                    # Skip copies left behind by the original YoloSplitter tool.
                    if images_dir == root_dir:
                        parts = image.relative_to(images_dir).parts
                        if parts and (parts[0].lower() in (
                                "split_lists", "combined_split_lists", "splitted_dataset")
                                or parts[0].lower().startswith("splitted_dataset_")):
                            stats["excluded_old_split_images"] += 1
                            continue
                    if ignore_empty_images and not nonempty_label(label_for(image)):
                        stats["excluded_empty_images"] += 1
                        continue
                    combined[kind].append(image)
        finally:
            for path, previous_bytes in original.items():
                if previous_bytes is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(previous_bytes)

    total = sum(len(group) for group in combined.values())
    if not total:
        raise ValueError("No eligible images found. Check your labels and the ignore-empty setting.")
    if not combined["train"]:
        raise ValueError("Train split is empty. Increase train percentage or add more images.")

    images = [image for group in combined.values() for image in group]
    names, placeholders = _autosplit_combined_names(sources, images, label_for)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for kind, group in combined.items():
        target = output_dir / f"{kind}.txt"
        target.write_text("".join(f"{p.as_posix()}\n" for p in group), encoding="utf-8")
        results[kind] = str(target)
        print(f"{kind}: {len(group)} images -> {target}")
        if weights[raw_names.index(kind)] > 0 and not group:
            print(f"WARNING: {kind} received no images. This is a random split; try more images.")

    yaml_data = {
        "path": str(output_dir),
        "train": "train.txt",
        "val": "val.txt",
        "names": names,
    }
    if weights[2] > 0:
        yaml_data["test"] = "test.txt"
    yaml_path = output_dir / "autosplit_data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8")

    if placeholders:
        print("NOTE: No shared class names found in source data.yaml files. "
              "Verify/edit the class names in autosplit_data.yaml before training.")
    print(f"Total eligible image paths: {total}")
    print(f"Excluded missing/empty/whitespace labels: {stats['excluded_empty_images']}")
    print(f"Deduplicated overlapping source paths: {stats['excluded_duplicates']}")
    print(f"YOLO dataset configuration: {yaml_path}")
    print("Source images, labels and pre-existing AutoSplit files were left unchanged.")
    return {
        **results,
        "yaml": str(yaml_path),
        "counts": {kind: len(group) for kind, group in combined.items()},
        "source_folders": [str(p) for p in selected_paths],
        "output_dir": str(output_dir),
        **stats,
    }


if __name__ == "__main__":
    # Original default still starts the YOLOSplitter training setup.
    # `python DSL_TOOLS_multi_folder.py --autosplit` opens the multi-folder GUI.
    if "--autosplit" in sys.argv[1:]:
        autosplit_dataset()
    else:
        split_data_and_train_model()
