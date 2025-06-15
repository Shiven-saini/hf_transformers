import torch

def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)

        # Get total memory in MB
        total_memory = properties.total_memory / (1024 ** 2)

        # Get number of streaming multiprocessors
        multiprocessors = properties.multi_processor_count

        # Estimate CUDA cores per multiprocessor based on architecture
        cores_per_multiprocessor = {
            # Architecture and cores per multiprocessor mapping
            'sm_2x': 32,   # Fermi
            'sm_3x': 192,  # Kepler
            'sm_5x': 128,  # Maxwell
            'sm_6x': 64,   # Pascal
            'sm_7x': 64,   # Volta and Turing
            'sm_8x': 128,  # Ampere
            'sm_89': 128,  # Ada Lovelace
        }

        # Compute capability
        compute_capability = f"sm_{properties.major}{properties.minor}"
        cores_per_sm = cores_per_multiprocessor.get(compute_capability, "Unknown")

        if cores_per_sm != "Unknown":
            total_cuda_cores = multiprocessors * cores_per_sm
            print(f"GPU is available: {gpu_name}")
            print(f"Device Name: {device}")
            print(f"Total Memory: {total_memory:.2f} MB")
            print(f"Streaming Multiprocessors: {multiprocessors}")
            print(f"CUDA Cores per Multiprocessor: {cores_per_sm}")
            print(f"Total CUDA Cores: {total_cuda_cores}")
        else:
            print(f"GPU architecture ({compute_capability}) is not recognized.")
    else:
        print("No GPU available on this system.")

if __name__ == "__main__":
    check_gpu()