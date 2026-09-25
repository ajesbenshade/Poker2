# Deterministic cuBLAS needs this set before the first CUDA matmul in the process.
# Some tests switch torch.use_deterministic_algorithms on to compare runs bit for bit.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
