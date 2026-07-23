import pytest
import torch

import torch_infini  # noqa: F401

from .backends import AcceleratorBackend


INFINI_BACKEND = AcceleratorBackend("infini", "infini", torch.infini)
CUDA_BACKEND = AcceleratorBackend("cuda", "cuda", torch.cuda)


@pytest.fixture(
    params=[
        INFINI_BACKEND,
        pytest.param(
            CUDA_BACKEND,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available",
            ),
        ),
    ],
    ids=lambda backend: backend.name,
)
def accelerator_backend(request):
    return request.param


@pytest.fixture
def available_accelerator_backends():
    return tuple(
        backend for backend in (INFINI_BACKEND, CUDA_BACKEND) if backend.is_available()
    )
