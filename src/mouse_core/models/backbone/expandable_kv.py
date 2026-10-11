"""Growable K/V buffer for the paged decode pool.

On CUDA the token dimension grows by mapping more physical pages under a
reserved virtual address (existing tokens are not copied). Elsewhere the
buffer is a normal tensor and growth copies the prefix into a larger one.
"""

from __future__ import annotations

import ctypes
from ctypes import c_int, c_size_t, c_ubyte, c_uint64, c_ushort, c_void_p
from typing import Any

import torch

# cuda-bindings publishes no free-threaded 3.15 wheel, so the page pool
# calls libcuda itself. Layout matches cuda.h (CUDA 13).
_CUDA_SUCCESS = 0
_CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
_CU_MEM_HANDLE_TYPE_NONE = 0x0
_CU_MEM_LOCATION_TYPE_DEVICE = 0x1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
_CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1


class _CUmemLocation(ctypes.Structure):
    _fields_ = (("type", c_int), ("id", c_int))


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = (
        ("compressionType", c_ubyte),
        ("gpuDirectRDMACapable", c_ubyte),
        ("usage", c_ushort),
        ("reserved", c_ubyte * 4),
    )


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = (
        ("type", c_int),
        ("requestedHandleTypes", c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", c_void_p),
        ("allocFlags", _CUmemAllocFlags),
    )


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = (("location", _CUmemLocation), ("flags", c_int))


class _CudaDriver:
    """Subset of the CUDA driver API the page pool maps with."""

    def __init__(self) -> None:
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuMemGetAllocationGranularity.argtypes = [
            ctypes.POINTER(c_size_t),
            ctypes.POINTER(_CUmemAllocationProp),
            c_int,
        ]
        lib.cuMemGetAllocationGranularity.restype = c_int
        lib.cuMemAddressReserve.argtypes = [
            ctypes.POINTER(c_uint64),
            c_size_t,
            c_size_t,
            c_uint64,
            c_uint64,
        ]
        lib.cuMemAddressReserve.restype = c_int
        lib.cuMemAddressFree.argtypes = [c_uint64, c_size_t]
        lib.cuMemAddressFree.restype = c_int
        lib.cuMemCreate.argtypes = [
            ctypes.POINTER(c_uint64),
            c_size_t,
            ctypes.POINTER(_CUmemAllocationProp),
            c_uint64,
        ]
        lib.cuMemCreate.restype = c_int
        lib.cuMemRelease.argtypes = [c_uint64]
        lib.cuMemRelease.restype = c_int
        lib.cuMemMap.argtypes = [c_uint64, c_size_t, c_size_t, c_uint64, c_uint64]
        lib.cuMemMap.restype = c_int
        lib.cuMemUnmap.argtypes = [c_uint64, c_size_t]
        lib.cuMemUnmap.restype = c_int
        lib.cuMemSetAccess.argtypes = [
            c_uint64,
            c_size_t,
            ctypes.POINTER(_CUmemAccessDesc),
            c_size_t,
        ]
        lib.cuMemSetAccess.restype = c_int
        self._lib = lib

    def cuMemGetAllocationGranularity(self, prop: _CUmemAllocationProp, option: int) -> tuple[int, int]:
        gran = c_size_t()
        err = self._lib.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), option)
        return err, int(gran.value)

    def cuMemAddressReserve(self, size: int, alignment: int, addr: int, flags: int) -> tuple[int, int]:
        ptr = c_uint64()
        err = self._lib.cuMemAddressReserve(ctypes.byref(ptr), size, alignment, addr, flags)
        return err, int(ptr.value)

    def cuMemAddressFree(self, ptr: int, size: int) -> tuple[int]:
        return (self._lib.cuMemAddressFree(ptr, size),)

    def cuMemCreate(self, size: int, prop: _CUmemAllocationProp, flags: int) -> tuple[int, int]:
        handle = c_uint64()
        err = self._lib.cuMemCreate(ctypes.byref(handle), size, ctypes.byref(prop), flags)
        return err, int(handle.value)

    def cuMemRelease(self, handle: int) -> tuple[int]:
        return (self._lib.cuMemRelease(handle),)

    def cuMemMap(self, ptr: int, size: int, offset: int, handle: int, flags: int) -> tuple[int]:
        return (self._lib.cuMemMap(ptr, size, offset, handle, flags),)

    def cuMemUnmap(self, ptr: int, size: int) -> tuple[int]:
        return (self._lib.cuMemUnmap(ptr, size),)

    def cuMemSetAccess(
        self,
        ptr: int,
        size: int,
        descs: list[_CUmemAccessDesc],
        count: int,
    ) -> tuple[int]:
        array = (_CUmemAccessDesc * count)(*descs)
        return (self._lib.cuMemSetAccess(ptr, size, array, count),)


_DRIVER: _CudaDriver | None = None


def _cuda_driver() -> _CudaDriver:
    global _DRIVER
    if _DRIVER is None:
        try:
            _DRIVER = _CudaDriver()
        except OSError as exc:
            raise RuntimeError("libcuda.so.1 is not available") from exc
    return _DRIVER


def _prod(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _check_drv(result: tuple[int, ...], label: str) -> int:
    err = result[0]
    if err != _CUDA_SUCCESS:
        raise RuntimeError(f"{label}: {err}")
    if len(result) == 1:
        return 0
    return result[1]


def _raw_shape(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Token-major storage layout ``[T, layers, kv_heads, head_dim]``.

    Growing T does not change the inner strides, so existing tokens stay
    at the same addresses. A contiguous ``[layers, kv_heads, T, head_dim]``
    buffer would change those strides on grow and reread the prefix wrong.
    """
    layers, kv_heads, n_tokens, head_dim = shape
    return (n_tokens, layers, kv_heads, head_dim)


class ExpandableKvTensor:
    """One K or V pool: ``[layers, kv_heads, n_tokens, head_dim]``.

    ``grow_token_dim`` extends the token axis. On a VMM-backed buffer the
    ``data_ptr`` stays put until the reserved virtual range is outgrown
    (then mappings move to a larger reservation, still without a copy).
    """

    def __init__(
        self,
        shape: tuple[int, int, int, int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if len(shape) != 4:
            raise ValueError(f"ExpandableKvTensor shape must be 4-D, got {shape}.")
        if any(d < 1 for d in shape):
            raise ValueError(f"ExpandableKvTensor dims must be >= 1, got {shape}.")
        self._shape = shape
        self.dtype = dtype
        if device.type == "cuda":
            torch.cuda.init()
            device = torch.device("cuda", _device_index(device))
        self.device = device
        self._itemsize = torch.empty((), dtype=dtype).element_size()
        self._vmm = False
        self._closed = False
        self._drv: Any = None
        self._prop: Any = None
        self._access: list[Any] = []
        self._gran = 0
        self._base = 0
        self._reserve = 0
        self._mapped = 0
        self._handles: list[tuple[int, int, Any]] = []  # (offset, size, handle)
        self._raw = torch.empty(0, dtype=dtype, device=device)
        self._tensor = self._raw
        if device.type == "cuda":
            try:
                self._init_vmm(shape)
                return
            except (ImportError, RuntimeError):
                self._release_vmm()
                self._vmm = False
        self._init_copy(shape)

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def uses_vmm(self) -> bool:
        return self._vmm

    @property
    def committed_bytes(self) -> int:
        if self._vmm:
            return self._mapped
        return int(self._tensor.untyped_storage().nbytes())

    def grow_token_dim(self, n_tokens: int) -> torch.Tensor:
        """Extend dim 2 to ``n_tokens``. Prefix is unchanged."""
        if self._closed:
            raise RuntimeError("ExpandableKvTensor.grow_token_dim after close.")
        if n_tokens < self._shape[2]:
            raise ValueError(
                f"grow_token_dim cannot shrink {self._shape[2]} -> {n_tokens}."
            )
        if n_tokens == self._shape[2]:
            return self._tensor
        new_shape = (self._shape[0], self._shape[1], n_tokens, self._shape[3])
        if self._vmm:
            self._grow_vmm(new_shape)
        else:
            self._grow_copy(new_shape)
        return self._tensor

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
        self._raw = torch.empty(0, dtype=self.dtype, device=self.device)
        self._tensor = self._raw
        self._release_vmm()
        self._vmm = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _nbytes(self, shape: tuple[int, int, int, int]) -> int:
        return _prod(shape) * self._itemsize

    def _init_copy(self, shape: tuple[int, int, int, int]) -> None:
        self._tensor = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self._shape = shape

    def _grow_copy(self, new_shape: tuple[int, int, int, int]) -> None:
        new = torch.zeros(new_shape, dtype=self.dtype, device=self.device)
        old_len = self._shape[2]
        new[:, :, :old_len].copy_(self._tensor)
        self._tensor = new
        self._shape = new_shape

    def _init_vmm(self, shape: tuple[int, int, int, int]) -> None:
        drv = _cuda_driver()

        torch.cuda.init()
        idx = _device_index(self.device)
        torch.cuda.set_device(idx)
        # Touch the caching allocator so the process has a current context.
        torch.zeros(1, device=self.device)

        prop = _CUmemAllocationProp()
        prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = idx
        prop.requestedHandleTypes = _CU_MEM_HANDLE_TYPE_NONE
        gran = int(
            _check_drv(
                drv.cuMemGetAllocationGranularity(
                    prop,
                    _CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                ),
                "cuMemGetAllocationGranularity",
            )
        )
        if gran < 1:
            raise RuntimeError("CUDA VMM granularity is 0.")

        access = _CUmemAccessDesc()
        access.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = idx
        access.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        need = self._nbytes(shape)
        free, _total = torch.cuda.mem_get_info(idx)
        reserve = _align_up(max(need, int(free)), gran)
        try:
            base = int(_check_drv(drv.cuMemAddressReserve(reserve, gran, 0, 0), "cuMemAddressReserve"))
        except RuntimeError:
            reserve = _align_up(max(need * 2, gran), gran)
            base = int(_check_drv(drv.cuMemAddressReserve(reserve, gran, 0, 0), "cuMemAddressReserve"))

        self._drv = drv
        self._prop = prop
        self._access = [access]
        self._gran = gran
        self._base = base
        self._reserve = reserve
        self._mapped = 0
        self._handles = []
        self._commit(need)
        self._set_view(shape)
        self._tensor.zero_()
        self._vmm = True

    def _grow_vmm(self, new_shape: tuple[int, int, int, int]) -> None:
        old_len = self._shape[2]
        need = self._nbytes(new_shape)
        if need > self._reserve:
            self._remap(_align_up(max(need, self._reserve * 2), self._gran))
        self._commit(need)
        self._set_view(new_shape)
        self._raw[old_len:].zero_()

    def _commit(self, need: int) -> None:
        need = _align_up(need, self._gran)
        if need <= self._mapped:
            return
        if need > self._reserve:
            raise RuntimeError(
                f"VMM commit {need} bytes exceeds reserved virtual range {self._reserve}."
            )
        add = need - self._mapped
        handle = _check_drv(self._drv.cuMemCreate(add, self._prop, 0), "cuMemCreate")
        addr = self._base + self._mapped
        mapped = False
        try:
            _check_drv(self._drv.cuMemMap(addr, add, 0, handle, 0), "cuMemMap")
            mapped = True
            _check_drv(
                self._drv.cuMemSetAccess(addr, add, self._access, len(self._access)),
                "cuMemSetAccess",
            )
        except BaseException:
            if mapped:
                self._drv.cuMemUnmap(addr, add)
            self._drv.cuMemRelease(handle)
            raise
        self._handles.append((self._mapped, add, handle))
        self._mapped = need

    def _remap(self, new_reserve: int) -> None:
        new_reserve = _align_up(new_reserve, self._gran)
        new_base = int(
            _check_drv(
                self._drv.cuMemAddressReserve(new_reserve, self._gran, 0, 0),
                "cuMemAddressReserve(grow)",
            )
        )
        try:
            for offset, size, handle in self._handles:
                _check_drv(self._drv.cuMemUnmap(self._base + offset, size), "cuMemUnmap(remap)")
                _check_drv(
                    self._drv.cuMemMap(new_base + offset, size, 0, handle, 0),
                    "cuMemMap(remap)",
                )
                _check_drv(
                    self._drv.cuMemSetAccess(
                        new_base + offset, size, self._access, len(self._access)
                    ),
                    "cuMemSetAccess(remap)",
                )
        except BaseException:
            self._drv.cuMemAddressFree(new_base, new_reserve)
            raise
        _check_drv(self._drv.cuMemAddressFree(self._base, self._reserve), "cuMemAddressFree(old)")
        self._base = new_base
        self._reserve = new_reserve

    def _set_view(self, shape: tuple[int, int, int, int]) -> None:
        nbytes = self._nbytes(shape)
        storage = torch._C._construct_storage_from_data_pointer(self._base, self.device, nbytes)
        self._raw.set_(storage, 0, _raw_shape(shape))  # type: ignore[call-overload]
        self._tensor = self._raw.permute(1, 2, 0, 3)
        self._shape = shape

    def _release_vmm(self) -> None:
        if self._base == 0 or self._drv is None:
            self._handles = []
            self._mapped = 0
            self._base = 0
            self._reserve = 0
            return
        drv = self._drv
        for offset, size, handle in self._handles:
            try:
                drv.cuMemUnmap(self._base + offset, size)
            except Exception:
                pass
            try:
                drv.cuMemRelease(handle)
            except Exception:
                pass
        try:
            drv.cuMemAddressFree(self._base, self._reserve)
        except Exception:
            pass
        self._handles = []
        self._mapped = 0
        self._base = 0
        self._reserve = 0
        self._drv = None
