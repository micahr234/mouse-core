"""Growable K/V buffer for the paged decode pool.

On CUDA the token dimension grows by mapping more physical pages under a
reserved virtual address (existing tokens are not copied). Elsewhere the
buffer is a normal tensor and growth copies the prefix into a larger one.
"""

from __future__ import annotations

from typing import Any

import torch


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


def _check_drv(result: Any, label: str) -> Any:
    from cuda.bindings import driver as drv  # type: ignore[attr-defined]

    if not isinstance(result, tuple):
        result = (result,)
    err = result[0]
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{label}: {err}")
    return result[1] if len(result) > 1 else None


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
        from cuda.bindings import driver as drv  # type: ignore[attr-defined]

        torch.cuda.init()
        idx = _device_index(self.device)
        torch.cuda.set_device(idx)
        # Touch the caching allocator so the process has a current context.
        torch.zeros(1, device=self.device)

        prop = drv.CUmemAllocationProp()
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = idx
        prop.requestedHandleTypes = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        gran = int(
            _check_drv(
                drv.cuMemGetAllocationGranularity(
                    prop,
                    drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                ),
                "cuMemGetAllocationGranularity",
            )
        )
        if gran < 1:
            raise RuntimeError("CUDA VMM granularity is 0.")

        access = drv.CUmemAccessDesc()
        access.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = idx
        access.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

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
