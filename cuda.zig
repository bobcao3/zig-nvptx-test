//! Host access to an NVIDIA GPU through the CUDA driver API. The functions load the driver
//! library (`libcuda.so.1` or `nvcuda.dll`) at run time, so programs build without the CUDA
//! toolkit.

const builtin = @import("builtin");
const std = @import("std");

// The CUDA driver API declares its functions `__stdcall` on Windows. Compilers ignore
// `__stdcall` on x86_64 and aarch64, so `callconv(.c)` matches the driver on every supported host.
comptime {
    if (builtin.os.tag == .windows and builtin.cpu.arch == .x86) @compileError("32-bit Windows has no CUDA driver");
}

pub const library_name = if (builtin.os.tag == .windows) "nvcuda.dll" else "libcuda.so.1";

pub const DevicePointer = u64;
pub const Handle = ?*anyopaque;
pub const Result = c_int;
pub const error_not_ready = 600;

pub const Error = DriverError || error{ DriverNotFound, Unexpected };

/// The driver results that `check` turns into errors. The name of each error is the name of its
/// `CUresult` value without the `CUDA_ERROR_` prefix, converted to TitleCase.
pub const DriverError = error{
    InvalidValue,
    OutOfMemory,
    NotInitialized,
    NoDevice,
    InvalidDevice,
    InvalidImage,
    InvalidContext,
    NoBinaryForGpu,
    InvalidPtx,
    UnsupportedPtxVersion,
    InvalidHandle,
    NotFound,
    IllegalAddress,
    LaunchOutOfResources,
    LaunchTimeout,
    Assert,
    HardwareStackError,
    IllegalInstruction,
    MisalignedAddress,
    InvalidAddressSpace,
    InvalidPc,
    LaunchFailed,
    CooperativeLaunchTooLarge,
    NotSupported,
};

pub const JitOption = enum(c_int) {
    error_log_buffer = 5,
    error_log_buffer_size_bytes = 6,
    _,

    pub const c_prefix = "CU_JIT_";
};

pub const JitInput = enum(c_int) {
    cubin = 0,
    ptx = 1,
    fatbinary = 2,
    object = 3,
    library = 4,
    _,

    pub const c_prefix = "CU_JIT_INPUT_";
};

pub const TensorMapDataType = enum(c_int) {
    uint8 = 0,
    uint16 = 1,
    uint32 = 2,
    int32 = 3,
    uint64 = 4,
    int64 = 5,
    float16 = 6,
    float32 = 7,
    float64 = 8,
    bfloat16 = 9,
    float32_ftz = 10,
    tfloat32 = 11,
    tfloat32_ftz = 12,
    @"16u4_align8b" = 13,
    @"16u4_align16b" = 14,
    @"16u6_align16b" = 15,
    _,

    pub const c_prefix = "CU_TENSOR_MAP_DATA_TYPE_";
};

pub const TensorMapInterleave = enum(c_int) {
    none = 0,
    @"16b" = 1,
    @"32b" = 2,

    pub const c_prefix = "CU_TENSOR_MAP_INTERLEAVE_";
};

pub const TensorMapSwizzle = enum(c_int) {
    none = 0,
    @"32b" = 1,
    @"64b" = 2,
    @"128b" = 3,
    @"128b_atom_32b" = 4,
    @"128b_atom_32b_flip_8b" = 5,
    @"128b_atom_64b" = 6,

    pub const c_prefix = "CU_TENSOR_MAP_SWIZZLE_";
};

pub const TensorMapL2Promotion = enum(c_int) {
    none = 0,
    l2_64b = 1,
    l2_128b = 2,
    l2_256b = 3,

    pub const c_prefix = "CU_TENSOR_MAP_L2_PROMOTION_";
};

pub const TensorMapFloatOobFill = enum(c_int) {
    none = 0,
    nan_request_zero_fma = 1,

    pub const c_prefix = "CU_TENSOR_MAP_FLOAT_OOB_FILL_";
};

pub const TensorMap = extern struct {
    opaque_words: [16]u64 align(128),

    pub const c_type = "CUtensorMap";

    pub const Tiled = struct {
        data_type: TensorMapDataType,
        address: DevicePointer,
        /// Lists dimensions from the fastest-varying axis to the slowest-varying axis.
        dims: []const u64,
        /// Lists byte strides for every axis after the fastest-varying axis.
        strides: []const u64,
        box: []const u32,
        element_strides: ?[]const u32 = null,
        interleave: TensorMapInterleave = .none,
        swizzle: TensorMapSwizzle = .none,
        l2_promotion: TensorMapL2Promotion = .none,
        oob_fill: TensorMapFloatOobFill = .none,
    };

    pub fn encodeTiled(config: Tiled) Error!TensorMap {
        if (config.address == 0 or config.dims.len == 0 or config.dims.len > 5 or
            config.strides.len != config.dims.len - 1 or config.box.len != config.dims.len)
            return error.InvalidValue;
        const ones: [5]u32 = @splat(1);
        const element_strides = config.element_strides orelse ones[0..config.dims.len];
        if (element_strides.len != config.dims.len) return error.InvalidValue;
        const encode = api.cuTensorMapEncodeTiled orelse return error.NotSupported;
        var map: TensorMap = undefined;
        try check(encode(
            &map,
            config.data_type,
            @intCast(config.dims.len),
            @ptrFromInt(config.address),
            config.dims.ptr,
            config.strides.ptr,
            config.box.ptr,
            element_strides.ptr,
            config.interleave,
            config.swizzle,
            config.l2_promotion,
            config.oob_fill,
        ));
        return map;
    }
};

pub const MemoryAttach = enum(c_uint) {
    global = 1,
    _,

    pub const c_prefix = "CU_MEM_ATTACH_";
};

pub const StreamFlags = enum(c_uint) {
    default = 0,
    non_blocking = 1,

    pub const c_prefix = "CU_STREAM_";
};

pub const DeviceAttribute = enum(c_int) {
    max_threads_per_block = 1,
    max_block_dim_x = 2,
    max_block_dim_y = 3,
    max_block_dim_z = 4,
    max_grid_dim_x = 5,
    max_grid_dim_y = 6,
    max_grid_dim_z = 7,
    max_shared_memory_per_block = 8,
    total_constant_memory = 9,
    warp_size = 10,
    max_pitch = 11,
    max_registers_per_block = 12,
    clock_rate = 13,
    texture_alignment = 14,
    gpu_overlap = 15,
    multiprocessor_count = 16,
    kernel_exec_timeout = 17,
    integrated = 18,
    can_map_host_memory = 19,
    compute_mode = 20,
    maximum_texture1d_width = 21,
    maximum_texture2d_width = 22,
    maximum_texture2d_height = 23,
    maximum_texture3d_width = 24,
    maximum_texture3d_height = 25,
    maximum_texture3d_depth = 26,
    maximum_texture2d_layered_width = 27,
    maximum_texture2d_layered_height = 28,
    maximum_texture2d_layered_layers = 29,
    surface_alignment = 30,
    concurrent_kernels = 31,
    ecc_enabled = 32,
    pci_bus_id = 33,
    pci_device_id = 34,
    tcc_driver = 35,
    memory_clock_rate = 36,
    global_memory_bus_width = 37,
    l2_cache_size = 38,
    max_threads_per_multiprocessor = 39,
    async_engine_count = 40,
    unified_addressing = 41,
    maximum_texture1d_layered_width = 42,
    maximum_texture1d_layered_layers = 43,
    can_tex2d_gather = 44,
    maximum_texture2d_gather_width = 45,
    maximum_texture2d_gather_height = 46,
    maximum_texture3d_width_alternate = 47,
    maximum_texture3d_height_alternate = 48,
    maximum_texture3d_depth_alternate = 49,
    pci_domain_id = 50,
    texture_pitch_alignment = 51,
    maximum_texturecubemap_width = 52,
    maximum_texturecubemap_layered_width = 53,
    maximum_texturecubemap_layered_layers = 54,
    maximum_surface1d_width = 55,
    maximum_surface2d_width = 56,
    maximum_surface2d_height = 57,
    maximum_surface3d_width = 58,
    maximum_surface3d_height = 59,
    maximum_surface3d_depth = 60,
    maximum_surface1d_layered_width = 61,
    maximum_surface1d_layered_layers = 62,
    maximum_surface2d_layered_width = 63,
    maximum_surface2d_layered_height = 64,
    maximum_surface2d_layered_layers = 65,
    maximum_surfacecubemap_width = 66,
    maximum_surfacecubemap_layered_width = 67,
    maximum_surfacecubemap_layered_layers = 68,
    maximum_texture1d_linear_width = 69,
    maximum_texture2d_linear_width = 70,
    maximum_texture2d_linear_height = 71,
    maximum_texture2d_linear_pitch = 72,
    maximum_texture2d_mipmapped_width = 73,
    maximum_texture2d_mipmapped_height = 74,
    compute_capability_major = 75,
    compute_capability_minor = 76,
    maximum_texture1d_mipmapped_width = 77,
    stream_priorities_supported = 78,
    global_l1_cache_supported = 79,
    local_l1_cache_supported = 80,
    max_shared_memory_per_multiprocessor = 81,
    max_registers_per_multiprocessor = 82,
    managed_memory = 83,
    multi_gpu_board = 84,
    multi_gpu_board_group_id = 85,
    host_native_atomic_supported = 86,
    single_to_double_precision_perf_ratio = 87,
    pageable_memory_access = 88,
    concurrent_managed_access = 89,
    compute_preemption_supported = 90,
    can_use_host_pointer_for_registered_mem = 91,
    cooperative_launch = 95,
    max_shared_memory_per_block_optin = 97,
    can_flush_remote_writes = 98,
    host_register_supported = 99,
    pageable_memory_access_uses_host_page_tables = 100,
    direct_managed_mem_access_from_host = 101,
    virtual_address_management_supported = 102,
    handle_type_posix_file_descriptor_supported = 103,
    handle_type_win32_handle_supported = 104,
    handle_type_win32_kmt_handle_supported = 105,
    max_blocks_per_multiprocessor = 106,
    generic_compression_supported = 107,
    max_persisting_l2_cache_size = 108,
    max_access_policy_window_size = 109,
    gpu_direct_rdma_with_cuda_vmm_supported = 110,
    reserved_shared_memory_per_block = 111,
    _,

    pub const c_prefix = "CU_DEVICE_ATTRIBUTE_";
};

pub const FunctionAttribute = enum(c_int) {
    max_threads_per_block = 0,
    shared_size_bytes = 1,
    const_size_bytes = 2,
    local_size_bytes = 3,
    num_regs = 4,
    ptx_version = 5,
    binary_version = 6,
    cache_mode_ca = 7,
    max_dynamic_shared_size_bytes = 8,
    preferred_shared_memory_carveout = 9,
    _,

    pub const c_prefix = "CU_FUNC_ATTRIBUTE_";
};

pub const CacheConfig = enum(c_int) {
    prefer_none = 0,
    prefer_shared = 1,
    prefer_l1 = 2,
    prefer_equal = 3,

    pub const c_prefix = "CU_FUNC_CACHE_";
};

pub const Limit = enum(c_int) {
    stack_size = 0,
    printf_fifo_size = 1,
    malloc_heap_size = 2,
    _,

    pub const c_prefix = "CU_LIMIT_";
};

pub const ArrayFormat = enum(c_uint) {
    unsigned_int8 = 0x01,
    unsigned_int16 = 0x02,
    unsigned_int32 = 0x03,
    signed_int8 = 0x08,
    signed_int16 = 0x09,
    signed_int32 = 0x0a,
    half = 0x10,
    float = 0x20,
    _,

    /// Returns the size of one channel in bytes. Asserts that the format is one of the listed
    /// members.
    pub fn channelBytes(format: ArrayFormat) usize {
        return switch (format) {
            .unsigned_int8, .signed_int8 => 1,
            .unsigned_int16, .signed_int16, .half => 2,
            .unsigned_int32, .signed_int32, .float => 4,
            _ => unreachable,
        };
    }

    pub const c_prefix = "CU_AD_FORMAT_";
};

pub const AddressMode = enum(c_uint) {
    wrap = 0,
    clamp = 1,
    mirror = 2,
    border = 3,

    pub const c_prefix = "CU_TR_ADDRESS_MODE_";
};

pub const FilterMode = enum(c_uint) {
    point = 0,
    linear = 1,

    pub const c_prefix = "CU_TR_FILTER_MODE_";
};

/// `element_type` returns the stored values. `normalized_float` converts 8-bit and 16-bit
/// integer elements to floating point numbers in [0, 1] or [-1, 1].
pub const ReadMode = enum { element_type, normalized_float };

pub const TextureDescriptor = struct {
    address_mode: [3]AddressMode = @splat(.clamp),
    filter_mode: FilterMode = .point,
    read_mode: ReadMode = .element_type,
    normalized_coordinates: bool = false,
    srgb: bool = false,
    seamless_cubemap: bool = false,
    border_color: [4]f32 = @splat(0),
};

pub const ArrayDescriptor = extern struct {
    width: usize,
    height: usize = 0,
    format: ArrayFormat,
    channels: c_uint,

    pub const c_type = "CUDA_ARRAY_DESCRIPTOR";
};

pub const ArrayFlags = packed struct(c_uint) {
    layered: bool = false,
    surface_ldst: bool = false,
    cubemap: bool = false,
    texture_gather: bool = false,
    depth_texture: bool = false,
    color_attachment: bool = false,
    reserved: u26 = 0,

    pub const c_prefix = "CUDA_ARRAY3D_";
};

/// A `depth` of 0 describes a one-dimensional or two-dimensional array. With `layered`, `depth`
/// counts the layers and `height` is 0 for a layered one-dimensional array. A cubemap has equal
/// `width` and `height`, and a `depth` of 6 (a multiple of 6 when it is also layered).
pub const Array3dDescriptor = extern struct {
    width: usize,
    height: usize = 0,
    depth: usize = 0,
    format: ArrayFormat,
    channels: c_uint,
    flags: ArrayFlags = .{},

    pub const c_type = "CUDA_ARRAY3D_DESCRIPTOR";
};

pub const MemoryType = enum(c_uint) {
    host = 1,
    device = 2,
    array = 3,

    pub const c_prefix = "CU_MEMORYTYPE_";
};

pub const Memcpy2D = extern struct {
    src_x_bytes: usize = 0,
    src_y: usize = 0,
    src_memory_type: MemoryType,
    src_host: ?*const anyopaque = null,
    src_device: DevicePointer = 0,
    src_array: Handle = null,
    src_pitch: usize = 0,
    dst_x_bytes: usize = 0,
    dst_y: usize = 0,
    dst_memory_type: MemoryType,
    dst_host: ?*anyopaque = null,
    dst_device: DevicePointer = 0,
    dst_array: Handle = null,
    dst_pitch: usize = 0,
    width_bytes: usize,
    height: usize,

    pub const c_type = "CUDA_MEMCPY2D";
};

pub const Memcpy3D = extern struct {
    src_x_bytes: usize = 0,
    src_y: usize = 0,
    src_z: usize = 0,
    src_lod: usize = 0,
    src_memory_type: MemoryType,
    src_host: ?*const anyopaque = null,
    src_device: DevicePointer = 0,
    src_array: Handle = null,
    reserved0: ?*anyopaque = null,
    src_pitch: usize = 0,
    src_height: usize = 0,
    dst_x_bytes: usize = 0,
    dst_y: usize = 0,
    dst_z: usize = 0,
    dst_lod: usize = 0,
    dst_memory_type: MemoryType,
    dst_host: ?*anyopaque = null,
    dst_device: DevicePointer = 0,
    dst_array: Handle = null,
    reserved1: ?*anyopaque = null,
    dst_pitch: usize = 0,
    dst_height: usize = 0,
    width_bytes: usize,
    height: usize,
    depth: usize,

    pub const c_type = "CUDA_MEMCPY3D";
};

pub const ResourceKind = enum(c_uint) {
    array = 0,
    linear = 2,
    pitch2d = 3,

    pub const c_prefix = "CU_RESOURCE_TYPE_";
};

pub const RawResourceDescriptor = extern struct {
    kind: ResourceKind,
    res: extern union {
        array: extern struct { array: Handle },
        linear: extern struct {
            address: DevicePointer,
            format: ArrayFormat,
            channels: c_uint,
            size_bytes: usize,
        },
        pitch2d: extern struct {
            address: DevicePointer,
            format: ArrayFormat,
            channels: c_uint,
            width: usize,
            height: usize,
            pitch_bytes: usize,
        },
        reserved: [32]c_int,
    },
    flags: c_uint = 0,

    pub const c_type = "CUDA_RESOURCE_DESC";
};

pub const TextureFlags = packed struct(c_uint) {
    read_as_integer: bool,
    normalized_coordinates: bool,
    reserved_low: u2 = 0,
    srgb: bool,
    disable_trilinear_optimization: bool = false,
    seamless_cubemap: bool,
    reserved_high: u25 = 0,

    pub const c_prefix = "CU_TRSF_";
};

pub const RawTextureDescriptor = extern struct {
    address_mode: [3]AddressMode,
    filter_mode: FilterMode,
    flags: TextureFlags,
    max_anisotropy: c_uint = 0,
    mipmap_filter_mode: FilterMode = .point,
    mipmap_level_bias: f32 = 0,
    min_mipmap_level_clamp: f32 = 0,
    max_mipmap_level_clamp: f32 = 0,
    border_color: [4]f32,
    reserved: [12]c_int = @splat(0),

    pub const c_type = "CUDA_TEXTURE_DESC";
};

pub const Raw = struct {
    cuInit: *const fn (flags: c_uint) callconv(.c) Result,
    cuGetErrorName: *const fn (result: Result, name: *[*:0]const u8) callconv(.c) Result,
    cuGetErrorString: *const fn (result: Result, text: *[*:0]const u8) callconv(.c) Result,
    cuDeviceGet: *const fn (device: *c_int, ordinal: c_int) callconv(.c) Result,
    cuDeviceGetName: *const fn (name: [*]u8, len: c_int, device: c_int) callconv(.c) Result,
    cuDeviceGetAttribute: *const fn (value: *c_int, attribute: DeviceAttribute, device: c_int) callconv(.c) Result,
    cuDeviceTotalMem_v2: *const fn (bytes: *usize, device: c_int) callconv(.c) Result,
    cuDevicePrimaryCtxRetain: *const fn (context: *Handle, device: c_int) callconv(.c) Result,
    cuDevicePrimaryCtxRelease_v2: *const fn (device: c_int) callconv(.c) Result,
    cuCtxSetCurrent: *const fn (context: Handle) callconv(.c) Result,
    cuCtxSynchronize: *const fn () callconv(.c) Result,
    cuCtxSetLimit: *const fn (limit: Limit, value: usize) callconv(.c) Result,
    cuCtxGetLimit: *const fn (value: *usize, limit: Limit) callconv(.c) Result,
    cuModuleLoadData: *const fn (module: *Handle, image: *const anyopaque) callconv(.c) Result,
    cuModuleLoadDataEx: *const fn (
        module: *Handle,
        image: *const anyopaque,
        options_len: c_uint,
        options: [*]const JitOption,
        option_values: [*]const ?*anyopaque,
    ) callconv(.c) Result,
    cuModuleUnload: *const fn (module: Handle) callconv(.c) Result,
    cuModuleGetFunction: *const fn (function: *Handle, module: Handle, name: [*:0]const u8) callconv(.c) Result,
    cuModuleGetGlobal_v2: *const fn (pointer: *DevicePointer, size: *usize, module: Handle, name: [*:0]const u8) callconv(.c) Result,
    cuLinkCreate_v2: *const fn (
        options_len: c_uint,
        options: [*]const JitOption,
        option_values: [*]const ?*anyopaque,
        link: *Handle,
    ) callconv(.c) Result,
    cuLinkAddFile_v2: *const fn (
        link: Handle,
        input_type: JitInput,
        path: [*:0]const u8,
        options_len: c_uint,
        options: ?[*]const JitOption,
        option_values: ?[*]const ?*anyopaque,
    ) callconv(.c) Result,
    cuLinkComplete: *const fn (link: Handle, cubin: **anyopaque, size: *usize) callconv(.c) Result,
    cuLinkDestroy: *const fn (link: Handle) callconv(.c) Result,
    cuFuncGetAttribute: *const fn (value: *c_int, attribute: FunctionAttribute, function: Handle) callconv(.c) Result,
    cuFuncSetAttribute: *const fn (function: Handle, attribute: FunctionAttribute, value: c_int) callconv(.c) Result,
    cuFuncSetCacheConfig: *const fn (function: Handle, config: CacheConfig) callconv(.c) Result,
    cuOccupancyMaxActiveBlocksPerMultiprocessor: *const fn (
        blocks: *c_int,
        function: Handle,
        block_size: c_int,
        dynamic_shared_memory_size: usize,
    ) callconv(.c) Result,
    cuOccupancyMaxPotentialBlockSize: *const fn (
        min_grid_size: *c_int,
        block_size: *c_int,
        function: Handle,
        block_size_to_dynamic_shared_memory_size: ?*const fn (block_size: c_int) callconv(.c) usize,
        dynamic_shared_memory_size: usize,
        block_size_limit: c_int,
    ) callconv(.c) Result,
    cuMemAlloc_v2: *const fn (pointer: *DevicePointer, size: usize) callconv(.c) Result,
    cuMemAllocManaged: *const fn (pointer: *DevicePointer, size: usize, flags: c_uint) callconv(.c) Result,
    cuMemFree_v2: *const fn (pointer: DevicePointer) callconv(.c) Result,
    cuMemsetD8_v2: *const fn (pointer: DevicePointer, value: u8, count: usize) callconv(.c) Result,
    cuMemsetD8Async: *const fn (pointer: DevicePointer, value: u8, count: usize, stream: Handle) callconv(.c) Result,
    cuMemsetD32_v2: *const fn (pointer: DevicePointer, value: u32, count: usize) callconv(.c) Result,
    cuMemcpyHtoD_v2: *const fn (destination: DevicePointer, source: *const anyopaque, size: usize) callconv(.c) Result,
    cuMemcpyDtoH_v2: *const fn (destination: *anyopaque, source: DevicePointer, size: usize) callconv(.c) Result,
    cuMemcpyDtoD_v2: *const fn (destination: DevicePointer, source: DevicePointer, size: usize) callconv(.c) Result,
    cuMemcpyHtoDAsync_v2: *const fn (destination: DevicePointer, source: *const anyopaque, size: usize, stream: Handle) callconv(.c) Result,
    cuMemcpyDtoHAsync_v2: *const fn (destination: *anyopaque, source: DevicePointer, size: usize, stream: Handle) callconv(.c) Result,
    cuArrayCreate_v2: *const fn (array: *Handle, descriptor: *const ArrayDescriptor) callconv(.c) Result,
    cuArray3DCreate_v2: *const fn (array: *Handle, descriptor: *const Array3dDescriptor) callconv(.c) Result,
    cuArrayDestroy: *const fn (array: Handle) callconv(.c) Result,
    cuMemcpy2D_v2: *const fn (copy: *const Memcpy2D) callconv(.c) Result,
    cuMemcpy3D_v2: *const fn (copy: *const Memcpy3D) callconv(.c) Result,
    cuTexObjectCreate: *const fn (
        texture: *TextureObject,
        resource: *const RawResourceDescriptor,
        texture_descriptor: *const RawTextureDescriptor,
        view_descriptor: ?*const anyopaque,
    ) callconv(.c) Result,
    cuTexObjectDestroy: *const fn (texture: TextureObject) callconv(.c) Result,
    cuSurfObjectCreate: *const fn (surface: *SurfaceObject, resource: *const RawResourceDescriptor) callconv(.c) Result,
    cuSurfObjectDestroy: *const fn (surface: SurfaceObject) callconv(.c) Result,
    cuStreamCreate: *const fn (stream: *Handle, flags: c_uint) callconv(.c) Result,
    cuStreamDestroy_v2: *const fn (stream: Handle) callconv(.c) Result,
    cuStreamSynchronize: *const fn (stream: Handle) callconv(.c) Result,
    cuStreamQuery: *const fn (stream: Handle) callconv(.c) Result,
    cuEventCreate: *const fn (event: *Handle, flags: c_uint) callconv(.c) Result,
    cuEventDestroy_v2: *const fn (event: Handle) callconv(.c) Result,
    cuEventRecord: *const fn (event: Handle, stream: Handle) callconv(.c) Result,
    cuEventSynchronize: *const fn (event: Handle) callconv(.c) Result,
    cuEventElapsedTime: *const fn (milliseconds: *f32, start: Handle, end: Handle) callconv(.c) Result,
    cuLaunchKernel: *const fn (
        function: Handle,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_memory_size: c_uint,
        stream: Handle,
        params: [*]const *const anyopaque,
        extra: ?[*]const ?*anyopaque,
    ) callconv(.c) Result,
    cuLaunchCooperativeKernel: *const fn (
        function: Handle,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_memory_size: c_uint,
        stream: Handle,
        params: [*]const *const anyopaque,
    ) callconv(.c) Result,
    cuTensorMapEncodeTiled: ?*const fn (
        map: *TensorMap,
        data_type: TensorMapDataType,
        rank: u32,
        address: *anyopaque,
        dims: [*]const u64,
        strides: [*]const u64,
        box: [*]const u32,
        element_strides: [*]const u32,
        interleave: TensorMapInterleave,
        swizzle: TensorMapSwizzle,
        l2_promotion: TensorMapL2Promotion,
        oob_fill: TensorMapFloatOobFill,
    ) callconv(.c) Result,

    pub fn load() error{DriverNotFound}!Raw {
        var library = Library.open() catch return error.DriverNotFound;
        var raw: Raw = undefined;
        const info = @typeInfo(Raw).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(raw, name) = if (@typeInfo(T) == .optional)
                library.lookup(@typeInfo(T).optional.child, name)
            else
                library.lookup(T, name) orelse return error.DriverNotFound;
        }
        return raw;
    }

    pub fn errorName(raw: *const Raw, result: Result) []const u8 {
        var name: [*:0]const u8 = undefined;
        if (raw.cuGetErrorName(result, &name) != 0) return "unknown CUDA error";
        return std.mem.span(name);
    }

    pub fn errorString(raw: *const Raw, result: Result) []const u8 {
        var text: [*:0]const u8 = undefined;
        if (raw.cuGetErrorString(result, &text) != 0) return "unknown CUDA error";
        return std.mem.span(text);
    }

    pub fn check(raw: *const Raw, result: Result, comptime what: []const u8) void {
        if (result != 0) std.process.fatal("{s} failed: {s}", .{ what, raw.errorName(result) });
    }

    pub fn initDevice(raw: *const Raw) void {
        raw.check(raw.cuInit(0), "cuInit");
        var device: c_int = undefined;
        raw.check(raw.cuDeviceGet(&device, 0), "cuDeviceGet");
        var context: Handle = null;
        raw.check(raw.cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
        raw.check(raw.cuCtxSetCurrent(context), "cuCtxSetCurrent");
    }

    pub fn computeCapability(raw: *const Raw) struct { major: u32, minor: u32 } {
        var device: c_int = undefined;
        raw.check(raw.cuDeviceGet(&device, 0), "cuDeviceGet");
        var major: c_int = undefined;
        var minor: c_int = undefined;
        raw.check(raw.cuDeviceGetAttribute(&major, .compute_capability_major, device), "cuDeviceGetAttribute");
        raw.check(raw.cuDeviceGetAttribute(&minor, .compute_capability_minor, device), "cuDeviceGetAttribute");
        return .{ .major = @intCast(major), .minor = @intCast(minor) };
    }

    pub fn loadModule(raw: *const Raw, ptx: [:0]const u8, ptx_path: [:0]const u8, link_paths: []const [:0]const u8) Handle {
        var log: [16 * 1024]u8 = @splat(0);
        const options = [_]JitOption{ .error_log_buffer, .error_log_buffer_size_bytes };
        const option_values = [_]?*anyopaque{ &log, @ptrFromInt(log.len) };
        var module: Handle = null;
        if (link_paths.len == 0) {
            const result = raw.cuModuleLoadDataEx(&module, ptx.ptr, options.len, &options, &option_values);
            if (result != 0) std.process.fatal("cuModuleLoadDataEx failed: {s}\n{s}", .{
                raw.errorName(result), std.mem.sliceTo(&log, 0),
            });
            return module;
        }
        var link: Handle = null;
        raw.check(raw.cuLinkCreate_v2(options.len, &options, &option_values, &link), "cuLinkCreate");
        defer raw.check(raw.cuLinkDestroy(link), "cuLinkDestroy");
        for (0..link_paths.len + 1) |index| {
            const path = if (index == 0) ptx_path else link_paths[index - 1];
            const result = raw.cuLinkAddFile_v2(link, .ptx, path, 0, null, null);
            if (result != 0) std.process.fatal("cuLinkAddFile {s} failed: {s}\n{s}", .{
                path, raw.errorName(result), std.mem.sliceTo(&log, 0),
            });
        }
        var cubin: *anyopaque = undefined;
        var cubin_size: usize = undefined;
        const result = raw.cuLinkComplete(link, &cubin, &cubin_size);
        if (result != 0) std.process.fatal("cuLinkComplete failed: {s}\n{s}", .{
            raw.errorName(result), std.mem.sliceTo(&log, 0),
        });
        raw.check(raw.cuModuleLoadData(&module, cubin), "cuModuleLoadData");
        return module;
    }
};

var api: Raw = undefined;

fn check(result: Result) Error!void {
    if (result == 0) return;
    var name: [*:0]const u8 = undefined;
    var text: [*:0]const u8 = undefined;
    if (api.cuGetErrorName(result, &name) != 0) name = "unknown CUDA error";
    if (api.cuGetErrorString(result, &text) != 0) text = "";
    std.log.err("CUDA driver: {s}: {s}", .{ name, text });
    inline for (@typeInfo(DriverError).error_set.error_names.?) |error_name| {
        if (std.mem.eql(u8, "CUDA_ERROR_" ++ comptime screamingSnakeCase(error_name), std.mem.span(name))) {
            return @field(DriverError, error_name);
        }
    }
    return error.Unexpected;
}

pub const Device = struct {
    ordinal: c_int,
    context: Handle,

    pub fn deinit(device: Device) void {
        _ = api.cuDevicePrimaryCtxRelease_v2(device.ordinal);
    }

    pub fn attribute(device: Device, which: DeviceAttribute) Error!i32 {
        var value: c_int = undefined;
        try check(api.cuDeviceGetAttribute(&value, which, device.ordinal));
        return value;
    }

    /// Writes the device name into `buffer` and returns the written part.
    pub fn name(device: Device, buffer: []u8) Error![]const u8 {
        try check(api.cuDeviceGetName(buffer.ptr, @intCast(buffer.len), device.ordinal));
        return std.mem.sliceTo(buffer, 0);
    }

    /// Returns the size of the device memory in bytes.
    pub fn totalMemory(device: Device) Error!usize {
        var bytes: usize = undefined;
        try check(api.cuDeviceTotalMem_v2(&bytes, device.ordinal));
        return bytes;
    }

    pub fn computeCapability(device: Device) Error!struct { major: i32, minor: i32 } {
        return .{
            .major = try device.attribute(.compute_capability_major),
            .minor = try device.attribute(.compute_capability_minor),
        };
    }
};

/// Loads the driver and makes the primary context of device `ordinal` current on the calling
/// thread.
pub fn init(ordinal: c_int) Error!Device {
    api = try Raw.load();
    try check(api.cuInit(0));
    var device: Device = .{ .ordinal = undefined, .context = null };
    try check(api.cuDeviceGet(&device.ordinal, ordinal));
    try check(api.cuDevicePrimaryCtxRetain(&device.context, device.ordinal));
    try check(api.cuCtxSetCurrent(device.context));
    return device;
}

pub fn synchronize() Error!void {
    try check(api.cuCtxSynchronize());
}

pub fn setLimit(limit: Limit, value: usize) Error!void {
    try check(api.cuCtxSetLimit(limit, value));
}

pub fn getLimit(limit: Limit) Error!usize {
    var value: usize = undefined;
    try check(api.cuCtxGetLimit(&value, limit));
    return value;
}

/// Returns the number of cores of one multiprocessor of the given compute capability, as the
/// CUDA samples do. A compute capability that the table lacks gets the count of the newest entry.
pub fn coresPerMultiprocessor(major: i32, minor: i32) i32 {
    const table = [_]struct { u8, i32 }{
        .{ 0x30, 192 }, .{ 0x32, 192 }, .{ 0x35, 192 }, .{ 0x37, 192 }, .{ 0x50, 128 },
        .{ 0x52, 128 }, .{ 0x53, 128 }, .{ 0x60, 64 },  .{ 0x61, 128 }, .{ 0x62, 128 },
        .{ 0x70, 64 },  .{ 0x72, 64 },  .{ 0x75, 64 },  .{ 0x80, 64 },  .{ 0x86, 128 },
        .{ 0x87, 128 }, .{ 0x89, 128 }, .{ 0x90, 128 }, .{ 0xa0, 128 }, .{ 0xa1, 128 },
        .{ 0xa3, 128 }, .{ 0xa7, 128 }, .{ 0xb0, 128 }, .{ 0xc0, 128 }, .{ 0xc1, 128 },
    };
    for (table) |entry| {
        if (entry[0] == (major << 4) + minor) return entry[1];
    }
    std.log.warn("MapSMtoCores for SM {d}.{d} is undefined.  Default to use {d} Cores/SM", .{
        major,
        minor,
        table[table.len - 1][1],
    });
    return table[table.len - 1][1];
}

/// `len` elements of type `T` in device memory.
pub fn Buffer(comptime T: type) type {
    return struct {
        address: DevicePointer,
        len: usize,
        /// The driver allocates managed memory, which the host can access as `hostSlice()`.
        managed: bool = false,

        const Self = @This();
        pub const Element = T;

        pub fn free(buffer: Self) void {
            _ = api.cuMemFree_v2(buffer.address);
        }

        pub fn bytes(buffer: Self) usize {
            return buffer.len * @sizeOf(T);
        }

        pub fn hostSlice(buffer: Self) []T {
            std.debug.assert(buffer.managed);
            const many: [*]T = @ptrFromInt(buffer.address);
            return many[0..buffer.len];
        }

        pub fn copyFrom(buffer: Self, host: []const T) Error!void {
            std.debug.assert(host.len <= buffer.len);
            try check(api.cuMemcpyHtoD_v2(buffer.address, host.ptr, host.len * @sizeOf(T)));
        }

        pub fn copyTo(buffer: Self, host: []T) Error!void {
            std.debug.assert(host.len <= buffer.len);
            try check(api.cuMemcpyDtoH_v2(host.ptr, buffer.address, host.len * @sizeOf(T)));
        }

        pub fn copyFromBuffer(buffer: Self, source: Self) Error!void {
            std.debug.assert(source.len <= buffer.len);
            try check(api.cuMemcpyDtoD_v2(buffer.address, source.address, source.bytes()));
        }

        pub fn copyFromAsync(buffer: Self, host: []const T, stream: Stream) Error!void {
            std.debug.assert(host.len <= buffer.len);
            try check(api.cuMemcpyHtoDAsync_v2(buffer.address, host.ptr, host.len * @sizeOf(T), stream.handle));
        }

        pub fn copyToAsync(buffer: Self, host: []T, stream: Stream) Error!void {
            std.debug.assert(host.len <= buffer.len);
            try check(api.cuMemcpyDtoHAsync_v2(host.ptr, buffer.address, host.len * @sizeOf(T), stream.handle));
        }

        /// Sets every byte of the buffer to `value`.
        pub fn memset(buffer: Self, value: u8) Error!void {
            try check(api.cuMemsetD8_v2(buffer.address, value, buffer.bytes()));
        }

        pub fn memsetAsync(buffer: Self, value: u8, stream: Stream) Error!void {
            try check(api.cuMemsetD8Async(buffer.address, value, buffer.bytes(), stream.handle));
        }

        /// Sets every 32-bit word of the buffer to `value`. Asserts that the size of the buffer
        /// is a multiple of 4 bytes.
        pub fn memset32(buffer: Self, value: u32) Error!void {
            std.debug.assert(buffer.bytes() % 4 == 0);
            try check(api.cuMemsetD32_v2(buffer.address, value, buffer.bytes() / 4));
        }
    };
}

pub fn alloc(comptime T: type, len: usize) Error!Buffer(T) {
    var address: DevicePointer = undefined;
    try check(api.cuMemAlloc_v2(&address, @max(len, 1) * @sizeOf(T)));
    return .{ .address = address, .len = len };
}

/// Allocates managed memory, which the host and the device can both access.
pub fn allocManaged(comptime T: type, len: usize) Error!Buffer(T) {
    var address: DevicePointer = undefined;
    try check(api.cuMemAllocManaged(&address, @max(len, 1) * @sizeOf(T), @backingInt(MemoryAttach.global)));
    return .{ .address = address, .len = len, .managed = true };
}

/// Allocates device memory and copies `host` into it.
pub fn dupe(comptime T: type, host: []const T) Error!Buffer(T) {
    const buffer = try alloc(T, host.len);
    errdefer buffer.free();
    try buffer.copyFrom(host);
    return buffer;
}

pub const Module = struct {
    handle: Handle,

    /// Compiles PTX text for the current device.
    pub fn load(ptx: [:0]const u8) Error!Module {
        var log: [16 * 1024]u8 = @splat(0);
        const options = [_]JitOption{ .error_log_buffer, .error_log_buffer_size_bytes };
        const option_values = [_]?*anyopaque{ &log, @ptrFromInt(log.len) };
        var module: Module = .{ .handle = null };
        check(api.cuModuleLoadDataEx(&module.handle, ptx.ptr, options.len, &options, &option_values)) catch |err| {
            std.log.err("PTX JIT compiler: {s}", .{std.mem.sliceTo(&log, 0)});
            return err;
        };
        return module;
    }

    pub fn unload(module: Module) void {
        _ = api.cuModuleUnload(module.handle);
    }

    pub fn function(module: Module, name: [:0]const u8) Error!Function {
        var result: Function = .{ .handle = null };
        try check(api.cuModuleGetFunction(&result.handle, module.handle, name));
        return result;
    }

    /// Returns the module-scope variable `name` (a `__constant__` or `__device__` variable in
    /// CUDA C++) as a buffer of `T`.
    pub fn global(module: Module, comptime T: type, name: [:0]const u8) Error!Buffer(T) {
        var address: DevicePointer = undefined;
        var size: usize = undefined;
        try check(api.cuModuleGetGlobal_v2(&address, &size, module.handle, name));
        return .{ .address = address, .len = size / @sizeOf(T) };
    }
};

pub const Dim3 = struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,
};

pub const LaunchConfig = struct {
    grid: Dim3,
    block: Dim3,
    shared_memory_bytes: u32 = 0,
    stream: Stream = .default,
    /// Launches with `cuLaunchCooperativeKernel`, so that all work groups run at the same time.
    cooperative: bool = false,
};

pub const Function = struct {
    handle: Handle,

    pub fn attribute(function: Function, which: FunctionAttribute) Error!i32 {
        var value: c_int = undefined;
        try check(api.cuFuncGetAttribute(&value, which, function.handle));
        return value;
    }

    pub fn setAttribute(function: Function, which: FunctionAttribute, value: i32) Error!void {
        try check(api.cuFuncSetAttribute(function.handle, which, value));
    }

    /// Allows launches with more than 48 KiB of dynamic shared memory.
    pub fn setMaxDynamicSharedMemory(function: Function, bytes: u32) Error!void {
        try function.setAttribute(.max_dynamic_shared_size_bytes, @intCast(bytes));
    }

    pub fn setCacheConfig(function: Function, config: CacheConfig) Error!void {
        try check(api.cuFuncSetCacheConfig(function.handle, config));
    }

    /// Returns the block size that gives the highest occupancy, and the smallest grid size that
    /// reaches that occupancy on the whole device. A `block_size_limit` of 0 sets no limit.
    pub fn maxPotentialBlockSize(
        function: Function,
        dynamic_shared_memory_bytes: usize,
        block_size_limit: u32,
    ) Error!struct { min_grid_size: u32, block_size: u32 } {
        var min_grid_size: c_int = undefined;
        var block_size: c_int = undefined;
        try check(api.cuOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &block_size,
            function.handle,
            null,
            dynamic_shared_memory_bytes,
            @intCast(block_size_limit),
        ));
        return .{ .min_grid_size = @intCast(min_grid_size), .block_size = @intCast(block_size) };
    }

    pub fn maxActiveBlocksPerMultiprocessor(
        function: Function,
        block_size: u32,
        dynamic_shared_memory_bytes: usize,
    ) Error!u32 {
        var blocks: c_int = undefined;
        try check(api.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks,
            function.handle,
            @intCast(block_size),
            dynamic_shared_memory_bytes,
        ));
        return @intCast(blocks);
    }

    /// Launches the kernel. `args` is a tuple with one field per kernel parameter. A `Buffer`
    /// passes its device address, `null` and a null `?Buffer` pass a null device address, and
    /// every other field passes its bytes. A `TextureObject` or `SurfaceObject` passes its handle.
    pub fn launch(function: Function, config: LaunchConfig, args: anytype) Error!void {
        try function.launchKernel(config, config.cooperative, args);
    }

    /// Launches the kernel so that all work groups run at the same time, whatever
    /// `config.cooperative` says.
    pub fn launchCooperative(function: Function, config: LaunchConfig, args: anytype) Error!void {
        try function.launchKernel(config, true, args);
    }

    fn launchKernel(function: Function, config: LaunchConfig, cooperative: bool, args: anytype) Error!void {
        const types = @typeInfo(@TypeOf(args)).@"struct".field_types;
        const Params = comptime params: {
            var param_types: [types.len]type = undefined;
            for (types, &param_types, 0..) |T, *P, index| {
                switch (@typeInfo(T)) {
                    .comptime_int, .comptime_float => @compileError(std.fmt.comptimePrint(
                        "kernel argument {d} has no runtime type; cast it to the type of the kernel parameter",
                        .{index},
                    )),
                    else => P.* = if (isDevicePointerArgument(T)) DevicePointer else T,
                }
            }
            break :params @Tuple(&param_types);
        };
        var values: Params = undefined;
        var params: [types.len + 1]*const anyopaque = undefined;
        inline for (types, 0..) |T, index| {
            values[index] = if (comptime isDevicePointerArgument(T)) devicePointer(args[index]) else args[index];
            params[index] = &values[index];
        }
        const g = config.grid;
        const b = config.block;
        try check(if (cooperative)
            api.cuLaunchCooperativeKernel(function.handle, g.x, g.y, g.z, b.x, b.y, b.z, config.shared_memory_bytes, config.stream.handle, &params)
        else
            api.cuLaunchKernel(function.handle, g.x, g.y, g.z, b.x, b.y, b.z, config.shared_memory_bytes, config.stream.handle, &params, null));
    }
};

fn isBuffer(comptime T: type) bool {
    return @typeInfo(T) == .@"struct" and @hasDecl(T, "Element") and T == Buffer(T.Element);
}

fn isDevicePointerArgument(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .null => true,
        .optional => |optional| isBuffer(optional.child),
        else => isBuffer(T),
    };
}

fn devicePointer(argument: anytype) DevicePointer {
    return switch (@typeInfo(@TypeOf(argument))) {
        .null => 0,
        .optional => if (argument) |buffer| buffer.address else 0,
        else => argument.address,
    };
}

pub const Stream = struct {
    handle: Handle,

    pub const default: Stream = .{ .handle = null };

    pub fn create() Error!Stream {
        var stream: Stream = undefined;
        try check(api.cuStreamCreate(&stream.handle, @backingInt(StreamFlags.non_blocking)));
        return stream;
    }

    pub fn destroy(stream: Stream) void {
        _ = api.cuStreamDestroy_v2(stream.handle);
    }

    pub fn synchronize(stream: Stream) Error!void {
        try check(api.cuStreamSynchronize(stream.handle));
    }
};

pub const Event = struct {
    handle: Handle,

    pub fn create() Error!Event {
        var event: Event = undefined;
        try check(api.cuEventCreate(&event.handle, 0));
        return event;
    }

    pub fn destroy(event: Event) void {
        _ = api.cuEventDestroy_v2(event.handle);
    }

    pub fn record(event: Event, stream: Stream) Error!void {
        try check(api.cuEventRecord(event.handle, stream.handle));
    }

    pub fn synchronize(event: Event) Error!void {
        try check(api.cuEventSynchronize(event.handle));
    }

    /// Returns the milliseconds between two recorded events.
    pub fn elapsed(start: Event, end: Event) Error!f32 {
        var milliseconds: f32 = undefined;
        try check(api.cuEventElapsedTime(&milliseconds, start.handle, end.handle));
        return milliseconds;
    }
};

/// A CUDA array: device memory in an opaque layout that texture and surface objects access.
pub const Array = struct {
    handle: Handle,
    descriptor: Array3dDescriptor,

    /// Creates a one-dimensional array when `descriptor.height` is 0, and a two-dimensional array
    /// otherwise.
    pub fn create(descriptor: ArrayDescriptor) Error!Array {
        var array: Array = .{ .handle = null, .descriptor = .{
            .width = descriptor.width,
            .height = descriptor.height,
            .format = descriptor.format,
            .channels = descriptor.channels,
        } };
        try check(api.cuArrayCreate_v2(&array.handle, &descriptor));
        return array;
    }

    pub fn create3d(descriptor: Array3dDescriptor) Error!Array {
        var array: Array = .{ .handle = null, .descriptor = descriptor };
        try check(api.cuArray3DCreate_v2(&array.handle, &descriptor));
        return array;
    }

    pub fn destroy(array: Array) void {
        _ = api.cuArrayDestroy(array.handle);
    }

    /// Returns the size of one row of the array in bytes.
    pub fn rowBytes(array: Array) usize {
        const descriptor = array.descriptor;
        return descriptor.width * descriptor.channels * descriptor.format.channelBytes();
    }

    pub fn copyFromHost(array: Array, comptime T: type, host: []const T) Error!void {
        std.debug.assert(host.len * @sizeOf(T) == array.rowBytes() * array.rows());
        try array.copy(.to_array, .{ .host = @constCast(host.ptr) }, array.rowBytes());
    }

    pub fn copyToHost(array: Array, comptime T: type, host: []T) Error!void {
        std.debug.assert(host.len * @sizeOf(T) == array.rowBytes() * array.rows());
        try array.copy(.from_array, .{ .host = host.ptr }, array.rowBytes());
    }

    /// Copies `buffer` into the array. The rows of `buffer` are `pitch_bytes` apart, and the
    /// layers of a three-dimensional array follow each other.
    pub fn copyFromBuffer(array: Array, comptime T: type, buffer: Buffer(T), pitch_bytes: usize) Error!void {
        std.debug.assert(buffer.bytes() >= pitch_bytes * array.rows());
        try array.copy(.to_array, .{ .device = buffer.address }, pitch_bytes);
    }

    /// Copies the array into `buffer`, whose rows are `pitch_bytes` apart.
    pub fn copyToBuffer(array: Array, comptime T: type, buffer: Buffer(T), pitch_bytes: usize) Error!void {
        std.debug.assert(buffer.bytes() >= pitch_bytes * array.rows());
        try array.copy(.from_array, .{ .device = buffer.address }, pitch_bytes);
    }

    fn rows(array: Array) usize {
        return @max(array.descriptor.height, 1) * @max(array.descriptor.depth, 1);
    }

    const Direction = enum { to_array, from_array };
    const Endpoint = union(enum) { host: *anyopaque, device: DevicePointer };

    fn copy(array: Array, direction: Direction, endpoint: Endpoint, pitch_bytes: usize) Error!void {
        const descriptor = array.descriptor;
        const height = @max(descriptor.height, 1);
        const memory_type: MemoryType = switch (endpoint) {
            .host => .host,
            .device => .device,
        };
        const host: ?*anyopaque = switch (endpoint) {
            .host => |pointer| pointer,
            .device => null,
        };
        const device: DevicePointer = switch (endpoint) {
            .host => 0,
            .device => |address| address,
        };
        const to_array = direction == .to_array;
        if (descriptor.depth == 0) {
            try check(api.cuMemcpy2D_v2(&.{
                .src_memory_type = if (to_array) memory_type else .array,
                .src_host = if (to_array) host else null,
                .src_device = if (to_array) device else 0,
                .src_array = if (to_array) null else array.handle,
                .src_pitch = pitch_bytes,
                .dst_memory_type = if (to_array) .array else memory_type,
                .dst_host = if (to_array) null else host,
                .dst_device = if (to_array) 0 else device,
                .dst_array = if (to_array) array.handle else null,
                .dst_pitch = pitch_bytes,
                .width_bytes = array.rowBytes(),
                .height = height,
            }));
            return;
        }
        try check(api.cuMemcpy3D_v2(&.{
            .src_memory_type = if (to_array) memory_type else .array,
            .src_host = if (to_array) host else null,
            .src_device = if (to_array) device else 0,
            .src_array = if (to_array) null else array.handle,
            .src_pitch = pitch_bytes,
            .src_height = height,
            .dst_memory_type = if (to_array) .array else memory_type,
            .dst_host = if (to_array) null else host,
            .dst_device = if (to_array) 0 else device,
            .dst_array = if (to_array) array.handle else null,
            .dst_pitch = pitch_bytes,
            .dst_height = height,
            .width_bytes = array.rowBytes(),
            .height = height,
            .depth = descriptor.depth,
        }));
    }
};

pub const ResourceDescriptor = union(enum) {
    array: Array,
    /// Device memory that a one-dimensional texture reads.
    linear: struct {
        address: DevicePointer,
        format: ArrayFormat,
        channels: u32,
        size_bytes: usize,
    },
    /// Device memory with padded rows that a two-dimensional texture or surface accesses.
    pitch2d: struct {
        address: DevicePointer,
        format: ArrayFormat,
        channels: u32,
        width: usize,
        height: usize,
        pitch_bytes: usize,
    },

    fn raw(descriptor: ResourceDescriptor) RawResourceDescriptor {
        return switch (descriptor) {
            .array => |array| .{ .kind = .array, .res = .{ .array = .{ .array = array.handle } } },
            .linear => |linear| .{ .kind = .linear, .res = .{ .linear = .{
                .address = linear.address,
                .format = linear.format,
                .channels = linear.channels,
                .size_bytes = linear.size_bytes,
            } } },
            .pitch2d => |pitch2d| .{ .kind = .pitch2d, .res = .{ .pitch2d = .{
                .address = pitch2d.address,
                .format = pitch2d.format,
                .channels = pitch2d.channels,
                .width = pitch2d.width,
                .height = pitch2d.height,
                .pitch_bytes = pitch2d.pitch_bytes,
            } } },
        };
    }
};

/// A kernel receives the handle of a texture object as a 64-bit integer parameter.
pub const TextureObject = enum(u64) {
    _,

    pub fn create(resource: ResourceDescriptor, texture: TextureDescriptor) Error!TextureObject {
        const raw_resource = resource.raw();
        const raw_texture: RawTextureDescriptor = .{
            .address_mode = texture.address_mode,
            .filter_mode = texture.filter_mode,
            .flags = .{
                .read_as_integer = texture.read_mode == .element_type,
                .normalized_coordinates = texture.normalized_coordinates,
                .srgb = texture.srgb,
                .seamless_cubemap = texture.seamless_cubemap,
            },
            .border_color = texture.border_color,
        };
        var object: TextureObject = undefined;
        try check(api.cuTexObjectCreate(&object, &raw_resource, &raw_texture, null));
        return object;
    }

    pub fn destroy(object: TextureObject) void {
        _ = api.cuTexObjectDestroy(object);
    }
};

/// A kernel receives the handle of a surface object as a 64-bit integer parameter. An array
/// resource needs the `surface_ldst` flag.
pub const SurfaceObject = enum(u64) {
    _,

    pub fn create(resource: ResourceDescriptor) Error!SurfaceObject {
        const raw_resource = resource.raw();
        var object: SurfaceObject = undefined;
        try check(api.cuSurfObjectCreate(&object, &raw_resource));
        return object;
    }

    pub fn destroy(object: SurfaceObject) void {
        _ = api.cuSurfObjectDestroy(object);
    }
};

/// `std.DynLib` does not support Windows, so on Windows the loader functions of ntdll open the
/// library.
const Library = if (builtin.os.tag == .windows) struct {
    handle: *anyopaque,

    fn open() error{FileNotFound}!Library {
        const windows = std.os.windows;
        var handle: *anyopaque = undefined;
        const name = std.unicode.utf8ToUtf16LeStringLiteral(library_name);
        return switch (windows.ntdll.LdrLoadDll(null, null, &.init(name), &handle)) {
            .SUCCESS => .{ .handle = handle },
            else => error.FileNotFound,
        };
    }

    fn lookup(library: *Library, comptime T: type, name: [:0]const u8) ?T {
        var address: *anyopaque = undefined;
        return switch (std.os.windows.ntdll.LdrGetProcedureAddress(library.handle, &.init(name), 0, &address)) {
            .SUCCESS => @ptrCast(@alignCast(address)),
            else => null,
        };
    }
} else struct {
    dyn_lib: std.DynLib,

    fn open() error{FileNotFound}!Library {
        return .{ .dyn_lib = std.DynLib.open(library_name) catch return error.FileNotFound };
    }

    fn lookup(library: *Library, comptime T: type, name: [:0]const u8) ?T {
        return library.dyn_lib.lookup(T, name);
    }
};

test "every value, layout and function signature matches cuda.h" {
    @setEvalBranchQuota(100_000);
    const c = @import("cuda_h");
    inline for (@typeInfo(@This()).@"struct".decl_names) |decl_name| {
        const decl = @field(@This(), decl_name);
        if (@TypeOf(decl) != type) continue;
        switch (@typeInfo(decl)) {
            .@"enum" => if (@hasDecl(decl, "c_prefix")) try expectEnum(c, decl),
            .@"struct" => |info| {
                if (info.layout == .@"packed" and @hasDecl(decl, "c_prefix")) try expectFlags(c, decl);
                if (@hasDecl(decl, "c_type")) try expectLayout(decl, @field(c, decl.c_type));
            },
            .error_set => |info| if (decl == DriverError) inline for (info.error_names.?) |error_name| {
                const name = "CUDA_ERROR_" ++ comptime screamingSnakeCase(error_name);
                if (!@hasDecl(c, name)) return missing(name);
            },
            else => {},
        }
    }
    const api_info = @typeInfo(Raw).@"struct";
    inline for (api_info.field_names, api_info.field_types) |name, Pointer| {
        if (!@hasDecl(c, name)) return missing(name);
        const P = if (@typeInfo(Pointer) == .optional) @typeInfo(Pointer).optional.child else Pointer;
        expectAbi(@typeInfo(P).pointer.child, @TypeOf(@field(c, name))) catch |err| {
            std.debug.print("CUDA signature mismatch: {s}\n", .{name});
            return err;
        };
    }
    try std.testing.expectEqual(c.CUDA_ERROR_NOT_READY, error_not_ready);
}

fn expectAbi(comptime Ours: type, comptime Theirs: type) anyerror!void {
    if (@typeInfo(Ours) == .optional) return expectAbi(@typeInfo(Ours).optional.child, Theirs);
    if (@typeInfo(Theirs) == .optional) return expectAbi(Ours, @typeInfo(Theirs).optional.child);
    const ours = @typeInfo(Ours);
    const theirs = @typeInfo(Theirs);
    switch (ours) {
        .@"fn" => |info| {
            if (theirs != .@"fn") return error.SignatureMismatch;
            const other = theirs.@"fn";
            try std.testing.expectEqual(other.attrs.@"callconv", info.attrs.@"callconv");
            try std.testing.expectEqual(other.attrs.varargs, info.attrs.varargs);
            try std.testing.expectEqual(other.param_types.len, info.param_types.len);
            if (info.param_types.len != other.param_types.len) return;
            inline for (info.param_types, other.param_types) |Our, Their| try expectAbi(Our.?, Their.?);
            try expectAbi(info.return_type.?, other.return_type.?);
            return;
        },
        .pointer => |info| {
            if (theirs != .pointer) return error.SignatureMismatch;
            try expectAbi(info.child, theirs.pointer.child);
        },
        .@"opaque" => {
            if (theirs != .@"opaque" and theirs != .@"struct") return error.SignatureMismatch;
            return;
        },
        .int, .@"enum" => if (theirs != .int and theirs != .@"enum") return error.SignatureMismatch,
        .float => if (theirs != .float) return error.SignatureMismatch,
        .array => |info| {
            if (theirs != .array) return error.SignatureMismatch;
            try std.testing.expectEqual(theirs.array.len, info.len);
            try expectAbi(info.child, theirs.array.child);
        },
        .@"struct" => |info| {
            if (info.layout == .@"packed") return expectAbi(info.backing_integer.?, Theirs);
            if (theirs != .@"struct") return error.SignatureMismatch;
            try expectLayout(Ours, Theirs);
        },
        .void => if (theirs != .void) return error.SignatureMismatch,
        else => @compileError("unsupported CUDA ABI type: " ++ @typeName(Ours)),
    }
    try std.testing.expectEqual(@sizeOf(Theirs), @sizeOf(Ours));
    try std.testing.expectEqual(@alignOf(Theirs), @alignOf(Ours));
}

fn expectEnum(comptime c: type, comptime E: type) !void {
    const info = @typeInfo(E).@"enum";
    inline for (info.field_names, info.field_values) |field_name, value| {
        const name = E.c_prefix ++ comptime screamingSnakeCase(field_name);
        if (!@hasDecl(c, name)) return missing(name);
        try std.testing.expectEqual(@field(c, name), value);
    }
}

fn expectFlags(comptime c: type, comptime Flags: type) !void {
    const Bits = @typeInfo(Flags).@"struct".backing_integer.?;
    inline for (@typeInfo(Flags).@"struct".field_names) |field_name| {
        if (comptime std.mem.startsWith(u8, field_name, "reserved")) continue;
        const name = Flags.c_prefix ++ comptime screamingSnakeCase(field_name);
        if (!@hasDecl(c, name)) return missing(name);
        var flags: Flags = @bitCast(@as(Bits, 0));
        @field(flags, field_name) = true;
        try std.testing.expectEqual(@as(Bits, @field(c, name)), @as(Bits, @bitCast(flags)));
    }
}

fn expectLayout(comptime Ours: type, comptime Theirs: type) !void {
    try std.testing.expectEqual(@sizeOf(Theirs), @sizeOf(Ours));
    try std.testing.expectEqual(@alignOf(Theirs), @alignOf(Ours));
    const ours = @typeInfo(Ours).@"struct";
    const theirs = @typeInfo(Theirs).@"struct";
    try std.testing.expectEqual(theirs.field_names.len, ours.field_names.len);
    if (ours.field_names.len != theirs.field_names.len) return;
    inline for (ours.field_names, theirs.field_names, ours.field_types, theirs.field_types) |our_name, their_name, Our, Their| {
        try std.testing.expectEqual(@offsetOf(Theirs, their_name), @offsetOf(Ours, our_name));
        try std.testing.expectEqual(@sizeOf(Their), @sizeOf(Our));
    }
}

fn missing(name: []const u8) error{MissingFromHeader} {
    std.log.err("cuda.h does not declare {s}", .{name});
    return error.MissingFromHeader;
}

/// Converts `snake_case` and `TitleCase` names to the `SCREAMING_SNAKE_CASE` of the CUDA headers.
fn screamingSnakeCase(comptime name: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(100_000);
        var result: []const u8 = "";
        for (name, 0..) |char, i| {
            if (std.ascii.isUpper(char) and i != 0) result = result ++ "_";
            result = result ++ .{std.ascii.toUpper(char)};
        }
        return result;
    }
}
