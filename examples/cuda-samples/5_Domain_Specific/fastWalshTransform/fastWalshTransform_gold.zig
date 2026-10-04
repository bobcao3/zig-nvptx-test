/// CPU Fast Walsh Transform.
pub fn fwtCpu(h_output: []f32, h_input: []const f32, log2n: u5) void {
    const n = @as(usize, 1) << log2n;

    @memcpy(h_output[0..n], h_input[0..n]);

    // Cycle through stages with different butterfly strides
    var stride = n / 2;
    while (stride >= 1) : (stride >>= 1) {
        // Cycle through subvectors of (2 * stride) elements
        var base: usize = 0;
        while (base < n) : (base += 2 * stride) {
            // Butterfly index within subvector of (2 * stride) size
            for (0..stride) |j| {
                const index0 = base + j;
                const index1 = base + j + stride;

                const t1 = h_output[index0];
                const t2 = h_output[index1];
                h_output[index0] = t1 + t2;
                h_output[index1] = t1 - t2;
            }
        }
    }
}

/// Straightforward Walsh Transform: used to test both CPU and GPU FWT. Slow. Uses doubles because
/// of straightforward accumulation.
pub fn slowWtCpu(h_output: []f32, h_input: []const f32, log2n: u5) void {
    const n = @as(usize, 1) << log2n;

    for (0..n) |i| {
        var sum: f64 = 0;

        for (0..n) |j| {
            // Walsh-Hadamard quotient
            var q: f64 = 1.0;

            var t = i & j;
            while (t != 0) : (t >>= 1) {
                if (t & 1 != 0) q = -q;
            }

            sum += q * h_input[j];
        }

        h_output[i] = @floatCast(sum);
    }
}

/// Reference CPU dyadic convolution. Extremely slow because of non-linear memory access patterns
/// (cache thrashing).
pub fn dyadicConvolutionCpu(
    h_result: []f32,
    h_data: []const f32,
    h_kernel: []const f32,
    log2data_n: u5,
    log2kernel_n: u5,
) void {
    const data_n = @as(usize, 1) << log2data_n;
    const kernel_n = @as(usize, 1) << log2kernel_n;

    for (0..data_n) |i| {
        var sum: f64 = 0;

        for (0..kernel_n) |j| {
            sum += h_data[i ^ j] * h_kernel[j];
        }

        h_result[i] = @floatCast(sum);
    }
}
