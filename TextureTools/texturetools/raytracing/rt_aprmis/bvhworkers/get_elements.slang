[AutoPyBindCUDA]
[CUDAKernel]
void generateElements(
    TensorView<float> vert, TensorView<int> v_indx,
    TensorView<int> ele_primitiveIdx, TensorView<float> ele_aabb
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    if (dispatchIdx.x >= v_indx.size(0)) return;
    int primitiveIndex = dispatchIdx.x;
    float3 minV = float3(1e9, 1e9, 1e9);
    float3 maxV = float3(-1e9, -1e9, -1e9);
    for (int i = 0; i < 3; i++)
    {
        int v_i = v_indx[primitiveIndex, i];
        float3 v = float3(vert[v_i, 0], vert[v_i, 1], vert[v_i, 2]);

        minV[0] = min(minV[0], v[0]);
        minV[1] = min(minV[1], v[1]);
        minV[2] = min(minV[2], v[2]);

        maxV[0] = max(maxV[0], v[0]);
        maxV[1] = max(maxV[1], v[1]);
        maxV[2] = max(maxV[2], v[2]);
    }

    float3 aabb_min = float3(min(minV[0], maxV[0]), min(minV[1], maxV[1]), min(minV[2], maxV[2]));
    float3 aabb_max = float3(max(minV[0], maxV[0]), max(minV[1], maxV[1]), max(minV[2], maxV[2]));

    ele_primitiveIdx[primitiveIndex] = primitiveIndex;
    ele_aabb[primitiveIndex, 0] = aabb_min.x;
    ele_aabb[primitiveIndex, 1] = aabb_min.y;
    ele_aabb[primitiveIndex, 2] = aabb_min.z;

    ele_aabb[primitiveIndex, 3] = aabb_max.x;
    ele_aabb[primitiveIndex, 4] = aabb_max.y;
    ele_aabb[primitiveIndex, 5] = aabb_max.z;
}

int findMSB(uint val) {
    if (val == 0) return -1;
    int msb = 31;  // 32 uint
    while (!bool((val >> msb) & 1)) {
        msb--;
    }
    return msb;
}
uint expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
uint morton3D(float x, float y, float z) {
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    uint xx = expandBits(uint(x));
    uint yy = expandBits(uint(y));
    uint zz = expandBits(uint(z));
    return xx * 4 + yy * 2 + zz;
}

[AutoPyBindCUDA]
[CUDAKernel]
void debug_cb(
    TensorView<float> a,
    TensorView<int> b
    )
{
    b[0] = morton3D(a[0], a[1], a[2]);
}