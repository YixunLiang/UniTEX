// input for the builder (normally a triangle or some other kind of primitive); it is necessary to allocate and fill the buffer
struct Element {
    uint primitiveIdx;// the id of the primitive; this primitive id is copied to the leaf nodes of the  LBVHNode
    float aabbMinX;// aabb of the primitive
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
};

struct pushConstantsMortonCodes {
    int g_num_elements;
    float g_min_x; // AABB that contains the entire model
    float g_min_y;
    float g_min_z;
    float g_max_x;
    float g_max_y;
    float g_max_z;
};

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
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
void morton_codes(pushConstantsMortonCodes pc,
                  TensorView<float> ele_aabb,
                  TensorView<int> morton_codes_ele
    )
{
    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint gID = dispatchIdx.x;
    if (gID >= pc.g_num_elements) return;
    Element element;
    element.primitiveIdx = 0;
    element.aabbMinX = ele_aabb[gID, 0];
    element.aabbMinY = ele_aabb[gID, 1];
    element.aabbMinZ = ele_aabb[gID, 2];

    element.aabbMaxX = ele_aabb[gID, 3];
    element.aabbMaxY = ele_aabb[gID, 4];
    element.aabbMaxZ = ele_aabb[gID, 5];

    float3 aabbMin = float3(element.aabbMinX, element.aabbMinY, element.aabbMinZ);
    float3 aabbMax = float3(element.aabbMaxX, element.aabbMaxY, element.aabbMaxZ);

    // calculate center
    float3 center = (aabbMin + 0.5 * (aabbMax - aabbMin));
    // map to unit cube
    float3 g_min = float3(pc.g_min_x, pc.g_min_y, pc.g_min_z);
    float3 g_max = float3(pc.g_max_x, pc.g_max_y, pc.g_max_z);
    float3 mappedCenter = (center - g_min) / (g_max - g_min);
    // assign morton code
    uint mortonCode = morton3D(mappedCenter.x, mappedCenter.y, mappedCenter.z);
    uint elementIdx = gID;

    morton_codes_ele[gID, 0] = int(mortonCode);
    morton_codes_ele[gID, 1] = int(elementIdx);
}