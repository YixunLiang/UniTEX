// output of the builder; it is necessary to allocate the (empty) buffer
struct LBVHNode {
    int left;          // pointer to the left child or INVALID_POINTER in case of leaf
    int right;         // pointer to the right child or INVALID_POINTER in case of leaf
    uint primitiveIdx; // custom value that is copied from the input Element or 0 in case of inner node
    float aabbMinX;    // aabb of the node
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
};

struct LBVHConstructionInfo {
    uint parent;         // pointer to the parent
    int visitationCount; // number of threads that arrived
};

void aabbUnion(float3 minA, float3 maxA, float3 minB, float3 maxB, out float3 minAABB, out float3 maxAABB) {
    minAABB = min(minA, minB);
    maxAABB = max(maxA, maxB);
}

[AutoPyBindCUDA]
[CUDAKernel]
void bounding_boxes(uint g_num_elements,
                TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
               TensorView<int> g_lbvh_construction_infos)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint gID = dispatchIdx.x;
    if (gID >= g_num_elements) return;
    uint g_absolute_pointers = 1;
    const int LEAF_OFFSET = int(g_num_elements) - 1;

    uint nodeIdx = g_lbvh_construction_infos[LEAF_OFFSET + gID, 0];
    while (true) {
        int visitations;
        g_lbvh_construction_infos.InterlockedAdd(uint2(nodeIdx, 1), 1, visitations);
        if (visitations < 1) {
            // this is the first thread that arrived at this node -> finished
            break;
        }
        // this is the second thread that arrived at this node, both children are computed -> compute aabb union and continue
        LBVHNode bvhNode;
        bvhNode.left = g_lbvh_info[nodeIdx, 0];
        bvhNode.right = g_lbvh_info[nodeIdx, 1];
        bvhNode.primitiveIdx = g_lbvh_info[nodeIdx, 2];
        bvhNode.aabbMinX = g_lbvh_aabb[nodeIdx, 0];
        bvhNode.aabbMinY = g_lbvh_aabb[nodeIdx, 1];
        bvhNode.aabbMinZ = g_lbvh_aabb[nodeIdx, 2];
        bvhNode.aabbMaxX = g_lbvh_aabb[nodeIdx, 3];
        bvhNode.aabbMaxY = g_lbvh_aabb[nodeIdx, 4];
        bvhNode.aabbMaxZ = g_lbvh_aabb[nodeIdx, 5];

        LBVHNode bvhNodeChildA;
        LBVHNode bvhNodeChildB;
        if (g_absolute_pointers != 0) {

            bvhNodeChildA.left = g_lbvh_info[bvhNode.left, 0];
            bvhNodeChildA.right = g_lbvh_info[bvhNode.left, 1];
            bvhNodeChildA.primitiveIdx = g_lbvh_info[bvhNode.left, 2];
            bvhNodeChildA.aabbMinX = g_lbvh_aabb[bvhNode.left, 0];
            bvhNodeChildA.aabbMinY = g_lbvh_aabb[bvhNode.left, 1];
            bvhNodeChildA.aabbMinZ = g_lbvh_aabb[bvhNode.left, 2];
            bvhNodeChildA.aabbMaxX = g_lbvh_aabb[bvhNode.left, 3];
            bvhNodeChildA.aabbMaxY = g_lbvh_aabb[bvhNode.left, 4];
            bvhNodeChildA.aabbMaxZ = g_lbvh_aabb[bvhNode.left, 5];

            bvhNodeChildB.left = g_lbvh_info[bvhNode.right, 0];
            bvhNodeChildB.right = g_lbvh_info[bvhNode.right, 1];
            bvhNodeChildB.primitiveIdx = g_lbvh_info[bvhNode.right, 2];
            bvhNodeChildB.aabbMinX = g_lbvh_aabb[bvhNode.right, 0];
            bvhNodeChildB.aabbMinY = g_lbvh_aabb[bvhNode.right, 1];
            bvhNodeChildB.aabbMinZ = g_lbvh_aabb[bvhNode.right, 2];
            bvhNodeChildB.aabbMaxX = g_lbvh_aabb[bvhNode.right, 3];
            bvhNodeChildB.aabbMaxY = g_lbvh_aabb[bvhNode.right, 4];
            bvhNodeChildB.aabbMaxZ = g_lbvh_aabb[bvhNode.right, 5];

        } else {

            bvhNodeChildA.left = g_lbvh_info[nodeIdx + bvhNode.left, 0];
            bvhNodeChildA.right = g_lbvh_info[nodeIdx + bvhNode.left, 1];
            bvhNodeChildA.primitiveIdx = g_lbvh_info[nodeIdx + bvhNode.left, 2];
            bvhNodeChildA.aabbMinX = g_lbvh_aabb[nodeIdx + bvhNode.left, 0];
            bvhNodeChildA.aabbMinY = g_lbvh_aabb[nodeIdx + bvhNode.left, 1];
            bvhNodeChildA.aabbMinZ = g_lbvh_aabb[nodeIdx + bvhNode.left, 2];
            bvhNodeChildA.aabbMaxX = g_lbvh_aabb[nodeIdx + bvhNode.left, 3];
            bvhNodeChildA.aabbMaxY = g_lbvh_aabb[nodeIdx + bvhNode.left, 4];
            bvhNodeChildA.aabbMaxZ = g_lbvh_aabb[nodeIdx + bvhNode.left, 5];

            bvhNodeChildB.left = g_lbvh_info[nodeIdx + bvhNode.right, 0];
            bvhNodeChildB.right = g_lbvh_info[nodeIdx + bvhNode.right, 1];
            bvhNodeChildB.primitiveIdx = g_lbvh_info[nodeIdx + bvhNode.right, 2];
            bvhNodeChildB.aabbMinX = g_lbvh_aabb[nodeIdx + bvhNode.right, 0];
            bvhNodeChildB.aabbMinY = g_lbvh_aabb[nodeIdx + bvhNode.right, 1];
            bvhNodeChildB.aabbMinZ = g_lbvh_aabb[nodeIdx + bvhNode.right, 2];
            bvhNodeChildB.aabbMaxX = g_lbvh_aabb[nodeIdx + bvhNode.right, 3];
            bvhNodeChildB.aabbMaxY = g_lbvh_aabb[nodeIdx + bvhNode.right, 4];
            bvhNodeChildB.aabbMaxZ = g_lbvh_aabb[nodeIdx + bvhNode.right, 5];
        }

        float3 minA = float3(bvhNodeChildA.aabbMinX, bvhNodeChildA.aabbMinY, bvhNodeChildA.aabbMinZ);
        float3 maxA = float3(bvhNodeChildA.aabbMaxX, bvhNodeChildA.aabbMaxY, bvhNodeChildA.aabbMaxZ);
        float3 minB = float3(bvhNodeChildB.aabbMinX, bvhNodeChildB.aabbMinY, bvhNodeChildB.aabbMinZ);
        float3 maxB = float3(bvhNodeChildB.aabbMaxX, bvhNodeChildB.aabbMaxY, bvhNodeChildB.aabbMaxZ);

        if (bvhNode.left == 0)
        {
            minA = float3(1e9, 1e9, 1e9);
            maxA = float3(-1e9, -1e9, -1e9);
        }

        if (bvhNode.right == 0)
        {
            minB = float3(1e9, 1e9, 1e9);
            maxB = float3(-1e9, -1e9, -1e9);
        }

        float3 minAABB;
        float3 maxAABB;
        aabbUnion(minA, maxA, minB, maxB, minAABB, maxAABB);
        bvhNode.aabbMinX = minAABB.x;
        bvhNode.aabbMinY = minAABB.y;
        bvhNode.aabbMinZ = minAABB.z;
        bvhNode.aabbMaxX = maxAABB.x;
        bvhNode.aabbMaxY = maxAABB.y;
        bvhNode.aabbMaxZ = maxAABB.z;

        g_lbvh_info[nodeIdx, 0] = bvhNode.left;
        g_lbvh_info[nodeIdx, 1] = bvhNode.right;
        g_lbvh_info[nodeIdx, 2] = int(bvhNode.primitiveIdx);

        g_lbvh_aabb[nodeIdx, 0] = bvhNode.aabbMinX;
        g_lbvh_aabb[nodeIdx, 1] = bvhNode.aabbMinY;
        g_lbvh_aabb[nodeIdx, 2] = bvhNode.aabbMinZ;
        g_lbvh_aabb[nodeIdx, 3] = bvhNode.aabbMaxX;
        g_lbvh_aabb[nodeIdx, 4] = bvhNode.aabbMaxY;
        g_lbvh_aabb[nodeIdx, 5] = bvhNode.aabbMaxZ;

        if (nodeIdx == 0) {
            break;
        }
        nodeIdx = g_lbvh_construction_infos[nodeIdx, 0];
    }
    
}

[AutoPyBindCUDA]
[CUDAKernel]
void get_bvh_height(uint g_num_elements,
                    TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
                    TensorView<int> g_lbvh_construction_infos, TensorView<int> tree_heights)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint gID = dispatchIdx.x;
    if (gID >= g_num_elements) return;
    uint g_absolute_pointers = 1;
    const int LEAF_OFFSET = int(g_num_elements) - 1;

    uint nodeIdx = g_lbvh_construction_infos[LEAF_OFFSET + gID, 0];
    int height = 0;
    while (true) {
        if (nodeIdx == 0) {
            break;
        }
        height++;
        nodeIdx = g_lbvh_construction_infos[nodeIdx, 0];
    }
    tree_heights[gID] = height;
}

[AutoPyBindCUDA]
[CUDAKernel]
void get_bbox(uint g_num_elements, int expected_height,
                    TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
                    TensorView<int> g_lbvh_construction_infos)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint gID = dispatchIdx.x;
    if (gID >= g_num_elements) return;
    uint g_absolute_pointers = 1;
    const int LEAF_OFFSET = int(g_num_elements) - 1;

    uint nodeIdx = g_lbvh_construction_infos[LEAF_OFFSET + gID, 0];
    int height = 0;
    while (true) {
        if (nodeIdx == 0) {
            break;
        }
        height++;

        if (height > expected_height) break;
        if (height == expected_height)
        {  
            LBVHNode bvhNode;
            bvhNode.left = g_lbvh_info[nodeIdx, 0];
            bvhNode.right = g_lbvh_info[nodeIdx, 1];
            bvhNode.primitiveIdx = g_lbvh_info[nodeIdx, 2];
            bvhNode.aabbMinX = g_lbvh_aabb[nodeIdx, 0];
            bvhNode.aabbMinY = g_lbvh_aabb[nodeIdx, 1];
            bvhNode.aabbMinZ = g_lbvh_aabb[nodeIdx, 2];
            bvhNode.aabbMaxX = g_lbvh_aabb[nodeIdx, 3];
            bvhNode.aabbMaxY = g_lbvh_aabb[nodeIdx, 4];
            bvhNode.aabbMaxZ = g_lbvh_aabb[nodeIdx, 5];

            LBVHNode bvhNodeChildA;
            LBVHNode bvhNodeChildB;
            if (g_absolute_pointers != 0) {

                bvhNodeChildA.left = g_lbvh_info[bvhNode.left, 0];
                bvhNodeChildA.right = g_lbvh_info[bvhNode.left, 1];
                bvhNodeChildA.primitiveIdx = g_lbvh_info[bvhNode.left, 2];
                bvhNodeChildA.aabbMinX = g_lbvh_aabb[bvhNode.left, 0];
                bvhNodeChildA.aabbMinY = g_lbvh_aabb[bvhNode.left, 1];
                bvhNodeChildA.aabbMinZ = g_lbvh_aabb[bvhNode.left, 2];
                bvhNodeChildA.aabbMaxX = g_lbvh_aabb[bvhNode.left, 3];
                bvhNodeChildA.aabbMaxY = g_lbvh_aabb[bvhNode.left, 4];
                bvhNodeChildA.aabbMaxZ = g_lbvh_aabb[bvhNode.left, 5];

                bvhNodeChildB.left = g_lbvh_info[bvhNode.right, 0];
                bvhNodeChildB.right = g_lbvh_info[bvhNode.right, 1];
                bvhNodeChildB.primitiveIdx = g_lbvh_info[bvhNode.right, 2];
                bvhNodeChildB.aabbMinX = g_lbvh_aabb[bvhNode.right, 0];
                bvhNodeChildB.aabbMinY = g_lbvh_aabb[bvhNode.right, 1];
                bvhNodeChildB.aabbMinZ = g_lbvh_aabb[bvhNode.right, 2];
                bvhNodeChildB.aabbMaxX = g_lbvh_aabb[bvhNode.right, 3];
                bvhNodeChildB.aabbMaxY = g_lbvh_aabb[bvhNode.right, 4];
                bvhNodeChildB.aabbMaxZ = g_lbvh_aabb[bvhNode.right, 5];
            } else {

                bvhNodeChildA.left = g_lbvh_info[nodeIdx + bvhNode.left, 0];
                bvhNodeChildA.right = g_lbvh_info[nodeIdx + bvhNode.left, 1];
                bvhNodeChildA.primitiveIdx = g_lbvh_info[nodeIdx + bvhNode.left, 2];
                bvhNodeChildA.aabbMinX = g_lbvh_aabb[nodeIdx + bvhNode.left, 0];
                bvhNodeChildA.aabbMinY = g_lbvh_aabb[nodeIdx + bvhNode.left, 1];
                bvhNodeChildA.aabbMinZ = g_lbvh_aabb[nodeIdx + bvhNode.left, 2];
                bvhNodeChildA.aabbMaxX = g_lbvh_aabb[nodeIdx + bvhNode.left, 3];
                bvhNodeChildA.aabbMaxY = g_lbvh_aabb[nodeIdx + bvhNode.left, 4];
                bvhNodeChildA.aabbMaxZ = g_lbvh_aabb[nodeIdx + bvhNode.left, 5];

                bvhNodeChildB.left = g_lbvh_info[nodeIdx + bvhNode.right, 0];
                bvhNodeChildB.right = g_lbvh_info[nodeIdx + bvhNode.right, 1];
                bvhNodeChildB.primitiveIdx = g_lbvh_info[nodeIdx + bvhNode.right, 2];
                bvhNodeChildB.aabbMinX = g_lbvh_aabb[nodeIdx + bvhNode.right, 0];
                bvhNodeChildB.aabbMinY = g_lbvh_aabb[nodeIdx + bvhNode.right, 1];
                bvhNodeChildB.aabbMinZ = g_lbvh_aabb[nodeIdx + bvhNode.right, 2];
                bvhNodeChildB.aabbMaxX = g_lbvh_aabb[nodeIdx + bvhNode.right, 3];
                bvhNodeChildB.aabbMaxY = g_lbvh_aabb[nodeIdx + bvhNode.right, 4];
                bvhNodeChildB.aabbMaxZ = g_lbvh_aabb[nodeIdx + bvhNode.right, 5];
            }

            float3 minA = float3(bvhNodeChildA.aabbMinX, bvhNodeChildA.aabbMinY, bvhNodeChildA.aabbMinZ);
            float3 maxA = float3(bvhNodeChildA.aabbMaxX, bvhNodeChildA.aabbMaxY, bvhNodeChildA.aabbMaxZ);
            float3 minB = float3(bvhNodeChildB.aabbMinX, bvhNodeChildB.aabbMinY, bvhNodeChildB.aabbMinZ);
            float3 maxB = float3(bvhNodeChildB.aabbMaxX, bvhNodeChildB.aabbMaxY, bvhNodeChildB.aabbMaxZ);

            if (bvhNode.left == 0)
            {
                minA = float3(1e9, 1e9, 1e9);
                maxA = float3(-1e9, -1e9, -1e9);
            }

            if (bvhNode.right == 0)
            {
                minB = float3(1e9, 1e9, 1e9);
                maxB = float3(-1e9, -1e9, -1e9);
            }

            float3 minAABB;
            float3 maxAABB;
            aabbUnion(minA, maxA, minB, maxB, minAABB, maxAABB);
            bvhNode.aabbMinX = minAABB.x;
            bvhNode.aabbMinY = minAABB.y;
            bvhNode.aabbMinZ = minAABB.z;
            bvhNode.aabbMaxX = maxAABB.x;
            bvhNode.aabbMaxY = maxAABB.y;
            bvhNode.aabbMaxZ = maxAABB.z;

            g_lbvh_info[nodeIdx, 0] = bvhNode.left;
            g_lbvh_info[nodeIdx, 1] = bvhNode.right;
            g_lbvh_info[nodeIdx, 2] = int(bvhNode.primitiveIdx);

            g_lbvh_aabb[nodeIdx, 0] = bvhNode.aabbMinX;
            g_lbvh_aabb[nodeIdx, 1] = bvhNode.aabbMinY;
            g_lbvh_aabb[nodeIdx, 2] = bvhNode.aabbMinZ;
            g_lbvh_aabb[nodeIdx, 3] = bvhNode.aabbMaxX;
            g_lbvh_aabb[nodeIdx, 4] = bvhNode.aabbMaxY;
            g_lbvh_aabb[nodeIdx, 5] = bvhNode.aabbMaxZ;

            break;
        }

        nodeIdx = g_lbvh_construction_infos[nodeIdx, 0];
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void set_root(
              TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
              )
{
    uint nodeIdx = 0;
    uint g_absolute_pointers = 0;

    LBVHNode bvhNode;
    bvhNode.left = g_lbvh_info[nodeIdx, 0];
    bvhNode.right = g_lbvh_info[nodeIdx, 1];
    bvhNode.primitiveIdx = g_lbvh_info[nodeIdx, 2];
    bvhNode.aabbMinX = g_lbvh_aabb[nodeIdx, 0];
    bvhNode.aabbMinY = g_lbvh_aabb[nodeIdx, 1];
    bvhNode.aabbMinZ = g_lbvh_aabb[nodeIdx, 2];
    bvhNode.aabbMaxX = g_lbvh_aabb[nodeIdx, 3];
    bvhNode.aabbMaxY = g_lbvh_aabb[nodeIdx, 4];
    bvhNode.aabbMaxZ = g_lbvh_aabb[nodeIdx, 5];

    LBVHNode bvhNodeChildA;
    LBVHNode bvhNodeChildB;
    if (g_absolute_pointers != 0) {

        bvhNodeChildA.left = g_lbvh_info[bvhNode.left, 0];
        bvhNodeChildA.right = g_lbvh_info[bvhNode.left, 1];
        bvhNodeChildA.primitiveIdx = g_lbvh_info[bvhNode.left, 2];
        bvhNodeChildA.aabbMinX = g_lbvh_aabb[bvhNode.left, 0];
        bvhNodeChildA.aabbMinY = g_lbvh_aabb[bvhNode.left, 1];
        bvhNodeChildA.aabbMinZ = g_lbvh_aabb[bvhNode.left, 2];
        bvhNodeChildA.aabbMaxX = g_lbvh_aabb[bvhNode.left, 3];
        bvhNodeChildA.aabbMaxY = g_lbvh_aabb[bvhNode.left, 4];
        bvhNodeChildA.aabbMaxZ = g_lbvh_aabb[bvhNode.left, 5];

        bvhNodeChildB.left = g_lbvh_info[bvhNode.right, 0];
        bvhNodeChildB.right = g_lbvh_info[bvhNode.right, 1];
        bvhNodeChildB.primitiveIdx = g_lbvh_info[bvhNode.right, 2];
        bvhNodeChildB.aabbMinX = g_lbvh_aabb[bvhNode.right, 0];
        bvhNodeChildB.aabbMinY = g_lbvh_aabb[bvhNode.right, 1];
        bvhNodeChildB.aabbMinZ = g_lbvh_aabb[bvhNode.right, 2];
        bvhNodeChildB.aabbMaxX = g_lbvh_aabb[bvhNode.right, 3];
        bvhNodeChildB.aabbMaxY = g_lbvh_aabb[bvhNode.right, 4];
        bvhNodeChildB.aabbMaxZ = g_lbvh_aabb[bvhNode.right, 5];
    } else {

        bvhNodeChildA.left = g_lbvh_info[nodeIdx + bvhNode.left, 0];
        bvhNodeChildA.right = g_lbvh_info[nodeIdx + bvhNode.left, 1];
        bvhNodeChildA.primitiveIdx = g_lbvh_info[nodeIdx + bvhNode.left, 2];
        bvhNodeChildA.aabbMinX = g_lbvh_aabb[nodeIdx + bvhNode.left, 0];
        bvhNodeChildA.aabbMinY = g_lbvh_aabb[nodeIdx + bvhNode.left, 1];
        bvhNodeChildA.aabbMinZ = g_lbvh_aabb[nodeIdx + bvhNode.left, 2];
        bvhNodeChildA.aabbMaxX = g_lbvh_aabb[nodeIdx + bvhNode.left, 3];
        bvhNodeChildA.aabbMaxY = g_lbvh_aabb[nodeIdx + bvhNode.left, 4];
        bvhNodeChildA.aabbMaxZ = g_lbvh_aabb[nodeIdx + bvhNode.left, 5];

        bvhNodeChildB.left = g_lbvh_info[nodeIdx + bvhNode.right, 0];
        bvhNodeChildB.right = g_lbvh_info[nodeIdx + bvhNode.right, 1];
        bvhNodeChildB.primitiveIdx = g_lbvh_info[nodeIdx + bvhNode.right, 2];
        bvhNodeChildB.aabbMinX = g_lbvh_aabb[nodeIdx + bvhNode.right, 0];
        bvhNodeChildB.aabbMinY = g_lbvh_aabb[nodeIdx + bvhNode.right, 1];
        bvhNodeChildB.aabbMinZ = g_lbvh_aabb[nodeIdx + bvhNode.right, 2];
        bvhNodeChildB.aabbMaxX = g_lbvh_aabb[nodeIdx + bvhNode.right, 3];
        bvhNodeChildB.aabbMaxY = g_lbvh_aabb[nodeIdx + bvhNode.right, 4];
        bvhNodeChildB.aabbMaxZ = g_lbvh_aabb[nodeIdx + bvhNode.right, 5];
    }

    float3 minA = float3(bvhNodeChildA.aabbMinX, bvhNodeChildA.aabbMinY, bvhNodeChildA.aabbMinZ);
    float3 maxA = float3(bvhNodeChildA.aabbMaxX, bvhNodeChildA.aabbMaxY, bvhNodeChildA.aabbMaxZ);
    float3 minB = float3(bvhNodeChildB.aabbMinX, bvhNodeChildB.aabbMinY, bvhNodeChildB.aabbMinZ);
    float3 maxB = float3(bvhNodeChildB.aabbMaxX, bvhNodeChildB.aabbMaxY, bvhNodeChildB.aabbMaxZ);

    float3 minAABB;
    float3 maxAABB;
    aabbUnion(minA, maxA, minB, maxB, minAABB, maxAABB);
    bvhNode.aabbMinX = minAABB.x;
    bvhNode.aabbMinY = minAABB.y;
    bvhNode.aabbMinZ = minAABB.z;
    bvhNode.aabbMaxX = maxAABB.x;
    bvhNode.aabbMaxY = maxAABB.y;
    bvhNode.aabbMaxZ = maxAABB.z;

    g_lbvh_info[nodeIdx, 0] = bvhNode.left;
    g_lbvh_info[nodeIdx, 1] = bvhNode.right;
    g_lbvh_info[nodeIdx, 2] = int(bvhNode.primitiveIdx);

    g_lbvh_aabb[nodeIdx, 0] = bvhNode.aabbMinX;
    g_lbvh_aabb[nodeIdx, 1] = bvhNode.aabbMinY;
    g_lbvh_aabb[nodeIdx, 2] = bvhNode.aabbMinZ;
    g_lbvh_aabb[nodeIdx, 3] = bvhNode.aabbMaxX;
    g_lbvh_aabb[nodeIdx, 4] = bvhNode.aabbMaxY;
    g_lbvh_aabb[nodeIdx, 5] = bvhNode.aabbMaxZ;

}