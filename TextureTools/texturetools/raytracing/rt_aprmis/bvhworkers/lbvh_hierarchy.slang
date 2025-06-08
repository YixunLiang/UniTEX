#define INVALID_POINTER 0x0

struct Element {
    uint primitiveIdx; // the id of the primitive; this primitive id is copied to the leaf nodes of the  LBVHNode
    float aabbMinX;    // aabb of the primitive
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
};

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

int findMSB(uint val) {
    if (val == 0) return -1;
    int msb = 31; // 32 uint
    while (!bool((val >> msb) & 1)) {
        msb--;
    }
    return msb;
}

int delta(int i, uint codeI, int j, uint g_num_elements, TensorView<int> g_sorted_morton_codes) {
    if (j < 0 || j > g_num_elements - 1) {
        return -1;
    }
    uint codeJ = g_sorted_morton_codes[j, 0];
    if (codeI == codeJ) {
        // handle duplicate morton codes
        uint elementIdxI = i;// g_sorted_morton_codes[i].elementIdx;
        uint elementIdxJ = j;// g_sorted_morton_codes[j].elementIdx;
        // add 32 for common prefix of codeI ^ codeJ
        return 32 + 31 - findMSB(elementIdxI ^ elementIdxJ);
    }
    return 31 - findMSB(codeI ^ codeJ);
}

void determineRange(int idx, out int lower, out int upper, uint g_num_elements, TensorView<int> g_sorted_morton_codes) {
    // determine direction of the range (+1 or -1)
    const uint code = g_sorted_morton_codes[idx, 0];
    const int deltaL = delta(idx, code, idx - 1, g_num_elements, g_sorted_morton_codes);
    const int deltaR = delta(idx, code, idx + 1, g_num_elements, g_sorted_morton_codes);
    const int d = (deltaR >= deltaL) ? 1 : -1;

    // compute upper bound for the length of the range
    const int deltaMin = min(deltaL, deltaR);// delta(idx, code, idx - d);
    int lMax = 2;
    while (delta(idx, code, idx + lMax * d, g_num_elements, g_sorted_morton_codes) > deltaMin) {
        lMax = lMax << 1;
    }

    // find the other end using binary search
    int l = 0;
    for (int t = lMax >> 1; t > 0; t >>= 1) {
        if (delta(idx, code, idx + (l + t) * d, g_num_elements, g_sorted_morton_codes) > deltaMin) {
            l += t;
        }
    }
    int jdx = idx + l * d;

    // ensure idx < jdx
    lower = min(idx, jdx);
    upper = max(idx, jdx);
}

int findSplit(int first, int last, uint g_num_elements, TensorView<int> g_sorted_morton_codes) {
    uint firstCode = g_sorted_morton_codes[first, 0];

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.
    int commonPrefix = delta(first, firstCode, last, g_num_elements, g_sorted_morton_codes);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    int split = first;// initial guess
    int stride = last - first;
    do {
        stride = (stride + 1) >> 1;// exponential decrease
        int newSplit = split + stride;// proposed new position
        if (newSplit < last) {
            int splitPrefix = delta(first, firstCode, newSplit, g_num_elements, g_sorted_morton_codes);
            if (splitPrefix > commonPrefix) {
                split = newSplit;// accept proposal
            }
        }
    } while (stride > 1);

    return split;
}

[AutoPyBindCUDA]
[CUDAKernel]
void hierarchy(uint g_num_elements, TensorView<int> ele_primitiveIdx, TensorView<float> ele_aabb,
               TensorView<int> g_sorted_morton_codes, TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
               TensorView<int> g_lbvh_construction_infos)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint gID = dispatchIdx.x;
    if (gID >= g_num_elements) return;
    uint g_absolute_pointers = 1;
    const int LEAF_OFFSET = int(g_num_elements) - 1;
    // construct leaf nodes
    if (gID < g_num_elements) {
        Element element;
        element.primitiveIdx = ele_primitiveIdx[g_sorted_morton_codes[gID, 1], 0];
        element.aabbMinX = ele_aabb[g_sorted_morton_codes[gID, 1], 0];
        element.aabbMinY = ele_aabb[g_sorted_morton_codes[gID, 1], 1];
        element.aabbMinZ = ele_aabb[g_sorted_morton_codes[gID, 1], 2];

        element.aabbMaxX = ele_aabb[g_sorted_morton_codes[gID, 1], 3];
        element.aabbMaxY = ele_aabb[g_sorted_morton_codes[gID, 1], 4];
        element.aabbMaxZ = ele_aabb[g_sorted_morton_codes[gID, 1], 5];

        LBVHNode lbvhNode;
        lbvhNode.left = INVALID_POINTER;
        lbvhNode.right = INVALID_POINTER;
        lbvhNode.primitiveIdx = element.primitiveIdx;
        lbvhNode.aabbMinX = element.aabbMinX;
        lbvhNode.aabbMinY = element.aabbMinY;
        lbvhNode.aabbMinZ = element.aabbMinZ;
        lbvhNode.aabbMaxX = element.aabbMaxX;
        lbvhNode.aabbMaxY = element.aabbMaxY;
        lbvhNode.aabbMaxZ = element.aabbMaxZ;
        g_lbvh_info[LEAF_OFFSET + gID, 0] = lbvhNode.left;
        g_lbvh_info[LEAF_OFFSET + gID, 1] = lbvhNode.right;
        g_lbvh_info[LEAF_OFFSET + gID, 2] = int(lbvhNode.primitiveIdx);

        g_lbvh_aabb[LEAF_OFFSET + gID, 0] = lbvhNode.aabbMinX;
        g_lbvh_aabb[LEAF_OFFSET + gID, 1] = lbvhNode.aabbMinY;
        g_lbvh_aabb[LEAF_OFFSET + gID, 2] = lbvhNode.aabbMinZ;
        g_lbvh_aabb[LEAF_OFFSET + gID, 3] = lbvhNode.aabbMaxX;
        g_lbvh_aabb[LEAF_OFFSET + gID, 4] = lbvhNode.aabbMaxY;
        g_lbvh_aabb[LEAF_OFFSET + gID, 5] = lbvhNode.aabbMaxZ;
    }

    // construct internal nodes
    if (gID < g_num_elements - 1) {
        // Find out which range of objects the node corresponds to.
        // (This is where the magic happens!)
        int first;
        int last;
        determineRange(int(gID), first, last, g_num_elements, g_sorted_morton_codes);

        // determine where to split the range
        int split = findSplit(first, last, g_num_elements, g_sorted_morton_codes);

        // select childA
        int childA = -1;
        if (split == first) {
            childA = LEAF_OFFSET + split; // pointer to leaf node
        } else {
            childA = split; // pointer to internal node
        }

        // select childB
        int childB = -1;
        if (split + 1 == last) {
            childB = LEAF_OFFSET + split + 1; // pointer to leaf node
        } else {
            childB = split + 1; // pointer to internal node
        }

        // record parent-child relationships
        if (g_absolute_pointers != 0) {
            LBVHNode lbvhNode;
            lbvhNode.left = childA;
            lbvhNode.right = childB;
            lbvhNode.primitiveIdx = 0;
            lbvhNode.aabbMinX = 1e9;
            lbvhNode.aabbMinY = 1e9;
            lbvhNode.aabbMinZ = 1e9;
            lbvhNode.aabbMaxX = -1e9;
            lbvhNode.aabbMaxY = -1e9;
            lbvhNode.aabbMaxZ = -1e9;
            g_lbvh_info[gID, 0] = lbvhNode.left;
            g_lbvh_info[gID, 1] = lbvhNode.right;
            g_lbvh_info[gID, 2] = int(lbvhNode.primitiveIdx);

            g_lbvh_aabb[gID, 0] = lbvhNode.aabbMinX;
            g_lbvh_aabb[gID, 1] = lbvhNode.aabbMinY;
            g_lbvh_aabb[gID, 2] = lbvhNode.aabbMinZ;
            g_lbvh_aabb[gID, 3] = lbvhNode.aabbMaxX;
            g_lbvh_aabb[gID, 4] = lbvhNode.aabbMaxY;
            g_lbvh_aabb[gID, 5] = lbvhNode.aabbMaxZ;
        } else {
            LBVHNode lbvhNode;
            lbvhNode.left = childA - int(gID);
            lbvhNode.right = childB - int(gID);
            lbvhNode.primitiveIdx = 0;
            lbvhNode.aabbMinX = 1e9;
            lbvhNode.aabbMinY = 1e9;
            lbvhNode.aabbMinZ = 1e9;
            lbvhNode.aabbMaxX = -1e9;
            lbvhNode.aabbMaxY = -1e9;
            lbvhNode.aabbMaxZ = -1e9;
            g_lbvh_info[gID, 0] = lbvhNode.left;
            g_lbvh_info[gID, 1] = lbvhNode.right;
            g_lbvh_info[gID, 2] = int(lbvhNode.primitiveIdx);

            g_lbvh_aabb[gID, 0] = lbvhNode.aabbMinX;
            g_lbvh_aabb[gID, 1] = lbvhNode.aabbMinY;
            g_lbvh_aabb[gID, 2] = lbvhNode.aabbMinZ;
            g_lbvh_aabb[gID, 3] = lbvhNode.aabbMaxX;
            g_lbvh_aabb[gID, 4] = lbvhNode.aabbMaxY;
            g_lbvh_aabb[gID, 5] = lbvhNode.aabbMaxZ;
        }
        LBVHConstructionInfo bvh_con_info;
        bvh_con_info.parent = gID;
        bvh_con_info.visitationCount = 0;

        g_lbvh_construction_infos[childA, 0] = bvh_con_info.parent;
        g_lbvh_construction_infos[childA, 1] = bvh_con_info.visitationCount;

        g_lbvh_construction_infos[childB, 0] = bvh_con_info.parent;
        g_lbvh_construction_infos[childB, 1] = bvh_con_info.visitationCount;
    }

    // node 0 is the root
    if (gID == 0) {
        LBVHConstructionInfo bvh_con_info;
        bvh_con_info.parent = 0;
        bvh_con_info.visitationCount = 0;

        g_lbvh_construction_infos[0, 0] = bvh_con_info.parent;
        g_lbvh_construction_infos[0, 1] = bvh_con_info.visitationCount;
    }
}