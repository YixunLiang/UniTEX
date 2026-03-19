#define MAX_STACK_SIZE 64

struct GPUBvhNode {
    int index;
    int left;          // pointer to the left child or INVALID_POINTER in case of leaf
    int right;         // pointer to the right child or INVALID_POINTER in case of leaf
    uint primitiveIdx; // custom value that is copied from the input Element or 0 in case of inner node
};

/*
intersection code from https://github.com/maxilevi/raytracer/tree/master
*/

bool aabb_hit(float3 rayo, float3 rayd, float t_min, float t_max, float3 bbox_min, float3 bbox_max)
{
    for (int i = 0; i < 3; ++i)
    {
        float ray_d_i = rayd[i];
        if (ray_d_i == 0.f) ray_d_i = 0.000001f;
        float inverse_direction = 1.0 / ray_d_i;
        float t0 = (bbox_min[i] - rayo[i]) * inverse_direction;
        float t1 = (bbox_max[i] - rayo[i]) * inverse_direction;
        if (inverse_direction < 0.0) {
            float tmp = t1;
            t1 = t0;
            t0 = tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max < t_min)
            return false;
    }
    return true;

}

bool triangle_hit(float3 rayo, float3 rayd, float t_min, float t_max, float3 v0, float3 v1, float3 v2, inout float t_hit, inout float3 pos)
{
    float epsilon = 1e-9;
    float3 E1 = v1 - v0;
    float3 E2 = v2 - v0;
    float3 P = cross(rayd, E2);
    float det = dot(E1, P);
    if (det > -epsilon && det < epsilon)
        return false;
    float invDet = 1 / det;

    float3 T = rayo - v0;
    float u = dot(T,P) * invDet;
    if (u < 0 || u > 1) return false;

    float3 Q = cross(T, E1);
    float v = dot(rayd, Q) * invDet;
    if (v < 0 || u + v > 1) return false;

    float t = dot(E2, Q) * invDet;
    t_hit = t;
    pos = rayo + t * rayd;
    return true;
}

bool bvh_hit(TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
             TensorView<float> vert, TensorView<int> v_indx,
    float3 rayo, float3 rayd, float t_min, float t_max, inout float t_hit, inout float3 pos)
{
    GPUBvhNode stack[MAX_STACK_SIZE];
    int count = 0;

    int root_index = 0;
    /* Add root node */
    GPUBvhNode now_node;
    now_node.index = root_index;
    now_node.left = g_lbvh_info[root_index, 0];
    now_node.right = g_lbvh_info[root_index, 1];
    now_node.primitiveIdx = g_lbvh_info[root_index, 2];
    stack[count++] = now_node;
    float closest_so_far = t_max;
    bool any_hit = false;
    float hit_t = 0.f;
    float3 hit_pos = float3(0.f);

    while (count > 0) {
        /* Pop the last node from the stack */
        GPUBvhNode node = stack[--count];
        float aabbMinX = g_lbvh_aabb[node.index, 0];
        float aabbMinY = g_lbvh_aabb[node.index, 1];
        float aabbMinZ = g_lbvh_aabb[node.index, 2];
        float aabbMaxX = g_lbvh_aabb[node.index, 3];
        float aabbMaxY = g_lbvh_aabb[node.index, 4];
        float aabbMaxZ = g_lbvh_aabb[node.index, 5];

        if (!aabb_hit(rayo, rayd, t_min, closest_so_far, float3(aabbMinX, aabbMinY, aabbMinZ), float3(aabbMaxX, aabbMaxY, aabbMaxZ)))
            continue;

        if (node.left != 0 && node.right !=0) {
            /* Add both nodes to the stack for further processing */
            GPUBvhNode left_node;
            left_node.index = node.left;
            left_node.left = g_lbvh_info[node.left, 0];
            left_node.right = g_lbvh_info[node.left, 1];
            left_node.primitiveIdx = g_lbvh_info[node.left, 2];

            GPUBvhNode right_node;
            right_node.index = node.right;
            right_node.left = g_lbvh_info[node.right, 0];
            right_node.right = g_lbvh_info[node.right, 1];
            right_node.primitiveIdx = g_lbvh_info[node.right, 2];

            stack[count++] = left_node;
            stack[count++] = right_node;
        } else if (node.left == 0 && node.right == 0)
        {
            /* It's a child node, we should test against the triangles and save the closest */
            //any_hit = true;
            uint now_pri_idx = node.primitiveIdx;
            int3 v_in = int3(v_indx[now_pri_idx, 0], v_indx[now_pri_idx, 1], v_indx[now_pri_idx, 2]);
            float3 tri_v0 = float3(vert[v_in.x, 0], vert[v_in.x, 1], vert[v_in.x, 2]);
            float3 tri_v1 = float3(vert[v_in.y, 0], vert[v_in.y, 1], vert[v_in.y, 2]);
            float3 tri_v2 = float3(vert[v_in.z, 0], vert[v_in.z, 1], vert[v_in.z, 2]);
            float now_t_hit = 0.f;
            float3 now_pos_hit = float3(0.f);
            bool hit = triangle_hit(rayo, rayd, t_min, closest_so_far, tri_v0, tri_v1, tri_v2, now_t_hit, now_pos_hit);
            closest_so_far = hit ? min(now_t_hit, closest_so_far) : closest_so_far;
            if (hit) {
                any_hit = true;
                hit_t = closest_so_far;
                hit_pos = rayo + hit_t * rayd;
            }
            //hit_pos = tri_v;
        }
    }
    if (any_hit)
    {
        t_hit = hit_t;
        pos = hit_pos;
    }
    return any_hit;
}

[PreferRecompute]
bool triangle_hit_with_normal(float3 rayo, float3 rayd, float t_min, float t_max, float3 v0, float3 v1, float3 v2, inout float t_hit, inout float3 pos, inout float3 normal)
{
    float epsilon = 1e-15;
    float3 E1 = v1 - v0;
    float3 E2 = v2 - v0;
    float3 P = cross(rayd, E2);
    float det = dot(E1, P);
    if (det > -epsilon && det < epsilon)
        return false;
    float invDet = 1 / det;

    float3 T = rayo - v0;
    float u = dot(T,P) * invDet;
    if (u < 0 || u > 1) return false;

    float3 Q = cross(T, E1);
    float v = dot(rayd, Q) * invDet;
    if (v < 0 || u + v > 1) return false;

    float t = dot(E2, Q) * invDet;
    t_hit = t;
    pos = rayo + t * rayd;

    float3 face_normal = normalize(cross(E1, E2));
    float r = 1.0 - u - v;
    normal = u * face_normal + v * face_normal + r * face_normal;
    if(dot(-rayd, normal)<0)
    {
        normal = -normal;
    }
    normal = normalize(normal);

    return true;
}

[PreferRecompute]
bool bvh_hit_with_normal(TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
             TensorView<float> vert, TensorView<int> v_indx,
    float3 rayo, float3 rayd, float t_min, float t_max, inout float t_hit, inout float3 pos, inout float3 normal)
{
    rayd = normalize(rayd);
    GPUBvhNode stack[MAX_STACK_SIZE];
    int count = 0;

    int root_index = 0;
    /* Add root node */
    GPUBvhNode now_node;
    now_node.index = root_index;
    now_node.left = g_lbvh_info[root_index, 0];
    now_node.right = g_lbvh_info[root_index, 1];
    now_node.primitiveIdx = g_lbvh_info[root_index, 2];
    stack[count++] = now_node;
    float closest_so_far = t_max;
    bool any_hit = false;
    float hit_t = 0.f;
    float3 hit_pos = float3(0.f);
    float3 hit_normal = float3(1.f);

    while (count > 0) {
        /* Pop the last node from the stack */
        GPUBvhNode node = stack[--count];
        float aabbMinX = g_lbvh_aabb[node.index, 0];
        float aabbMinY = g_lbvh_aabb[node.index, 1];
        float aabbMinZ = g_lbvh_aabb[node.index, 2];
        float aabbMaxX = g_lbvh_aabb[node.index, 3];
        float aabbMaxY = g_lbvh_aabb[node.index, 4];
        float aabbMaxZ = g_lbvh_aabb[node.index, 5];

        if (!aabb_hit(rayo, rayd, t_min, closest_so_far, float3(aabbMinX, aabbMinY, aabbMinZ), float3(aabbMaxX, aabbMaxY, aabbMaxZ)))
            continue;

        if (node.left != 0 && node.right !=0) {
            /* Add both nodes to the stack for further processing */
            GPUBvhNode left_node;
            left_node.index = node.left;
            left_node.left = g_lbvh_info[node.left, 0];
            left_node.right = g_lbvh_info[node.left, 1];
            left_node.primitiveIdx = g_lbvh_info[node.left, 2];

            GPUBvhNode right_node;
            right_node.index = node.right;
            right_node.left = g_lbvh_info[node.right, 0];
            right_node.right = g_lbvh_info[node.right, 1];
            right_node.primitiveIdx = g_lbvh_info[node.right, 2];

            stack[count++] = left_node;
            stack[count++] = right_node;
        } else if (node.left == 0 && node.right == 0)
        {
            /* It's a child node, we should test against the triangles and save the closest */
            //any_hit = true;
            uint now_pri_idx = node.primitiveIdx;
            int3 v_in = int3(v_indx[now_pri_idx, 0], v_indx[now_pri_idx, 1], v_indx[now_pri_idx, 2]);
            float3 tri_v0 = float3(vert[v_in.x, 0], vert[v_in.x, 1], vert[v_in.x, 2]);
            float3 tri_v1 = float3(vert[v_in.y, 0], vert[v_in.y, 1], vert[v_in.y, 2]);
            float3 tri_v2 = float3(vert[v_in.z, 0], vert[v_in.z, 1], vert[v_in.z, 2]);
            float now_t_hit = 0.f;
            float3 now_pos_hit = float3(0.f);
            float3 now_normal_hit = float3(1.f);
            bool hit = triangle_hit_with_normal(rayo, rayd, t_min, closest_so_far, tri_v0, tri_v1, tri_v2, now_t_hit, now_pos_hit, now_normal_hit);
            closest_so_far = hit ? min(now_t_hit, closest_so_far) : closest_so_far;
            if (hit) {
                any_hit = true;
                hit_t = closest_so_far;
                hit_pos = rayo + hit_t * rayd;
                if(now_t_hit<=closest_so_far)
                    hit_normal = now_normal_hit;  
            }
            //hit_pos = tri_v;
        }
    }
    if (any_hit)
    {
        t_hit = hit_t;
        pos = hit_pos;
        normal = hit_normal;
    }
    return any_hit;
}

[AutoPyBindCUDA]
[CUDAKernel]
void intersect(int num_rays, TensorView<float> rays_o, TensorView<float> rays_d,
               TensorView<int> g_lbvh_info, TensorView<float> g_lbvh_aabb,
               TensorView<float> vert, TensorView<int> v_indx,
               TensorView<int> hit_map, TensorView<float> hit_pos_map)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint ray_index = dispatchIdx.x;
    if (ray_index >= num_rays) return;

    float3 rayo = float3(rays_o[ray_index, 0], rays_o[ray_index, 1], rays_o[ray_index, 2]);
    float3 rayd = float3(rays_d[ray_index, 0], rays_d[ray_index, 1], rays_d[ray_index, 2]);
    rayd = normalize(rayd);

    float t_min = 0.f;
    float t_max = 1e9;
    float t_hit = 0.f;
    float3 pos = float3(0.f);
    bool hit = bvh_hit(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
                       rayo, rayd, t_min, t_max, t_hit, pos);
    //float3 normal = float3(0.f);
    //bool hit = bvh_hit_with_normal(g_lbvh_info, g_lbvh_aabb, vert, v_indx,
    //                   rayo, rayd, t_min, t_max, t_hit, pos, normal);

    if (hit)
    {
        hit_map[ray_index, 0] = 1;
        hit_pos_map[ray_index, 0] = pos.x;
        hit_pos_map[ray_index, 1] = pos.y;
        hit_pos_map[ray_index, 2] = pos.z;

        //hit_pos_map[ray_index, 0] = normal.x*0.5+0.5;
        //hit_pos_map[ray_index, 1] = normal.y*0.5+0.5;
        //hit_pos_map[ray_index, 2] = normal.z*0.5+0.5;
    }
}