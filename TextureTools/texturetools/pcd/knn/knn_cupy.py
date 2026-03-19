import warnings
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx.scipy.spatial.distance import distance_matrix
from cupyx.scipy.sparse import coo_matrix
import numpy as np


################################################
# copied from https://github.com/cupy/cupy/blob/main/cupyx/scipy/spatial/_kdtree_utils.py


def _get_typename(dtype):
    typename = get_typename(dtype)
    if cupy.dtype(dtype).kind == 'c':
        typename = 'thrust::' + typename
    elif typename == 'float16':
        if runtime.is_hip:
            # 'half' in name_expressions weirdly raises
            # HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID in getLoweredName() on
            # ROCm
            typename = '__half'
        else:
            typename = 'half'
    return typename


FLOAT_TYPES = [cupy.float16, cupy.float32, cupy.float64]
INT_TYPES = [cupy.int8, cupy.int16, cupy.int32, cupy.int64]
UNSIGNED_TYPES = [cupy.uint8, cupy.uint16, cupy.uint32, cupy.uint64]
COMPLEX_TYPES = [cupy.complex64, cupy.complex128]
TYPES = FLOAT_TYPES + INT_TYPES + UNSIGNED_TYPES + COMPLEX_TYPES  # type: ignore  # NOQA
TYPE_NAMES = [_get_typename(t) for t in TYPES]


KD_KERNEL = r'''
#include <cupy/math_constants.h>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

__device__ long long sb(
        const long long s_level, const int n,
        const int num_levels, const long long s) {
    long long num_settled = (1 << s_level) - 1;
    long long num_remaining = num_levels - s_level;

    long long first_node = num_settled;
    long long nls_s = s - first_node;
    long long num_to_left = nls_s * ((1 << num_remaining) - 1);
    long long num_to_left_last = nls_s * (1 << (num_remaining - 1));

    long long total_last = n - ((1 << (num_levels - 1)) - 1);
    long long num_left = min(total_last, num_to_left_last);
    long long num_missing = num_to_left_last - num_left;

    long long sb_s_l = num_settled + num_to_left - num_missing;
    return sb_s_l;
}

__device__ long long ss(
        const int n, const int num_levels,
        const long long s) {

    if(s >= n) {
        return 0;
    }

    long long level = 63 - __clzll(s + 1);
    long long num_level_subtree = num_levels - level;

    long long first = (s + 1) << (num_level_subtree - 1);
    long long on_last = (1 << (num_level_subtree - 1)) - 1;
    long long fllc_s = first + on_last;

    long long val = fllc_s - n;
    long long hi = 1 << (num_level_subtree - 1);
    long long lowest_level = max(min(val, hi), 0ll);

    long long num_nodes = (1 << num_level_subtree) - 1;
    long long ss_s = num_nodes - lowest_level;
    return ss_s;
}

__global__ void update_tags(
        const int n, const int level, long long* tags) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int level_size = (1 << level) - 1;
    if(idx >= n || idx < level_size) {
        return;
    }

    const int num_levels = 32 - __clz(n);

    long long tag = tags[idx];
    long long left_child = 2 * tag + 1;
    long long right_child = 2 * tag + 2;
    long long subtree_size = ss(n, num_levels, left_child);
    long long segment_begin = sb(level, n, num_levels, tag);
    long long pivot_pos = segment_begin + subtree_size;
    if(idx < pivot_pos) {
        tags[idx] = left_child;
    } else if(idx > pivot_pos) {
        tags[idx] = right_child;
    }

}

__device__ half max(half a, half b) {
    return __hmax(a, b);
}

__device__ half min(half a, half b) {
    return __hmin(a, b);
}

template<typename T>
__global__ void compute_bounds(
        const int n, const int n_dims,
        const int level, const int level_sz,
        const T* __restrict__ tree,
        T* bounds) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= level_sz) {
        return;
    }

    int level_start = (1 << level) - 1;
    idx += level_start;

    if(idx >= n) {
        return;
    }

    const int l_child = 2 * idx + 1;
    const int r_child = 2 * idx + 2;

    T* this_bounds = bounds + 2 * n_dims * idx;
    T* left_bounds = bounds + 2 * n_dims * l_child;
    T* right_bounds = bounds + 2 * n_dims * r_child;

    if(l_child >= n && r_child >= n) {
        const T* tree_node = tree + n_dims * idx;
        for(int dim = 0; dim < n_dims; dim++) {
            T* dim_bounds = this_bounds + 2 * dim;
            dim_bounds[0] = tree_node[dim];
            dim_bounds[1] = tree_node[dim];
        }
    } else if (l_child >= n || r_child >= n) {
        T* to_copy = right_bounds;
        if(r_child >= n) {
            to_copy = left_bounds;
        }

        for(int dim = 0; dim < n_dims; dim++) {
            T* dim_bounds = this_bounds + 2 * dim;
            T* to_copy_dim_bounds = to_copy + 2 * dim;
            dim_bounds[0] = to_copy_dim_bounds[0];
            dim_bounds[1] = to_copy_dim_bounds[1];
        }
    } else {
        for(int dim = 0; dim < n_dims; dim++) {
            T* dim_bounds = this_bounds + 2 * dim;
            T* left_dim_bounds = left_bounds + 2 * dim;
            T* right_dim_bounds = right_bounds + 2 * dim;

            dim_bounds[0] = min(left_dim_bounds[0], right_dim_bounds[0]);
            dim_bounds[1] = max(left_dim_bounds[1], right_dim_bounds[1]);
        }
    }
}

__global__ void tag_pairs(
        const int n, const int n_pairs,
        const long long* __restrict__ pair_count,
        const long long* __restrict__ pairs,
        const long long* __restrict__ out_off,
        long long* out) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_pairs) {
        return;
    }

    const long long cur_count = pair_count[idx];
    const long long prev_off = idx == 0 ? 0 : out_off[idx - 1];
    const long long* cur_pairs = pairs + n * idx;
    long long* cur_out = out + prev_off * 2;

    for(int i = 0; i < cur_count; i++) {
        cur_out[2 * i] = idx;
        cur_out[2 * i + 1] = cur_pairs[i];
    }
}
'''

KNN_KERNEL = r'''
#include <cupy/math_constants.h>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

__device__ unsigned long long abs(unsigned long long x) {
    return x;
}

__device__ unsigned int abs(unsigned int x) {
    return x;
}

__device__ half abs(half x) {
    return __habs(x);
}

template<typename T>
__device__ double compute_distance_inf(
        const T* __restrict__ point1, const T* __restrict__ point2,
        const double* __restrict__ box_bounds,
        const int n_dims, const double p, const int stride) {

    double dist = p == CUDART_INF ? -CUDART_INF : CUDART_INF;
    for(int i = 0; i < n_dims; i++) {
        double diff = abs(point1[i] - point2[i * stride]);
        double dim_bound = box_bounds[i];

        if(diff > dim_bound - diff) {
            diff = dim_bound - diff;
        }

        if(p == CUDART_INF) {
            dist = max(dist, diff);
        } else {
            dist = min(dist, diff);
        }
    }
    return dist;
}

template<typename T>
__device__ double compute_distance(
        const T* __restrict__ point1, const T* __restrict__ point2,
        const double* __restrict__ box_bounds,
        const int n_dims, const double p, const int stride,
        const bool take_root) {

    if(abs(p) == CUDART_INF) {
        return compute_distance_inf<T>(
            point1, point2, box_bounds, n_dims, p, stride);
    }

    double dist = 0.0;
    for(int i = 0; i < n_dims; i++) {
        double diff = abs(point1[i] - point2[i * stride]);
        double dim_bound = box_bounds[i];
        if(diff > dim_bound - diff) {
            diff = dim_bound - diff;
        }
        dist += pow(diff, p);
    }

    if(take_root) {
        dist = pow(dist, 1.0 / p);
    }
    return dist;
}

template<typename T>
__device__ T insort(
        const long long curr, const T dist, const int k, const int n,
        T* distances, long long* nodes, bool check) {

    if(check && dist > distances[k - 1]) {
        return distances[k - 1];
    }

    long long left = 0;
    long long right = k - 1;

    while(left != right) {
        long long pos = (left + right) / 2;
        if(distances[pos] < dist) {
            left = pos + 1;
        } else {
            right = pos;
        }
    }

    long long node_to_insert = curr;
    T dist_to_insert = dist;
    T dist_to_return = dist;

    for(long long i = left; i < k; i++) {
        long long node_tmp = nodes[i];
        T dist_tmp = distances[i];

        nodes[i] = node_to_insert;
        distances[i] = dist_to_insert;

        dist_to_return = max(dist_to_return, distances[i]);
        node_to_insert = node_tmp;
        dist_to_insert = dist_tmp;

    }

    return dist_to_return;
}

template<typename T>
__device__ double min_bound_dist(
        const T* __restrict__ point_bounds, const T point_dim,
        const double dim_bound, const int dim) {
    const T min_bound = point_bounds[0];
    const T max_bound = point_bounds[1];

    double min_dist = abs(min_bound - point_dim);
    min_dist = min(min_dist, dim_bound - min_dist);

    double max_dist = abs(max_bound - point_dim);
    max_dist = min(max_dist, dim_bound - max_dist);
    return min(min_dist, max_dist);
}

template<typename T>
__device__ void compute_knn(
        const int k, const int n, const int n_dims, const double eps,
        const double p, const double dist_bound, const bool periodic,
        const T* __restrict__ point, const T* __restrict__ tree,
        const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        const T* __restrict__ tree_bounds,
        double* distances, long long* nodes) {

    volatile long long prev = -1;
    volatile long long curr = 0;
    volatile double radius = !isinf(p) ? pow(dist_bound, p) : dist_bound;
    int visit_count = 0;

    double epsfac = 1.0;
    if(eps != 0) {
        if(p == 2) {
            epsfac = 1.0 / ((1 + eps) * (1 + eps));
        } else if(isinf(p) || p == 1) {
            epsfac = 1.0 / (1 + eps);
        } else {
            epsfac = 1.0 / pow(1 + eps, p);
        }
    }

    while(true) {
        const long long parent = (curr + 1) / 2 - 1;
        if(curr >= n) {
            prev = curr;
            curr = parent;
            continue;
        }

        const long long child = 2 * curr + 1;
        const long long r_child = 2 * curr + 2;

        const bool from_child = prev >= child;
        const T* cur_point = tree + n_dims * curr;

        if(!from_child) {
            const double dist = compute_distance(
                point, cur_point, box_bounds, n_dims, p, 1, false);

            if(dist <= radius) {
                radius = insort<double>(
                    index[curr], dist, k, n, distances, nodes, true);
            }
        }

        const long long cur_level = 63 - __clzll(curr + 1);
        const long long cur_dim = cur_level % n_dims;
        double curr_dim_dist = abs(point[cur_dim] - cur_point[cur_dim]);
        double overflow_dist = box_bounds[cur_dim] - curr_dim_dist;
        bool overflow = curr_dim_dist > overflow_dist;
        curr_dim_dist = overflow ? overflow_dist : curr_dim_dist;
        curr_dim_dist = !isinf(p) ? pow(curr_dim_dist, p) : curr_dim_dist;

        volatile long long cur_close_child = child;
        volatile long long cur_far_child = r_child;

        if(point[cur_dim] > cur_point[cur_dim]) {
            cur_close_child = r_child;
            cur_far_child = child;
        }

        long long next = -1;
        if(prev == cur_close_child) {
            if(periodic) {
                const T* close_child = tree + n_dims * cur_close_child;
                const T* far_child = tree + n_dims * cur_far_child;
                const T* close_bounds = (
                    tree_bounds + 2 * n_dims * cur_close_child + 2 * cur_dim);
                const T* far_bounds = (
                    tree_bounds + 2 * n_dims * cur_far_child + 2 * cur_dim);

                double far_dist = CUDART_INF;
                double close_dist = CUDART_INF;
                double far_bound_dist = CUDART_INF;
                double close_bound_dist = CUDART_INF;

                double curr_dist = compute_distance(
                    point, cur_point, box_bounds, n_dims, p, 1, false);

                if(cur_far_child < n) {
                    far_dist = compute_distance(
                        point, far_child, box_bounds, n_dims, p, 1, false);

                    far_bound_dist = min_bound_dist(
                        far_bounds, point[cur_dim], box_bounds[cur_dim],
                        cur_dim);
                }

                close_dist = compute_distance(
                    point, close_child, box_bounds, n_dims, p, 1, false);

                close_bound_dist = min_bound_dist(
                    close_bounds, point[cur_dim], box_bounds[cur_dim],
                    cur_dim);

                next
                = ((cur_far_child < n) &&
                   ((curr_dim_dist <= radius * epsfac) ||
                    (far_bound_dist <= curr_dim_dist * epsfac) ||
                    (far_dist <= close_dist * epsfac) ||
                    (far_bound_dist <= close_bound_dist + epsfac) ||
                    (far_bound_dist <= radius * epsfac)))
                ? cur_far_child
                : parent;
            } else {
                next
                = ((cur_far_child < n) &&
                   (curr_dim_dist <= radius * epsfac))
                ? cur_far_child
                : parent;
            }
        } else if (prev == cur_far_child) {
            next = parent;
        } else {
            next = (child < n) ? cur_close_child : parent;
        }

        if(next == -1) {
            return;
        }

        prev = curr;
        curr = next;
    }
}

template<typename T>
__global__ void knn(
        const int k, const int n, const int points_size, const int n_dims,
        const double eps, const double p, const double dist_bound,
        const T* __restrict__ points, const T* __restrict__ tree,
        const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        const T* __restrict__ tree_bounds,
        double* all_distances, long long* all_nodes) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= points_size) {
        return;
    }

    const T* point = points + n_dims * idx;
    double* distances = all_distances + k * idx;
    long long* nodes = all_nodes + k * idx;

    compute_knn<T>(k, n, n_dims, eps, p, dist_bound, false, point,
                   tree, index, box_bounds, tree_bounds,
                   distances, nodes);
}

__device__ void adjust_to_box(
        double* point, const int n_dims,
        const double* __restrict__ box_bounds) {
    for(int i = 0; i < n_dims; i++) {
        double dim_value = point[i];
        const double dim_box_bounds = box_bounds[i];
        if(dim_box_bounds > 0) {
            const double r = floor(dim_value / dim_box_bounds);
            double x1 = dim_value - r * dim_box_bounds;
            while(x1 >= dim_box_bounds) x1 -= dim_box_bounds;
            while(x1 < 0) x1 += dim_box_bounds;
            point[i] = x1;
        }
    }
}

__global__ void knn_periodic(
        const int k, const int n, const int points_size, const int n_dims,
        const double eps, const double p, const double dist_bound,
        double* __restrict__ points, const double* __restrict__ tree,
        const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        const double* __restrict__ tree_bounds,
        double* all_distances, long long* all_nodes) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= points_size) {
        return;
    }

    double* point = points + n_dims * idx;
    double* distances = all_distances + k * idx;
    long long* nodes = all_nodes + k * idx;

    adjust_to_box(point, n_dims, box_bounds);
    compute_knn<double>(k, n, n_dims, eps, p, dist_bound, true, point,
                        tree, index, box_bounds, tree_bounds,
                        distances, nodes);
}

template<typename T>
__device__ long long compute_query_ball(
        const int n, const int n_dims, const double radius, const double eps,
        const double p, bool periodic, int sort, const T* __restrict__ point,
        const T* __restrict__ tree, const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        const T* __restrict__ tree_bounds,
        long long* nodes) {

    volatile long long prev = -1;
    volatile long long curr = 0;
    long long node_count = 0;
    double radius_p = !isinf(p) ? pow(radius, p) : radius;

    while(true) {
        const long long parent = (curr + 1) / 2 - 1;
        if(curr >= n) {
            prev = curr;
            curr = parent;
            continue;
        }

        const long long child = 2 * curr + 1;
        const long long r_child = 2 * curr + 2;

        const bool from_child = prev >= child;
        const T* cur_point = tree + n_dims * curr;

        if(!from_child) {
            const double dist = compute_distance(
                point, cur_point, box_bounds, n_dims, p, 1, false);

            if(dist <= radius_p) {
                if(sort) {
                    insort<long long>(
                        index[curr], index[curr], n, n, nodes, nodes, false);
                } else {
                    nodes[node_count] = index[curr];
                }

                node_count++;
            }
        }

        const long long cur_level = 63 - __clzll(curr + 1);
        const long long cur_dim = cur_level % n_dims;

        double curr_dim_dist = abs(point[cur_dim] - cur_point[cur_dim]);
        double overflow_dist = box_bounds[cur_dim] - curr_dim_dist;
        bool overflow = curr_dim_dist > overflow_dist;
        curr_dim_dist = overflow ? overflow_dist : curr_dim_dist;
        curr_dim_dist = !isinf(p) ? pow(curr_dim_dist, p) : curr_dim_dist;

        volatile long long cur_close_child = child;
        volatile long long cur_far_child = r_child;

        if(point[cur_dim] > cur_point[cur_dim]) {
            cur_close_child = r_child;
            cur_far_child = child;
        }

        long long next = -1;
        if(prev == cur_close_child) {
            if(periodic) {
                const T* close_child = tree + n_dims * cur_close_child;
                const T* far_child = tree + n_dims * cur_far_child;
                const T* close_bounds = (
                    tree_bounds + 2 * n_dims * cur_close_child + 2 * cur_dim);
                const T* far_bounds = (
                    tree_bounds + 2 * n_dims * cur_far_child + 2 * cur_dim);

                double far_dist = CUDART_INF;
                double close_dist = CUDART_INF;
                double far_bound_dist = CUDART_INF;
                double close_bound_dist = CUDART_INF;

                double curr_dist = compute_distance(
                    point, cur_point, box_bounds, n_dims, p, 1, false);

                if(cur_far_child < n) {
                    far_dist = compute_distance(
                        point, far_child, box_bounds, n_dims, p, 1, false);

                    far_bound_dist = min_bound_dist(
                        far_bounds, point[cur_dim], box_bounds[cur_dim],
                        cur_dim);
                }

                close_dist = compute_distance(
                    point, close_child, box_bounds, n_dims, p, 1, false);

                close_bound_dist = min_bound_dist(
                    close_bounds, point[cur_dim], box_bounds[cur_dim],
                    cur_dim);

                next
                = ((cur_far_child < n) &&
                   ((curr_dim_dist <= radius_p * (1 + eps)) ||
                    (far_bound_dist <= curr_dim_dist * (1 + eps)) ||
                    (far_dist <= close_dist * (1 + eps)) ||
                    (far_bound_dist <= close_bound_dist + (1 + eps)) ||
                    (far_bound_dist <= radius_p * (1 + eps))))
                ? cur_far_child
                : parent;
            } else {
                next
                = ((cur_far_child < n) &&
                   (curr_dim_dist <= radius_p * (1 + eps)))
                ? cur_far_child
                : parent;
            }
        } else if (prev == cur_far_child) {
            next = parent;
        } else {
            next = (child < n) ? cur_close_child : parent;
        }

        prev = curr;
        curr = next;

        if(next == -1) {
            return node_count;
        }
    }
}

template<typename T>
__global__ void query_ball(
        const int n, const int points_size, const int n_dims,
        const double radius, const double eps, const double p, const int sort,
        const T* __restrict__ points, const T* __restrict__ tree,
        const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        const T* __restrict__ tree_bounds,
        long long* all_nodes, long long* node_count) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= points_size) {
        return;
    }

    const T* point = points + n_dims * idx;
    long long* nodes = all_nodes + n * idx;

    long long count = compute_query_ball<T>(
        n, n_dims, radius, eps, p, false, sort, point, tree, index, box_bounds,
        tree_bounds, nodes);

    node_count[idx] = count;
}

__global__ void query_ball_periodic(
        const int n, const int points_size, const int n_dims,
        const double radius, const double eps, const double p, const int sort,
        double* __restrict__ points, const double* __restrict__ tree,
        const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        const double* __restrict__ tree_bounds,
        long long* all_nodes, long long* node_count) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= points_size) {
        return;
    }

    double* point = points + n_dims * idx;
    long long* nodes = all_nodes + n * idx;

    adjust_to_box(point, n_dims, box_bounds);
    long long count = compute_query_ball<double>(
        n, n_dims, radius, eps, p, true, sort, point, tree, index, box_bounds,
        tree_bounds, nodes);

    node_count[idx] = count;
}
'''


KD_MODULE = cupy.RawModule(
    code=KD_KERNEL, options=('-std=c++11',),
    name_expressions=['update_tags', 'tag_pairs'] + [
        f'compute_bounds<{x}>' for x in TYPE_NAMES])

KNN_MODULE = cupy.RawModule(
    code=KNN_KERNEL, options=('-std=c++11',),
    name_expressions=['knn_periodic', 'query_ball_periodic'] +
    [f'knn<{x}>' for x in TYPE_NAMES] +
    [f'query_ball<{x}>' for x in TYPE_NAMES])


def _get_module_func(module, func_name, *template_args):
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


def asm_kd_tree(points):
    """
    Build an array-based KD-Tree from a given set of points.

    Parameters
    ----------
    points: ndarray
        Array input of size (m, n) which contains `m` points with dimension
        `n`.

    Returns
    -------
    tree: ndarray
        An array representation of a left balanced, dimension alternating
        KD-Tree of the input points.
    indices: ndarray
        An index array that maps the original input to its corresponding
        KD-Tree representation.

    Notes
    -----
    This algorithm is derived from [1]_.

    References
    ----------
    .. [1] Wald, I., GPU-friendly, Parallel, and (Almost-)In-Place
           Construction of Left-Balanced k-d Trees, 2022.
           doi:10.48550/arXiv.2211.00120.
    """
    x = points.copy()
    track_idx = cupy.arange(x.shape[0], dtype=cupy.int64)
    tags = cupy.zeros(x.shape[0], dtype=cupy.int64)
    length = x.shape[0]
    dims = x.shape[1]
    n_iter = int(np.log2(length))

    block_sz = 128
    n_blocks = (length + block_sz - 1) // block_sz
    update_tags = KD_MODULE.get_function('update_tags')
    x_tags = cupy.empty((2, length), dtype=x.dtype)

    level = 0
    for level in range(n_iter):
        dim = level % dims
        x_tags[0, :] = x[:, dim]
        x_tags[1, :] = tags
        idx = cupy.lexsort(x_tags)
        x = x[idx]
        tags = tags[idx]
        track_idx = track_idx[idx]
        update_tags((n_blocks,), (block_sz,), (length, level, tags))

    if n_iter > 1:
        level += 1

    dim = level % dims
    x_tags[0, :] = x[:, dim]
    x_tags[1, :] = tags
    idx = cupy.lexsort(x_tags)
    x = x[idx]
    track_idx = track_idx[idx]
    return x, track_idx


def compute_tree_bounds(tree):
    n, n_dims = tree.shape
    bounds = cupy.empty((n, n_dims, 2), dtype=tree.dtype)
    n_levels = int(np.log2(n))
    compute_bounds = _get_module_func(KD_MODULE, 'compute_bounds', tree)

    block_sz = 128
    for level in range(n_levels, -1, -1):
        level_sz = 2 ** level
        n_blocks = (level_sz + block_sz - 1) // block_sz
        compute_bounds(
            (n_blocks,), (block_sz,),
            (n, n_dims, level, level_sz, tree, bounds))
    return bounds


def compute_knn(points, tree, index, boxdata, bounds, k=1, eps=0.0, p=2.0,
                distance_upper_bound=cupy.inf, adjust_to_box=False):
    max_k = int(np.max(k))
    points_shape = points.shape

    if points.ndim > 2:
        points = points.reshape(-1, points_shape[-1])
        if not points.flags.c_contiguous:
            points = points.copy()

    if points.ndim == 1:
        n_points = 1
        n_dims = points.shape[0]
    else:
        n_points, n_dims = points.shape

    if n_dims != tree.shape[-1]:
        raise ValueError('The number of dimensions of the query points must '
                         'match with the tree ones. '
                         f'Expected {tree.shape[-1]}, got: {n_dims}')

    if cupy.dtype(points.dtype) is not cupy.dtype(tree.dtype):
        raise ValueError('Query points dtype must match the tree one.')

    distances = cupy.full((n_points, max_k), cupy.inf, dtype=cupy.float64)
    nodes = cupy.full((n_points, max_k), tree.shape[0], dtype=cupy.int64)

    block_sz = 128
    n_blocks = (n_points + block_sz - 1) // block_sz
    knn_fn, fn_args = (
        ('knn', (points,)) if not adjust_to_box else ('knn_periodic', tuple()))
    knn = _get_module_func(KNN_MODULE, knn_fn, *fn_args)
    knn((n_blocks,), (block_sz,),
        (max_k, tree.shape[0], n_points, n_dims, eps, p, distance_upper_bound,
         points, tree, index, boxdata, bounds, distances, nodes))

    if not isinstance(k, int):
        indices = [k_i - 1 for k_i in k]
        distances = distances[:, indices]
        nodes = nodes[:, indices]

    if len(points_shape) > 2:
        distances = distances.reshape(*points_shape[:-1], -1)
        nodes = nodes.reshape(*points_shape[:-1], -1)

    if len(points_shape) == 1:
        distances = cupy.squeeze(distances, 0)
        nodes = cupy.squeeze(nodes, 0)

    if k == 1 and len(points_shape) > 1:
        distances = cupy.squeeze(distances, -1)
        nodes = cupy.squeeze(nodes, -1)

    if not cupy.isinf(p):
        distances = distances ** (1.0 / p)
    return distances, nodes


def find_nodes_in_radius(points, tree, index, boxdata, bounds, r,
                         p=2.0, eps=0, return_sorted=None, return_length=False,
                         adjust_to_box=False, return_tuples=False):
    points_shape = points.shape
    tree_length = tree.shape[0]

    if points.ndim > 2:
        points = points.reshape(-1, points_shape[-1])
        if not points.flags.c_contiguous:
            points = points.copy()

    if points.ndim == 1:
        n_points = 1
        n_dims = points.shape[0]
    else:
        n_points, n_dims = points.shape

    if n_dims != tree.shape[-1]:
        raise ValueError('The number of dimensions of the query points must '
                         'match with the tree ones. '
                         f'Expected {tree.shape[-1]}, got: {n_dims}')

    if points.dtype != tree.dtype:
        raise ValueError('Query points dtype must match the tree one.')

    nodes = cupy.full((n_points, tree_length), tree.shape[0], dtype=cupy.int64)
    total_nodes = cupy.empty((n_points,), cupy.int64)

    return_sorted = 1 if return_sorted is None else return_sorted

    block_sz = 128
    n_blocks = (n_points + block_sz - 1) // block_sz
    query_ball_fn, fn_args = (
        ('query_ball', (points,)) if not adjust_to_box else
        ('query_ball_periodic', tuple()))
    query_ball = _get_module_func(KNN_MODULE, query_ball_fn, *fn_args)
    query_ball((n_blocks,), (block_sz,),
               (tree_length, n_points, n_dims, float(r), eps, float(p),
                int(return_sorted),
                points, tree, index, boxdata, bounds, nodes,
                total_nodes))

    if return_length:
        return total_nodes
    elif not return_tuples:
        split_nodes = cupy.array_split(
            nodes[nodes != tree_length], total_nodes.cumsum().tolist())
        split_nodes = split_nodes[:n_points]
        return split_nodes
    else:
        cum_total = total_nodes.cumsum()
        n_pairs = int(cum_total[-1])
        result = cupy.empty((n_pairs, 2), dtype=cupy.int64)
        tag_pairs = KD_MODULE.get_function('tag_pairs')
        tag_pairs((n_blocks,), (block_sz,),
                  (tree_length, n_points, total_nodes, nodes,
                   cum_total, result))
        return result[result[:, 0] < result[:, 1]]


################################################
# copied from https://github.com/cupy/cupy/blob/main/cupyx/scipy/spatial/_kdtree.py


def broadcast_contiguous(x, shape, dtype):
    """Broadcast ``x`` to ``shape`` and make contiguous, possibly by copying"""
    # Avoid copying if possible
    try:
        if x.shape == shape:
            return cupy.ascontiguousarray(x, dtype)
    except AttributeError:
        pass
    # Assignment will broadcast automatically
    ret = cupy.empty(shape, dtype)
    ret[...] = x
    return ret


class KDTree:
    """
    KDTree(data, leafsize=16, compact_nodes=True, copy_data=False,
            balanced_tree=True, boxsize=None)

    kd-tree for quick nearest-neighbor lookup

    This class provides an index into a set of k-dimensional points
    which can be used to rapidly look up the nearest neighbors of any
    point.

    Parameters
    ----------
    data : array_like, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles, and so modifying this data will result in
        bogus results. The data are also copied if the kd-tree is built
        with copy_data=True.
    leafsize : positive int, optional
        The number of points at which the algorithm switches over to
        brute-force. Default: 16.
    compact_nodes : bool, optional
        If True, the kd-tree is built to shrink the hyperrectangles to
        the actual data range. This usually gives a more compact tree that
        is robust against degenerated input data and gives faster queries
        at the expense of longer build time. Default: True.
    copy_data : bool, optional
        If True the data is always copied to protect the kd-tree against
        data corruption. Default: False.
    balanced_tree : bool, optional
        If True, the median is used to split the hyperrectangles instead of
        the midpoint. This usually gives a more compact tree and
        faster queries at the expense of longer build time. Default: True.
    boxsize : array_like or scalar, optional
        Apply a m-d toroidal topology to the KDTree.. The topology is generated
        by :math:`x_i + n_i L_i` where :math:`n_i` are integers and :math:`L_i`
        is the boxsize along i-th dimension. The input data shall be wrapped
        into :math:`[0, L_i)`. A ValueError is raised if any of the data is
        outside of this bound.

    Notes
    -----
    The algorithm used is described in Wald, I. 2022 [1]_.
    The general idea is that the kd-tree is a binary tree, each of whose
    nodes represents an axis-aligned hyperrectangle. Each node specifies
    an axis and splits the set of points based on whether their coordinate
    along that axis is greater than or less than a particular value.

    The tree can be queried for the r closest neighbors of any given point
    (optionally returning only those within some maximum distance of the
    point). It can also be queried, with a substantial gain in efficiency,
    for the r approximate closest neighbors. See [2]_ for more information
    regarding the implementation.

    For large dimensions (20 is already large) do not expect this to run
    significantly faster than brute force. High-dimensional nearest-neighbor
    queries are a substantial open problem in computer science.

    Attributes
    ----------
    data : ndarray, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles. The data are also copied if the kd-tree is built
        with `copy_data=True`.
    leafsize : positive int
        The number of points at which the algorithm switches over to
        brute-force.
    m : int
        The dimension of a single data-point.
    n : int
        The number of data points.
    maxes : ndarray, shape (m,)
        The maximum value in each dimension of the n data points.
    mins : ndarray, shape (m,)
        The minimum value in each dimension of the n data points.
    tree : ndarray
        This attribute exposes the array representation of the tree.
    size : int
        The number of nodes in the tree.

    References
    ----------
    .. [1] Wald, I., GPU-friendly, Parallel, and (Almost-)In-Place
           Construction of Left-Balanced k-d Trees, 2022.
           doi:10.48550/arXiv.2211.00120.
    .. [2] Wald, I., A Stack-Free Traversal Algorithm for Left-Balanced
           k-d Trees, 2022. doi:10.48550/arXiv.2210.12859.
    """

    def __init__(self, data, leafsize=10, compact_nodes=True, copy_data=False,
                 balanced_tree=True, boxsize=None):
        self.data = data
        if copy_data:
            self.data = self.data.copy()

        if not balanced_tree:
            warnings.warn('balanced_tree=False is not supported by the GPU '
                          'implementation of KDTree, skipping.')

        self.copy_query_points = False
        self.n, self.m = self.data.shape

        self.boxsize = cupy.full(self.m, cupy.inf, dtype=cupy.float64)
        # self.boxsize_data = None

        if boxsize is not None:
            # self.boxsize_data = cupy.empty(self.m, dtype=data.dtype)
            self.copy_query_points = True
            boxsize = broadcast_contiguous(boxsize, shape=(self.m,),
                                           dtype=cupy.float64)
            # self.boxsize_data[:self.m] = boxsize
            # self.boxsize_data[self.m:] = 0.5 * boxsize

            self.boxsize = boxsize
            periodic_mask = self.boxsize > 0
            if ((self.data >= self.boxsize[None, :])[:, periodic_mask]).any():
                raise ValueError(
                    "Some input data are greater than the size of the "
                    "periodic box.")
            if ((self.data < 0)[:, periodic_mask]).any():
                raise ValueError("Negative input data are outside of the "
                                 "periodic box.")

        self.tree, self.index = asm_kd_tree(self.data)
        self.bounds = cupy.empty((0,))
        if self.copy_query_points:
            if self.data.dtype != cupy.float64:
                raise ValueError('periodic KDTree is only available '
                                 'on float64')
            self.bounds = compute_tree_bounds(self.tree)

        self.mins = cupy.min(self.tree, axis=0)
        self.maxes = cupy.max(self.tree, axis=0)

    def query(self, x, k=1, eps=0.0, p=2.0, distance_upper_bound=cupy.inf):
        r"""
        Query the kd-tree for nearest neighbors.

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : list of integer or integer
            The list of k-th nearest neighbors to return. If k is an
            integer it is treated as a list of [1, ... k] (range(1, k+1)).
            Note that the counting starts from 1.
        eps : non-negative float
            Return approximate nearest neighbors; the k-th returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real k-th nearest neighbor.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
            A finite large p may cause a ValueError if overflow can occur.
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance.  This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.

        Returns
        -------
        d : array of floats
            The distances to the nearest neighbors.
            If ``x`` has shape ``tuple+(self.m,)``, then ``d`` has shape
            ``tuple+(k,)``. When k == 1, the last dimension of the output is
            squeezed. Missing neighbors are indicated with infinite distances.
        i : ndarray of ints
            The index of each neighbor in ``self.data``.
            If ``x`` has shape ``tuple+(self.m,)``, then ``i`` has shape
            ``tuple+(k,)``. When k == 1, the last dimension of the output is
            squeezed. Missing neighbors are indicated with ``self.n``.

        Notes
        -----
        If the KD-Tree is periodic, the position ``x`` is wrapped into the
        box.

        When the input k is a list, a query for arange(max(k)) is performed,
        but only columns that store the requested values of k are preserved.
        This is implemented in a manner that reduces memory usage.

        Examples
        --------
        >>> import cupy as cp
        >>> from cupyx.scipy.spatial import KDTree
        >>> x, y = cp.mgrid[0:5, 2:8]
        >>> tree = KDTree(cp.c_[x.ravel(), y.ravel()])

        To query the nearest neighbours and return squeezed result, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=1)
        >>> print(dd, ii, sep='\n')
        [2.         0.2236068]
        [ 0 13]

        To query the nearest neighbours and return unsqueezed result, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1])
        >>> print(dd, ii, sep='\n')
        [[2.        ]
         [0.2236068]]
        [[ 0]
         [13]]

        To query the second nearest neighbours and return unsqueezed result,
        use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[2])
        >>> print(dd, ii, sep='\n')
        [[2.23606798]
         [0.80622577]]
        [[ 6]
         [19]]

        To query the first and second nearest neighbours, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=2)
        >>> print(dd, ii, sep='\n')
        [[2.         2.23606798]
         [0.2236068  0.80622577]]
        [[ 0  6]
         [13 19]]

        or, be more specific

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1, 2])
        >>> print(dd, ii, sep='\n')
        [[2.         2.23606798]
         [0.2236068  0.80622577]]
        [[ 0  6]
         [13 19]]

        """
        if self.copy_query_points:
            if x.dtype != cupy.float64:
                raise ValueError('periodic KDTree is only available '
                                 'on float64')
            x = x.copy()

        common_dtype = cupy.result_type(self.tree.dtype, x.dtype)
        tree = self.tree
        if cupy.dtype(self.tree.dtype) is not common_dtype:
            tree = self.tree.astype(common_dtype)
        if cupy.dtype(x.dtype) is not common_dtype:
            x = x.astype(common_dtype)

        if not isinstance(k, list):
            try:
                k = int(k)
            except TypeError:
                raise ValueError('k must be an integer or list of integers')

        return compute_knn(
            x, tree, self.index, self.boxsize, self.bounds, k=k,
            eps=float(eps), p=float(p),
            distance_upper_bound=distance_upper_bound,
            adjust_to_box=self.copy_query_points)

    def query_ball_point(self, x, r, p=2., eps=0, return_sorted=None,
                         return_length=False):
        """
        Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : array_like, float
            The radius of points to return, shall broadcast to the length of x.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
            A finite large p may cause a ValueError if overflow can occur.
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.
        return_sorted : bool, optional
            Sorts returned indices if True and does not sort them if False. If
            None, does not sort single point queries, but does sort
            multi-point queries which was the behavior before this option
            was added in SciPy.
        return_length: bool, optional
            Return the number of points inside the radius instead of a list
            of the indices.

        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.

        Notes
        -----
        If you have many points whose neighbors you want to find, you may save
        substantial amounts of time by putting them in a KDTree and using
        query_ball_tree.

        Examples
        --------
        >>> import cupy as cp
        >>> from cupyx.scipy import spatial
        >>> x, y = cp.mgrid[0:4, 0:4]
        >>> points = cp.c_[x.ravel(), y.ravel()]
        >>> tree = spatial.KDTree(points)
        >>> tree.query_ball_point([2, 0], 1)
        [4, 8, 9, 12]
        """
        if self.copy_query_points:
            if x.dtype != cupy.float64:
                raise ValueError('periodic KDTree is only available '
                                 'on float64')
            x = x.copy()

        common_dtype = cupy.result_type(self.tree.dtype, x.dtype)
        tree = self.tree
        if cupy.dtype(self.tree.dtype) is not common_dtype:
            tree = self.tree.astype(common_dtype)
        if cupy.dtype(x.dtype) is not common_dtype:
            x = x.astype(common_dtype)

        return find_nodes_in_radius(
            x, tree, self.index, self.boxsize, self.bounds,
            r, p=p, eps=eps, return_sorted=return_sorted,
            return_length=return_length, adjust_to_box=self.copy_query_points)

    def query_ball_tree(self, other, r, p=2.0, eps=0.0):
        """
        Find all pairs of points between `self` and `other` whose distance
        is at most r.

        Parameters
        ----------
        other : KDTree instance
            The tree containing points to search against.
        r : float
            The maximum distance, has to be positive.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
            A finite large p may cause a ValueError if overflow can occur.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.

        Returns
        -------
        results : list of ndarrays
            For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            list of the indices of its neighbors in ``other.data``.

        Examples
        --------
        You can search all pairs of points between two kd-trees within a
        distance:

        >>> import matplotlib.pyplot as plt
        >>> import cupy as cp
        >>> from cupyx.scipy.spatial import KDTree
        >>> points1 = cp.random.rand((15, 2))
        >>> points2 = cp.random.rand((15, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points1[:, 0], points1[:, 1], "xk", markersize=14)
        >>> plt.plot(points2[:, 0], points2[:, 1], "og", markersize=14)
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
        >>> for i in range(len(indexes)):
        ...     for j in indexes[i]:
        ...         plt.plot([points1[i, 0], points2[j, 0]],
        ...             [points1[i, 1], points2[j, 1]], "-r")
        >>> plt.show()

        """
        return other.query_ball_point(
            self.data, r, p=p, eps=eps, return_sorted=True)

    def query_pairs(self, r, p=2.0, eps=0, output_type='ndarray'):
        """
        Find all pairs of points in `self` whose distance is at most r.

        Parameters
        ----------
        r : positive float
            The maximum distance.
        p : float, optional
            Which Minkowski norm to use.  ``p`` has to meet the condition
            ``1 <= p <= infinity``.
            A finite large p may cause a ValueError if overflow can occur.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        output_type : string, optional
            Choose the output container, 'set' or 'ndarray'. Default: 'ndarray'
            Note: 'set' output is not supported.

        Returns
        -------
        results : ndarray
            An ndarray of size ``(total_pairs, 2)``, containing each pair
            ``(i,j)``, with ``i < j``, for which the corresponding
            positions are close.

        Notes
        -----
        This method does not support the `set` output type.

        Examples
        --------
        You can search all pairs of points in a kd-tree within a distance:

        >>> import matplotlib.pyplot as plt
        >>> import cupy as cp
        >>> from cupyx.scipy.spatial import KDTree
        >>> points = cp.random.rand((20, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points[:, 0], points[:, 1], "xk", markersize=14)
        >>> kd_tree = KDTree(points)
        >>> pairs = kd_tree.query_pairs(r=0.2)
        >>> for (i, j) in pairs:
        ...     plt.plot([points[i, 0], points[j, 0]],
        ...             [points[i, 1], points[j, 1]], "-r")
        >>> plt.show()

        """
        if output_type == 'set':
            warnings.warn("output_type='set' is not supported by the GPU "
                          "implementation of KDTree, resorting back to "
                          "'ndarray'.")

        x = self.data
        if self.copy_query_points:
            if x.dtype != cupy.float64:
                raise ValueError('periodic KDTree is only available '
                                 'on float64')
            x = x.copy()

        common_dtype = cupy.result_type(self.tree.dtype, x.dtype)
        tree = self.tree
        if cupy.dtype(self.tree.dtype) is not common_dtype:
            tree = self.tree.astype(common_dtype)
        if cupy.dtype(x.dtype) is not common_dtype:
            x = x.astype(common_dtype)

        return find_nodes_in_radius(
            x, tree, self.index, self.boxsize, self.bounds,
            r, p=p, eps=eps, return_sorted=True, return_tuples=True,
            adjust_to_box=self.copy_query_points)

    def count_neighbors(self, other, r, p=2.0, weights=None, cumulative=True):
        """
        Count how many nearby pairs can be formed.

        Count the number of pairs ``(x1,x2)`` can be formed, with ``x1`` drawn
        from ``self`` and ``x2`` drawn from ``other``, and where
        ``distance(x1, x2, p) <= r``.

        Data points on ``self`` and ``other`` are optionally weighted by the
        ``weights`` argument. (See below)

        This is adapted from the "two-point correlation" algorithm described by
        Gray and Moore [1]_.  See notes for further discussion.

        Parameters
        ----------
        other : KDTree instance
            The other tree to draw points from, can be the same tree as self.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched with
            a single tree traversal.
            If the count is non-cumulative(``cumulative=False``), ``r`` defines
            the edges of the bins, and must be non-decreasing.
        p : float, optional
            1<=p<=infinity.
            Which Minkowski p-norm to use.
            Default 2.0.
            A finite large p may cause a ValueError if overflow can occur.
        weights : tuple, array_like, or None, optional
            If None, the pair-counting is unweighted.
            If given as a tuple, weights[0] is the weights of points in
            ``self``, and weights[1] is the weights of points in ``other``;
            either can be None to indicate the points are unweighted.
            If given as an array_like, weights is the weights of points in
            ``self`` and ``other``. For this to make sense, ``self`` and
            ``other`` must be the same tree. If ``self`` and ``other`` are two
            different trees, a ``ValueError`` is raised.
            Default: None
        cumulative : bool, optional
            Whether the returned counts are cumulative. When cumulative is set
            to ``False`` the algorithm is optimized to work with a large number
            of bins (>10) specified by ``r``. When ``cumulative`` is set to
            True, the algorithm is optimized to work with a small number of
            ``r``. Default: True

        Returns
        -------
        result : scalar or 1-D array
            The number of pairs. For unweighted counts, the result is integer.
            For weighted counts, the result is float.
            If cumulative is False, ``result[i]`` contains the counts with
            ``(-inf if i == 0 else r[i-1]) < R <= r[i]``

        """
        raise NotImplementedError('count_neighbors is not available on CuPy')

    def sparse_distance_matrix(self, other, max_distance, p=2.0,
                               output_type='coo_matrix'):
        """
        Compute a sparse distance matrix

        Computes a distance matrix between two KDTrees, leaving as zero
        any distance greater than max_distance.

        Parameters
        ----------
        other : KDTree
        max_distance : positive float
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            A finite large p may cause a ValueError if overflow can occur.
        output_type : string, optional
            Which container to use for output data. Options:
            'coo_matrix' or 'ndarray'. Default: 'coo_matrix'.

        Returns
        -------
        result : coo_matrix or ndarray
            Sparse matrix representing the results in "dictionary of keys"
            format. If output_type is 'ndarray' an NxM distance matrix will be
            returned.

        Examples
        --------
        You can compute a sparse distance matrix between two kd-trees:

        >>> import cupy
        >>> from cupyx.scipy.spatial import KDTree
        >>> points1 = cupy.random.rand((5, 2))
        >>> points2 = cupy.random.rand((5, 2))
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> sdm = kd_tree1.sparse_distance_matrix(kd_tree2, 0.3)
        >>> sdm.toarray()
        array([[0.        , 0.        , 0.12295571, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.28942611, 0.        , 0.        , 0.2333084 , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.24617575, 0.29571802, 0.26836782, 0.        , 0.        ]])

        You can check distances above the `max_distance` are zeros:

        >>> from cupyx.scipy.spatial import distance_matrix
        >>> distance_matrix(points1, points2)
        array([[0.56906522, 0.39923701, 0.12295571, 0.8658745 , 0.79428925],
           [0.37327919, 0.7225693 , 0.87665969, 0.32580855, 0.75679479],
           [0.28942611, 0.30088013, 0.6395831 , 0.2333084 , 0.33630734],
           [0.31994999, 0.72658602, 0.71124834, 0.55396483, 0.90785663],
           [0.24617575, 0.29571802, 0.26836782, 0.57714465, 0.6473269 ]])
        """
        if output_type not in {'coo_matrix', 'ndarray'}:
            raise ValueError(
                "sparse_distance_matrix only supports 'coo_matrix' and "
                "'ndarray' outputs")

        dist = distance_matrix(self.data, other.data, p)
        dist[dist > max_distance] = 0

        if output_type == 'coo_matrix':
            return coo_matrix(dist)
        return dist





