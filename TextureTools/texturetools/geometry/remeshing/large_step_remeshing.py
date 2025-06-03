import torch
from torch.autograd import Function
from cholespy import CholeskySolverF, MatrixType

class CholeskySolver:
    """
    Cholesky solver.

    Precomputes the Cholesky decomposition of the system matrix and solves the
    system by back-substitution.
    """
    def __init__(self, M):
        self.solver = CholeskySolverF(M.shape[0], M.indices()[0], M.indices()[1], M.values(), MatrixType.COO)

    def solve(self, b, backward=False):
        x = torch.zeros_like(b)
        self.solver.solve(b.detach(), x)
        return x

class ConjugateGradientSolver:
    """
    Conjugate gradients solver.
    """
    def __init__(self, M):
        """
        Initialize the solver.

        Parameters
        ----------
        M : torch.sparse_coo_tensor
            Linear system matrix.
        """
        self.guess_fwd = None
        self.guess_bwd = None
        self.M = M

    def solve_axis(self, b, x0):
        """
        Solve a single linear system with Conjugate Gradients.

        Parameters:
        -----------
        b : torch.Tensor
            The right hand side of the system Ax=b.
        x0 : torch.Tensor
            Initial guess for the solution.
        """
        x = x0
        r = self.M @ x - b
        p = -r
        r_norm = r.norm()
        while r_norm > 1e-5:
            Ap = self.M @ p
            r2 = r_norm.square()
            alpha = r2 / (p * Ap).sum(dim=0)
            x = x + alpha*p
            r_old = r
            r_old_norm = r_norm
            r = r + alpha*Ap
            r_norm = r.norm()
            beta = r_norm.square() / r2
            p = -r + beta*p
        return x

    def solve(self, b, backward=False):
        """
        Solve the sparse linear system.

        There is actually one linear system to solve for each axis in b
        (typically x, y and z), and we have to solve each separately with CG.
        Therefore this method calls self.solve_axis for each individual system
        to form the solution.

        Parameters
        ----------
        b : torch.Tensor
            The right hand side of the system Ax=b.
        backward : bool
            Whether we are in the backward or the forward pass.
        """
        if self.guess_fwd is None:
            # Initialize starting guesses in the first run
            self.guess_bwd = torch.zeros_like(b)
            self.guess_fwd = torch.zeros_like(b)

        if backward:
            x0 = self.guess_bwd
        else:
            x0 = self.guess_fwd

        if len(b.shape) != 2:
            raise ValueError(f"Invalid array shape {b.shape} for ConjugateGradientSolver.solve: expected shape (a, b)")

        x = torch.zeros_like(b)
        for axis in range(b.shape[1]):
            # We have to solve for each axis separately for CG to converge
            x[:, axis] = self.solve_axis(b[:, axis], x0[:, axis])

        if backward:
            # Update initial guess for next iteration
            self.guess_bwd = x
        else:
            self.guess_fwd = x

        return x

class DifferentiableSolve(Function):
    """
    Differentiable function to solve the linear system.

    This simply calls the solve methods implemented by the Solver classes.
    """
    @staticmethod
    def forward(ctx, solver, b):
        ctx.solver = solver
        return solver.solve(b, backward=False)

    @staticmethod
    def backward(ctx, grad_output):
        solver_grad = None # We have to return a gradient per input argument in forward
        b_grad = None
        if ctx.needs_input_grad[1]:
            b_grad = ctx.solver.solve(grad_output.contiguous(), backward=True)
        return (solver_grad, b_grad)

def to_differential(L, v):
    """
    Convert vertex coordinates to the differential parameterization.

    Parameters
    ----------
    L : torch.sparse.Tensor
        (I + l*L) matrix
    v : torch.Tensor
        Vertex coordinates
    """
    return L @ v

def get_differential_solver(L, method='Cholesky'):
    """
    Parameters
    ----------
    method : {'Cholesky', 'CG'}
        Solver to use.
    """
    if method == 'Cholesky':
        return CholeskySolver(L)
    elif method == 'CG':
        return ConjugateGradientSolver(L)
    else:
        raise ValueError(f"Unknown solver type '{method}'.")

def from_differential(solver, u):
    """
    Convert differential coordinates back to Cartesian.

    If this is the first time we call this function on a given matrix L, the
    solver is cached. It will be destroyed once the matrix is garbage collected.

    Parameters
    ----------
    solver: differential solver
    u : torch.Tensor
        Differential coordinates
    """
    return DifferentiableSolve.apply(solver, u)

def laplacian_cot(verts, faces):
    """
    Compute the cotangent laplacian

    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """

    # V = sum(V_n), F = sum(F_n)
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=0).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    return L

def laplacian_uniform(verts, faces):
    """
    Compute the uniform laplacian

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device='cuda', dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()

def compute_matrix(verts, faces, lambda_, alpha=None, cotan=False):
    """
    Build the parameterization matrix.

    If alpha is defined, then we compute it as (1-alpha)*I + alpha*L otherwise
    as I + lambda*L as in the paper. The first definition can be slightly more
    convenient as it the scale of the resulting matrix doesn't change much
    depending on alpha.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions
    faces : torch.Tensor
        Triangle faces
    lambda_ : float
        Hyperparameter lambda of our method, used to compute the
        parameterization matrix as (I + lambda_ * L)
    alpha : float in [0, 1[
        Alternative hyperparameter, used to compute the parameterization matrix
        as ((1-alpha) * I + alpha * L)
    cotan : bool
        Compute the cotangent laplacian. Otherwise, compute the combinatorial one
    """
    if cotan:
        L = laplacian_cot(verts, faces)
    else:
        L = laplacian_uniform(verts, faces)

    idx = torch.arange(verts.shape[0], dtype=torch.long, device='cuda')
    eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), torch.ones(verts.shape[0], dtype=torch.float, device='cuda'), (verts.shape[0], verts.shape[0]))
    if alpha is None:
        M = torch.add(eye, lambda_*L) # M = I + lambda_ * L
    else:
        if alpha < 0.0 or alpha >= 1.0:
            raise ValueError(f"Invalid value for alpha: {alpha} : it should take values between 0 (included) and 1 (excluded)")
        M = torch.add((1-alpha)*eye, alpha*L) # M = (1-alpha) * I + alpha * L
    return M.coalesce()

class AdamUniform(torch.optim.Optimizer):
    """
    Variant of Adam with uniform scaling by the second moment.

    Instead of dividing each component by the square root of its second moment,
    we divide all of them by the max.
    """
    def __init__(self, params, lr=0.1, betas=(0.9,0.999)):
        defaults = dict(lr=lr, betas=betas)
        super(AdamUniform, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamUniform, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            for p in group["params"]:
                state = self.state[p]
                # Lazy initialization
                if len(state)==0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1
                grad = p.grad.data

                g1.mul_(b1).add_(grad, alpha=1-b1)
                g2.mul_(b2).add_(grad.square(), alpha=1-b2)
                m1 = g1 / (1-(b1**state["step"]))
                m2 = g2 / (1-(b2**state["step"]))
                # This is the only modification we make to the original Adam algorithm
                gr = m1 / (1e-8 + m2.sqrt().max())
                p.data.sub_(gr, alpha=lr)

