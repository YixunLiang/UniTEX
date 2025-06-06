import torch
from .src.cr_impl import remesh, spatial_smooth

def lerp_unbiased(a:torch.Tensor,b:torch.Tensor,weight:float,step:int):
    """lerp with adam's bias correction"""
    c_prev = 1-weight**(step-1)
    c = 1-weight**step
    a_weight = weight*c_prev/c
    b_weight = (1-weight)/c
    a.mul_(a_weight).add_(b, alpha=b_weight)

class MeshOptimizer:
    """Use this like a pytorch Optimizer, but after calling opt.step(), do vertices,faces = opt.remesh()."""

    def __init__(self, 
            vertices:torch.Tensor, #V,3
            faces:torch.Tensor, #F,3
            lr=0.3, #learning rate
            betas=(0.8,0.8,0), #betas[0:2] are the same as in Adam, betas[2] may be used to time-smooth the relative velocity nu
            gammas=(0,0,0), #optional spatial smoothing for m1,m2,nu, values between 0 (no smoothing) and 1 (max. smoothing)
            nu_ref=0.3, #reference velocity for edge length controller
            edge_len_lims=(.01,.15), #smallest and largest allowed reference edge length
            edge_len_tol=.5, #edge length tolerance for split and collapse
            gain=.2,  #gain value for edge length controller
            laplacian_weight=.02, #for laplacian smoothing/regularization
            ramp=1, #learning rate ramp, actual ramp width is ramp/(1-betas[0])
            grad_lim=10., #gradients are clipped to m1.abs()*grad_lim
            remesh_interval=1, #larger intervals are faster but with worse mesh quality
            local_edgelen=True, #set to False to use a global scalar reference edge length instead
            ):
        self._vertices = vertices
        self._faces = faces
        self._lr = lr
        self._betas = betas
        self._gammas = gammas
        self._nu_ref = nu_ref
        self._edge_len_lims = edge_len_lims
        self._edge_len_tol = edge_len_tol
        self._gain = gain
        self._laplacian_weight = laplacian_weight
        self._ramp = ramp
        self._grad_lim = grad_lim
        self._remesh_interval = remesh_interval
        self._local_edgelen = local_edgelen
        self._step = 0

        V = self._vertices.shape[0]
        # prepare continuous tensor for all vertex-based data 
        self._vertices_etc = torch.zeros([V,9],device=vertices.device)
        self._split_vertices_etc()
        self.vertices.copy_(vertices) #initialize vertices
        self._vertices.requires_grad_()
        self._ref_len.fill_(edge_len_lims[1])

    @property
    def vertices(self):
        return self._vertices

    @property
    def faces(self):
        return self._faces

    def _split_vertices_etc(self):
        self._vertices = self._vertices_etc[:,:3]
        self._m2 = self._vertices_etc[:,3]
        self._nu = self._vertices_etc[:,4]
        self._m1 = self._vertices_etc[:,5:8]
        self._ref_len = self._vertices_etc[:,8]
        
        with_gammas = any(g!=0 for g in self._gammas)
        self._smooth = self._vertices_etc[:,:8] if with_gammas else self._vertices_etc[:,:3]

    def zero_grad(self):
        self._vertices.grad = None

    @torch.no_grad()
    def step(self):
        eps = 1e-8
        self._step += 1

        # spatial smoothing
        neighbor_smooth = spatial_smooth(self._smoot, self._faces) #V,S

        #apply optional smoothing of m1,m2,nu
        if self._gammas[0]:
            self._m1.lerp_(neighbor_smooth[:,5:8],self._gammas[0])
        if self._gammas[1]:
            self._m2.lerp_(neighbor_smooth[:,3],self._gammas[1])
        if self._gammas[2]:
            self._nu.lerp_(neighbor_smooth[:,4],self._gammas[2])

        #add laplace smoothing to gradients
        laplace = self._vertices - neighbor_smooth[:,:3]
        grad = torch.addcmul(self._vertices.grad, laplace, self._nu[:,None], value=self._laplacian_weight)

        #gradient clipping
        if self._step>1:
            grad_lim = self._m1.abs().mul_(self._grad_lim)
            grad.clamp_(min=-grad_lim,max=grad_lim)

        # moment updates
        lerp_unbiased(self._m1, grad, self._betas[0], self._step)
        lerp_unbiased(self._m2, (grad**2).sum(dim=-1), self._betas[1], self._step)

        velocity = self._m1 / self._m2[:,None].sqrt().add_(eps) #V,3
        speed = velocity.norm(dim=-1) #V

        if self._betas[2]:
            lerp_unbiased(self._nu,speed,self._betas[2],self._step) #V
        else:
            self._nu.copy_(speed) #V

        # update vertices
        ramped_lr = self._lr * min(1,self._step * (1-self._betas[0]) / self._ramp)
        self._vertices.add_(velocity * self._ref_len[:,None], alpha=-ramped_lr)

        # update target edge length
        if self._step % self._remesh_interval == 0:
            if self._local_edgelen:
                len_change = (1 + (self._nu - self._nu_ref) * self._gain)
            else:
                len_change = (1 + (self._nu.mean() - self._nu_ref) * self._gain)
            self._ref_len *= len_change
            self._ref_len.clamp_(*self._edge_len_lims)

    def remesh(self, flip:bool=True)->tuple[torch.Tensor,torch.Tensor]:
        min_edge_len = self._ref_len * (1 - self._edge_len_tol)
        max_edge_len = self._ref_len * (1 + self._edge_len_tol)
            
        self._vertices_etc,self._faces = remesh(
            self._vertices_etc[:, :3],
            self._faces,
            vertices_etc=self._vertices_etc[:, 3:],
            min_edge_len=min_edge_len,
            max_edge_len=max_edge_len,
            flip=flip,
        )

        self._split_vertices_etc()
        self._vertices.requires_grad_()

        return self._vertices, self._faces
