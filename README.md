# solve_anything
Solve any (invertible) sparse linear system with very little memory.


This is a proof of concept and is hilariously slow. You might wonder why bother. Because it's fun.


# Problem statement

Sparse direct solvers can take an essentially unbounded amount of memory to solve a sparse linear system even
with extreme sparsity levels. The classic example is 3D+ nearest-neighbor stencil problems (e.g. many PDEs)
where graph separators grow like N^(d-1) for N^d, even though the originating matrix has a tiny fraction of
nonzeros compared to the amount of data stored in the separator schur complements (fill-in). 

On the other hand preconditioned iterative methods, which spare a significant amount of memory
at the cost of needing to iterate (each iteration requiring a full evaluation over the matrix and any preconditioner)
especially when met with highly indefinite or very badly conditioned systems may never reach a satisfactory solution in our lifetime.

The difficulties in preconditioned iterative methods, particularly the black-box variety which start with no outside
information about the matrix except for the matrix itself, arise both from purely mathematical facts about convergence
but also practical details such as the underlying precision which we can actually perform computations.

# A funny solution


to be filled out when code is ready.
