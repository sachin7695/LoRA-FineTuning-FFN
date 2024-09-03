import torch
import numpy as np
torch.manual_seed(1337)

d, k = 10, 10

# generating a rank-deficient matrix
W_rank = 2
W = torch.randn(d,W_rank) @ torch.randn(W_rank,k)
# print(W)

W_rank = np.linalg.matrix_rank(W)
# print(f'Rank of W: {W_rank}')

# SVD decomposition of the W matrix.
# SVD on W (W = UxSxV^T)
U, S, V = torch.svd(W)

# For rank-r factorization, keep only the first r singular values (and corresponding columns of U and V)
U_r = U[:, :W_rank]
S_r = torch.diag(S[:W_rank])
V_r = V[:, :W_rank].t()  # Transpose V_r to get the right dimensions

# Compute B = U_r * S_r and A = V_r
B = U_r @ S_r
A = V_r
print(f'Shape of B: {B.shape}')
print(f'Shape of A: {A.shape}')

# Generate random bias and input
bias = torch.randn(d)
x = torch.randn(d)

# Compute y = Wx + bias
y = W @ x + bias
# Compute y' = (B*A)x + bias
y_prime = (B @ A) @ x + bias

print("Original y using W:\n", y)
print("")
print("y' computed using BA:\n", y_prime)

print("Total parameters of W: ", W.nelement())
print("Total parameters of B and A: ", B.nelement() + A.nelement())