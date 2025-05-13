# 定義 A 矩陣與 b 向量
A = [
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 1, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
]
b = [0, -1, 9, 4, 8, 6]

# 工具函式
def dot(v1, v2):
    return sum(v1[i] * v2[i] for i in range(len(v1)))

def mat_vec_mult(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]

def vec_add(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

def vec_sub(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]

def scalar_mult(c, v):
    return [c * v[i] for i in range(len(v))]

def norm(v):
    return sum(x * x for x in v) ** 0.5

# Jacobi Method
def jacobi(A, b, max_iter=25):
    n = len(A)
    x = [0] * n
    for _ in range(max_iter):
        x_new = x[:]
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        x = x_new
    return x

# Gauss-Seidel Method
def gauss_seidel(A, b, max_iter=25):
    n = len(A)
    x = [0] * n
    for _ in range(max_iter):
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
    return x

# SOR Method
def sor(A, b, omega=1.25, max_iter=25):
    n = len(A)
    x = [0] * n
    for _ in range(max_iter):
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x[i] = (1 - omega) * x[i] + omega * (b[i] - s1 - s2) / A[i][i]
    return x

# Conjugate Gradient Method
def conjugate_gradient(A, b, max_iter=25):
    n = len(A)
    x = [0] * n
    r = vec_sub(b, mat_vec_mult(A, x))
    p = r[:]
    for _ in range(max_iter):
        Ap = mat_vec_mult(A, p)
        alpha = dot(r, r) / dot(p, Ap)
        x = vec_add(x, scalar_mult(alpha, p))
        r_new = vec_sub(r, scalar_mult(alpha, Ap))
        if norm(r_new) < 1e-6:
            break
        beta = dot(r_new, r_new) / dot(r, r)
        p = vec_add(r_new, scalar_mult(beta, p))
        r = r_new
    return x

# 執行各方法
jacobi_res = jacobi(A, b)
gs_res = gauss_seidel(A, b)
sor_res = sor(A, b)
cg_res = conjugate_gradient(A, b)

# 輸出結果
print("Jacobi Method:")
print(jacobi_res)
print("\nGauss-Seidel Method:")
print(gs_res)
print("\nSOR Method:")
print(sor_res)
print("\nConjugate Gradient Method:")
print(cg_res)
