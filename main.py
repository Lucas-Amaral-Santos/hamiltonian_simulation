from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library import RXGate, PhaseGate
from qiskit.quantum_info import Operator

from dataclasses import dataclass
from typing import List, Tuple
from scipy.sparse import csr_matrix, diags, identity, lil_matrix


# ============================================================
# 1) Gerar H d-esparso com d = log2(N) = n, pesos inteiros positivos
# ============================================================

def desired_degree(N: int) -> int:
    """d = log2(N), saturated to N-1."""
    if N <= 1 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of 2 (e.g., 4, 8, 16).")
    d = int(np.log2(N))-1
    return min(d, N - 1)

def edges_complete_graph(n: int):
    """Edges of K_n as undirected pairs (i<j)."""
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def edges_k_regular(n: int, k: int, seed: int = 0, max_tries: int = 20000):
    """
    Generate edges of a simple undirected k-regular graph (configuration model with rejection).
    Returns list of undirected edges (i<j).
    Requirements: n*k even.
    """
    if k < 0 or k >= n:
        raise ValueError("Require 0 <= k < n.")
    if (n * k) % 2 != 0:
        raise ValueError("Require n*k even for k-regular graph.")
    rng = np.random.default_rng(seed)

    for _ in range(max_tries):
        stubs = np.repeat(np.arange(n), k)
        rng.shuffle(stubs)

        seen = set()
        edges = []
        ok = True

        for i in range(0, len(stubs), 2):
            u = int(stubs[i]); v = int(stubs[i+1])
            if u == v:
                ok = False; break
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                ok = False; break
            seen.add((a, b))
            edges.append((a, b))

        if ok:
            return edges

    raise RuntimeError("Failed to generate a simple k-regular graph. Try another seed or increase max_tries.")

def dd_spd_matrix_from_edges(
    n: int,
    edges,
    w_min: int,
    w_max: int,
    s_min: int,
    s_max: int,
    seed: int
) -> csr_matrix:
    """
    Build symmetric positive integer matrix:
      A_ij = A_ji = w_ij for edges (i,j)
      A_ii = sum_{j!=i} A_ij + s_i   (strict diagonal dominance -> SPD)
    """
    rng = np.random.default_rng(seed)

    rows, cols, data = [], [], []
    for (i, j) in edges:
        w = int(rng.integers(w_min, w_max + 1))
        rows += [i, j]
        cols += [j, i]
        data += [w, w]

    W = csr_matrix((np.array(data, dtype=np.int64), (rows, cols)), shape=(n, n), dtype=np.int64)

    r = np.array(W.sum(axis=1)).reshape(-1).astype(np.int64)  # off-diagonal row sums
    s = rng.integers(s_min, s_max + 1, size=n, dtype=np.int64)

    A = W.copy()
    A.setdiag(r + s)
    A.eliminate_zeros()

    # Hermitian check (real symmetric)
    if (A != A.T).nnz != 0:
        raise AssertionError("Not symmetric (Hermitian).")

    return A

def generate_dd_spd_positive_integer_matrix(
    N: int,
    w_min: int = 1,
    w_max: int = 2,
    s_min: int = 4,
    s_max: int = 6,
    seed: int = 0
):
    """
    Main generator:
      - chooses d = log2(N)
      - constructs a simple d-regular graph (exact degree)
      - builds Hermitian SPD matrix with positive integer entries
    Returns: (A, d)
    """
    d = desired_degree(N)

    # If d == N-1, use complete graph (always possible)
    if d == N - 1:
        edges = edges_complete_graph(N)
    else:
        edges = edges_k_regular(N, d, seed=seed)

    A = dd_spd_matrix_from_edges(
        n=N, edges=edges,
        w_min=w_min, w_max=w_max,
        s_min=s_min, s_max=s_max,
        seed=seed
    )
    return A, d

# -----------------------------
# Demo: N = 4, 8, 16 (d = 2, 3, 4)
# -----------------------------



@dataclass(frozen=True)
class DSparseInstance:
    n: int
    d: int
    H: np.ndarray
    edges: List[Tuple[int, int]]
    diag: np.ndarray
    w_max_effective: int  # útil p/ definir quantos bits m usar no Oh

def generate_d_sparse_hamiltonian(
    n: int,
    seed: int = 0,
    w_max: int = 2,
    include_diag: bool = True,
    # parâmetros do seu gerador SPD:
    w_min: int = 1,
    s_min: int = 4,
    s_max: int = 6,
):
    """
    Versão integrada: usa seu gerador dd-SPD (dominância diagonal estrita),
    devolvendo no mesmo formato do pipeline.

    Observações:
      - include_diag deve ser True para SPD (diagonal sempre existe).
      - d vem da sua desired_degree(N) (log2(N)-1 saturado).
      - w_max_effective pode ser bem maior que w_max porque diag = sum(offdiag)+s.
    """
    if not include_diag:
        raise ValueError("Para SPD por dominância diagonal, include_diag deve ser True.")

    N = 2**n

    # --- chama seu gerador ---
    A_csr, d = generate_dd_spd_positive_integer_matrix(
        N=N,
        w_min=w_min, w_max=w_max,
        s_min=s_min, s_max=s_max,
        seed=seed
    )

    # csr -> denso (float) para seu expm_iHt etc.
    H = A_csr.toarray().astype(float)

    # extrair diagonal
    diag = np.diag(H).copy()

    # extrair edges offdiag (i<j)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if H[i, j] != 0:
                edges.append((i, j))

    # útil para dimensionar registrador h no Oh:
    # offdiag <= w_max, mas diag pode ser até d*w_max + s_max
    w_max_effective = int(np.max(H))

    return DSparseInstance(
        n=n, d=int(d), H=H, edges=edges, diag=diag,
        w_max_effective=w_max_effective
    )



# ============================================================
# 2) Coloração de arestas (matching por cor) — objeto que o Berry precisa
#    (para matriz explícita: coloração gulosa é suficiente p/ validar)
# ============================================================

def greedy_edge_coloring(N: int, edges: List[Tuple[int, int]]) -> Tuple[Dict[Tuple[int,int], int], int]:
    used_at = [set() for _ in range(N)]
    edge_color: Dict[Tuple[int,int], int] = {}
    maxc = -1
    for (u, v) in edges:
        c = 0
        while c in used_at[u] or c in used_at[v]:
            c += 1
        edge_color[(u, v)] = c
        used_at[u].add(c)
        used_at[v].add(c)
        maxc = max(maxc, c)
    return edge_color, maxc + 1


# ============================================================
# 3) Construir termos 1-esparsos (cada cor -> um matching)
#    + termo diagonal separado
# ============================================================

@dataclass(frozen=True)
class OneSparseTerm:
    n: int
    f: Dict[int, int]                      # involução (fixpoints ok)
    w: Dict[Tuple[int, int], int]          # (min,max) ou (x,x) -> peso inteiro
    w_max: int                             

def build_terms_from_edge_coloring(inst: DSparseInstance, w_max: int) -> Tuple[List[OneSparseTerm], OneSparseTerm]:
    N = 2**inst.n
    edge_color, num_colors = greedy_edge_coloring(N, inst.edges)

    # montar, para cada cor, um matching parcial
    terms: List[OneSparseTerm] = []
    for c in range(num_colors):
        f = {x: x for x in range(N)}  # fixpoints default
        w: Dict[Tuple[int,int], int] = {}  # só edges desse termo
        for (u, v), col in edge_color.items():
            if col != c:
                continue
            f[u] = v
            f[v] = u
            w[(min(u, v), max(u, v))] = int(inst.H[u, v])
        terms.append(OneSparseTerm(n=inst.n, f=f, w=w, w_max=w_max))

    # termo diagonal separado (1-esparso também)
    fD = {x: x for x in range(N)}
    wD: Dict[Tuple[int,int], int] = {}
    for x in range(N):
        if inst.diag[x] != 0:
            wD[(x, x)] = int(inst.diag[x])
    diag_term = OneSparseTerm(n=inst.n, f=fD, w=wD, w_max=w_max)

    return terms, diag_term


# ============================================================
# 4) Oráculos por permutação: Of, Oh, Ogt, Oeq
# ============================================================

def make_perm_unitary(mapping: List[int]) -> np.ndarray:
    dim = len(mapping)
    U = np.zeros((dim, dim), dtype=complex)
    for i, j in enumerate(mapping):
        U[j, i] = 1.0
    return U

def build_Of_gate(n: int, f: Dict[int, int], label: str = "Of") -> UnitaryGate:
    N = 2**n
    dim = 2**(2*n)
    mapping = list(range(dim))
    for x in range(N):
        fx = f[x]
        for y in range(N):
            inp = x | (y << n)
            out = x | ((y ^ fx) << n)
            mapping[inp] = out
    return UnitaryGate(make_perm_unitary(mapping), label=label)

def build_Oh_gate(n: int, m: int, f: Dict[int, int], w: Dict[Tuple[int,int], int], label: str = "Oh") -> UnitaryGate:
    N = 2**n
    dim = 2**(2*n + m)
    Hdim = 2**m
    mapping = list(range(dim))
    for x in range(N):
        fx = f[x]
        for y in range(N):
            if y == fx:
                key = (min(x, y), max(x, y))
                hxy = int(w.get(key, 0))
            else:
                hxy = 0
            for hreg in range(Hdim):
                inp = x | (y << n) | (hreg << (2*n))
                out = x | (y << n) | ((hreg ^ hxy) << (2*n))
                mapping[inp] = out
    return UnitaryGate(make_perm_unitary(mapping), label=label)

def build_Ogt_gate(n: int, label: str = "Ogt") -> UnitaryGate:
    N = 2**n
    dim = 2**(2*n + 1)
    mapping = list(range(dim))
    for x in range(N):
        for y in range(N):
            gt = 1 if x > y else 0
            for b in (0, 1):
                inp = x | (y << n) | (b << (2*n))
                out = x | (y << n) | ((b ^ gt) << (2*n))
                mapping[inp] = out
    return UnitaryGate(make_perm_unitary(mapping), label=label)

def build_Oeq_gate(n: int, label: str = "Oeq") -> UnitaryGate:
    N = 2**n
    dim = 2**(2*n + 1)
    mapping = list(range(dim))
    for x in range(N):
        for y in range(N):
            eq = 1 if x == y else 0
            for e in (0, 1):
                inp = x | (y << n) | (e << (2*n))
                out = x | (y << n) | ((e ^ eq) << (2*n))
                mapping[inp] = out
    return UnitaryGate(make_perm_unitary(mapping), label=label)


# ============================================================
# 5) Simulador 1-esparso (com diagonal) via Of/Oh + rotações controladas
#    Retorna U_eff (NxN) atuando só no registro x.
# ============================================================

def build_one_sparse_circuit_with_diagonal(term: OneSparseTerm, t: float) -> Tuple[QuantumCircuit, int]:
    n = term.n
    w_max = term.w_max
    m = max(1, math.ceil(math.log2(w_max + 1)))

    x = QuantumRegister(n, "x")
    y = QuantumRegister(n, "y")
    hreg = QuantumRegister(m, "h")
    b = QuantumRegister(1, "b")      # também usado como qubit de mistura offdiag
    e = QuantumRegister(1, "e")      # flag x==y
    neq = QuantumRegister(1, "neq")  # NOT e
    p = QuantumRegister(1, "p")      # ancila de fase (kickback)

    qc = QuantumCircuit(x, y, hreg, b, e, neq, p)

    Of = build_Of_gate(n, term.f, "Of")
    Oh = build_Oh_gate(n, m, term.f, term.w, "Oh")
    Ogt = build_Ogt_gate(n, "Ogt")
    Oeq = build_Oeq_gate(n, "Oeq")

    # p = |1>
    qc.x(p[0])

    # y = f(x)
    qc.append(Of, list(x) + list(y))

    # b ^= [x>y]
    qc.append(Ogt, list(x) + list(y) + [b[0]])

    # canoniza: se b=1 troca x<->y => x=min, y=max (para pares)
    for k in range(n):
        qc.cswap(b[0], x[k], y[k])

    # e ^= [x==y]
    qc.append(Oeq, list(x) + list(y) + [e[0]])

    # neq = NOT e (neq começa em |0>)
    qc.x(neq[0])        # 1
    qc.cx(e[0], neq[0]) # se e=1 => neq=0

    # carrega h
    qc.append(Oh, list(x) + list(y) + list(hreg))

    # OFFDIAG: Rx(2*h*t) em b condicionado em (neq AND h_k)
    for k in range(m):
        angle = 2.0 * t * (2**k)
        crx2 = RXGate(angle).control(2)
        qc.append(crx2, [neq[0], hreg[k], b[0]])

    # DIAG: fase exp(-i*h*t) em p condicionado em (e AND h_k)
    for k in range(m):
        lam = -t * (2**k)
        cph2 = PhaseGate(lam).control(2)
        qc.append(cph2, [e[0], hreg[k], p[0]])

    # uncompute h
    qc.append(Oh.inverse(), list(x) + list(y) + list(hreg))

    # uncompute neq,e
    qc.cx(e[0], neq[0])
    qc.x(neq[0])
    qc.append(Oeq.inverse(), list(x) + list(y) + [e[0]])

    # desfaz canonização
    for k in range(n):
        qc.cswap(b[0], x[k], y[k])

    # desfaz b e y
    qc.append(Ogt.inverse(), list(x) + list(y) + [b[0]])
    qc.append(Of.inverse(), list(x) + list(y))

    # p volta a |0>
    qc.x(p[0])

    return qc, m

from qiskit.quantum_info import Statevector

def extract_U_eff_on_x_statevector(qc: QuantumCircuit, n: int, m: int) -> np.ndarray:
    """
    Extrai U_eff (N x N) usando simulação de Statevector:
      U_eff[:, xin] = projeção em ancillas=0 do estado final ao evoluir |xin, anc=0>.
    Layout: x(n), y(n), h(m), b, e, neq, p  => total = 2n+m+4
    """
    N = 2**n
    total = 2*n + m + 4
    dim = 2**total

    def idx(xv, yv, hv, bv, ev, nqv, pv):
        return (
            xv
            | (yv << n)
            | (hv << (2*n))
            | (bv << (2*n + m))
            | (ev << (2*n + m + 1))
            | (nqv << (2*n + m + 2))
            | (pv << (2*n + m + 3))
        )

    U_eff = np.zeros((N, N), dtype=complex)

    for xin in range(N):
        ket = np.zeros(dim, dtype=complex)
        ket[idx(xin, 0, 0, 0, 0, 0, 0)] = 1.0
        out = Statevector(ket).evolve(qc).data
        for xout in range(N):
            U_eff[xout, xin] = out[idx(xout, 0, 0, 0, 0, 0, 0)]

    return U_eff



# ============================================================
# 6) Exato, Trotter e erro
# ============================================================

def expm_iHt(H: np.ndarray, t: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(-1j * evals * t)
    return (evecs @ np.diag(phases) @ evecs.T).astype(complex)

def unitary_error(U_sim: np.ndarray, U_exact: np.ndarray) -> dict:
    D = U_sim - U_exact
    return {
        "frobenius": float(np.linalg.norm(D, ord="fro")),
        "spectral_2": float(np.linalg.norm(D, ord=2)),
    }

def trotter_first_order_from_terms_statevector(
    terms: List[OneSparseTerm],
    diag_term: OneSparseTerm,
    t: float,
    r: int,
) -> np.ndarray:
    """
    U ≈ ( U_diag(dt) * Π_j U_j(dt) )^r  (1ª ordem)
    mas extraindo U_j via Statevector (robusto).
    """
    n = diag_term.n
    N = 2**n
    dt = t / r

    # cache U de cada termo nesse dt
    cache: List[np.ndarray] = []

    # diagonal
    qcD, mD = build_one_sparse_circuit_with_diagonal(diag_term, dt)
    U_diag = extract_U_eff_on_x_statevector(qcD, n=n, m=mD)

    # termos offdiag
    for term in terms:
        qc, m = build_one_sparse_circuit_with_diagonal(term, dt)
        Uj = extract_U_eff_on_x_statevector(qc, n=n, m=m)
        cache.append(Uj)

    # monta U_step
    U_step = U_diag.copy()
    for Uj in cache:
        U_step = Uj @ U_step

    return np.linalg.matrix_power(U_step, r)


def strang_second_order_from_terms_statevector(
    terms: List[OneSparseTerm],
    diag_term: OneSparseTerm,
    t: float,
    r: int,
) -> np.ndarray:
    """
    Strang (2ª ordem) usando U_j extraído via Statevector.
    """
    n = diag_term.n
    N = 2**n
    dt = t / r
    half = dt / 2

    # U_diag(half)
    qcD, mD = build_one_sparse_circuit_with_diagonal(diag_term, half)
    U_diag_h = extract_U_eff_on_x_statevector(qcD, n=n, m=mD)

    # U_j(half) cache
    U_half = []
    for term in terms:
        qc, m = build_one_sparse_circuit_with_diagonal(term, half)
        Uj = extract_U_eff_on_x_statevector(qc, n=n, m=m)
        U_half.append(Uj)

    # monta um passo Strang
    U_step = U_diag_h.copy()
    for Uj in U_half:
        U_step = Uj @ U_step
    for Uj in reversed(U_half):
        U_step = Uj @ U_step
    U_step = U_diag_h @ U_step

    return np.linalg.matrix_power(U_step, r)



# ============================================================
# 7) Rodar experimento: d=n, coloração, Trotter, erro
# ============================================================

if __name__ == "__main__":
    n = 3
    seed = 12
    t = 0.9
    r_list = [1, 2, 4, 8, 16]
    cond = "MOD"
    w_min = 1
    w_max = 2
    
    if cond == "GOOD":
        s_min = 4
        s_max = 6
    elif cond == "MOD":
        s_min = 1
        s_max = 2

    inst = generate_d_sparse_hamiltonian(
        n=n, seed=seed,
        w_min=w_min, w_max=w_max,
        s_min=s_min, s_max=s_max,
        include_diag=True
    )

    terms, diag_term = build_terms_from_edge_coloring(inst, w_max=inst.w_max_effective)


    print(f"n={n} N={2**n} d={inst.d} |edges|={len(inst.edges)} |terms(offdiag)|={len(terms)} + diag")

    U_exact = expm_iHt(inst.H, t)

    print("\n r |  Trotter ||.||2   Strang ||.||2   (Frobenius também se quiser)")

    NORM_KEY = "spectral_2"  # ou "frobenius"

    r_arr = np.asarray(r_list, dtype=float)

    trotter_err = np.empty(len(r_list), dtype=float)
    strang_err  = np.empty(len(r_list), dtype=float)

    for i, r in enumerate(r_list):
        U_trot = trotter_first_order_from_terms_statevector(terms, diag_term, t=t, r=r)
        trotter_err[i] = unitary_error(U_trot, U_exact)[NORM_KEY]

        U_str = strang_second_order_from_terms_statevector(terms, diag_term, t=t, r=r)
        strang_err[i] = unitary_error(U_str, U_exact)[NORM_KEY]
        print(f"{r:2d} | {trotter_err[i]:14.6e}  {strang_err[i]:14.6e}")


        

