import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector



n = 2
d = 2

neighbors = {
    0: [1, 3],
    1: [0, 2],
    2: [1, 3],
    3: [0, 2],
}

H = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
], dtype=float)


def f_y_classical(H, x, i):
    return neighbors[x][i]



def apply_open_pattern(qc, reg, value):
    """Converte controle em |value> para controle em todos-1."""
    for k, q in enumerate(reg):
        if ((value >> k) & 1) == 0:
            qc.x(q)


def undo_open_pattern(qc, reg, value):
    apply_open_pattern(qc, reg, value)


def controlled_xor_const(qc, controls, target_reg, value):
    for b, q in enumerate(target_reg):
        if ((value >> b) & 1) == 1:
            qc.mcx(controls, q)


def append_eq_const_flag(qc, reg, value, flag):
    apply_open_pattern(qc, reg, value)
    qc.mcx(list(reg), flag)
    undo_open_pattern(qc, reg, value)


def append_eq_regs_flag(qc, reg_a, reg_b, flag):
    nbits = len(reg_a)
    assert len(reg_b) == nbits
    for v in range(2**nbits):
        apply_open_pattern(qc, reg_a, v)
        apply_open_pattern(qc, reg_b, v)
        qc.mcx(list(reg_a) + list(reg_b), flag)
        undo_open_pattern(qc, reg_b, v)
        undo_open_pattern(qc, reg_a, v)


def append_lt_regs_flag(qc, reg_a, reg_b, flag):
    nbits = len(reg_a)
    assert len(reg_b) == nbits
    for a in range(2**nbits):
        for b in range(2**nbits):
            if a < b:
                apply_open_pattern(qc, reg_a, a)
                apply_open_pattern(qc, reg_b, b)
                qc.mcx(list(reg_a) + list(reg_b), flag)
                undo_open_pattern(qc, reg_b, b)
                undo_open_pattern(qc, reg_a, a)


def append_copy_reg_if_flag(qc, flag, src_reg, dst_reg):
    assert len(src_reg) == len(dst_reg)
    for s, d in zip(src_reg, dst_reg):
        qc.cx(s, d)
        qc.ccx(flag, d, d) 


def append_cnot_reg_if_flag(qc, flag, src_reg, dst_reg):
    assert len(src_reg) == len(dst_reg)
    for s, d in zip(src_reg, dst_reg):
        qc.ccx(flag, s, d)


def append_xor_const_if_flag(qc, flag, dst_reg, value):
    for b, q in enumerate(dst_reg):
        if ((value >> b) & 1) == 1:
            qc.cx(flag, q)


def append_swap_1bit_if_flag(qc, flag, q1, q2):
    qc.cswap(flag, q1, q2)



def append_Of_from_neighbors(qc, reg_x, reg_i, reg_y, neighbors):
    """
    Implementa:
        |x>|i>|0> -> |x>|i>|f_y(x,i)>
    """
    n_x = len(reg_x)
    n_i = len(reg_i)

    for x in range(2**n_x):
        for i in range(2**n_i):
            if i >= len(neighbors[x]):
                y = x
            else:
                y = neighbors[x][i]

            apply_open_pattern(qc, reg_x, x)
            apply_open_pattern(qc, reg_i, i)

            controls = list(reg_x) + list(reg_i)
            controlled_xor_const(qc, controls, reg_y, y)

            undo_open_pattern(qc, reg_i, i)
            undo_open_pattern(qc, reg_x, x)



def append_berry_color_circuit_n2d2(
    qc,
    reg_x, reg_i, reg_j,
    reg_yi, reg_yj,
    reg_ican, reg_jcan, reg_nu, reg_valid,
    aux
):

    eq_yjx  = aux[0]
    lt_x_yi = aux[1]
    lt_yi_x = aux[2]
    eq_yi_x = aux[3]
    eq_i_j  = aux[4]
    case2   = aux[5]
    case3   = aux[6]
    diag    = aux[7]


    append_Of_from_neighbors(qc, reg_x, reg_i, reg_yi, neighbors)


    append_Of_from_neighbors(qc, reg_yi, reg_j, reg_yj, neighbors)


    append_eq_regs_flag(qc, reg_yj, reg_x, eq_yjx)     
    append_lt_regs_flag(qc, reg_x, reg_yi, lt_x_yi)    
    append_lt_regs_flag(qc, reg_yi, reg_x, lt_yi_x)    
    append_eq_regs_flag(qc, reg_yi, reg_x, eq_yi_x)    
    append_eq_regs_flag(qc, reg_i, reg_j, eq_i_j)      



    qc.ccx(eq_yjx, lt_x_yi, case2)


    qc.ccx(eq_yjx, lt_yi_x, case3)


    qc.ccx(eq_yi_x, eq_i_j, diag)


    qc.cx(diag, reg_valid[0])
    qc.cx(case2, reg_valid[0])
    qc.cx(case3, reg_valid[0])



    qc.ccx(diag,  reg_i[0], reg_ican[0])
    qc.ccx(diag,  reg_j[0], reg_jcan[0])

    qc.ccx(case2, reg_i[0], reg_ican[0])
    qc.ccx(case2, reg_j[0], reg_jcan[0])


    qc.ccx(case3, reg_j[0], reg_ican[0])
    qc.ccx(case3, reg_i[0], reg_jcan[0])



    append_cnot_reg_if_flag(qc, case2, reg_x,  reg_nu)
    append_cnot_reg_if_flag(qc, case3, reg_yi, reg_nu)



def berry_color_classical_n2d2(H, x, i, j):
    yi = f_y_classical(H, x, i)
    yj = f_y_classical(H, yi, j)


    if yi == x and i == j:
        return (i, j, 0, 1, yi, yj)


    if yi > x and yj == x:
    
        return (i, j, x, 1, yi, yj)


    if yi < x and yj == x:
    
    
        return (j, i, yi, 1, yi, yj)

    return (0, 0, 0, 0, yi, yj)



def test_berry_color_quantum_n2d2(H, x_val, i_val, j_val):
    reg_x     = QuantumRegister(2, 'x')
    reg_i     = QuantumRegister(1, 'i')
    reg_j     = QuantumRegister(1, 'j')
    reg_yi    = QuantumRegister(2, 'yi')
    reg_yj    = QuantumRegister(2, 'yj')
    reg_ican  = QuantumRegister(1, 'ican')
    reg_jcan  = QuantumRegister(1, 'jcan')
    reg_nu    = QuantumRegister(2, 'nu')
    reg_valid = QuantumRegister(1, 'valid')
    aux       = QuantumRegister(8, 'aux')

    c_yi    = ClassicalRegister(2, 'c_yi')
    c_yj    = ClassicalRegister(2, 'c_yj')
    c_ican  = ClassicalRegister(1, 'c_ican')
    c_jcan  = ClassicalRegister(1, 'c_jcan')
    c_nu    = ClassicalRegister(2, 'c_nu')
    c_valid = ClassicalRegister(1, 'c_valid')

    qc = QuantumCircuit(
        reg_x, reg_i, reg_j, reg_yi, reg_yj,
        reg_ican, reg_jcan, reg_nu, reg_valid, aux,
        c_yi, c_yj, c_ican, c_jcan, c_nu, c_valid
    )


    for k in range(2):
        if ((x_val >> k) & 1) == 1:
            qc.x(reg_x[k])

    if i_val & 1:
        qc.x(reg_i[0])
    if j_val & 1:
        qc.x(reg_j[0])

    append_berry_color_circuit_n2d2(
        qc,
        reg_x, reg_i, reg_j,
        reg_yi, reg_yj,
        reg_ican, reg_jcan, reg_nu, reg_valid,
        aux
    )

    print(qc.draw())
    qc.measure(reg_yi, c_yi)
    qc.measure(reg_yj, c_yj)
    qc.measure(reg_ican, c_ican)
    qc.measure(reg_jcan, c_jcan)
    qc.measure(reg_nu, c_nu)
    qc.measure(reg_valid, c_valid)

    sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))

    idx_yi    = [qc.find_bit(q).index for q in reg_yi]
    idx_yj    = [qc.find_bit(q).index for q in reg_yj]
    idx_ican  = [qc.find_bit(q).index for q in reg_ican]
    idx_jcan  = [qc.find_bit(q).index for q in reg_jcan]
    idx_nu    = [qc.find_bit(q).index for q in reg_nu]
    idx_valid = [qc.find_bit(q).index for q in reg_valid]

    yi_q    = int(np.argmax(sv.probabilities(qargs=idx_yi)))
    yj_q    = int(np.argmax(sv.probabilities(qargs=idx_yj)))
    ican_q  = int(np.argmax(sv.probabilities(qargs=idx_ican)))
    jcan_q  = int(np.argmax(sv.probabilities(qargs=idx_jcan)))
    nu_q    = int(np.argmax(sv.probabilities(qargs=idx_nu)))
    valid_q = int(np.argmax(sv.probabilities(qargs=idx_valid)))

    ican_c, jcan_c, nu_c, valid_c, yi_c, yj_c = berry_color_classical_n2d2(H, x_val, i_val, j_val)

    print("=" * 40)
    print(f"TESTE x={x_val} i={i_val} j={j_val}")
    print("yi_quântico   =", yi_q,   "| yi_clássico   =", yi_c)
    print("yj_quântico   =", yj_q,   "| yj_clássico   =", yj_c)
    print("ican_quântico =", ican_q, "| ican_clássico =", ican_c)
    print("jcan_quântico =", jcan_q, "| jcan_clássico =", jcan_c)
    print("nu_quântico   =", nu_q,   "| nu_clássico   =", nu_c)
    print("valid_quântico=", valid_q, "| valid_clássico=", valid_c)

    assert yi_q == yi_c, "ERRO em yi"
    assert yj_q == yj_c, "ERRO em yj"
    assert ican_q == ican_c, "ERRO em i_can"
    assert jcan_q == jcan_c, "ERRO em j_can"
    assert nu_q == nu_c, "ERRO em nu"
    assert valid_q == valid_c, "ERRO em valid"

    print(f"SUCESSO! Cor canônica = ({ican_q}, {jcan_q}, {nu_q}), valid={valid_q}")
    return qc


def run_all_tests_n2d2(H):
    for x in range(4):
        for i in range(2):
            for j in range(2):
                test_berry_color_quantum_n2d2(H, x, i, j)



if __name__ == "__main__":
    run_all_tests_n2d2(H)