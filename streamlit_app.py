import streamlit as st
import numpy as np

st.set_page_config(layout="wide")

st.title("MC Error Lower Bound Explorer")
st.caption("Interactive verification of theoretical Monte Carlo lower bounds of error in $L^p$ approximation of classes containg $\\tanh$ neural networks.")


# helper for sidebar inputs with optional ∞ / auto locking
def num_input(name, min_val, default, inf_possible=True, max_val=None, step=1, auto_possible=True):
    if name not in st.session_state:
        st.session_state[name] = default
        if inf_possible:
            st.session_state[f"{name}_inf"] = False
        elif auto_possible:
            st.session_state[f"{name}_auto"] = True

    col1, col2 = st.sidebar.columns([3,1.5])

    # formatting rule depending on variable type
    if name == "$\\varepsilon_{p}$":
        fmt = "%.2e"
    elif isinstance(step, int):
        fmt = "%d"
    else:
        fmt = "%.2f"

    is_inf = st.session_state.get(f"{name}_inf", False)
    is_auto = st.session_state.get(f"{name}_auto", False)

    # lock value if automatically determined
    if is_auto:
        st.session_state[name] = default

    with col1:
        val = st.number_input(
            name,
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=name,
            disabled=is_inf or is_auto,
            format=fmt
        )

    # right-side toggles (∞ or auto mode)
    with col2:
        if inf_possible:
            st.checkbox("∞", key=f"{name}_inf")
        elif auto_possible:
            st.checkbox("auto", key=f"{name}_auto")

    return np.inf if (inf_possible and is_inf) else val


# formatting helper for display
def pretty(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)


# --- model / bound components ---
def c_comp(q,B,c):
    return B**(1-2/q)*c


def constant_before_m(B, c, q, p, s, omega):
    term1 = np.sqrt((c_comp(q,B,c))**2 - 1) / (4 * B**(1 - 2/q))
    term2 = (omega / (2**(1 + 2/s) * np.sqrt(s)))**(s/p)
    return term1 * term2


def Theta(B,q,c):
    B_tanh = B **(-1/q) * np.tanh(c/2)
    return np.tanh(B_tanh) / (20 * np.cosh(B_tanh)**2)


def Omega(B,q,c):
    B_tanh = B **(-1/q) * (c - np.tanh(c/2))
    return 3/5 * np.tanh(B_tanh) / B_tanh


# auxiliary coefficient for asymptotic bound
def rho(tilde_c):
    if tilde_c <= 1:
        return 0
    rho = 1 / (np.sqrt(tilde_c ** 2 - 4 * tilde_c**2/ (tilde_c + 1)**2 * (np.arccosh(np.sqrt(tilde_c)))**2) + 1)
    return rho


# index constraint for j
def j_assump(q,B,c, tilde_c, rho_c):
    if tilde_c <= 1:
        return 0
    nom = 5 * B ** (3/q) * np.arccosh(np.sqrt(tilde_c)) * rho_c
    cosh_a = (np.cosh(2*np.arccosh(np.sqrt(tilde_c))*rho_c))**2
    den = c ** 2 * (np.tanh(2*B**(-1/q)*np.tanh(c/2))) * (np.tanh(B**(-1/q)*(c - np.tanh(c/2))))**2
    return np.log(nom / den)/ np.log(tilde_c / cosh_a)


# auxiliary exponent function
def pi(c, rho_c):
    x = np.sqrt(c) + np.sqrt(c-1)
    return x ** (4*(1-rho_c))


# index constraint for k
def k_assump(e_p, c, tilde_c, rho_c):
    if tilde_c <= 1:
        return 0
    p = pi(c, rho_c)
    nom = 4*c*tilde_c / (e_p * p * (1+1/p)**2)
    den = (np.cosh(tilde_c * (p-1)/(p+1)))**2 / tilde_c
    return 3 + np.log(nom) / np.log(den)


# exponent constraint for s
def s_assump(m_max, theta, q, B, c, rho_c, tilded_c, j):
    cosh_a = (np.cosh(2*np.arccosh(np.sqrt(tilded_c)) * rho_c))**2
    x = tilded_c / cosh_a
    y = theta * c ** 2 * (c - np.tanh(c/2)) ** 2
    z = y / (16 * B ** (5/q) * np.arccosh(np.sqrt(tilded_c)) * rho_c)
    return 2*np.log(4*m_max) / (j * np.log(x) + np.log(z))


# --- UI section ---
st.sidebar.markdown("## Parameters")

# reset all inputs
if st.sidebar.button("Reset parameters", type="primary"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


# fixed hyperparameters
p = num_input("$p$", 1, 4, auto_possible=False)
q = num_input("$q$", 1, 4, auto_possible=False)
d = num_input("$d$", 1, 15, inf_possible=False, auto_possible=False)
B = num_input("$B$", 1, 45, inf_possible=False, auto_possible=False)
c = num_input("$c$", 0.0, 2.0, step=0.01, inf_possible=False, auto_possible=False)


# derived structural quantity
tilde_c = c_comp(q,B,c)
rho_c = rho(tilde_c)

st.divider()
st.markdown("### Structural assumptions")

c_formula = r"B^{1-\frac2q}c"
ass_on_B = r"\qquad \text{and} \qquad B \ge 2d"

st.latex(rf"\tilde c = {c_formula} = {round(tilde_c,2)} > 1, \quad B \ge 2d")

# assumptions check
if tilde_c <= 1:
    st.error(r"Violation: $\tilde c \le 1$")
elif B < 2*d:
    st.error(r"Violation: $B < 2d$")
else:
    st.success("All structural assumptions satisfied.")

    e_p = num_input("$\\varepsilon_{p}$", 0.0, 1e-16, step = 1e-16, inf_possible=False, auto_possible=False)

    j_ass = j_assump(q, B, c, tilde_c, rho_c)
    j_min = max(1, int(np.ceil(j_ass)))
    k_ass = k_assump(e_p, c, tilde_c, rho_c)
    k_min = max(3, int(np.ceil(k_ass)))
    L_min = k_min+j_min + 3
    L = num_input("$L$", min_val=L_min, default=max(L_min, 12), inf_possible=False, auto_possible=False)

    # default automatic allocation
    k = k_min
    j_max = L - 3 - k
    j = j_max

    mode = st.checkbox("Set $j,k$ and $s$ automatically", value=True)

    if mode:
        k, j = k_min, j_max

    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            k = st.number_input(
                "$k$",
                min_value=k_min,
                max_value=L-3,
                value=k_min
            )

        j_max = L - 3 - k

        with col2:
            j = st.number_input(
                "$j$",
                min_value=j_min,
                max_value=j_max,
                value=j_max
            )


    m_max = num_input(r"$m_{\\max}$", 1, 100000, inf_possible=False, auto_possible=False)

    theta = Theta(B,q,c)
    omega = Omega(B,q,c)

    s_ass = s_assump(m_max, theta, q, B, c, rho_c, tilde_c, j)

    s_min = max(1, int(np.ceil(s_ass)))
    if mode:
        s = s_min
    else:
        with col3:
            s = st.number_input(
                "$s$",
                min_value=s_min,
                max_value=d,
                value=s_min
            )

    P = (L-2)*B**2 + (L+d)*B + 1

    # validity of parameters check
    if s_ass < 0:
        st.error("Unable to get $s$ positive, adjust the inputs")
    elif s_ass > d:
        st.error("$s$ has to be greater than $d$ in order to satisfy the results for all $m \\leq m_{\\max}$, adjust the inputs")
    else:
        # ---- Output ----
        st.divider()
        st.markdown("### Final bound")

        omega = Omega(B, q, c)
        final_const = constant_before_m(B, c, q, p, s, omega)
        m_form = r"\cdot m^{-\frac1p}"
        error_formula = r"\operatorname{err}_m^{MC}\!\left(U, L^p([0,1]^d)\right)\;\ge\;"
        worst_error_formula = r"\operatorname{err}_{m_{\max}}^{MC}\!\left(U, L^p([0,1]^d)\right)\;\ge\;"
        lower_bound_formula = r"c\frac{\sqrt{\tilde c^{\,2}-1}}{4\tilde c}\cdot\left(\frac{\Omega}{2^{1+\frac{2}{s}}\sqrt{s}}\right)^{\frac{s}{p}}m^{-\frac{1}{p}}"
        mantissa, exp = f"{final_const:.2e}".split("e")
        exp = int(exp)
        worst_lower_bound = final_const * m_max ** (-1/p)

        st.markdown(
            "Every algorithm with precision $\\varepsilon_p$ using $m \\leq m_{\\max}$ samples "
            "approximating the class of neural networks with input dimension $d$, width $B$, "
            "depth $L$, and $\\ell^q$-bounded weights by $c$ incurs an $L^p$ error of at least:"
        )

        if final_const < e_p:
            st.latex(rf"{error_formula} 0")
            st.error("Lower bound is smaller than machine precision $\\varepsilon_{p}$ for every $m$")
        else:
            st.latex(rf"{error_formula}{lower_bound_formula} = {mantissa} \cdot 10^{{{exp}}} {m_form}")
            if worst_lower_bound < e_p:
                st.error("Lower bound is smaller than machine precision for some $m$")


# optional showing the expressions
with st.expander("Show computation parameters"):
    k_assump_formula = r"3 + \frac{\ln\!\left( \frac{4c \tilde c}{\varepsilon_p \pi(\tilde c)\left(1+\pi(\tilde c)^{-1}\right)^2} \right)}{\ln \!\left( \frac{\cosh^2\!\left(\tilde c\frac{\pi(\tilde c)-1}{\pi(\tilde c)+1}\right)}{\tilde c} \right)} \le k"

    st.markdown("#### Layer allocation")
    st.write("Assumption on $3\\le k \\in \\mathbb N$:")
    st.latex(rf"{round(k_ass,2)} = {k_assump_formula} \, .")

    st.write("Assumptions on $j \\in \mathbb N$:")

    j_assump_formula = r"\frac{\ln\left( \frac{5 B^{\frac3q}\operatorname{arccosh}\left(\sqrt{\tilde c}\right)\,\rho(\tilde c)}{c^2\tanh(2B^{-1/q}\tanh(\frac c2))\tanh^2\left[B^{-1/q}(c - \tanh(\frac c2))\right]} \right)}{\ln\left(  \frac{\tilde c}{\cosh^2\left[2\operatorname{arccosh}(\sqrt{\tilde c})\rho(\tilde c)\right]} \right)} \leq j"
    st.latex(rf"{round(j_ass, 2)} = {j_assump_formula} \, .")
    st.write("Hence:")

    L_formula = r"L = 3 + k + j"
    st.latex(rf"{L_formula} \ge {L_min} \, .")

    st.markdown("#### Dimension constraint")
    st.write("Assumption on $s \\in \\mathbb N$:")

    s_formula = r"d \;\ge\; s \;\ge\;\frac{2\ln(4m_{\max})}{j\,\ln\!\Big( \frac{\tilde c}{\cosh^2\!\left[2\operatorname{arccosh}(\sqrt{\tilde c})\,\rho(\tilde c)\right]} \Big)+\ln\!\left(\frac{ \Theta\,c^2(c-\tanh(c/2))^2}{16\,B^{5/q}\operatorname{arccosh}(\sqrt{\tilde c})\,\rho(\tilde c)}\right)}"
    st.latex(rf"{s_formula} = {round(s_ass, 2)} \, .")

    st.markdown("##### Total number of weight parameters:")
    st.latex(rf"P = B^2(L-2) + B(L+d) + 1 = {P}")
