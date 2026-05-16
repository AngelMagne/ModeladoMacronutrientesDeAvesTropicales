import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Optimización de Dieta en Aves Neotropicales",
    page_icon="🐦",
    layout="wide"
)
st.title("🐦 Optimización de Dieta en Aves Neotropicales")
st.markdown("### Métodos Numéricos para el Balance de Macronutrientes")
st.markdown("---")


# ============================================
# DEFINICIÓN DE MÉTODOS NUMÉRICOS
# ============================================

def gradiente_conjugado(A, b, tol=1e-6, max_iter=1000):
    """Método del Gradiente Conjugado"""
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r

    historial_x = [x.copy()]
    historial_error = []

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        error = np.linalg.norm(r)

        historial_x.append(x.copy())
        historial_error.append(error)

        if error < tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, historial_error, len(historial_error)


def sor(A, b, omega=1.25, tol=1e-6, max_iter=1000):
    """Método de Sobrerelajación Sucesiva (SOR)"""
    n = len(b)
    x = np.zeros(n)
    historial_error = []

    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            suma1 = sum(A[i][j] * x[j] for j in range(i))
            suma2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (1 - omega) * x_old[i] + (omega / A[i][i]) * (b[i] - suma1 - suma2)

        error = np.linalg.norm(x - x_old, np.inf)
        historial_error.append(error)

        if error < tol:
            break

    return x, historial_error, len(historial_error)


def gauss_seidel(A, b, tol=1e-6, max_iter=1000):
    """Método de Gauss-Seidel"""
    n = len(b)
    x = np.zeros(n)
    historial_error = []

    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            suma1 = sum(A[i][j] * x[j] for j in range(i))
            suma2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - suma1 - suma2) / A[i][i]

        error = np.linalg.norm(x - x_old, np.inf)
        historial_error.append(error)

        if error < tol:
            break

    return x, historial_error, len(historial_error)


def jacobi(A, b, tol=1e-6, max_iter=1000):
    """Método de Jacobi"""
    n = len(b)
    x = np.zeros(n)
    x_old = np.zeros(n)
    historial_error = []

    for k in range(max_iter):
        for i in range(n):
            suma = 0
            for j in range(n):
                if j != i:
                    suma += A[i][j] * x_old[j]
            x[i] = (b[i] - suma) / A[i][i]

        error = np.linalg.norm(x - x_old, np.inf)
        historial_error.append(error)

        if error < tol:
            break

        x_old = x.copy()

    return x, historial_error, len(historial_error)


# ============================================
# BARRA LATERAL - CONFIGURACIÓN
# ============================================
st.sidebar.header("⚙️ Configuración del Sistema")

# Selección de escenario o personalizado
opcion = st.sidebar.radio(
    "Seleccionar escenario:",
    ["Caso Ideal", "Caso Estrés", "Caso Mal Condicionado", "Personalizado"]
)

# Escenarios predefinidos
if opcion == "Caso Ideal":
    A_def = [[3, 1, 1], [1, 4, 1], [1, 1, 5]]
    b_def = [13, 18, 25]
    st.sidebar.info("**Caso Ideal:** Sistema bien condicionado (κ ≈ 8.5)")

elif opcion == "Caso Estrés":
    A_def = [[300, 5, 5], [5, 400, 5], [1, 1, 5]]
    b_def = [635, 1230, 25]
    st.sidebar.warning("**Caso Estrés:** Coeficientes amplificados (κ ≈ 80)")

elif opcion == "Caso Mal Condicionado":
    A_def = [[1, 2, 3], [1.001, 2.001, 3.001], [4, 5, 6]]
    b_def = [20, 20.009, 47]
    st.sidebar.error("**Caso Mal Condicionado:** Filas casi paralelas (κ ≈ 10⁵)")

else:
    A_def = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    b_def = [1, 1, 1]

# Entrada de matriz A
st.sidebar.subheader("Matriz A (3x3)")
A_input = []
for i in range(3):
    row = []
    for j in range(3):
        val = st.sidebar.number_input(
            f"A[{i + 1}][{j + 1}]",
            value=float(A_def[i][j]),
            key=f"A_{i}_{j}",
            format="%.6f"
        )
        row.append(val)
    A_input.append(row)

# Entrada de vector b
st.sidebar.subheader("Vector b")
b_input = []
for i in range(3):
    val = st.sidebar.number_input(
        f"b[{i + 1}]",
        value=float(b_def[i]),
        key=f"b_{i}",
        format="%.6f"
    )
    b_input.append(val)

# Parámetros del método
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Parámetros de convergencia")
tolerancia = st.sidebar.select_slider(
    "Tolerancia:",
    options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    value=1e-6
)

omega = st.sidebar.slider(
    "Factor de relajación ω (SOR):",
    min_value=0.5,
    max_value=1.9,
    value=1.25,
    step=0.05,
    help="1 < ω < 2 acelera convergencia"
)

# Selección de método
metodo_elegido = st.sidebar.selectbox(
    "Seleccionar método iterativo:",
    ["Gradiente Conjugado", "SOR", "Gauss-Seidel", "Jacobi"]
)

# Botón para resolver
st.sidebar.markdown("---")
resolver = st.sidebar.button("🚀 Resolver sistema", type="primary")

# ============================================
# ÁREA PRINCIPAL
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Sistema de Ecuaciones")
    st.latex(r"""
    \begin{cases}
    a_{11}x_1 + a_{12}x_2 + a_{13}x_3 = b_1 \\
    a_{21}x_1 + a_{22}x_2 + a_{23}x_3 = b_2 \\
    a_{31}x_1 + a_{32}x_2 + a_{33}x_3 = b_3
    \end{cases}
    """)

    # Mostrar el sistema actual
    st.write("**Sistema actual:**")
    A_mat = np.array(A_input)
    b_vec = np.array(b_input)

    for i in range(3):
        ecuacion = f"{A_mat[i, 0]:.4f}·x₁ + {A_mat[i, 1]:.4f}·x₂ + {A_mat[i, 2]:.4f}·x₃ = {b_vec[i]:.4f}"
        st.latex(ecuacion)

with col2:
    st.subheader("🔍 Solución Exacta (LU)")
    try:
        lu_and_piv = lu_factor(A_mat)
        x_lu = lu_solve(lu_and_piv, b_vec)
        st.write("**Solución de referencia:**")
        st.write(f"🐔 **Proteínas (x₁):** {x_lu[0]:.6f} g/día")
        st.write(f"🐟 **Lípidos (x₂):** {x_lu[1]:.6f} g/día")
        st.write(f"🌾 **Carbohidratos (x₃):** {x_lu[2]:.6f} g/día")
    except Exception as e:
        st.error(f"Error en factorización LU: {e}")

# ============================================
# RESOLUCIÓN DEL SISTEMA
# ============================================

if resolver:
    st.markdown("---")
    st.subheader(f"📈 Resultados del método: {metodo_elegido}")

    with st.spinner("Resolviendo sistema..."):
        try:
            if metodo_elegido == "Gradiente Conjugado":
                x, errores, iteraciones = gradiente_conjugado(A_mat, b_vec, tol=tolerancia)
                nombre_metodo = "Gradiente Conjugado"
            elif metodo_elegido == "SOR":
                x, errores, iteraciones = sor(A_mat, b_vec, omega=omega, tol=tolerancia)
                nombre_metodo = f"SOR (ω = {omega})"
            elif metodo_elegido == "Gauss-Seidel":
                x, errores, iteraciones = gauss_seidel(A_mat, b_vec, tol=tolerancia)
                nombre_metodo = "Gauss-Seidel"
            else:
                x, errores, iteraciones = jacobi(A_mat, b_vec, tol=tolerancia)
                nombre_metodo = "Jacobi"

            # Mostrar resultados
            col_a, col_b = st.columns(2)

            with col_a:
                st.success(f"✅ **Convergencia alcanzada en {iteraciones} iteraciones**")
                st.write("**Solución encontrada:**")
                st.write(f"🐔 **x₁ (Proteínas):** {x[0]:.6f} g/día")
                st.write(f"🐟 **x₂ (Lípidos):** {x[1]:.6f} g/día")
                st.write(f"🌾 **x₃ (Carbohidratos):** {x[2]:.6f} g/día")

            with col_b:
                st.write("**Comparación con solución exacta (LU):**")
                try:
                    diff = np.abs(x - x_lu)
                    st.write(f"📐 Diferencia x₁: {diff[0]:.2e}")
                    st.write(f"📐 Diferencia x₂: {diff[1]:.2e}")
                    st.write(f"📐 Diferencia x₃: {diff[2]:.2e}")
                    st.metric("Error máximo", f"{np.max(diff):.2e}")
                except:
                    st.write("(No se pudo comparar con LU)")

            # Gráfica de convergencia
            st.subheader("📉 Curva de convergencia")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.semilogy(range(1, len(errores) + 1), errores, 'b-o', linewidth=2, markersize=4)
            ax.axhline(y=tolerancia, color='r', linestyle='--', label=f'Tolerancia = {tolerancia}')
            ax.set_xlabel("Iteraciones", fontsize=12)
            ax.set_ylabel("Error", fontsize=12)
            ax.set_title(f"Convergencia del método {nombre_metodo}", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Tabla de iteraciones
            with st.expander("📋 Ver historial de errores"):
                st.write("**Evolución del error por iteración:**")
                tabla_errores = {"Iteración": [], "Error": []}
                for i, err in enumerate(errores[:50]):  # Mostrar primeras 50
                    tabla_errores["Iteración"].append(i + 1)
                    tabla_errores["Error"].append(f"{err:.2e}")
                st.table(tabla_errores)
                if len(errores) > 50:
                    st.info(f"Se muestran las primeras 50 iteraciones de {len(errores)} totales.")

        except Exception as e:
            st.error(f"Error durante la resolución: {e}")
            st.info("El método puede no converger para este sistema. Prueba con otro método o ajusta los coeficientes.")

else:
    st.info("👈 Configura el sistema en la barra lateral y presiona 'Resolver sistema'")

# ============================================
# PIE DE PÁGINA
# ============================================
st.markdown("---")
st.caption("Desafío: Optimización de Dieta en Aves Nestotropicales | Métodos Numéricos | UMSA")