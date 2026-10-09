import pandas as pd
from shiny import App, ui, render

# Cargar el dataset fuera de cualquier función (se ejecuta una sola vez)
df  = pd.read_csv("../hr_ibm.csv")


# Preparar las opciones para los tres selectores
opts_dept = ["Todos"] + sorted(df["Department"].unique().tolist())
opts_genero = ["Todos"] + sorted(df["Gender"].unique().tolist())
opts_edu = ["Todos"] + sorted(df["EducationField"].unique().tolist())


# 'Todos' siempre es la primera opción — significa "sin filtro aplicado"
print(f"Data set: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Generos Disponibles: {df['Gender'].unique().tolist()}")
print(f"Campos Educativos: {df['EducationField'].unique().tolist()}")
print(f"Salario Promedio Global: {df['MonthlyIncome'].mean():,.0f}")

# ── Interfaz de usuario ───────────────────────────────────────
app_ui = ui.page_fluid(
    ui.h2("Explorador de empleados"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Filtros"),
            ui.input_select("dept", "Departamento", choices=opts_dept),
            ui.input_select("genero", "Género", choices=opts_genero),
            ui.input_select("edu", "Campo Educativo", choices=opts_edu)
        ),
        #salidas
        ui.output_text("conteo_texto"),
        ui.output_table("tabla_resultados"),
    )
)

# ── Servidor ──────────────────────────────────────────────────
def server(input, output, session):

    def _filtrar():
        d = df.copy() #copiamos el dataset

        if input.dept() != "Todos":
            d = d[d["Department"] == input.dept()]

        if input.genero() != "Todos":
            d = d[d["Gender"] == input.genero()]

        if input.edu() != "Todos":
            d = d[d["EducationField"] == input.edu()]

        return d

    @render.text
    def conteo_texto():
        d =_filtrar()
        n = len(d)
        if n == 0:
            return "Sin resultados, no se encontraron combinaciones posibles"
        sal = d["MonthlyIncome"].mean()
        return f"{n} empleados encontrados   ---- Salario Promedio: ${sal:,.0f}"

    @render.table
    def tabla_resultados():
        cols = ["EmployeeNumber", "Department", "Gender", "EducationField", "JobRole", "Attrition" , "MonthlyIncome"]
        return _filtrar()[cols].head(10)

app = App(app_ui, server)