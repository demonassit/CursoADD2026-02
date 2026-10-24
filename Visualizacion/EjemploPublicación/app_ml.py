
from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

# ── Dataset historico (para las graficas de contexto) ────────
df = pd.read_csv("hr_ibm.csv")
roles = sorted(df["JobRole"].unique().tolist())   # mismo orden que JobRole_n

# plotly.js se carga UNA vez en el <head>; si cada grafica lo trae por CDN,
# la primera pestana intenta dibujar antes de que termine de descargarse.
from plotly.offline import get_plotlyjs_version
plotly_js = ui.head_content(ui.tags.script(
    src=f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"))

def grafica(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=40, r=20, t=50, b=40),
                      paper_bgcolor="white", plot_bgcolor="white")
    return ui.HTML(fig.to_html(include_plotlyjs=False, full_html=False))

# ── Cargar modelos al arrancar la app (una sola vez) ─────────
modelo_lr   = joblib.load("modelo_clasificador_lr.pkl")
modelo_perm = joblib.load("modelo_permanencia.pkl")
modelo_sal  = joblib.load("modelo_salario.pkl")

with open("features_clasificador.json") as f: import json; feats_clf = json.load(f)
with open("features_permanencia.json")  as f: feats_perm = json.load(f)
with open("features_salario.json")      as f: feats_sal  = json.load(f)

# ── CSS ──────────────────────────────────────────────────────
css_lines = [
    "    body { background:#F4F6F9; font-family:Segoe UI,Arial,sans-serif; }",
    "    .navbar { background:#1F3864 !important; }",
    "    .navbar-brand, .nav-link { color:white !important; font-weight:500; }",
    "    .pred-box { background:white; border-radius:10px; padding:24px;",
    "               box-shadow:0 2px 8px rgba(0,0,0,.1); text-align:center; }",
    "    .pred-valor { font-size:2.4em; font-weight:bold; }",
    "    .pred-label { font-size:.9em; color:#666; margin-top:6px; }",
]
css = ui.tags.style(chr(10).join(css_lines))

# ── UI ───────────────────────────────────────────────────────
app_ui = ui.page_navbar(

    # PESTANA 1 -- CLASIFICADOR
    ui.nav_panel("Predictor de renuncia",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Perfil del empleado"),
                ui.input_slider("age",       "Edad",              18, 60, 35),
                ui.input_slider("jobsat",    "Satisfaccion (1-4)", 1,  4,  3),
                ui.input_slider("envsat",    "Sat. Ambiente (1-4)",1,  4,  3),
                ui.input_slider("wlb",       "Bal. Vida-Trabajo",  1,  4,  3),
                ui.input_slider("jobinv",    "Involucramiento",    1,  4,  3),
                ui.input_slider("stock",     "Stock Options",      0,  3,  0),
                ui.input_slider("years_co",  "Años en empresa",    0, 40,  2),
                ui.input_slider("income",    "Salario mensual",  1000,20000,5000,step=500),
                ui.input_slider("prev_cos",  "Empresas previas",  0,  9,  1),
                ui.input_slider("total_yrs", "Años experiencia",  0, 40,  5),
                ui.input_slider("yrs_mgr",   "Años con manager",  0, 17,  2),
                ui.input_select("overtime",  "Horas extra",
                                {"No":"No","Yes":"Si (Yes)"}),
                ui.input_select("marital",   "Estado civil",
                                {"Single":"Soltero","Married":"Casado","Divorced":"Divorciado"}),
                ui.input_select("travel",    "Viajes de trabajo",
                                {"Non-Travel":"No viaja",
                                 "Travel_Rarely":"Raramente",
                                 "Travel_Frequently":"Frecuente"}),
            ),
            ui.output_ui("pred_renuncia"),
            ui.output_ui("comparacion_score"),
            ui.layout_columns(
                ui.card(ui.output_ui("graf_gauge")),
                ui.card(ui.output_ui("graf_factores")),
                col_widths=[5, 7], fill=False,
            ),
        )
    ),

    # PESTANA 2 -- PERMANENCIA
    ui.nav_panel("Permanencia estimada",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Perfil del empleado"),
                ui.input_slider("p_age",     "Edad",              18, 60, 30),
                ui.input_slider("p_joblvl",  "Nivel de puesto",    1,  5,  2),
                ui.input_slider("p_jobsat",  "Satisfaccion (1-4)", 1,  4,  3),
                ui.input_slider("p_wlb",     "Bal. Vida-Trabajo",  1,  4,  3),
                ui.input_slider("p_income",  "Salario mensual", 1000,20000,4000,step=500),
                ui.input_slider("p_numcos",  "Empresas previas",   0,  9,  1),
                ui.input_slider("p_totalyrs","Años experiencia",   0, 40,  5),
            ),
            ui.output_ui("pred_permanencia"),
            ui.layout_columns(
                ui.card(ui.output_ui("graf_hist_perm")),
                ui.card(ui.output_ui("graf_sens_perm")),
                col_widths=[6, 6],
            ),
        )
    ),

    # PESTANA 3 -- SALARIO
    ui.nav_panel("Salario estimado",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h5("Perfil del empleado"),
                ui.input_slider("s_joblvl",  "Nivel de puesto",    1,  5,  2),
                ui.input_slider("s_totalyrs","Años experiencia",   0, 40,  5),
                ui.input_slider("s_educ",    "Educacion (1-5)",    1,  5,  3),
                ui.input_select("s_jobrole", "Puesto",
                    {"0":"Healthcare Rep","1":"Human Resources",
                     "2":"Laboratory Tech","3":"Manager",
                     "4":"Manufacturing Dir","5":"Research Dir",
                     "6":"Research Scientist","7":"Sales Exec","8":"Sales Rep"}),
            ),
            ui.output_ui("pred_salario"),
            ui.layout_columns(
                ui.card(ui.output_ui("graf_box_sal")),
                ui.card(ui.output_ui("graf_sens_sal")),
                col_widths=[6, 6],
            ),
        )
    ),

    title="TechNova Solutions -- Predictor de RH con ML",
    header=ui.TagList(plotly_js, css),
)

# ── SERVER ───────────────────────────────────────────────────
def server(input, output, session):

    @render.ui
    def pred_renuncia():
        le_ot = {"No":0,"Yes":1}
        le_ms = {"Single":2,"Married":1,"Divorced":0}
        le_bt = {"Non-Travel":0,"Travel_Frequently":1,"Travel_Rarely":2}

        fila = pd.DataFrame([[
            input.age(), input.jobsat(), input.envsat(), input.wlb(),
            input.jobinv(), input.stock(), input.years_co(), input.income(),
            input.prev_cos(), input.total_yrs(), input.yrs_mgr(),
            le_ot[input.overtime()], le_ms[input.marital()], le_bt[input.travel()]
        ]], columns=feats_clf)

        prob  = modelo_lr.predict_proba(fila)[0][1]
        pred  = modelo_lr.predict(fila)[0]

        if prob >= 0.70:   color = "#C62828"; nivel = "RIESGO ALTO"
        elif prob >= 0.40: color = "#F57F17"; nivel = "RIESGO MEDIO"
        else:              color = "#2E7D32"; nivel = "RIESGO BAJO"

        # Score manual para comparar
        ot_val  = 100 if input.overtime()=="Yes" else 0
        js_val  = (4 - input.jobsat()) / 3 * 100
        yr_val  = max(0, (3 - input.years_co()) / 3 * 100)
        wlb_val = (4 - input.wlb()) / 3 * 100
        es_val  = (4 - input.envsat()) / 3 * 100
        score   = (ot_val*0.30 + js_val*0.25 + yr_val*0.20 +
                   wlb_val*0.15 + es_val*0.10)

        return ui.div(
            ui.row(
                ui.column(6, ui.div(
                    ui.div(f"{prob*100:.1f} %", class_="pred-valor",
                           style=f"color:{color};"),
                    ui.div(f"Probabilidad de renuncia -- {nivel}", class_="pred-label"),
                    class_="pred-box")),
                ui.column(6, ui.div(
                    ui.div(f"{score:.1f}/100", class_="pred-valor",
                           style="color:#1A5F8A;"),
                    ui.div("Score manual (metodo S5)", class_="pred-label"),
                    class_="pred-box")),
            )
        )

    @render.ui
    def comparacion_score():
        return ui.div(
            ui.p("Columna izquierda: modelo ML (aprendio de los datos historicos). "
                 "Columna derecha: score manual construido en S5 con reglas del analista. "
                 "¿Coinciden? ¿Cual confias mas?",
                 style="color:#666;font-size:13px;margin-top:12px;")
        )

    @render.ui
    def pred_permanencia():
        fila = pd.DataFrame([[
            input.p_age(), input.p_joblvl(), input.p_jobsat(),
            input.p_wlb(), input.p_income(), input.p_numcos(), input.p_totalyrs()
        ]], columns=feats_perm)
        anos = max(0, modelo_perm.predict(fila)[0])
        if anos >= 5:   color = "#2E7D32"
        elif anos >= 2: color = "#F57F17"
        else:           color = "#C62828"
        return ui.div(ui.div(
            ui.div(f"{anos:.1f} años", class_="pred-valor", style=f"color:{color};"),
            ui.div("Permanencia estimada en la empresa", class_="pred-label"),
            class_="pred-box"), style="max-width:400px;margin:40px auto;")

    @render.ui
    def pred_salario():
        fila = pd.DataFrame([[
            int(input.s_joblvl()), int(input.s_totalyrs()),
            int(input.s_educ()), int(input.s_jobrole())
        ]], columns=feats_sal)
        sal = max(0, modelo_sal.predict(fila)[0])
        return ui.div(ui.div(
            ui.div(f"${sal:,.0f}", class_="pred-valor", style="color:#6A1B9A;"),
            ui.div("Salario mensual estimado (USD)", class_="pred-label"),
            class_="pred-box"), style="max-width:400px;margin:40px auto;")

    # ── GRAFICAS PESTANA 1 ───────────────────────────────────
    @reactive.calc
    def fila_clf():
        le_ot = {"No":0,"Yes":1}
        le_ms = {"Single":2,"Married":1,"Divorced":0}
        le_bt = {"Non-Travel":0,"Travel_Frequently":1,"Travel_Rarely":2}
        return pd.DataFrame([[
            input.age(), input.jobsat(), input.envsat(), input.wlb(),
            input.jobinv(), input.stock(), input.years_co(), input.income(),
            input.prev_cos(), input.total_yrs(), input.yrs_mgr(),
            le_ot[input.overtime()], le_ms[input.marital()], le_bt[input.travel()]
        ]], columns=feats_clf)

    @render.ui
    def graf_gauge():
        prob = modelo_lr.predict_proba(fila_clf())[0][1] * 100
        tasa = (df["Attrition"] == "Yes").mean() * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=prob,
            number={"suffix":" %", "valueformat":".1f"},
            delta={"reference":tasa, "valueformat":".1f", "suffix":" pts",
                   "increasing":{"color":"#C62828"}, "decreasing":{"color":"#2E7D32"}},
            title={"text":f"Probabilidad de renuncia<br><sup>vs tasa historica {tasa:.1f} %</sup>"},
            gauge={"axis":{"range":[0,100]}, "bar":{"color":"#1F3864"},
                   "steps":[{"range":[0,40],  "color":"#C8E6C9"},
                            {"range":[40,70], "color":"#FFE0B2"},
                            {"range":[70,100],"color":"#FFCDD2"}]}))
        return grafica(fig)

    @render.ui
    def graf_factores():
        # Aporte de cada factor al score manual (mismos pesos que pred_renuncia)
        factores = pd.DataFrame({
            "Factor": ["Horas extra", "Satisfaccion", "Antiguedad",
                       "Bal. Vida-Trabajo", "Sat. Ambiente"],
            "Puntos": [
                (100 if input.overtime()=="Yes" else 0) * 0.30,
                (4 - input.jobsat()) / 3 * 100 * 0.25,
                max(0, (3 - input.years_co()) / 3 * 100) * 0.20,
                (4 - input.wlb()) / 3 * 100 * 0.15,
                (4 - input.envsat()) / 3 * 100 * 0.10,
            ],
            "Maximo": [30, 25, 20, 15, 10],
        }).sort_values("Puntos")
        fig = go.Figure()
        fig.add_bar(y=factores["Factor"], x=factores["Maximo"], orientation="h",
                    name="Maximo posible", marker_color="#E0E0E0")
        fig.add_bar(y=factores["Factor"], x=factores["Puntos"], orientation="h",
                    name="Aporte actual", marker_color="#0C9BF3",
                    text=factores["Puntos"].round(1), textposition="outside")
        fig.update_layout(barmode="overlay", title="Aporte de cada factor al score manual",
                          xaxis_title="Puntos", legend=dict(orientation="h", y=-0.2))
        return grafica(fig)

    # ── GRAFICAS PESTANA 2 ───────────────────────────────────
    def fila_perm(total_yrs=None):
        return [input.p_age(), input.p_joblvl(), input.p_jobsat(), input.p_wlb(),
                input.p_income(), input.p_numcos(),
                input.p_totalyrs() if total_yrs is None else total_yrs]

    @render.ui
    def graf_hist_perm():
        anos = max(0, modelo_perm.predict(pd.DataFrame([fila_perm()], columns=feats_perm))[0])
        fig = px.histogram(df, x="YearsAtCompany", nbins=40,
                           color_discrete_sequence=["#8BEEB4"],
                           title="Antiguedad real de los empleados (historico)",
                           labels={"YearsAtCompany":"Años en la empresa"})
        fig.add_vline(x=anos, line_width=3, line_dash="dash", line_color="#C62828",
                      annotation_text=f"Estimado: {anos:.1f}", annotation_position="top right")
        fig.update_layout(yaxis_title="Empleados", bargap=0.05)
        return grafica(fig)

    @render.ui
    def graf_sens_perm():
        # Como cambia la estimacion si solo varian los años de experiencia
        rango = list(range(0, 41))
        filas = pd.DataFrame([fila_perm(t) for t in rango], columns=feats_perm)
        pred  = np.clip(modelo_perm.predict(filas), 0, None)
        actual = max(0, modelo_perm.predict(pd.DataFrame([fila_perm()], columns=feats_perm))[0])
        fig = px.line(x=rango, y=pred, title="Permanencia estimada segun años de experiencia",
                      labels={"x":"Años de experiencia", "y":"Permanencia estimada (años)"})
        fig.update_traces(line_color="#1F3864")
        fig.add_scatter(x=[input.p_totalyrs()], y=[actual], mode="markers",
                        marker=dict(size=14, color="#860000"), name="Perfil actual")
        fig.update_layout(showlegend=False)
        return grafica(fig)

    # ── GRAFICAS PESTANA 3 ───────────────────────────────────
    @render.ui
    def graf_box_sal():
        rol = roles[int(input.s_jobrole())]
        d = df[df["JobRole"] == rol]
        sal = max(0, modelo_sal.predict(pd.DataFrame([[
            int(input.s_joblvl()), int(input.s_totalyrs()),
            int(input.s_educ()), int(input.s_jobrole())]], columns=feats_sal))[0])
        fig = px.box(d, x="JobLevel", y="MonthlyIncome", points="all",
                     color_discrete_sequence=["#5B05C5"],
                     title=f"Salario real por nivel -- {rol}",
                     labels={"JobLevel":"Nivel de puesto", "MonthlyIncome":"Salario mensual (USD)"})
        fig.add_scatter(x=[int(input.s_joblvl())], y=[sal], mode="markers",
                        marker=dict(size=16, color="#C0F416", symbol="star"),
                        name="Estimado")
        fig.update_layout(showlegend=False, xaxis=dict(dtick=1))
        return grafica(fig)

    @render.ui
    def graf_sens_sal():
        # Curva del salario estimado por nivel de puesto vs años de experiencia
        rango = list(range(0, 41))
        fig = go.Figure()
        for nivel in range(1, 6):
            filas = pd.DataFrame([[nivel, t, int(input.s_educ()), int(input.s_jobrole())]
                                  for t in rango], columns=feats_sal)
            actual = nivel == int(input.s_joblvl())
            fig.add_scatter(x=rango, y=np.clip(modelo_sal.predict(filas), 0, None),
                            mode="lines", name=f"Nivel {nivel}",
                            line=dict(width=4 if actual else 1.5,
                                      color="#6A1B9A" if actual else "#B0BEC5"))
        sal = max(0, modelo_sal.predict(pd.DataFrame([[
            int(input.s_joblvl()), int(input.s_totalyrs()),
            int(input.s_educ()), int(input.s_jobrole())]], columns=feats_sal))[0])
        fig.add_scatter(x=[int(input.s_totalyrs())], y=[sal], mode="markers",
                        marker=dict(size=14, color="#C62828"), name="Perfil actual")
        fig.update_layout(title="Salario estimado segun experiencia y nivel",
                          xaxis_title="Años de experiencia", yaxis_title="Salario mensual (USD)")
        return grafica(fig)

app = App(app_ui, server)
