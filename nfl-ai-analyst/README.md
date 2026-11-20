# NFL AI Analyst

Predicciones de partidos de NFL con un modelo sencillo, más el reporte oficial de lesiones y las noticias recientes de cada equipo. Todo corre en tu computadora y solo cuando lo pides: no hay base de datos, servidor ni tareas automáticas.

## Qué hace

- **Cálculo:** probabilidad de ganar, margen proyectado (spread) y total de puntos para cada partido.
- **Lesiones:** el último reporte oficial de la NFL de ambos equipos (vía nflverse).
- **Noticias:** titulares de los últimos 3 días (Google News) y, si configuras tu clave de Anthropic, un resumen en español hecho por Claude. Claude solo resume; nunca cambia los números del modelo.
- **UI local:** una página para ver qué datos hay cargados, la semana, cada partido, los ratings y la validación.

## Cómo funciona el modelo

Dos ratings por equipo que se actualizan partido por partido, en orden, desde 2010:

1. **Elo** (estilo FiveThirtyEight): ventaja de local, ajuste por margen de victoria y regreso de 1/3 a la media entre temporadas. La diferencia de Elo entre 25 da el margen esperado.
2. **Puntos:** promedio exponencial de puntos anotados y permitidos (el último partido pesa 10%). Da los puntos esperados de cada equipo y, sumados, el total.

El margen final promedia ambos (50/50, el mejor peso en 2015–2022). La probabilidad de ganar sale de una distribución normal del margen con desviación de 13.5 puntos.

Cada predicción usa solo partidos jugados antes de ella, así que la validación no hace trampa. Resultados en 2023–2025 (855 partidos):

| Métrica | Modelo | Línea de cierre del mercado |
|---|---|---|
| Aciertos de ganador | 64.8% | |
| Brier score | 0.221 | |
| Error medio del margen | 10.20 pts | 9.79 pts |
| Error medio del total | 10.62 pts | 10.12 pts |

El mercado sigue siendo más preciso. El modelo no sabe de lesiones; por eso el reporte las muestra aparte.

## Instalación (macOS)

Necesitas Python 3.11 o más reciente.

```bash
cd nfl-ai-analyst
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # opcional: pon tu ANTHROPIC_API_KEY para el resumen de noticias
```

La primera ejecución descarga los datos de nflverse (unos segundos) y los guarda en `data/cache`, que se refresca cada 6 horas.

## Uso

```bash
nfl-analyst week                 # predicciones de la próxima semana
nfl-analyst week --week 12       # una semana específica
nfl-analyst game KC ARI          # un partido: cálculo + lesiones + noticias
nfl-analyst game KC ARI --no-news
nfl-analyst ratings              # Elo y puntos de cada equipo
nfl-analyst backtest             # validación en las últimas 3 temporadas
```

Los equipos usan los códigos de nflverse: ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN, DET, GB, HOU, IND, JAX, KC, LA, LAC, LV, MIA, MIN, NE, NO, NYG, NYJ, PHI, PIT, SEA, SF, TB, TEN, WAS.

### UI

```bash
streamlit run app.py
```

Abre http://localhost:8501. Pestañas: Datos cargados, Semana, Partido, Ratings y Validación.

## Estructura

```
nfl-ai-analyst/
├── app.py                 UI local (Streamlit)
├── nfl_analyst/
│   ├── data.py            descarga y caché de nflverse
│   ├── model.py           Elo + modelo de puntos, validación
│   ├── news.py            lesiones, titulares y resumen con Claude
│   ├── analysis.py        une datos, modelo y noticias
│   ├── report.py          salida en Markdown
│   ├── teams.py           códigos y nombres de equipos
│   └── cli.py             comandos
└── tests/
```

## Pruebas

```bash
pip install -e ".[dev]"
pytest
```
