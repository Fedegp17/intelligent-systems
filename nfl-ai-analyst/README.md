# NFL AI Analyst

Predicciones de partidos de NFL con un modelo sencillo, más el reporte oficial de lesiones y las noticias recientes de cada equipo. Todo corre en tu computadora y solo cuando lo pides: no hay base de datos, servidor ni tareas automáticas.

## Qué hace

- **Cálculo:** probabilidad de ganar, margen proyectado (spread) y total de puntos para cada partido.
- **Estrellas y bajas:** detecta el QB titular y las estrellas de cada equipo, revisa si se pierden el partido y ajusta el margen.
- **Lesiones:** el último reporte oficial de la NFL de ambos equipos (vía nflverse).
- **Noticias:** titulares de los últimos 3 días (Google News) y, si configuras tu clave de Anthropic, un resumen en español hecho por Claude. Claude solo resume; nunca cambia los números del modelo.
- **UI local:** una página para ver qué datos hay cargados, la semana, cada partido, los ratings y la validación.

## Cómo funciona el modelo

Dos ratings por equipo que se actualizan partido por partido, en orden, desde 2010:

1. **Elo** (estilo FiveThirtyEight): ventaja de local, ajuste por margen de victoria y regreso de 1/3 a la media entre temporadas. La diferencia de Elo entre 25 da el margen esperado.
2. **Puntos:** promedio exponencial de puntos anotados y permitidos (el último partido pesa 10%). Da los puntos esperados de cada equipo y, sumados, el total.

El margen final promedia ambos (50/50, el mejor peso en 2015–2022) y luego resta un castigo por cada estrella que no va a jugar. La probabilidad de ganar sale de una distribución normal del margen con desviación de 13.5 puntos.

### Estrellas y bajas

Antes de cada partido, con los últimos 10 partidos del equipo:

- **QB titular:** el quarterback con más dropbacks.
- **2 estrellas ofensivas:** los RB, WR o TE con más puntos de fantasy PPR.
- **2 estrellas defensivas:** los defensivos con más jugadas de impacto (sacks, golpes al QB, tackles para pérdida, intercepciones, pases defendidos, balones forzados).

Una estrella cuenta como baja si el reporte oficial la pone Out o Doubtful, o si el roster semanal la tiene en la lista de lesionados (IR), liberada o ya no está en el equipo. Los inactivos del día del partido no se usan porque salen 90 minutos antes.

Castigo por jugador ausente, en puntos de margen: **QB 2.0, estrella ofensiva 1.1, estrella defensiva 0.6**. Se ajustaron con mínimos cuadrados sobre los errores del modelo en 2013–2022. El castigo del QB parece bajo porque muchas bajas son largas y los ratings ya absorbieron parte del efecto. Probé también separar bajas nuevas de bajas largas y ponderar por la calidad del QB, y casi no mejoraba, así que se quedó simple. Las bajas no cambian el total proyectado (tampoco mejoraba).

### Validación

Cada predicción usa solo partidos jugados antes de ella, así que la validación no hace trampa. Resultados en 2023–2025 (855 partidos):

| Métrica | Sin bajas | Con bajas | Línea de cierre del mercado |
|---|---|---|---|
| Aciertos de ganador | 64.8% | 66.8% | |
| Brier score | 0.221 | 0.217 | |
| Error medio del margen | 10.20 pts | 10.08 pts | 9.79 pts |
| Error medio del total | 10.62 pts | 10.62 pts | 10.12 pts |

El mercado sigue siendo más preciso: sabe de lesiones de jugadores que no son estrellas, del clima y de noticias de último momento.

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
│   ├── model.py           Elo + modelo de puntos, castigo por bajas, validación
│   ├── availability.py    estrellas de cada equipo y si juegan
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
