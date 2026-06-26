import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import traceback

from exchanges import fetch_all_markets, fetch_all_tickers
from matcher import find_common_instruments, add_volume, get_stats

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Cross-Exchange Finder",
)

cached_markets: pd.DataFrame = pd.DataFrame()
cached_tickers: pd.DataFrame = pd.DataFrame()
cached_result: pd.DataFrame = pd.DataFrame()
cached_stats: pd.DataFrame = pd.DataFrame()

layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Cross-Exchange Instrument Finder",
                    className="text-center my-3",
                )
            )
        ),
        dbc.Row(
            dbc.Col(
                html.P(
                    "Инструменты на Binance, Bybit и OKX — спот и perpetual. "
                    "Сортировка по 24h объёму.",
                    className="text-center text-muted mb-4",
                )
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        "Загрузить рынки",
                        id="btn-load",
                        color="primary",
                        size="lg",
                        className="w-100",
                    ),
                    width=3,
                ),
                dbc.Col(
                    [
                        html.Label("Мин. объём 24h (USDT)", style={"color": "#ccc"}),
                        dcc.Dropdown(
                            id="dropdown-volume",
                            options=[
                                {"label": "Любой", "value": 0},
                                {"label": "> 1K", "value": 3},
                                {"label": "> 10K", "value": 4},
                                {"label": "> 100K", "value": 5},
                                {"label": "> 1M", "value": 6},
                                {"label": "> 10M", "value": 7},
                            ],
                            value=4,
                            clearable=False,
                            style={"backgroundColor": "#333", "color": "#000"},
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    dbc.Input(
                        id="input-search",
                        type="text",
                        placeholder="Поиск по названию...",
                        debounce=True,
                    ),
                    width=3,
                ),
                dbc.Col(
                    html.Div(id="status-text", className="pt-2 text-muted"),
                    width=2,
                ),
            ],
            className="mb-4",
        ),
        dcc.Loading(
            id="loading",
            type="circle",
            children=html.Div(id="loading-output"),
        ),
        html.Div(id="results-container"),
    ],
    fluid=True,
    className="py-3",
)

app.layout = layout


def format_volume(vol: float) -> str:
    """Форматировать объём для отображения."""
    if vol >= 1_000_000:
        return f"{vol / 1_000_000:.1f}M"
    if vol >= 1_000:
        return f"{vol / 1_000:.1f}K"
    return f"{vol:.0f}"


@callback(
    Output("loading-output", "children"),
    Output("status-text", "children"),
    Output("results-container", "children"),
    Input("btn-load", "n_clicks"),
    Input("input-search", "value"),
    Input("dropdown-volume", "value"),
    prevent_initial_call=True,
)
def update_results(n_clicks, search_text, volume_level):
    global cached_markets, cached_tickers, cached_result, cached_stats

    try:
        if cached_markets.empty or (
            dash.ctx.triggered_id == "btn-load" and n_clicks
        ):
            cached_markets = fetch_all_markets()
            if cached_markets.empty:
                return "", "Ошибка загрузки данных", html.Div()

            cached_tickers = fetch_all_tickers()
            common = find_common_instruments(cached_markets)
            cached_result = add_volume(common, cached_tickers)
            cached_stats = get_stats(cached_markets, cached_result)

        if cached_result.empty:
            return "", "Общих инструментов не найдено", html.Div()

        result = cached_result.copy()
        stats = cached_stats

        # Фильтр по объёму
        vol_threshold = 10 ** (volume_level or 4)
        result = result[result["total_volume_24h"] >= vol_threshold]

        # Фильтр по поиску
        if search_text:
            mask = result["base"].str.contains(search_text.upper(), case=False, na=False)
            result = result[mask]

        count = len(result)
        total_vol = result["total_volume_24h"].sum()
        status = f"{count} инст. | сумм. объём: {format_volume(total_vol)}"
        table = build_table(result, stats)
        return "", status, table

    except Exception as e:
        tb = traceback.format_exc()
        error_card = dbc.Alert(
            [
                html.H5("Ошибка", className="alert-heading"),
                html.Pre(tb, style={"whiteSpace": "pre-wrap", "fontSize": "12px"}),
            ],
            color="danger",
        )
        return "", str(e), error_card


def build_table(result: pd.DataFrame, stats: pd.DataFrame) -> html.Div:
    if result.empty:
        return dbc.Alert("Нет результатов", color="warning")

    rows = []
    for rank, (_, row) in enumerate(result.iterrows(), 1):
        base = row["base"]
        vol = row.get("total_volume_24h", 0)
        stat_row = stats[stats["base"] == base]
        stat_row = stat_row.iloc[0] if len(stat_row) > 0 else {}

        badges = []
        for ex in ["binance", "bybit", "okx"]:
            for mt, label, color in [("spot", "S", "success"), ("swap", "P", "danger")]:
                col = f"{ex}_{mt}"
                if col in row.index and row[col]:
                    symbol = stat_row.get(f"{col}_symbol", "")
                    market_id = stat_row.get(f"{col}_id", "")
                    tip = f"{symbol} ({market_id})" if market_id else symbol
                    badges.append(
                        dbc.Badge(
                            f"{ex[0].upper()}{label}",
                            color=color,
                            className="me-1",
                            title=tip,
                            style={"fontSize": "11px"},
                        )
                    )

        # Объём по биржам
        vol_parts = []
        for ex in ["binance", "bybit", "okx"]:
            v = row.get(f"vol_{ex}", 0)
            if v > 0:
                vol_parts.append(f"{ex[0].upper()}: {format_volume(v)}")

        rows.append(
            html.Tr([
                html.Td(str(rank), className="text-center text-muted", style={"width": "4%"}),
                html.Td(html.Strong(base), className="fs-6", style={"width": "10%"}),
                html.Td(html.Div(badges), style={"width": "25%"}),
                html.Td(format_volume(vol), className="text-end", style={"width": "12%"}),
                html.Td(
                    html.Small(" | ".join(vol_parts), className="text-muted"),
                    style={"width": "30%"},
                ),
            ], className="align-middle")
        )

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("#", className="text-center", style={"width": "4%"}),
                html.Th("Актив", style={"width": "10%"}),
                html.Th("Биржи (S=spot, P=perp)", style={"width": "25%"}),
                html.Th("Объём 24h", className="text-end", style={"width": "12%"}),
                html.Th("По биржам", style={"width": "30%"}),
            ])),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        size="sm",
        className="mt-2",
    )

    return html.Div([
        dbc.Card(dbc.CardBody([
            html.H5(f"Найдено: {len(result)} инструментов", className="card-title"),
            html.P(
                "Binance + Bybit + OKX | спот + perpetual | "
                "отсортировано по 24h объёму (quote volume, USDT)",
                className="card-text text-muted small",
            ),
        ]), className="mb-3"),
        table,
    ])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=True)
