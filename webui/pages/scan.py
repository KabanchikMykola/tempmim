"""Scan Pairs page — выбор периода и пар для загрузки."""

import dash
from dash import html, dcc, dash_table, Input, Output, State, callback_context

dash.register_page(__name__, path="/", name="Scan", title="Scan Pairs")


def layout():
    return html.Div([

        html.H2("🔍 Scan Pairs", className="mb-4"),

        # ── Period ──
        html.Div([
            html.Div([html.Strong("Период")], className="card-header"),
            html.Div([
                html.Div([
                    dcc.Input(
                        id="start-date",
                        type="text",
                        value="2025-01-01",
                        placeholder="YYYY-MM-DD",
                        className="form-control",
                        style={"maxWidth": "200px", "color": "white", "backgroundColor": "#505050"},
                    ),
                    html.Button(
                        "🔍 Scan Pairs",
                        id="scan-button",
                        n_clicks=0,
                        className="btn btn-primary ms-3",
                    ),
                ], className="d-flex align-items-center"),
                html.Div(id="scan-status", className="mt-2"),
            ], className="card-body"),
        ], className="card mb-3"),

        # ── Pairs table ──
        html.Div([
            html.Div([html.Strong("Пары")], className="card-header"),
            html.Div([
                html.Div([
                    html.Div("Quick select top:", className="me-2"),
                    html.Div(
                        dcc.Dropdown(
                            id="top-n-dropdown",
                            options=[
                                {"label": "10", "value": 10},
                                {"label": "25", "value": 25},
                                {"label": "50", "value": 50},
                                {"label": "100", "value": 100},
                                {"label": "All", "value": -1},
                            ],
                            value=None,
                            placeholder="Top N...",
                            style={"width": 120, "color": "#000"},
                            clearable=True,
                        ),
                        className="me-3",
                    ),
                    html.Div(id="summary-line", className="text-light"),
                ], className="d-flex align-items-center mb-3"),

                html.Div([
                    dash_table.DataTable(
                        id="pairs-table",
                        columns=[
                            {"name": "Symbol", "id": "symbol"},
                            {"name": "Base", "id": "baseAsset"},
                            {"name": "Volume 24h", "id": "volume_str"},
                            {"name": "Listed", "id": "listed_date"},
                        ],
                        data=[],
                        row_selectable="multi",
                        selected_rows=[],
                        page_size=50,
                        style_header={
                            "backgroundColor": "rgb(30, 30, 30)",
                            "color": "white",
                            "fontWeight": "bold",
                        },
                        style_cell={
                            "backgroundColor": "rgb(50, 50, 50)",
                            "color": "white",
                            "fontFamily": "monospace",
                        },
                        style_data_conditional=[{
                            "if": {"state": "selected"},
                            "backgroundColor": "rgba(0, 123, 255, 0.3)",
                        }],
                    ),
                ]),
            ], className="card-body"),
        ], className="card mb-3"),

        dcc.Store(id="pair-store"),
    ])


# ── Callbacks ──

@dash.callback(
    Output("pair-store", "data"),
    Output("scan-status", "children"),
    Output("scan-button", "disabled"),
    Input("scan-button", "n_clicks"),
    State("start-date", "value"),
    prevent_initial_call=True,
)
def scan_pairs(n_clicks, start_date):
    from datetime import datetime, timezone
    from data_fetcher.menu import get_common_symbols, check_listing_dates, get_volumes

    if not start_date:
        start_date = "2025-01-01"

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return [], "❌ Формат даты: YYYY-MM-DD", False

    common = get_common_symbols(start)
    if not common:
        return [], "❌ Нет общих пар", False

    filtered = check_listing_dates(common, start)
    if not filtered:
        return [], "❌ Все пары отсеяны по дате", False

    with_volumes = get_volumes(filtered)

    for row in with_volumes:
        vol = row["volume_usd"]
        row["volume_str"] = f"${vol/1e6:.1f}M" if vol >= 1e6 else f"${vol:,.0f}"
        ts = row.get("first_kline", 0)
        if ts:
            row["listed_date"] = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m")
        else:
            row["listed_date"] = "—"

    return with_volumes, f"✅ Найдено {len(with_volumes)} пар", False


@dash.callback(
    Output("pairs-table", "data"),
    Output("pairs-table", "selected_rows"),
    Input("pair-store", "data"),
    Input("top-n-dropdown", "value"),
)
def update_table(data, top_n):
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if not data:
        return [], []

    if triggered_id == "top-n-dropdown" and top_n is not None:
        n = len(data) if top_n == -1 else min(top_n, len(data))
        return data, list(range(n))

    return data, []


@dash.callback(
    Output("summary-line", "children"),
    Input("pairs-table", "selected_rows"),
    State("pair-store", "data"),
)
def update_summary(selected_rows, data):
    if not data:
        return "Загрузите пары через Scan"
    n = len(selected_rows)
    total = len(data)
    if n == 0:
        return f"Выбрано: 0/{total}"
    symbols = [data[i]["symbol"] for i in selected_rows]
    display = ", ".join(symbols[:5])
    if n > 5:
        display += "..."
    return f"Выбрано: {n}/{total} — {display}"
