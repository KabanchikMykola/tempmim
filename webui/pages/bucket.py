"""Bucket page — данные в HuggingFace Bucket."""

import dash
from dash import html, dash_table, Input, Output

from data_fetcher import config

dash.register_page(__name__, path="/bucket", name="Bucket", title="Данные в Bucket")


def scan_local_data():
    """Сканировать локальную папку fin_data/binance/."""
    results = []
    binance_dir = config.DATA_DIR / "binance"

    if not binance_dir.exists():
        return results

    for subdir, data_type in [
        ("ohlcv_spot", "OHLCV spot"),
        ("ohlcv_perp", "OHLCV perp"),
    ]:
        d = binance_dir / subdir
        if d.exists():
            for f in d.glob("*.parquet"):
                parts = f.stem.split("_")
                if len(parts) >= 3:
                    symbol, interval, year = parts[0], parts[1], parts[2]
                    size_mb = f.stat().st_size / (1024 * 1024)
                    results.append({
                        "symbol": symbol,
                        "data_type": f"{data_type} {interval}",
                        "years": year,
                        "size_str": f"{size_mb:.1f} MB",
                    })

    for subdir, data_type in [
        ("funding", "Funding"),
        ("metrics", "Metrics"),
    ]:
        d = binance_dir / subdir
        if d.exists():
            for f in d.glob("*.parquet"):
                symbol = f.stem.replace(f"_{subdir}", "")
                size_mb = f.stat().st_size / (1024 * 1024)
                results.append({
                    "symbol": symbol,
                    "data_type": data_type,
                    "years": "all",
                    "size_str": f"{size_mb:.1f} MB",
                })

    results.sort(key=lambda x: (x["symbol"], x["data_type"]))
    return results


def layout():
    return html.Div([

        html.H2("📦 Данные в Bucket", className="mb-4"),

        html.Div([
            html.Div([
                html.Strong("Локальные данные"),
                html.Button(
                    "🔄 Refresh",
                    id="refresh-bucket",
                    n_clicks=0,
                    className="btn btn-sm btn-outline-secondary float-end",
                ),
            ], className="card-header"),
            html.Div([
                html.Div(id="bucket-status", className="mb-2"),
                dash_table.DataTable(
                    id="bucket-table",
                    columns=[
                        {"name": "Symbol", "id": "symbol"},
                        {"name": "Data Type", "id": "data_type"},
                        {"name": "Years", "id": "years"},
                        {"name": "Size", "id": "size_str"},
                    ],
                    data=[],
                    page_size=20,
                    sort_action="native",
                    filter_action="native",
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
                ),
            ], className="card-body"),
        ], className="card mb-3"),

        dcc.Store(id="bucket-store"),
    ])


# ── Callbacks ──

@dash.callback(
    Output("bucket-store", "data"),
    Output("bucket-status", "children"),
    Input("refresh-bucket", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_bucket(n_clicks):
    data = scan_local_data()
    symbols = sorted(set(d["symbol"] for d in data))
    total_size = sum(float(d["size_str"].replace(" MB", "")) for d in data)
    status = f"📊 {len(symbols)} символов, {len(data)} файлов, {total_size:.0f} MB"
    return data, status


@dash.callback(
    Output("bucket-table", "data"),
    Input("bucket-store", "data"),
)
def update_bucket_table(data):
    if not data:
        return []
    return data
