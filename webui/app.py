"""WEBUI — Binance Data Downloader.

Multi-page Dash app.
Запуск: uv run python -m webui
"""

import dash
from dash import html
import dash_bootstrap_components as dbc

import webui.pages  # noqa: F401 — register pages


def create_app():
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="Binance Data Downloader",
        use_pages=True,
        suppress_callback_exceptions=True,
    )

    nav = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Scan", href="/")),
            dbc.NavItem(dbc.NavLink("Bucket", href="/bucket")),
        ],
        brand="Binance Data Downloader",
        brand_href="/",
        color="dark",
        dark=True,
    )

    app.layout = html.Div([
        nav,
        dash.page_container,
    ], className="container-fluid p-4")

    return app


def main():
    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=8051)


if __name__ == "__main__":
    main()
