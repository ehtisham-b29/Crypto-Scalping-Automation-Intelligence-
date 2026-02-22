"""
Live terminal display -- uses rich to show a real-time price dashboard.
Prices update at millisecond speed from the trades WebSocket.
Log messages from loguru are routed through the rich Console so they
print cleanly above the live table without garbling the display.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

# Shared console -- used by both the live display and the loguru sink
console = Console(highlight=False)

_SPINNER = ["|", "/", "-", "\\"]

_LEVEL_STYLE: dict[str, str] = {
    "DEBUG":    "dim white",
    "INFO":     "white",
    "SUCCESS":  "bold green",
    "WARNING":  "yellow",
    "ERROR":    "bold red",
    "CRITICAL": "bold red on white",
}


def loguru_sink(message) -> None:
    """
    Drop-in loguru sink that routes log lines through the rich Console.
    Rich's Live display will print them cleanly above the live table.
    """
    record = message.record
    level  = record["level"].name
    ts     = record["time"].strftime("%H:%M:%S")
    msg    = record["message"]
    style  = _LEVEL_STYLE.get(level, "white")

    text = Text()
    text.append(f"{ts} ", style="dim cyan")
    text.append(f"{level:<7} ", style=style)
    text.append(msg)
    console.print(text)


def build_table(states, portfolio, config, tick: int) -> Table:
    """Build the live price/status table rendered by the Live context."""
    spinner  = _SPINNER[tick % 4]
    mode     = "PAPER" if config.IS_PAPER else "LIVE"
    mode_col = "green" if config.IS_PAPER else "red"
    ts       = datetime.now().strftime("%H:%M:%S.%f")[:-4]   # HH:MM:SS.mm

    table = Table(
        title=(
            f"[bold cyan]{spinner}[/bold cyan]  "
            f"[bold white]AI Scalping Bot[/bold white]  "
            f"[dim]|[/dim]  [{mode_col}]{mode}[/{mode_col}]  "
            f"[dim]|[/dim]  [dim cyan]{ts}[/dim cyan]"
        ),
        title_justify="left",
        border_style="bright_blue",
        show_lines=False,
        expand=True,
    )
    table.add_column("Pair",       style="cyan",         min_width=16)
    table.add_column("Live Price", justify="right",      min_width=16)
    table.add_column("RSI",        justify="right",      min_width=6)
    table.add_column("Score",      justify="center",     min_width=7)
    table.add_column("Direction",  justify="center",     min_width=10)
    table.add_column("Status",     justify="left",       min_width=18)

    for state in states:
        pair = state.symbol.replace(":USDT", "")

        # Price -- updated from trades stream at ms level
        if state.last_price > 0:
            price_str = f"[bold bright_green]${state.last_price:>12,.4f}[/bold bright_green]"
        else:
            price_str = "[dim]connecting...[/dim]"

        # RSI
        rsi = state.cached_rsi
        if rsi > 0:
            if rsi < 35:
                rsi_str = f"[bold green]{rsi:.1f}[/bold green]"
            elif rsi > 65:
                rsi_str = f"[bold red]{rsi:.1f}[/bold red]"
            else:
                rsi_str = f"{rsi:.1f}"
        else:
            rsi_str = "[dim]--[/dim]"

        # Confluence score
        conf = state.cached_conf
        if rsi > 0:
            if abs(conf) >= config.MIN_CONFLUENCE_SCORE:
                conf_str = f"[bold green]{conf:+d}/7[/bold green]"
            else:
                conf_str = f"[dim]{conf:+d}/7[/dim]"
        else:
            conf_str = "[dim]--/7[/dim]"

        # Direction
        d = state.cached_dir
        if d == "long":
            dir_str = "[bold green]LONG[/bold green]"
        elif d == "short":
            dir_str = "[bold red]SHORT[/bold red]"
        else:
            dir_str = "[dim]NEUTRAL[/dim]"

        # Status
        if state.cached_tradeable:
            status_str = "[bold green blink]** SIGNAL! **[/bold green blink]"
        elif not state.ready:
            status_str = "[dim yellow]warming up...[/dim yellow]"
        else:
            status_str = "[dim]scanning...[/dim]"

        table.add_row(pair, price_str, rsi_str, conf_str, dir_str, status_str)

    # Portfolio summary row
    table.add_section()
    pnl     = portfolio.total_pnl
    capital = portfolio.capital
    s       = portfolio.summary()
    pnl_col = "green" if pnl >= 0 else "red"
    sign    = "+" if pnl >= 0 else ""

    table.add_row(
        "[bold white]Portfolio[/bold white]",
        f"[bold white]${capital:.2f}[/bold white]",
        "",
        f"[{pnl_col}]{sign}{pnl:.4f}[/{pnl_col}]",
        f"W:[green]{s['wins']}[/green] L:[red]{s['losses']}[/red]",
        f"[cyan]{s['target_progress_pct']:.1f}%[/cyan] of ${config.PROFIT_TARGET_USDT:.0f} target",
    )

    return table


async def run_live(get_states, get_portfolio, running) -> None:
    """
    Async task that runs the rich Live display at 5 fps (200ms refresh).
    - get_states:   callable -> list[MarketState]
    - get_portfolio: callable -> Portfolio
    - running:      callable -> bool  (False = stop)
    """
    import config as _config
    tick = 0
    with Live(
        build_table(get_states(), get_portfolio(), _config, tick),
        console=console,
        refresh_per_second=5,
        screen=False,
        auto_refresh=False,
    ) as live:
        while running():
            await asyncio.sleep(0.2)
            tick += 1
            live.update(
                build_table(get_states(), get_portfolio(), _config, tick)
            )
            live.refresh()
