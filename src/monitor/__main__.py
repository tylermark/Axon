"""CLI entry point for the autonomous training monitor.

Usage:
    python -m src.monitor once [--config monitor.yaml]   # single cycle
    python -m src.monitor watch [--config monitor.yaml]  # blocking daemon
"""

from __future__ import annotations

import json
import logging
import sys
import time

import click

from .config import load_config
from .orchestrator import MonitorOrchestrator


@click.group()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=False),
    help="Path to monitor YAML config file (default: monitor.yaml).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, verbose: bool) -> None:
    """Autonomous training monitor for Axon."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


@cli.command()
@click.pass_context
def once(ctx: click.Context) -> None:
    """Run a single poll-analyze-decide cycle and print results as JSON."""
    config = load_config(ctx.obj["config_path"])
    orchestrator = MonitorOrchestrator(config)
    result = orchestrator.run_once()

    # Output as JSON for Claude Code /loop consumption
    output = result.model_dump_json(indent=2)
    click.echo(output)


@cli.command()
@click.pass_context
def watch(ctx: click.Context) -> None:
    """Run as a blocking daemon, polling on a configurable interval."""
    config = load_config(ctx.obj["config_path"])
    orchestrator = MonitorOrchestrator(config)

    logger = logging.getLogger(__name__)
    logger.info("Starting monitor daemon (Ctrl+C to stop)")

    try:
        while True:
            result = orchestrator.run_once()

            sleep_sec = result.suggested_sleep_seconds
            logger.info(
                "Cycle complete. %d runs, %d controls written. "
                "Sleeping %ds.",
                len(result.snapshots),
                result.controls_written,
                sleep_sec,
            )
            time.sleep(sleep_sec)
    except KeyboardInterrupt:
        logger.info("Monitor stopped.")
        sys.exit(0)


if __name__ == "__main__":
    cli()
