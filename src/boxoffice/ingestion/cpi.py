"""Indice des prix à la consommation américain (FRED CPIAUCSL) pour les dollars constants."""
import argparse
import io
import logging
import os

import pandas as pd
import requests

from boxoffice import config

log = logging.getLogger(__name__)


def download(force: bool = False) -> None:
    if config.CPI_RAW.exists() and not force:
        return
    log.info("Téléchargement de %s", config.CPI_URL)
    resp = requests.get(config.CPI_URL, timeout=60)
    resp.raise_for_status()
    config.CPI_RAW.parent.mkdir(parents=True, exist_ok=True)
    part = config.CPI_RAW.with_suffix(".part")
    part.write_text(resp.text, encoding="utf-8")
    os.replace(part, config.CPI_RAW)


def yearly_index() -> pd.Series:
    download()
    table = pd.read_csv(config.CPI_RAW)
    table.columns = ["date", "cpi"]
    table["annee"] = pd.to_datetime(table["date"]).dt.year
    return table.groupby("annee")["cpi"].mean()


def factors(reference_year: int = config.CPI_REFERENCE_YEAR) -> pd.Series:
    index = yearly_index()
    return (index[reference_year] / index).rename("facteur")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Indice des prix pour les dollars constants.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    download(force=args.force)
    f = factors()
    print(f"dollars constants {config.CPI_REFERENCE_YEAR} : {f.index.min()}-{f.index.max()}")
    print(f.loc[[1995, 2005, 2015, 2023]].round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
