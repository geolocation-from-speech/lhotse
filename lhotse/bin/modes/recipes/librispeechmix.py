from typing import Optional, Sequence, Union

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.librispeechmix import prepare_librispeechmix
from lhotse.utils import Pathlike


__all__ = ["librispeechmix"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-s",
    "--sets",
    multiple=True,
    default=["test-clean-2mix", "test-clean-3mix"],
    help="Specify which test sets to prepare, e.g., "
    "        lhoste prepare librispeechmix -s test-clean-2mix -s test-clean-3mix"
)
def librispeechmix(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    sets: Optional[Union[str, Sequence[str]]],
):
    prepare_librispeechmix(corpus_dir, output_dir=output_dir, sets=sets)

