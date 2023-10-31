# =====================================================================================
#  C O P Y R I G H T
# -------------------------------------------------------------------------------------
#  Copyright (c) 2023 by Robert Bosch GmbH. All rights reserved.
#
#  Author(s):
#  - Markus Braun, :em engineering methods AG (contracted by Robert Bosch GmbH)
#  - Celina Adelhardt, :em engineering methods AG (contracted by Robert Bosch GmbH)
#  - Gergely Meszaros, Stream HPC B.V. (contracted by Advanced Micro Devices Inc.)
# =====================================================================================

# noqa: D301
"""
Entry module for the doxysphinx cli.

Defines click main command (:func:`cli`) and subcommands (:func:`build`), (:func:`clean`)

.. note::

    * Execute this script directly to start doxysphinx.

    * If you need to call a function to start doxysphinx (e.g. for vscode launch config etc.) use the
      :func:`cli` directly.

        Sphinx autodoc which created this documentation seems to have problems with decorated methods.
        The function signatures shown here in the documentation aren't correct. Just click on view source to
        see the correct signatures.
"""

import importlib.metadata as metadata
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional

import click
import click_log  # type: ignore

from doxysphinx.doxygen import (
    DoxygenOutputPathValidator,
    DoxygenSettingsValidator,
    read_doxyconfig,
)
from doxysphinx.process import Builder, Cleaner
from doxysphinx.utils.contexts import TimedContext

_logger = logging.getLogger()
click_log.basic_config(_logger)


@dataclass
class DoxygenContext:
    """
    Represent the options for doxygen that can be set via the cli.

    The doxygen projects are specified through INPUT (multiple possible). INPUT can be:

    * a doxygen configuration file (aka doxyfile)

    * a directory, which contains the generated doxygen html documentation.
      Note that specifying a directory will skip the config validation completely and is therefore considered
      "advanced stuff". You will typically want to use that if you're integrating doxysphinx in a ci build
      system. If unsure, use a doxyfile.
    """

    input: List[Path]
    doxygen_exe: str
    doxygen_cwd: Path


def _doxygen_context():
    def _(function):
        function = click.argument(
            "input", nargs=-1, type=click.Path(dir_okay=True, file_okay=True, exists=True, path_type=Path)
        )(function)
        function = click.option(
            "--doxygen_exe", type=str, default="doxygen", help="Name of the doxygen executable, default is 'doxygen'."
        )(function)
        function = click.option(
            "--doxygen_cwd",
            type=click.Path(file_okay=False, exists=True, dir_okay=True),
            default=Path.cwd(),
            help="Working directory in case another doxygen is used"
            "(because paths inside the doxyfile are relative to the directory from which doxygen is executed).",
        )(function)
        return function

    return _


@click.group()
@click.version_option()
@click_log.simple_verbosity_option(_logger)
def cli():
    """Integrates doxygen html documentation with sphinx.

    Doxysphinx typically should run right after doxygen. It will generate rst files out of doxygen's html
    files. This has the implication, that the doxygen html output directory (where the rst files are generated
    to) has to live inside sphinx's input tree.
    """
    click.secho(f"doxysphinx v{metadata.version('doxysphinx')}", fg="bright_white")


@cli.command()
@click.option(
    "--tagfile_toc/--no_tagfile_toc",
    type=bool,
    default=False,
    help="Parse the doxygen tagfile to create a more accurate table of contents for sphinx, off by default.",
)
@click.option(
    "--parallel/--sequential",
    default=True,
    help="parallel will separate the work among all available processor cores where possible while sequential "
    "won't. The default is 'parallel' - 'sequential' might come in handy when debugging or tracing problems "
    "because of the linear output.",
)
@click.argument("sphinx_source", type=click.Path(file_okay=False, exists=True, path_type=Path))
@click.argument("sphinx_output", type=click.Path(file_okay=False, path_type=Path))
@_doxygen_context()
def build(tagfile_toc: bool, parallel: bool, sphinx_source: Path, sphinx_output: Path, **kwargs):
    """
    Build rst and copy related files for doxygen projects.

    SPHINX_SOURCE specifies the root of the sphinx source directory tree while SPHINX_OUTPUT specifies the root of the
    sphinx output directory tree.
    \f

    .. warning::

       * when using ``sphinx-build -b html SOURCE_DIR OUTPUT_DIR ...`` the html output will be put to ``OUTPUT_DIR`` so
         so doxysphinx's ``SPHINX_OUTPUT`` should be ``OUTPUT_DIR``.
       * when using ``sphinx-build -M html`` the html output will be put to ``OUTPUT_DIR/html`` so doxysphinx's
         ``SPHINX_OUTPUT`` should be ``OUTPUT_DIR/html``.
    """
    doxy_context = DoxygenContext(**kwargs)
    _logger.info("starting build command...")
    with TimedContext() as timed_scope:
        builder = Builder(sphinx_source, sphinx_output, enable_tagfile_toc=tagfile_toc, parallel=parallel)
        for params in _get_doxygen_params(doxy_context, sphinx_source):
            builder.build(params.outdir, params.tagfile)
    _logger.info(f"build command done in {timed_scope.elapsed_humanized()} ({timed_scope.elapsed()}).")


@cli.command()
@click.option(
    "--parallel/--sequential",
    default=True,
    help="parallel will separate the work among all available processor cores where possible while sequential "
    "won't. The default is 'parallel' - 'sequential' might come in handy when debugging or tracing problems "
    "because of the linear output.",
)
@click.argument("sphinx_source", type=click.Path(file_okay=False, exists=True, path_type=Path))
@click.argument("sphinx_output", type=click.Path(file_okay=False, path_type=Path))
@_doxygen_context()
def clean(parallel: bool, sphinx_source: Path, sphinx_output: Path, **kwargs):
    r"""
    Clean up files created by doxysphinx.

    SPHINX_SOURCE specifies the root of the sphinx source directory tree while SPHINX_OUTPUT specifies the root of the
    sphinx output directory tree. The doxygen html outputs are specified through INPUT (multiple possible) either
    by pointing to the doxygen html output directory or by pointing to the doxygen config file (doxyfile).
    """
    doxy_context = DoxygenContext(**kwargs)
    _logger.info("starting clean command...")
    with TimedContext() as tc:
        cleaner = Cleaner(sphinx_source, sphinx_output, parallel=parallel)
        for params in _get_doxygen_params(doxy_context, sphinx_source):
            cleaner.cleanup(params.outdir)
    _logger.info(f"clean command done in {tc.elapsed_humanized()}.")


class DoxygenParams(NamedTuple):
    """Doxygen Parameters read from doxyfile or passed on the command line."""

    outdir: Path
    tagfile: Optional[Path]


def _get_doxygen_params(doxy_context: DoxygenContext, sphinx_source: Path) -> Iterator[DoxygenParams]:
    for i in doxy_context.input:
        if i.is_dir():
            yield _get_params_via_doxyoutputdir(i)
        else:
            yield _get_params_via_doxyfile(i, sphinx_source, doxy_context)


def _get_params_via_doxyfile(doxyfile: Path, sphinx_source: Path, doxy_context: DoxygenContext) -> DoxygenParams:
    config = read_doxyconfig(doxyfile, doxy_context.doxygen_exe, doxy_context.doxygen_cwd)

    validator = DoxygenSettingsValidator()
    if not validator.validate(config, sphinx_source, doxy_context.doxygen_cwd):
        if any(item for item in validator.validation_errors if not item.startswith("Hint:")):
            message = validator.validation_msg
            raise click.UsageError(
                f'The doxygen settings defined in "{doxyfile}"'
                f"do not match the mandatory settings necessary for doxysphinx:\n"
                f"{message}"
            )
        logging.warning("Not all optional doxygen settings are set correctly:\n")
        logging.warning(f"{validator.validation_msg}")

    return DoxygenParams(validator.absolute_out, validator.tagfile)


def _get_params_via_doxyoutputdir(doxygen_html_output_dir: Path) -> DoxygenParams:
    validator = DoxygenOutputPathValidator()
    if not validator.validate(doxygen_html_output_dir):
        raise click.UsageError(validator.validation_msg)

    return DoxygenParams(doxygen_html_output_dir, None)


if __name__ == "__main__":
    cli()
