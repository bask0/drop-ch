"""Main entry point for tuning and cross validation."""

from project import CLI


def main():
    """
    Initialize the CLI and executes cross-validation.

    This function orchestrates the following steps:
    1. Instantiates the CLI interface.
    2. Sets up the tuning engine with all available options.
    3. Performs cross-validation on the tuned models.
    """

    cli = CLI()
    cli.tuning_engine.setup(which="xval")
    cli.tuning_engine.xval()


if __name__ == "__main__":
    main()
