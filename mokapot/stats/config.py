import argparse

# todo: change to pydantic based configuration


def add_config_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tdc",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Specifies whether input comes from target decoy competition "
            "(default) or from separate search."
        ),
    )

    parser.add_argument(
        "--peps_error",
        default=False,
        action="store_true",
        help="Raise error when all PEPs values are equal to 1.",
    )

    parser.add_argument(
        "--peps_algorithm",
        default="qvality",
        choices=["qvality", "qvality_bin", "kde_nnls", "hist_nnls"],
        help=(
            "Specify the algorithm for pep computation. 'qvality_bin' works "
            "only if the qvality binary is on the search path"
        ),
    )

    parser.add_argument(
        "--pi0_algorithm",
        default="default",
        choices=[
            "default",
            "ratio",
            "slope",
            "fixed",
            "storey_smoother",
            "storey_fixed",
            "storey_bootstrap",
        ],
        help=("Specify the algorithm for pi0 estimation. "),
    )

    parser.add_argument(
        "--pi0_eval_lambda",
        type=float,
        default=0.5,
        help=(
            "Specify the lambda in Storey's pi0 estimation for evaluation "
            "(works currently only with storey_* pi0 algorithms."
        ),
    )

    parser.add_argument(
        "--pi0_value",
        type=float,
        default=1.0,
        help=(
            "Specify a fixed value for pi0. This has only an effect if the "
            "pi0_algorithm is set to 'fixed'"
        ),
    )

    parser.add_argument(
        "--qvalue_algorithm",
        default="default",
        choices=["default", "tdc", "from_counts", "storey"],
        help=(
            "Specify the algorithm for qvalue computation. If the `tdc` option"
            "is set to true (which is the default0 `default` evals to `tdc`, "
            "the original mokapot algorithm, which works only with tdc. "
            "Otherwise, it defaults to `storey`."
        ),
    )
