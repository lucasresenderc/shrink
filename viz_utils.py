import numpy as np  # Used for dummy data generation, imported here for completeness
from typing import List, Dict, Any, Union
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from itertools import product
import seaborn as sns
from collections import OrderedDict

key_format_dict = {
    "n": "$N$",
    "shrinkage_function": "Shrinkage Function",
    "delta": r"$\delta$",
    "contamination_level": r"$\varepsilon$",
    "base_estimator": r"\widehat{\kappa}",
    "quantile": "Error",
    "split_ratio": "Split ratio",
    "best_m": r"$m^\star$",
}
base_estimator_dict = OrderedDict([
    ("mean", "$\overline{X}$"),
    ("median", "$M$"),
    ("tm", "TM"),
    ("mom", "MoM")
])
shrinkage_function_dict = OrderedDict([
    ("none", "None"),
    ("atm", r"$w(t)=\mathbf{1}_{t<1}$"),
    ("win", r"$w(t)=1\wedge t^{-1}$"),
    ("lv", r"$w(t)=(1-t^p)_+$"),
    ("exp", r"$w(t)=e^{-t^p}$"),
    ("inv", r"$w(t)=(1+t^p)^{-1}$"),
    ("ln", r"$w(t)=(\ln(e+t))^{-1}$"),
    ("ln_sq", r"$w(t)=(\ln(e+t^2))^{-1}$"),
    ("sqrt", r"$w(t)=1-\sqrt{1-(1-t)_+^2}$"),
    ("inv_sqrt", r"$w(t)=(1+\sqrt{t})^{-1}$"),
])
distribution_dict = OrderedDict([
    ("a=inf,r=0.0", "N"),
    ("a=inf,r=0.99", "SN"),
    ("a=1.005,r=0.0", "T"),
    ("a=1.005,r=0.99", "ST"),
])
split_ratio_dict = OrderedDict([
    ("none", "No split"),
])
contamination_dict = OrderedDict([])
symmetrized_dict = OrderedDict([
    (False, "asym"),
    (True, "sym"),
])
normalized_dict = OrderedDict([
    (False, "unnorm"),
    (True, "norm"),
])
a_distribution_dict = OrderedDict([
    (np.inf, "Light-tailed"),
    (1.005, "Heavy-tailed"),
])
r_distribution_dict = OrderedDict([
    (0.0, "Symmetric"),
    (0.99, "Skewed"),
])
formatter_dict = {
    "base_estimator": base_estimator_dict,
    "shrinkage_function": shrinkage_function_dict,
    "distribution": distribution_dict,
    "split_ratio": split_ratio_dict,
    "contamination_level": contamination_dict,
    "is_symmetrized": symmetrized_dict,
    "is_normalized": normalized_dict,
    "a": a_distribution_dict,
    "r": r_distribution_dict,
}


def format_val(key, val):
    if key in formatter_dict:
        try:
            return formatter_dict[key][val]
        except KeyError:
            return f"${val}$"
    else:
        if val == np.inf:
            val = "\\infty"
        return f"${key}={val}$"


def sort_key(key, val_list):
    if key in formatter_dict:
        key_list = list(formatter_dict[key].keys())

        def sort_func(v):
            return (key_list.index(
                v) if v in key_list else len(key_list), v)
        return sorted(val_list, key=sort_func)
    else:
        try:
            return sorted(val_list)
        except TypeError:
            return sorted(val_list, key=str)


def precision_formatter(x: float, n: int) -> str:
    """Formats a float to a string with specified decimal precision."""
    format_str = f"{{:.{n}f}}"
    return format_str.format(x)


def line_plot_df(
    df: pl.DataFrame,
    x_key: str,
    y_key: str,
    graph_key: Union[str, List[str]],
    row_subplot_key: Union[str, List[str]],
    col_subplot_key: Union[str, List[str]],
    log_x_scale: bool = False,
    log_y_scale: bool = False,
    marker=True,
    sharey: bool = False,
    yticks: bool = False,
    figsize: tuple = (6, 4),
    fontsize: int = 10,
) -> plt.Figure:
    """
    Generates a line plot with a grid of subplots (facets), leveraging
    Polars for data processing.

    Args:
        df: The input Polars DataFrame.
        x_key: The column name for the X-axis.
        y_key: The column name for the Y-axis.
        graph_key: The column name(s) used to color/group the individual lines (curves).
        row_subplot_key: The column name(s) used to create the subplot rows.
        col_subplot_key: The column name(s) used to create the subplot columns.

    Returns:
        A Matplotlib Figure object.
    """
    from itertools import product

    def ensure_list(keys):
        return keys if isinstance(keys, list) else [keys]

    row_keys_list = [sort_key(k, df.get_column(k).unique().to_list())
                     for k in ensure_list(row_subplot_key)]
    col_keys_list = [sort_key(k, df.get_column(k).unique().to_list())
                     for k in ensure_list(col_subplot_key)]
    graph_keys_list = [sort_key(k, df.get_column(
        k).unique().to_list()) for k in ensure_list(graph_key)]

    row_keys = list(product(*row_keys_list))
    col_keys = list(product(*col_keys_list))
    graph_keys = list(product(*graph_keys_list))

    n_rows = len(row_keys)
    n_cols = len(col_keys)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=figsize,
        sharex=True,
        sharey=False
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = np.expand_dims(
            axes, axis=0) if n_cols == 1 else np.expand_dims(axes, axis=1)

    linestyles = ['-', '--', '-.', ':']
    marker_styles = ['o', 's', 'x', '.', '^', 'v']

    graph_keys_to_keep = []
    for graph_val in graph_keys:
        graph_filter = pl.lit(True)
        for k, v in zip(ensure_list(graph_key), graph_val):
            graph_filter &= (pl.col(k) == v)
        graph_df = df.filter(graph_filter)
        if graph_df.height > 0:
            graph_keys_to_keep.append(graph_val)
    graph_keys = graph_keys_to_keep
    colors = sns.color_palette()

    style_map = {
        key: {
            'color': colors[i],
            'linestyle': linestyles[i % len(linestyles)],
            'marker': marker_styles[i // len(linestyles) % len(marker_styles)] if marker else None
        }
        for i, key in enumerate(graph_keys)
    }

    for i, row_val in enumerate(row_keys):
        for j, col_val in enumerate(col_keys):
            ax = axes[i, j]

            # Build filter for current facet
            facet_filter = pl.lit(True)
            for k, v in zip(ensure_list(row_subplot_key), row_val):
                facet_filter &= (pl.col(k) == v)
            for k, v in zip(ensure_list(col_subplot_key), col_val):
                facet_filter &= (pl.col(k) == v)

            facet_df = df.filter(facet_filter).sort(x_key)
            for graph_val in graph_keys:
                graph_filter = pl.lit(True)
                for k, v in zip(ensure_list(graph_key), graph_val):
                    graph_filter &= (pl.col(k) == v)
                graph_df = facet_df.filter(graph_filter)
                if graph_df.height == 0:
                    continue
                label = ', '.join([format_val(k, v) for k, v in zip(
                    ensure_list(graph_key), graph_val) if format_val(k, v) != "None"])
                sns.lineplot(data=graph_df, x=x_key, y=y_key, ax=ax,
                             lw=1.5, label=label, **style_map[graph_val])

            if i == 0:
                title = ', '.join([format_val(k, v) for k, v in zip(
                    ensure_list(col_subplot_key), col_val)])
                ax.set_title(title)
            else:
                ax.set_title('')

            if j == 0:
                ylabel = key_format_dict[y_key]
                ax.set_ylabel(ylabel)
            elif j == n_cols-1:
                ylabel = ', '.join([format_val(k, v) for k, v in zip(
                    ensure_list(row_subplot_key), row_val)])
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel('')

            if i == n_rows - 1:
                ax.set_xlabel(key_format_dict[x_key])
            else:
                ax.set_xlabel('')

            if not yticks or (sharey and j > 0):
                ax.tick_params(labelleft=False)

            if i < n_rows - 1:
                ax.tick_params(labelbottom=False)

            if log_x_scale:
                ax.set_xscale('log')
            if log_y_scale:
                ax.set_yscale('log')
                if not yticks:
                    ax.set_yticks([], minor=True)

    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes.flatten():
        ax.get_legend().remove()

    fig.legend(
        handles,
        labels,
        # title=', '.join(ensure_list(graph_key)).replace('_', ' ').title(),
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        ncol=len(graph_keys)
    )

    # fig.suptitle(
    #     f"Faceted Line Plot: {key_format_dict[y_key]} vs {key_format_dict[x_key]}",
    #     fontsize=14, y=1.00
    # )

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])

    return fig


def polars_to_latex_table(
    df: pl.DataFrame,
    col_keys: List[str],
    row_keys: List[str],
    value_key: str,
    highlight_conditions: int = None,
    colsep: str = "1.5pt",
    reference_row_val=None,
    float_format: int = 2
) -> str:
    """
    Generates a LaTeX table with nested multicolumn headers (up to 3 levels)
    and nested row groups (up to 2 levels) from a Polars DataFrame.

    Args:
        df: The input Polars DataFrame, expected to be in long format.
        col_keys: List of column keys (max 3) for nested headers (e.g., ["n", "delta", "p"]).
        row_keys: List of row keys (max 2) for row groups (e.g., ["base_estimator", "shrinkage_function"]).
        value_key: The column containing the numerical/string values for the table entries.
        highlight_conditions: A dictionary mapping the value column name to a comparison function
                              for conditional highlighting (e.g., {"metric_value": lambda x: x < -70}).
                              If None, no highlighting is performed.

    Returns:
        A string containing the full LaTeX table environment.
    """
    if len(row_keys) > 2 or len(col_keys) > 3:
        return "Error: Function supports at most 2 row_keys and 3 col_keys."

    # 1. --- Data Preparation and Pivoting using Polars ---

    required_cols = row_keys + col_keys + [value_key]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in DataFrame.")

    cols_per_key = [len(df[col].unique())
                    for col in col_keys]
    total_multicols = np.prod(cols_per_key[:-1])

    table_str = ""
    table_str += "\\begin{table}[ht]\n\\centering\n"
    table_str += f"\\setlength{{\\tabcolsep}}{{{colsep}}}\n"
    table_str += "\\begin{tabular*}{0.95\\textwidth}{l@{\extracolsep{\\fill}}"
    table_str += total_multicols*f"|{'c'*cols_per_key[-1]}"
    table_str += "}\n"
    for i, col in enumerate(col_keys):
        col_key_str = ""
        sorted_unique_vals = sort_key(col,
                                      df.get_column(col).unique().to_list())
        if i < len(col_keys) - 1:
            multicol_span = np.prod(cols_per_key[i+1:])
            for val in sorted_unique_vals[:-1]:
                col_key_str += f"& \\multicolumn{{{multicol_span}}}{{c|}}{{{format_val(col,val)}}}"
            col_key_str += f"& \\multicolumn{{{multicol_span}}}{{c}}{{{format_val(col,sorted_unique_vals[-1])}}}"
        else:
            for val in sorted_unique_vals:
                col_key_str += f"& {format_val(col,val)}"
        table_str += np.prod(cols_per_key[:i]
                             ).astype(np.int64) * col_key_str + "\n"
        table_str += "\\\\\n"
    table_str += "\\hline\n"

    row_key_prod = list(product(
        *[sort_key(key, df[key].unique().to_list()) for key in row_keys]
    ))
    col_key_prod = list(product(
        *[sort_key(key, df[key].unique().to_list()) for key in col_keys]
    ))
    if highlight_conditions is not None:
        best_dict = {}
        for col_combo in col_key_prod:
            filter = pl.lit(True)
            for i in range(len(col_keys)):
                filter &= pl.col(col_keys[i]) == col_combo[i]
            filtered_df = df.filter(filter)
            filtered_df = filtered_df.sort(value_key, descending=highlight_conditions < 0)[
                :abs(highlight_conditions)]
            structs = filtered_df.select(pl.struct(row_keys).alias("row_keys"))[
                "row_keys"].to_list()
            tuples = [tuple(tup.values()) for tup in structs]
            best_dict[col_combo] = tuples

    old_row_val1 = None
    for row_combo in row_key_prod:
        row_val1 = row_combo[0]
        row_val2 = row_combo[1] if len(row_keys) > 1 else None
        if row_val2 is not None and (row_val1 != old_row_val1 and old_row_val1 is not None):
            table_str += "\\hline\n"
        row_label = " ".join(
            [format_val(row_keys[i], row_combo[i])
             for i in range(len(row_keys))]
        )
        table_str += f"{row_label}"
        for col_combo in col_key_prod:
            row_filter_condition = pl.lit(True)
            for i, key in enumerate(row_keys):
                row_filter_condition &= (pl.col(key) == row_combo[i])
            col_filter_condition = pl.lit(True)
            for i, key in enumerate(col_keys):
                col_filter_condition &= (pl.col(key) == col_combo[i])

            cell_df = df.filter(row_filter_condition & col_filter_condition)
            assert cell_df.height == 1, "DataFrame has more than one matching row for the given keys."
            if reference_row_val is None:
                cell_value = precision_formatter(
                    cell_df[value_key][0], float_format)
            else:
                reference_condition = pl.lit(True)
                for i, key in enumerate(row_keys):
                    if reference_row_val[i] is not None:
                        reference_val = reference_row_val[i]
                    else:
                        reference_val = row_combo[i]
                    reference_condition &= (
                        pl.col(key) == reference_val)
                reference_df = df.filter(
                    reference_condition & col_filter_condition)
                reference_value = reference_df[value_key][0]
                # relative difference percentage
                cell_value = (cell_df[value_key][0] -
                              reference_value) / abs(reference_value) * 100
                cell_value = precision_formatter(cell_value, float_format)
            if highlight_conditions is not None and best_dict.get(col_combo) and row_combo in best_dict[col_combo]:
                cell_str = rf"\(\mathbf{{{cell_value}}}\)"
            else:
                cell_str = rf"\({cell_value}\)"

            table_str += f" & {cell_str}"

        table_str += " \\\\\n"
        old_row_val1 = row_val1
    table_str += "\\hline\n"
    table_str += "\\end{tabular*}\n"
    table_str += "\\end{table}\n"
    return table_str
