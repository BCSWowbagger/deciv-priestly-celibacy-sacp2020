"""Weighted celibacy charts and Datawrapper CSVs: 2020 survey plus 2002 toplines.

Install: py -3.14 -m pip install pandas numpy matplotlib pyreadstat
Run (short labels):
  py -3.14 priest_celibacy_analysis.py sacp2020-ngo-subsample.SAV sacp2020-ocd-subsample.SAV
Run (full labels):
  py -3.14 priest_celibacy_analysis.py sacp2020-ngo-subsample.SAV sacp2020-ocd-subsample.SAV --labels full
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WEIGHT = 'WEIGHT_ANEKSRAKE_COMBINED'
CELIBACY = 'Q55'
STACK_CODES = [2, 1, 4, 3]  # bottom to top
STACK_LABELS = {
    2: 'Do not waver in vow',
    1: 'Try, but do not always succeed',
    4: 'Ongoing journey',
    3: 'Do not observe celibacy',
}
FULL_LABELS = {
    2: 'Celibacy is not a problem for me and I do not waver in my vow',
    1: 'Celibacy is a discipline I try to follow, but do not always succeed',
    4: 'Celibacy takes time to achieve and I consider it an ongoing journey',
    3: 'Celibacy is not relevant to my priesthood and I do not observe it',
}
COLORS = {2: '#24744e', 1: '#eda64b', 4: '#4c87b7', 3: '#b54c56'}
TOPLINE_2002 = {2: 0.337, 1: 0.147, 4: 0.494, 3: 0.021}

GROUPS = {
    'ordination': {
        'column': 'ORDINATION_CATEGORIES',
        'title': 'Celibacy attitudes by ordination cohort',
        'categories': [
            ('Pre-1981', [5]), ('1981–1990', [1]), ('1991–2000', [2]),
            ('2001–2010', [3]), ('Post-2010', [4]),
        ],
    },
    'politics': {
        'column': 'Q51',
        'title': 'Celibacy attitudes by political orientation',
        'categories': [
            ('Very liberal', [6]), ('Somewhat liberal', [4]),
            ('Middle-of-the-road', [1]), ('Somewhat conservative', [3]),
            ('Very conservative', [5]),
        ],
    },
    'birth_control': {
        'column': 'Q39',
        'title': 'Celibacy attitudes by views on artificial birth control',
        'categories': [('Always a sin', [1]), ('Often', [4]),
                       ('Seldom', [5]), ('Never', [3])],
    },
    'sexual_orientation': {
        'column': 'Q54',
        'title': 'Celibacy attitudes by self-described sexual orientation (with leaners)',
        'categories': [
            ('Heterosexual', [2, 5]),
            ('Bisexual', [1]),
            ('Homosexual / bisexual', [3, 6]),
        ],
    },
    'age': {
        'column': 'AGE_CATEGORIES',
        'title': 'Celibacy attitudes by age group',
        'categories': [
            ('Under 30', [9]), ('30–39', [1]), ('40–49', [2]),
            ('50–59', [3]), ('60–69', [4]), ('70–79', [5]),
            ('80+', [6, 7]),
        ],
    },
}


def selected_labels(style):
    """Single source for every legend and export's response wording."""
    return FULL_LABELS if style == 'full' else STACK_LABELS


def load_files(paths):
    frames = []
    for path in paths:
        frame = pd.read_spss(path, convert_categoricals=False)
        frame.columns = [str(c).upper() for c in frame.columns]
        frames.append(frame)
        print(f'Loaded {Path(path).name}: {len(frame):,} respondents')
    data = pd.concat(frames, ignore_index=True)
    needed = {WEIGHT, CELIBACY} | {spec['column'] for spec in GROUPS.values()}
    missing = needed - set(data.columns)
    if missing:
        raise ValueError(f'Missing variable(s): {sorted(missing)}. Available: {list(data.columns)}')
    for col in needed:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    bad = ~np.isfinite(data[WEIGHT]) | (data[WEIGHT] <= 0)
    if bad.any():
        print(f'Excluding {bad.sum()} respondents with missing, nonpositive, or infinite weights.')
    data.loc[bad, WEIGHT] = np.nan
    return data


def make_response_table(values_by_code, labels, *, weighted_counts=None, unweighted_counts=None):
    """Align numeric codes, never display strings, to prevent mismatched comparisons."""
    vals = pd.Series(values_by_code, dtype=float).reindex(STACK_CODES, fill_value=0)
    result = pd.DataFrame({
        'Code': STACK_CODES,
        'Response': [labels[c] for c in STACK_CODES],
        'Full response wording': [FULL_LABELS[c] for c in STACK_CODES],
        'Weighted percent': vals.to_numpy(),
    })
    if weighted_counts is not None:
        result['Weighted count'] = pd.Series(weighted_counts).reindex(STACK_CODES, fill_value=0).to_numpy()
    if unweighted_counts is not None:
        result['Unweighted n'] = pd.Series(unweighted_counts).reindex(STACK_CODES, fill_value=0).to_numpy()
    return result


def pie_data(data, labels):
    valid = data[data[CELIBACY].isin(STACK_CODES) & data[WEIGHT].notna()]
    totals = valid.groupby(CELIBACY)[WEIGHT].sum().reindex(STACK_CODES, fill_value=0.)
    if totals.sum() <= 0:
        raise ValueError('No eligible 2020 respondents after filtering Q55 and weight.')
    percentages = totals / totals.sum() * 100
    counts = valid[CELIBACY].value_counts().reindex(STACK_CODES, fill_value=0)
    return make_response_table(percentages, labels, weighted_counts=totals,
                               unweighted_counts=counts), len(valid)


def topline_data(labels):
    values = pd.Series(TOPLINE_2002).reindex(STACK_CODES)
    return make_response_table(values / values.sum() * 100, labels)


def add_legend(ax, labels, *, full=False):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in STACK_CODES]
    ax.legend(handles, [labels[c] for c in STACK_CODES],
              loc='lower center', bbox_to_anchor=(0.5, -0.18),
              ncol=1 if full else 2, frameon=False, fontsize=10 if full else 9)


def plot_pie(table, title, output, labels, *, full=False):
    values = table.set_index('Code').reindex(STACK_CODES)['Weighted percent'].to_numpy()
    fig, ax = plt.subplots(figsize=(10 if full else 8, 7))
    nonzero = values > 0
    ax.pie(values[nonzero],
           colors=[COLORS[c] for c, show in zip(STACK_CODES, nonzero) if show],
           startangle=90, counterclock=False,
           autopct=lambda pct: f'{pct:.1f}%' if pct >= 2 else '',
           pctdistance=0.75, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    add_legend(ax, labels, full=full)
    ax.set_title(title, pad=16)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def cross_tab(data, spec, labels):
    mapping = {code: name for name, codes in spec['categories'] for code in codes}
    group_order = [name for name, _ in spec['categories']]
    subset = data[data[CELIBACY].isin(STACK_CODES) & data[WEIGHT].notna()].copy()
    subset['Group'] = subset[spec['column']].map(mapping)
    subset = subset.dropna(subset=['Group'])
    weights = subset.pivot_table(index='Group', columns=CELIBACY, values=WEIGHT,
                                 aggfunc='sum', fill_value=0).reindex(
                                     index=group_order, columns=STACK_CODES, fill_value=0)
    denom = weights.sum(axis=1)
    percentages = weights.div(denom.replace(0, np.nan), axis=0).mul(100).fillna(0)
    percentages.columns = [labels[c] for c in STACK_CODES]
    table = percentages.reset_index()
    counts = subset.groupby('Group').size().reindex(group_order, fill_value=0)
    details = pd.DataFrame({'Group': group_order, 'Unweighted n': counts.to_numpy(),
                            'Weighted denominator': denom.to_numpy()})
    if (counts == 0).any():
        print(f"WARNING: {spec['title']} has empty categories: {list(counts[counts == 0].index)}")
    return table, details


def plot_bars(table, detail, spec, output, labels, *, full=False):
    fig, ax = plt.subplots(figsize=(13 if full else 12, 8 if full else 7))
    x = np.arange(len(table))
    bottom = np.zeros(len(table))
    for code in STACK_CODES:
        name = labels[code]
        vals = table[name].to_numpy(dtype=float)
        bars = ax.bar(x, vals, bottom=bottom, label=name,
                      color=COLORS[code], width=.69, edgecolor='white', linewidth=.6)
        for rect, pct, base in zip(bars, vals, bottom):
            if pct >= 7:
                ax.text(rect.get_x() + rect.get_width()/2, base + pct/2,
                        f'{pct:.0f}%', ha='center', va='center', fontsize=9,
                        color='white', weight='bold')
        bottom += vals
    ax.set_xticks(x, table['Group'])
    ax.tick_params(axis='x', labelrotation=18)
    for tick in ax.get_xticklabels():
        tick.set_ha('right')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Weighted share of respondents (%)')
    ax.set_title(spec['title'] + '\n100% stacked bars · excludes missing answers', pad=15)
    ax.spines[['top', 'right']].set_visible(False)
    add_legend(ax, labels, full=full)
    for i, n in enumerate(detail['Unweighted n']):
        ax.text(i, 101, f'n={n}', ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def comparison_data(table_2020, table_2002, labels):
    """Join by stable numeric code, not short/full response strings."""
    a = table_2020.set_index('Code')['Weighted percent'].reindex(STACK_CODES)
    b = table_2002.set_index('Code')['Weighted percent'].reindex(STACK_CODES)
    if a.isna().any() or b.isna().any():
        raise ValueError('Missing response codes for 2002 vs. 2020 comparison.')
    result = pd.DataFrame({'Year': ['2002', '2020']})
    for code in STACK_CODES:
        result[labels[code]] = [b.loc[code], a.loc[code]]
    return result


def plot_year_comparison_stacked(comparison, output, labels, *, full=False):
    fig, ax = plt.subplots(figsize=(12 if full else 10, 8 if full else 6))
    x = np.arange(2)
    bottom = np.zeros(2)
    for code in STACK_CODES:
        vals = comparison[labels[code]].to_numpy(dtype=float)
        bars = ax.bar(x, vals, bottom=bottom, color=COLORS[code],
                      edgecolor='white', linewidth=.8, width=.62)
        for rect, pct, base in zip(bars, vals, bottom):
            if pct >= 6:
                ax.text(rect.get_x()+rect.get_width()/2, base+pct/2,
                        f'{pct:.0f}%', ha='center', va='center', fontsize=11,
                        color='white', weight='bold')
        bottom += vals
    ax.set_xticks(x, comparison['Year'])
    ax.set_ylim(0, 105)
    ax.set_ylabel('Weighted share of respondents (%)')
    ax.set_title('How priests describe the role of celibacy in their lives\n'
                 '2002 vs. 2020 · 100% stacked bars', pad=15)
    ax.spines[['top', 'right']].set_visible(False)
    add_legend(ax, labels, full=full)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sav_files', nargs=2, help='NGO and OCD subsamples as .sav files')
    parser.add_argument('--labels', choices=('short', 'full'), default='short',
                        help='Response labels for ALL charts and Datawrapper CSVs (default: short)')
    parser.add_argument('--output', default=None,
                        help='Override default folder (short: celibacy_outputs; full: celibacy_outputs_long)')
    args = parser.parse_args(argv)
    full = args.labels == 'full'
    labels = selected_labels(args.labels)
    out = Path(args.output or ('celibacy_outputs_long' if full else 'celibacy_outputs'))
    out.mkdir(parents=True, exist_ok=True)

    data = load_files(args.sav_files)
    pie_2020, n = pie_data(data, labels)
    pie_2002 = topline_data(labels)
    pie_2020.to_csv(out / '01_celibacy_pie_details.csv', index=False, float_format='%.6f')
    pie_2020[['Response', 'Weighted percent']].to_csv(
        out / '01_celibacy_pie_datawrapper.csv', index=False, float_format='%.6f')
    plot_pie(pie_2020, 'How priests describe the role of celibacy in their lives\n'
             f'2020 · weighted percentages · valid unweighted n={n:,}',
             out / '01_celibacy_pie.png', labels, full=full)

    pie_2002[['Response', 'Weighted percent']].to_csv(
        out / '01b_celibacy_pie_2002_datawrapper.csv', index=False, float_format='%.6f')
    plot_pie(pie_2002, 'How priests describe the role of celibacy in their lives\n'
             'L.A. Times 2002 · weighted topline percentages only',
             out / '01b_celibacy_pie_2002.png', labels, full=full)

    comparison = comparison_data(pie_2020, pie_2002, labels)
    comparison.to_csv(out / '01d_celibacy_2002_vs_2020_datawrapper.csv',
                      index=False, float_format='%.6f')
    plot_year_comparison_stacked(comparison,
        out / '01d_celibacy_2002_vs_2020_stacked.png', labels, full=full)

    for i, (name, spec) in enumerate(GROUPS.items(), 2):
        table, detail = cross_tab(data, spec, labels)
        stem = f'{i:02d}_{name}'
        table.to_csv(out / f'{stem}_datawrapper.csv', index=False, float_format='%.6f')
        detail.to_csv(out / f'{stem}_sample_sizes.csv', index=False, float_format='%.6f')
        plot_bars(table, detail, spec, out / f'{stem}.png', labels, full=full)
        print(f"{name}: valid n={detail['Unweighted n'].sum():,}")

    print(f'Finished ({args.labels} labels). Charts and CSVs saved to: {out.resolve()}')
    print('Note: percentages weighted; sample sizes unweighted; no design-based uncertainty estimated.')


if __name__ == '__main__':
    main()
