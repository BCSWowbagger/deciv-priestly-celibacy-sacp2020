"""Weighted charts and Datawrapper CSV exports for 2020 American Catholic Priests survey.

Install: py -3.14 -m pip install pandas numpy matplotlib pyreadstat
Run: py -3.14 priest_celibacy_analysis.py sacp2020-ngo-subsample.SAV sacp2020-ocd-subsample.SAV
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WEIGHT = 'WEIGHT_ANEKSRAKE_COMBINED'
CELIBACY = 'Q55'
STACK_CODES = [2, 1, 4, 3]  # bottom to top, by substantive meaning
STACK_LABELS = {
    2: 'Do not waver in vow',
    1: 'Try, but do not always succeed',
    4: 'Ongoing journey',
    3: 'Do not observe celibacy',
}
FULL_LABELS = {
    1: 'Celibacy is a discipline I try to follow, but do not always succeed',
    2: 'Celibacy is not a problem for me and I do not waver in my vow',
    3: 'Celibacy is not relevant to my priesthood and I do not observe it',
    4: 'Celibacy takes time to achieve and I consider it an ongoing journey',
}
COLORS = {2: '#24744e', 1: '#eda64b', 4: '#4c87b7', 3: '#b54c56'}

# Question codes verified against ARDA codebooks. Numeric codes are not rank orders.
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
            ('Under 30', [9]),
            ('30–39', [1]),
            ('40–49', [2]),
            ('50–59', [3]),
            ('60–69', [4]),
            ('70–79', [5]),
            ('80+', [6, 7]),
        ],
    }
}


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
        raise ValueError(f'Missing variable(s): {sorted(missing)}. Available columns: {list(data.columns)}')
    for col in needed:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    bad_weights = data[WEIGHT].notna() & (~np.isfinite(data[WEIGHT]) | (data[WEIGHT] <= 0))
    if bad_weights.any():
        print(f'WARNING: {bad_weights.sum()} records with nonpositive/infinite weights excluded.')
    data.loc[~np.isfinite(data[WEIGHT]) | (data[WEIGHT] <= 0), WEIGHT] = np.nan
    return data


def pie_data(data):
    valid = data[data[CELIBACY].isin(STACK_CODES) & data[WEIGHT].notna()].copy()
    totals = valid.groupby(CELIBACY)[WEIGHT].sum().reindex(STACK_CODES, fill_value=0.0)
    percentages = 100 * totals / totals.sum() if totals.sum() > 0 else totals
    table = pd.DataFrame({
        'Response': [STACK_LABELS[c] for c in STACK_CODES],
        'Full response wording': [FULL_LABELS[c] for c in STACK_CODES],
        'Weighted percent': percentages.to_numpy(),
        'Weighted count': totals.to_numpy(),
        'Unweighted n': [int((valid[CELIBACY] == c).sum()) for c in STACK_CODES],
    })
    return table, len(valid)


def plot_pie(table, n, output):
    values = table['Weighted percent'].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 7))
    nonzero = values > 0
    ax.pie(values[nonzero], colors=[COLORS[c] for c, show in zip(STACK_CODES, nonzero) if show],
           startangle=90, counterclock=False,
           autopct=lambda pct: f'{pct:.1f}%' if pct >= 2 else '',
           pctdistance=0.75, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in STACK_CODES],
              [STACK_LABELS[c] for c in STACK_CODES], loc='lower center',
              bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
    ax.set_title('How priests describe the role of celibacy in their lives\n'
                 f'Weighted percentages · valid unweighted n={n:,}', pad=18)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def cross_tab(data, spec):
    # Category lookup also makes missing / NA responses automatically drop out.
    mapping = {code: label for label, codes in spec['categories'] for code in codes}
    labels = [label for label, codes in spec['categories']]
    col = spec['column']
    subset = data[data[CELIBACY].isin(STACK_CODES) & data[WEIGHT].notna()].copy()
    subset['Group'] = subset[col].map(mapping)
    subset = subset.dropna(subset=['Group'])
    weights = subset.pivot_table(index='Group', columns=CELIBACY, values=WEIGHT,
                                 aggfunc='sum', fill_value=0).reindex(index=labels, columns=STACK_CODES, fill_value=0)
    denominators = weights.sum(axis=1)
    percents = weights.div(denominators.replace(0, np.nan), axis=0).mul(100)
    percents = percents.fillna(0)
    percents.columns = [STACK_LABELS[c] for c in STACK_CODES]
    table = percents.reset_index()
    # Metadata separate from Datawrapper chart data, so all plotted columns are percentages.
    counts = subset.groupby('Group').size().reindex(labels, fill_value=0)
    detail = pd.DataFrame({'Group': labels, 'Unweighted n': counts.to_numpy(),
                           'Weighted denominator': denominators.to_numpy()})
    if (counts == 0).any():
        print(f"WARNING: {spec['title']} has empty categories: {list(counts[counts == 0].index)}")
    return table, detail


def plot_bars(table, detail, spec, output):
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(table))
    bottom = np.zeros(len(table))
    for code in STACK_CODES:
        label = STACK_LABELS[code]
        val = table[label].to_numpy(dtype=float)
        bars = ax.bar(x, val, bottom=bottom, label=label,
                      color=COLORS[code], width=0.69, edgecolor='white', linewidth=.6)
        for rect, pct, base in zip(bars, val, bottom):
            if pct >= 7:
                ax.text(rect.get_x() + rect.get_width()/2, base + pct/2,
                        f'{pct:.0f}%', ha='center', va='center', fontsize=9, color='white', weight='bold')
        bottom += val
    ax.set_xticks(x, table['Group'])
    ax.tick_params(axis='x', labelrotation=18)
    for t in ax.get_xticklabels():
        t.set_ha('right')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Weighted share of respondents (%)')
    ax.set_title(spec['title'] + '\n100% stacked bars · excludes missing answers', pad=15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, fontsize=9)
    for i, n in enumerate(detail['Unweighted n']):
        ax.text(i, 101, f'n={n}', ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sav_files', nargs='+', help='Both NGO and OCD SPSS .sav files')
    parser.add_argument('--output', default='celibacy_outputs', help='Output folder')
    args = parser.parse_args()
    if len(args.sav_files) != 2:
        parser.error('Provide exactly two .sav files (NGO and OCD subsamples).')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = load_files(args.sav_files)
    pie, n = pie_data(data)
    if n == 0:
        sys.exit('No eligible respondents; inspect variable coding and weights.')
    pie.to_csv(out / '01_celibacy_pie_details.csv', index=False, float_format='%.6f')
    pie[['Response', 'Weighted percent']].to_csv(out / '01_celibacy_pie_datawrapper.csv',
                                                index=False, float_format='%.6f')
    plot_pie(pie, n, out / '01_celibacy_pie.png')
    print(f'Pie chart: {n:,} valid unweighted observations')
    for i, (name, spec) in enumerate(GROUPS.items(), 2):
        table, detail = cross_tab(data, spec)
        stem = f'{i:02d}_{name}'
        table.to_csv(out / f'{stem}_datawrapper.csv', index=False, float_format='%.6f')
        detail.to_csv(out / f'{stem}_sample_sizes.csv', index=False, float_format='%.6f')
        plot_bars(table, detail, spec, out / f'{stem}.png')
        print(f"{name}: valid n={detail['Unweighted n'].sum():,} (across all groups)")
    print(f'Finished. Charts and CSVs saved to: {out.resolve()}')
    print('Note: percentages are weighted; sample sizes are unweighted. No design-based uncertainty estimated.')


if __name__ == '__main__':
    main()
