# Licensed under a 3-clause BSD style license - see LICENSE.rst
files = dict(
    model_spec='{{src.model}}/{{src.model}}',
    out_dir='{{src.model}}/{{src.outdir}}',
    out_values='{{src.model}}/{{src.outdir}}/values',
    out_states='{{src.model}}/{{src.outdir}}/states',
    fit_dir='{{src.model}}/{{src.outdir}}',
    index='{{src.model}}/{{src.outdir}}/index',
    fit_spec='{{src.model}}/{{src.outdir}}/{{src.model}}',
    fit_log='{{src.model}}/{{src.outdir}}/log',
    fit_resid='{{src.model}}/{{src.outdir}}/resid_{{src.msid}}_{{src.date_range}}',
    fit_hist='{{src.model}}/{{src.outdir}}/hist_{{src.select}}_{{src.msid}}_{{src.date_range}}',
    fit_data='{{src.model}}/{{src.outdir}}/data_{{src.date_range}}',
    fit_fit='{{src.model}}/{{src.outdir}}/fit_{{src.date_range}}',
    fit_quants=(
        '{{src.model}}/{{src.outdir}}/quantiles_{{src.select}}_{{src.date_range}}'
    ),
)
