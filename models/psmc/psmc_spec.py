{
    "comps": [
        {
            "class_name": "Node", 
            "init_args": [
                "1pin1at"
            ], 
            "init_kwargs": {
                "sigma": 1000.0
            }, 
            "name": "1pin1at"
        }, 
        {
            "class_name": "Node", 
            "init_args": [
                "1pdeaat"
            ], 
            "init_kwargs": {
                "quant": 2.5
                }, 
            "name": "1pdeaat"
        }, 
        {
            "class_name": "Pitch", 
            "init_args": [], 
            "init_kwargs": {}, 
            "name": "pitch"
        }, 
        {
            "class_name": "SimZ", 
            "init_args": [], 
            "init_kwargs": {}, 
            "name": "sim_z"
        }, 
        {
            "class_name": "Coupling", 
            "init_args": [
                "1pin1at", 
                "1pdeaat"
            ], 
            "init_kwargs": {
                "tau": 13.561590344337947
            }, 
            "name": "coupling__1pin1at__1pdeaat"
        }, 
        {
            "class_name": "Coupling", 
            "init_args": [
                "1pdeaat", 
                "1pin1at"
            ], 
            "init_kwargs": {
                "tau": 1.3444562773636255
            }, 
            "name": "coupling__1pdeaat__1pin1at"
        }, 
        {
            "class_name": "AcisPsmcSolarHeat", 
            "init_args": [
                "1pin1at", 
                "pitch", 
                "sim_z"
            ], 
            "init_kwargs": {
                "P_pitches": [
                    45, 
                    70, 
                    90, 
                    130, 
                    170
                ], 
                "P_vals": [
                    [
                        1.62, 
                        1.5975, 
                        1.5806, 
                        1.75, 
                        1.96259
                    ], 
                    [
                        2.029, 
                        2.029, 
                        1.4774, 
                        1.6, 
                        1.73676
                    ], 
                    [
                        2.8, 
                        2.55407, 
                        1.42066, 
                        1.45, 
                        1.476175
                    ]
                ]
            }, 
            "name": "psmc_solarheat__1pin1at"
        }, 
        {
            "class_name": "HeatSink", 
            "init_args": [
                "1pin1at"
            ], 
            "init_kwargs": {
                "T": -36.35223330470379, 
                "tau": 18.987574552683895
            }, 
            "name": "heatsink__1pin1at"
        }, 
        {
            "class_name": "CmdStatesData", 
            "init_args": [
                "fep_count"
            ], 
            "init_kwargs": {}, 
            "name": "fep_count"
        }, 
        {
            "class_name": "CmdStatesData", 
            "init_args": [
                "ccd_count"
            ], 
            "init_kwargs": {}, 
            "name": "ccd_count"
        }, 
        {
            "class_name": "CmdStatesData", 
            "init_args": [
                "vid_board"
            ], 
            "init_kwargs": {}, 
            "name": "vid_board"
        }, 
        {
            "class_name": "CmdStatesData", 
            "init_args": [
                "clocking"
            ], 
            "init_kwargs": {}, 
            "name": "clocking"
        }, 
        {
            "class_name": "AcisDpaStatePower", 
            "init_args": [
                "1pdeaat"
            ], 
            "init_kwargs": {
                "ccd_count": "ccd_count", 
                "clocking": "clocking", 
                "fep_count": "fep_count", 
                "vid_board": "vid_board"
            }, 
            "name": "dpa_power"
        }
    ], 
    "datestart": "2012:030:12:08:25.816", 
    "datestop": "2012:330:11:22:32.816", 
    "dt": 328.0, 
    "gui_config": {
        "filename": "/data/baffin/tom/git/xija/psmc_classic4.json", 
        "plot_names": [
            "1pdeaat data__time", 
            "1pdeaat resid__data"
        ], 
        "set_data_vals": {}, 
        "size": [
            1838, 
            800
        ]
    }, 
    "mval_names": [], 
    "name": "psmc", 
    "pars": [
        {
            "comp_name": "coupling__1pin1at__1pdeaat", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "coupling__1pin1at__1pdeaat__tau", 
            "max": 200.0, 
            "min": 2.0, 
            "name": "tau", 
            "val": 32.0152160862839
        }, 
        {
            "comp_name": "coupling__1pdeaat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "coupling__1pdeaat__1pin1at__tau", 
            "max": 200.0, 
            "min": 2.0, 
            "name": "tau", 
            "val": 14.377668666116021
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_hrcs_45", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_hrcs_45", 
            "val": 1.7136060000024824
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_hrcs_70", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_hrcs_70", 
            "val": 1.9948552742376484
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_hrcs_90", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_hrcs_90", 
            "val": 1.5446270599269409
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_hrcs_130", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_hrcs_130", 
            "val": 1.9821095636143924
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_hrcs_170", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_hrcs_170", 
            "val": 2.3977434061546274
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_hrci_45", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_hrci_45", 
            "val": 2.0385169184093135
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_hrci_70", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_hrci_70", 
            "val": 2.0174710238006077
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_hrci_90", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_hrci_90", 
            "val": 1.7061881661848459
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_hrci_130", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_hrci_130", 
            "val": 1.896790204570264
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_hrci_170", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_hrci_170", 
            "val": 2.130214393547039
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_acis_45", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_acis_45", 
            "val": 3.2849231687502121
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_acis_70", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_acis_70", 
            "val": 2.5876295451837974
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_acis_90", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_acis_90", 
            "val": 1.491357457413395
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_acis_130", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_acis_130", 
            "val": 1.6681537529578345
        }, 
        {
            "comp_name": "psmc_solarheat__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "psmc_solarheat__1pin1at__P_acis_170", 
            "max": 10.0, 
            "min": -10.0, 
            "name": "P_acis_170", 
            "val": 1.7155839829349957
        }, 
        {
            "comp_name": "heatsink__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "heatsink__1pin1at__T", 
            "max": 100.0, 
            "min": -100.0, 
            "name": "T", 
            "val": -24.306219637409995
        }, 
        {
            "comp_name": "heatsink__1pin1at", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "heatsink__1pin1at__tau", 
            "max": 200.0, 
            "min": 2.0, 
            "name": "tau", 
            "val": 17.59762711976278
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": false, 
            "full_name": "dpa_power__pow_0xxx", 
            "max": 60, 
            "min": 10, 
            "name": "pow_0xxx", 
            "val": 18.681980038267259
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": false, 
            "full_name": "dpa_power__pow_1xxx", 
            "max": 60, 
            "min": 15, 
            "name": "pow_1xxx", 
            "val": 25.17839229317271
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": false, 
            "full_name": "dpa_power__pow_2xxx", 
            "max": 80, 
            "min": 20, 
            "name": "pow_2xxx", 
            "val": 51.376512057113928
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": false, 
            "full_name": "dpa_power__pow_3xx0", 
            "max": 100, 
            "min": 20, 
            "name": "pow_3xx0", 
            "val": 40.788988399570151
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": false, 
            "full_name": "dpa_power__pow_3xx1", 
            "max": 100, 
            "min": 20, 
            "name": "pow_3xx1", 
            "val": 57.952819879437172
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": false, 
            "full_name": "dpa_power__pow_4xxx", 
            "max": 120, 
            "min": 20, 
            "name": "pow_4xxx", 
            "val": 56.279366050901693
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": false, 
            "full_name": "dpa_power__pow_5xxx", 
            "max": 120, 
            "min": 20, 
            "name": "pow_5xxx", 
            "val": 68.925799674533408
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": false, 
            "full_name": "dpa_power__pow_66x0", 
            "max": 140, 
            "min": 20, 
            "name": "pow_66x0", 
            "val": 52.171704150066574
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": false, 
            "full_name": "dpa_power__pow_6611", 
            "max": 140, 
            "min": 20, 
            "name": "pow_6611", 
            "val": 83.490187265820225
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": false, 
            "full_name": "dpa_power__pow_6xxx", 
            "max": 140, 
            "min": 20, 
            "name": "pow_6xxx", 
            "val": 78.385351798818206
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "dpa_power__mult", 
            "max": 2.0, 
            "min": 0.0, 
            "name": "mult", 
            "val": 1.5841324064332269
        }, 
        {
            "comp_name": "dpa_power", 
            "fmt": "{:.4g}", 
            "frozen": true, 
            "full_name": "dpa_power__bias", 
            "max": 100, 
            "min": 10, 
            "name": "bias", 
            "val": 15.041998236949997
        }
    ], 
    "tlm_code": null
}
