from src.registry import Registry

# What a log is called, as against `data.name` in its config, which is what its run directory is
# called. A log with nothing declared for it stops the run. Also the order logs are tabulated in,
# which `DATASETS.ordered` applies.
DATASETS = Registry[str](
    kind='dataset',
    where='DATASETS in src/visualization/labels/datasets.py',
    entries={
        'sepsis': 'Sepsis',
        'sepsis-log': 'Sepsis (log)',
        'bpic19': 'BPIC19',
        'bpic19-log': 'BPIC19 (log)',
        'bpic17': 'BPIC17',
        'bpic17-log': 'BPIC17 (log)',
    },
)
