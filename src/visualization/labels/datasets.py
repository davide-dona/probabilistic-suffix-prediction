from src.registry import Registry

# What a log is called, as against `data.name` in its config, which is what its run directory is
# called. A log with nothing declared for it stops the run. Also the order logs are tabulated in,
# which `DATASETS.ordered` applies.
DATASETS = Registry[str](
    kind='dataset',
    where='DATASETS in src/visualization/labels/datasets.py',
    entries={
        'bpic17': 'BPIC17',
        'bpic19': 'BPIC19',
        'sepsis': 'Sepsis',
    },
)
