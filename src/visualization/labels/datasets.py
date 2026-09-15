from src.registry import Registry

# Display labels and ordering for datasets.
DATASETS = Registry[str](
    kind='dataset',
    where='DATASETS in src/visualization/labels/datasets.py',
    entries={
        'sepsis': 'Sepsis',
        'bpic19': 'BPIC19',
        'bpic17': 'BPIC17',
        'bpic13': 'BPIC13',
    },
)
