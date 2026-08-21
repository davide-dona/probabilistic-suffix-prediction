# Training queue

Dataset configs staged for a batch training run. Copy in the configs to train, then hand them all
to the two GPUs at once:

```bash
cp config/datasets/bpic17.yaml config/datasets/bpic19.yaml config/queue/
scripts/train_queue.sh -w config/hardware/cuda-a6000.yaml
```

One training runs per GPU at a time, and a GPU that finishes picks up the next config rather than
waiting on the run beside it.

A config being trained right now is renamed `<name>.yaml.running`, and only a plain `.yaml` is ever
picked up, so nothing is trained twice.

A config that trained successfully is deleted from here. One whose run failed is renamed
`<name>.yaml.failed` and stays, since a config a freed GPU could claim again would be retried on
the spot rather than looked at. Its output is under `outputs/queue/`, named in the summary the
script prints. To try it again, drop the suffix:

```bash
mv config/queue/bpic17.yaml.failed config/queue/bpic17.yaml
```

A `.running` left behind after the script is gone is an interrupted run, and is re-queued the same
way.

Everything here is gitignored except this file.
