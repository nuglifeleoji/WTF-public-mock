# tau2 experiment data overlay

These four files preserve the tau2 data used by the source experiments. Apply
them over the `data/` directory from
[tau2-bench](https://github.com/sierra-research/tau2-bench/tree/0ed2fd8d830a20657d89ae9c2efcc94838aa7129)
at commit `0ed2fd8d830a20657d89ae9c2efcc94838aa7129`. The parent directory containing
the resulting `tau2/` directory is `TAU2_DATA_DIR`.

The airline and retail task files differ from that upstream revision in task
instructions, evaluation assertions/actions, and explicit reward bases. Replacing
them with the upstream versions changes the benchmark being run. The user
simulator guideline files preserve the original whitespace. The required domain
databases, policies, and telecom tasks are unchanged from the pinned checkout.

`MANIFEST.json` records every file's byte count, SHA-256 hash, and the SHA-256 hash
of the corresponding upstream file before the overlay. The four data files total
505,706 bytes. Apply only the paths listed in the manifest; this README, manifest,
and license are attribution/verification material, not runtime data.

This snapshot is derived from Sierra Research's tau2-bench, distributed under the
included [MIT license](LICENSE). It preserves the experiment's local snapshot;
it is not presented as an unmodified upstream benchmark release. No simulation
results, voice assets, or unused benchmark domains are included.
