# Experiment runtime compatibility patches

`manifest.json` records three small modifications present in the original
experiment environment, compared against the published PyPI wheels:

- [SWE-bench](https://github.com/SWE-bench/SWE-bench) 4.1.0: guard the legacy
  cgroup write and transfer files through Modal sandbox execution.
- [Harbor](https://github.com/harbor-framework/harbor) 0.4.0: create Modal log
  directories through execution, and preserve Claude Code's authentication
  token separately from its API key.

Each entry includes the upstream and modified file's SHA256 and exact source
replacements. These are experiment compatibility changes to third-party code;
they do not change task definitions, solver prompts, or grading criteria.

Run `python -m Task.patch_runtime swebench` or
`python -m Task.patch_runtime terminalbench` in the corresponding virtual
environment. The installer checks the package version and every target file
before writing, accepts files already patched, and rejects other local edits.
Reinstalling a dependency may require reapplying these patches.

The source snippets remain under their upstream licenses. The full licenses
from the published wheels are included as `LICENSE.swebench` (MIT, copyright
Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press,
and Karthik R Narasimhan) and `LICENSE.harbor` (Apache 2.0). The replacements
are modifications used by this experiment, not upstream release contents.
