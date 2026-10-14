# Contributing

Bug reports, fixes and documentation improvements are welcome.

Before you open a pull request, run the unit tests with `python -m unittest -v`. They need only the standard library
and skip the parts that need aiohttp, numpy, torch, transformers or mlx-vlm when those are missing. A change to the
prompts, the read-out or the AutoThink path must keep the probabilities of the golden test (`tests/test_golden.py`)
identical, or explain in the pull request why and by how much they move. A change that affects accuracy or latency
should come with numbers measured with `startlux-decide eval` or `startlux-decide bench` on the same hardware before
and after.

Keep the code in the style of the surrounding files and keep new dependencies optional where possible. By contributing
you agree that your contribution is licensed under the Apache License 2.0.
