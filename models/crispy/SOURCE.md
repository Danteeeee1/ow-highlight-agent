# Crispy pretrained kill-detection models

The `.npy` weight files in this directory are NOT committed to git.
Download them once:

    python run.py fetch-model            # overwatch (default)
    python run.py fetch-model --game all # overwatch + csgo2 + valorant

## Source and license

* Upstream project: https://github.com/Flowtter/crispy
* Files: `overwatch.npy`, `csgo2.npy`, `valorant.npy` and the two
  `*-mask.png` images come from the `windows.zip` asset of Crispy release
  `1.1.4` (https://github.com/Flowtter/crispy/releases/tag/1.1.4).
* License: MIT (see https://github.com/Flowtter/crispy/blob/master/LICENSE.md).
  Full credit for the training data, network architecture and weights belongs
  to the Crispy project. This repo only re-implements their inference
  pipeline (preprocessing + forward pass) in `owagent/model.py`.
* The masks (`csgo2-mask.png`, `valorant-mask.png`) are tiny image assets and
  ARE committed for convenience; they are also re-extracted by fetch-model.

## What the model does

A tiny 10000-120-15-2 MLP (numpy only, no deep-learning framework) trained on
real gameplay frames. Per frame it answers "is a kill announcement visible
right now?" with a probability. Our pipeline turns a rising edge over
`recognizer.confidence` into one highlight event, which then feeds the same
clustering / scoring / clipping stages as the heuristic detector.
