sft_runs/ — LoRA training artifacts grouped by experimental round

Layout (physical directories; large subtrees remain gitignored per .gitignore):

  round1/
    meter_only/     ← April 11 meter LoRA (was sft/meter_only_lora/flan-t5-large_lora_20260411_174009)
    rhyme_only/     ← April 11 rhyme LoRA (was sft/rhyme_only_lora/.../20260411_174908)
    combined/       ← April 11 combined (was sft_full/combined_lora/.../20260411_175033)

  round2/
    meter_only/     ← April 13 meter (was sft_full/meter_only_lora/.../20260413_210437)
    rhyme_only/     ← April 13 rhyme (was sft_full/rhyme_only_lora/.../20260413_210437)
    combined/       ← April 13 combined (was sft_full/combined_lora/.../20260413_212630)

  round3/
    meter_only_lr1e4/   ← Round-3 meter sweep (lr=1e-4)
    meter_only_lr5e5/   ← Round-3 meter sweep (lr=5e-5)

  outputs/
    Placeholder for NEW training (--output-root default: sft_runs/outputs/<task>_lora).
    Slurm: scripts/hpc/lora_train.slurm and scripts/hpc/submit_lora3.sh.

  archive/
    remainder_sft / remainder_sft_full — reserved for non-canonical moves (currently empty).

The old top-level folders sft/ and sft_full/ were removed after this migration.
If you still need other timestamps (e.g. extra rhyme attempts), recover them from Bouchet/rsync
into round3/ or outputs/ and add a short note here.

Hyperparameters: run_params.json inside each run directory.
Eval JSON + tables: results/ (separate from weights).
