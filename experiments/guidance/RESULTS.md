# Guided continuation results

Primary run: `/data1/dhuang/flowr_root/output/molsteer_guidance_20260923_fp32`.

Local [Chinese results](../../../output/MolSteer_Guidance_20260923/Results.zh-CN.md) and [English results](../../../output/MolSteer_Guidance_20260923/Results.en.md).

Final MMFF local-relaxation proxy (kcal/mol): unguided 19.635271; selection 19.623723; creativity 19.584905. All final graphs match; one consolidated structural-screening region remains. This is a single paired continuation, with reconstructed historical self-conditioning/RNG and exact restoration of saved state tensors.

The full output bundle includes bilingual risk reports, all molecular stage tensors/SDFs, gradient traces, live finite differences and an exact t=0.75 restart audit. `resume_start.pt` provides the complete shared starting runtime checkpoint. Use a fresh output path to reproduce; the supplied configuration preserves the actual experiment location for provenance.
