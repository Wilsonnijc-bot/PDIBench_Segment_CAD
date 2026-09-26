# Ten-case V1/V2 replay experiment

The supported command is `PYTHONPATH=src python -m pdi_eval.experiment`. See the [experiment interface README](../src/pdi_eval/experiment/README.md) for selection, preflight, run, resume, status, and outputs.

The frozen selection is `configs/v1_v2_replay_10.json`. It contains ten workbook-labeled videos from the 197-video source manifest. The source videos alone are used; motion-smoothness results are not inputs. Each of two workers completes one video through persistent masking, V1, V2, and replay. One GPU lock limits model calls to a single process while API requests and file work may overlap.

The current GPU profile is `configs/experiment_v1_v2_persistent_10.json`. Launch `scripts/record_v1_v2_replay_two.sh` under tmux to keep the work detached and write a durable log. Each case has one validated mask supplied to both scoring versions. For a positive case, the persistent full-video mask replaces only `link7` of the six-link base segmentation; a negative case with no confirmed deformation uses the base mask and records that policy. Invalid masking or source provenance fails the case.

V1 emits its standard MP4 replay. V2 also emits the [interactive rigidity replay](rigidity_replay_contract.md) with scoring point pairs, lines, deviation color, pair metrics, final score, and synchronized source video.
