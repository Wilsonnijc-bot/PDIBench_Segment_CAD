# Ten-case V1 replay experiment

The frozen selection is `configs/v1_replay_10.json`: ten workbook-labeled videos from the 197-video source manifest. The source videos alone are inputs; motion-smoothness results are not. Use the new `v1-persistent-10` run root and the profile `configs/experiment_v1_persistent_10.json`. The historical V1/V2 comparison folder remains unchanged.

Each of two workers completes one video through persistent masking, V1 scoring, and replay. A GPU lock limits model calls to one process while API requests and file work may overlap. A validated full-video mask replaces only `link7` of the six-link base segmentation. A negative case with no confirmed deformation uses the base mask and records that policy. Invalid masking or source provenance fails the case.

V1 emits a combined MP4 and an [interactive rigidity replay](rigidity_replay_contract.md) with scored point pairs, deviation colors, pair metrics, final score, and synchronized source video. Launch `scripts/record_v1_replay_two.sh` under tmux for a durable log and exit status. See the [experiment interface](../src/pdi_eval/experiment/README.md) for staging, preflight, run, resume, and export commands.
