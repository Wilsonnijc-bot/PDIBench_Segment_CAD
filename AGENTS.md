# Project instructions

Read [agent.md](agent.md) before scheduling GPU video experiments.

Maximize useful memory utilization and measured throughput on one H200, including
training. Five parallel videos is a baseline, not a ceiling. Configure concurrency
separately from GPU allocation and measure stage capacity. Honor a stop request by
cancelling active and dependent jobs.
