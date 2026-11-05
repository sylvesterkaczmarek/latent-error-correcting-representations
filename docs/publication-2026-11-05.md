# Publication recovery record

Version 0.2.0 was independently verified and merged at
210d02faafcbf9df495c59bdfc577147c193bccb. All eight final PR checks and all eight
exact-release verification jobs passed. The publication API then returned HTTP
500, including on a failed-job retry and an authenticated maintainer attempt.

The reviewed tag was created explicitly. The exact tested hosted distributions,
source archive, review records and checksum manifest were staged in a draft.
All eight staged asset hashes were checked. Publishing the same draft with the
optional latest-release designation disabled succeeded. No source, package,
research result or existing published artifact was replaced. The original failed
workflow attempts remain visible rather than being represented as successful.

The publication workflow now disables that optional designation for subsequent
releases. A separate read-only workflow verifies an existing published tag,
its source-archive identity, package hashes and full non-editable installation.
It neither rebuilds nor replaces public artifacts. Public installation results
are recorded in that workflow and the merged release PR.

Release tag v0.2.0 remains bound to the reviewed commit above. This later
workflow/documentation correction does not change the release's runtime code.
No PyPI publication was performed.
