## Design Context

### Users
The project owner inspects Link5 reconstruction and anomaly detection results.
The task is to compare the four-frame0 normal reference with actual test clouds
and see where the current checkpoint reports anomalies.

### Brand Personality
Direct, restrained, readable. Use the owner's explicit request for a
"super straight forward" replay with no stacked explanations or diagnostic tabs.

### Aesthetic Direction
Two large 3D views, plain labels, one test selector, and a short checkpoint status.
Retain a dark plotting canvas for point visibility. Use neutral normal points and
a readable anomaly scale. Avoid model architecture jargon, training tables,
prompt previews, artificial clouds and historical variants in this viewing path.

### Design Principles
- Put normal geometry and test anomaly colors together in the first screen.
- Keep measured scores, shared coordinates and point identities unchanged.
- Tie color contrast to the normal threshold; never imply that color proves accuracy.
- State a failed checkpoint status plainly without filling the page with diagnostics.
- Keep provenance in metadata, outside the user's inspection flow.
