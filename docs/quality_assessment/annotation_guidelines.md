# C3 feature annotation guidelines

## Scope and label versions

This is the current interpretation for C3 root/leaf-distribution research. It
supersedes provisional wording about balanced foliage in older visual reviews.
It does not edit the workbook, change buyer rules, or supply automatic labels.
Use composite identity `(Plant_ID, Sample_ID)`; current plant IDs are P01–P08.
Keep image IDs, view, original paths, source hashes, annotator, guideline version,
and adjudication date with future annotation exports. Never overwrite a released
experiment snapshot when labels are revised.

The researcher-confirmed root corrections are released as
`c3_features_v2_20261022`; see [dataset_version_history.md](dataset_version_history.md).
Current workbook labels are corrected ground truth for that release. Remaining
leaf-origin visibility flags do not reverse or automatically exclude expert labels.

## Leaf distribution: branch origin relative to the main stem

Applicable plant types are P02–P08 where annotated data exists; this is separate
from P01 leaf-count and leaf-height work.

- **Both Sides:** at least two qualifying branches originate from opposite sides
  of the main stem. Branch lengths, sizes and leaf counts can differ.
- **One Side:** qualifying branches originate from only one side of the main
  stem, including when multiple branches occur on that side.
- Leaf blades or petioles crossing the stem in the photograph do not establish
  Both Sides. Balanced foliage and equal leaf counts are not the criterion.

Review procedure:

1. Locate the main stem, including its cut end and curved portions. Trace each
   candidate branch back to its junction with that main stem. Do not substitute
   a petiole-to-shoot junction for a qualifying branch-to-main-stem junction.
2. Mark visible branch origins and their side relative to the main stem. Confirm
   the company's definition of a qualifying branch, including whether short buds,
   developing shoots, scars or stubs qualify. No size threshold is invented here.
3. Confirm Both Sides only from qualifying origins on opposite sides; the main
   stem's local direction and the viewing angle matter more than the image edge.
4. Assess whether foliage, overlapping shoots, the rear stem surface or cropping
   conceals other junctions. One visible branch alone is not proof that no hidden
   opposite branch exists. Review the whole available stem and auxiliary views.
5. Record visibility separately: `origins_visible`, `partially_obscured`, or
   `origins_obscured`; record qualifying-branch uncertainty independently. These
   are proposed review fields, not automatically added Excel columns.
6. When the image does not establish the criterion, retain the expert/sample
   annotation and flag image evidence for manual review. Do not manufacture a
   binary image label from foliage appearance. Additional views or expert
   adjudication may be needed to distinguish sample truth from image visibility.

Physical opposite sides may be ambiguous in a single 2D view. Confirm the
inspection orientation and what evidence suffices for rear-side junctions with
the expert. Review all eligible Front images under the same policy; do not
relabel or exclude only the samples a model got wrong. Neither an annotation
disagreement nor a saliency map establishes which features the model learned.

## Root distribution: variety-specific expert criteria

Labels are **Acceptable / Not Acceptable**. Root images are applicable to P01–P08
where verified images and expert annotations exist. These are feature labels,
not overall acceptance or general visual-quality labels.

Visible external root quantity, distribution across visible pot regions, root
structure, and any health criterion must be defined separately with experts for
each variety and rooting medium. Do not apply a universal visible-root count or
infer a variety's natural root requirement from the few available photographs.
P01 foam and P02–P07 net-pot images have different visibility constraints.

For a future blinded review, record visible root coverage/quantity, concentration
or evenness, obscured regions, pot/medium visibility, framing, lighting, and
whether alternate views were used. Qualitative visual bins are exploratory
observations, not expert quality thresholds or measurements of hidden roots.
Confirm capture stage and physical scale before comparing varieties. Never infer
root labels from overall quality, leaf appearance, or model predictions.

## Revisions and evaluation

Keep old and proposed labels, reason, evidence and reviewer in a separate change
log. Only an authorized expert review can approve source-annotation changes.
Freeze a new version after adjudication. For a controlled model comparison,
all compared models need the same target version and identical saved grouped
outer/inner partitions. Re-scoring old predictions against revised labels is a
post-hoc sensitivity check, not evidence of a model improvement. Current C3
analysis supplements remain under ignored processed data.
