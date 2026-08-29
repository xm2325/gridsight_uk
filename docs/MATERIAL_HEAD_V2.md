# Material head v2: UK transfer audit

## Fixed design

Material head v2 combines frozen SigLIP2 features from two public sources. From MPID it selects, before encoding, the largest eligible labelled box from each sampled image: 400 train and 60 development images for each of glass, porcelain/ceramic and polymer/composite. It also reuses the pinned Substation15 glass, porcelain and background features from material head v1. The four output classes are glass, porcelain/ceramic, polymer/composite and other.

The encoder remains frozen. A two-layer MLP is trained for 600 fixed full-batch steps. Acceptance requires sufficient native pixels, tight/context agreement, a development-derived logit margin and development embedding support. Scores are not probabilities. Thirteen source-assisted boxes on three previously observed UK assets were frozen before v2 inference and excluded from training and threshold calibration.

## Results

Roihu job **940239** completed with exit `0:0` in 56 seconds. Internal MPID/Substation15 development diagnostics were 78.2% coverage and 98.5% accepted accuracy, with one false acceptance among 40 background targets. The UK development diagnostic was only 46.2% coverage and 50.0% accepted accuracy: six of thirteen boxes were accepted and three were correct. In particular, all three source-supported porcelain strings on image 8090535 were accepted as glass. The complete source-supported porcelain string on 770272 was safely rejected to unknown.

The native-resolution intervention was frozen before inference. A CC BY-SA 2.0, 2560×1920 original of 8090535 was acquired from Wikimedia Commons with the API response, download URL, author, licence, file hashes and exact 4× coordinate transform retained. Job **940273** completed with exit `0:0` in 17 seconds. Four of six boxes were accepted and two were correct, again 50.0% accepted accuracy. Two porcelain strings remained accepted as glass. The material error is therefore not explained by the 640×480 derivative alone.

Three leave-one-asset-out adaptation folds were then fixed in advance. Each fold held out one complete asset group, used only the other two UK groups for adaptation, froze the encoder and first MLP layer, updated the last layer for 120 steps and recalibrated rejection only on academic development data. Job **940286** completed with exit `0:0` in 10 seconds. Aggregated coverage was 38.5% and accepted accuracy was 40.0%. The method is rejected; no best fold is selected.

## Capability decision

The direct MPID detector remains a proposal arm, not a deployable material labeler. Neither full-assembly cropping, a four-class academic head, more native pixels nor sparse last-layer UK adaptation provides safe Keen-style UK material labels. The English report preserves each failure and raw decision.

The next bounded route is a larger, provenance-preserved UK source pool at asset level, complete-string proposals, and encoder-level adaptation evaluated by asset-separated folds. Polymer/composite requires actual UK examples. Until an acceptance gate is validated on unseen UK asset groups, the interface must return `unknown` rather than a polished material percentage.
