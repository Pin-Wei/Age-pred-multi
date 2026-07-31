#!/usr/bin/env bash
#
# Run voxel-wise statistical analysis of DTI data using FSL's Tract-Based Spatial Statistics
# - see: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/TBSS(2f)UserGuide.html
# 
# FA images were stored in $DATA_ROOT/FA and named sub-????_FA.nii.gz
# MD images were stored in $DATA_ROOT/MD and named sub-????_MD.nii.gz

set -euo pipefail
: "${FSLDIR:?Error: FSLDIR is not set.}"

DATA_ROOT=~/pinwei/Age_pred_multi/data/dti
STAGE=${1:-all}

if [[ ! -d $DATA_ROOT/tbss && -d $DATA_ROOT/FA ]]; then
	mv "$DATA_ROOT/FA" "$DATA_ROOT/tbss"
	cd $DATA_ROOT/tbss
	for f in sub-????_FA.nii.gz; do 
		[ -e "$f" ] && mv "$f" "${f/_FA.nii.gz/.nii.gz}"
	done
	
	if [ -d ../MD ]; then
		mv ../MD .
		cd MD
		for f in sub-????_MD.nii.gz; do 
			[ -e "$f" ] && mv "$f" "${f/_MD.nii.gz/.nii.gz}"
		done
		cd ..
	fi
fi

if [[ "$STAGE" = all || "$STAGE" = 1 ]]; then
    echo "=== TBSS stage 1: prepare FA maps ==="
	cd $DATA_ROOT/tbss
	tbss_1_preproc *.nii.gz
fi

if [[ "$STAGE" = all || "$STAGE" = 2 ]]; then
    echo "=== TBSS stage 2: registration (this is the slow step) ==="
    cd $DATA_ROOT/tbss
	tbss_2_reg -T
fi

if [[ "$STAGE" = all || "$STAGE" = 3 ]]; then
    echo "=== TBSS stage 3: mean FA + skeleton ==="
    cd $DATA_ROOT/tbss
	tbss_3_postreg -S
fi

if [[ "$STAGE" = all || "$STAGE" = 4 ]]; then
    echo "=== TBSS stage 4: project onto skeleton ==="
    cd $DATA_ROOT/tbss
	tbss_4_prestats 0.2
fi

if [[ "$STAGE" = all || "$STAGE" = md ]]; then
    echo "=== TBSS non-FA: project MD onto skeleton ==="
	cd $DATA_ROOT/tbss	
	[[ -f stats/thresh.txt ]] || { echo "ERROR: stage 4 not run yet (stats/thresh.txt missing)"; exit 1; }
	tbss_non_FA MD
fi

## ----------------------------------------------------------------------------------------------------------

if [[ "$STAGE" = fix_md ]]; then
	cd $DATA_ROOT/tbss/stats
	tbss_skeleton -i mean_FA -p `cat thresh.txt` \
		mean_FA_skeleton_mask_dst \
		$FSLDIR/data/standard/LowerCingulum_1mm \
		all_FA all_MD_skeletonised -a all_MD
fi