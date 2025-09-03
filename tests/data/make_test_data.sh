# resample bert to fsaverage3
# lh.white.stripped is created with nibabel without saving volume information
cp $FREESURFER_HOME/subjects/bert/mri/T1.mgz .
for hemi in lh rh
do
    TARGET_REG=$FREESURFER_HOME/subjects/fsaverage4/surf/$hemi.sphere
    for surf in white pial
    do
        echo $hemi : $surf
        mris_resample --atlas_reg $TARGET_REG --subject_reg $FREESURFER_HOME/subjects/bert/surf/$hemi.sphere.reg --subject_surf $FREESURFER_HOME/subjects/bert/surf/$hemi.$surf --out $hemi.$surf --annot_in $FREESURFER_HOME/subjects/bert/label/$hemi.aparc.annot --annot_out $PWD/$hemi.aparc.annot
        # ensure correct volume geometry
        mris_convert --vol-geom $FREESURFER_HOME/subjects/bert/mri/wm.mgz $hemi.$surf $hemi.$surf
        mris_convert --to-scanner $hemi.$surf $hemi.$surf.scanner
        mris_convert $hemi.$surf $hemi.$surf.gii
    done
done
