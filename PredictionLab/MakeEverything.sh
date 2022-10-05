#!/usr/bin/env bash

# Use the two following statements on the terminal to decide to test whether the Savchenko model or our own.
#export MODEL_IS_SAVCHENKO=1
#unset MODEL_IS_SAVCHENKO

if test $MODEL_IS_SAVCHENKO
then
  echo "Variable MODEL_IS_SAVCHENKO is set. Training only Savchenko models."
  MODELS=/Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/ASavchenko-ENet-b0_7
else
  MODELS="\
   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/MobileNet-Affnet-CropRot-220623 \
   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/MNet_Afnet_Crop_20220802_124446 \
   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/MNet_Afnet_Crop11_20220914_155408 \
   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/MNet_Afnet_Crop12_20220914_155447 \
   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/MNet_Afnet_CNormCropRot_20220919_050811 \
   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/MNet_Afnet_CNormCrop11_20220914_155447 \
   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/MNet_Afnet_CNormCrop12_20220804_190816 \
   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/MNet_Afnet_HFlip_CNormCrop11Rot_20221003 \
   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/MNet_Afnet_HFlip_CropRot_20221003 \
   "
fi

#    /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/ResNet50-AffNet-nopreproc-210921 \
#   /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/ResNet50-AfNet-CropRot-220701 \
#    /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/models/EfNet_AfNet_20220802_123441 \

make_count=0

for img_preproc in NoProc CropRot Crop11Rot Crop12Rot CNormCropRot CNormCrop11Rot CNormCrop12Rot
do
  echo 11111 $img_preproc
  for model_dir in $MODELS
  do
    echo 2222 Model $model_dir
    for data_dir in /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/data/EkmanReferences \
      /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/data/DaFEx \
      /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/data/FePh \
      /Users/fanu01-admin/Nextcloud/Documents/Projects/EASIER/EkmanClassifierSharedData/data/AffectNetVal
    do
      echo 333 Data $data_dir
      ((make_count+=1))
      echo Make number $make_count
      DATADIR=$data_dir MODELDIR=$model_dir IMG_PREPROC=$img_preproc make test
    done
  done
done

