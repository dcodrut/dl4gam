WD=../data/external/wd/s2_alps_plus/
mkdir -p $WD
wget -N https://huggingface.co/datasets/dcodrut/dl4gam_alps/resolve/main/data/glacier_wide/inv.tar.gz -P $WD
wget -N https://huggingface.co/datasets/dcodrut/dl4gam_alps/resolve/main/data/glacier_wide/2023.tar.gz -P $WD
mkdir -p $WD/inv/glacier_wide/
tar -xvzf $WD/inv.tar.gz -C $WD/inv/glacier_wide/
mkdir -p $WD/2023/glacier_wide/
tar -xvzf $WD/2023.tar.gz -C $WD/2023/glacier_wide/
printf "\nFinished extracting the rasters to $WD\n"
