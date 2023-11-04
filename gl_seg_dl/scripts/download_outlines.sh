mkdir -p ../data/outlines/s2/raw
wget -N https://store.pangaea.de/Publications/PaulF-etal_2019/c3s_gi_rgi11_s2_2015_v2.zip -P ../data/outlines/s2/raw
unzip  -o -d ../data/outlines/s2/raw ../data/outlines/s2/raw/c3s_gi_rgi11_s2_2015_v2.zip
printf "\nOutlines from Paul et. al. 2020 downloaded in data/outlines\n"