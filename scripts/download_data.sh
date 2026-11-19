#!/bin/bash
# Download the public-domain USGS side-scan mosaics (1 m GeoTIFF) and sea-floor texture maps
# used for training into data/raw (about 1.7 GB zipped, 9 GB unpacked).
set -e
cd "$(dirname "$0")/.." && mkdir -p data/raw && cd data/raw
get() {  # get <folder> <url>
  mkdir -p "$1"; [ -f "$1/.done" ] && return
  curl -sSL --retry 4 -C - -o "$1/data.zip" "$2" && unzip -o -q "$1/data.zip" -d "$1" && rm "$1/data.zip" && touch "$1/.done"
  echo "ok $1"
}
# side-scan backscatter mosaics
get ofr2007_1373_klein https://pubs.usgs.gov/of/2007/1373/GIS/raster/backscatter/klein_bs1m.zip
get ofr2012_1006_vs    https://pubs.usgs.gov/of/2012/1006/GIS/raster/backscatter/VS_Backscatter_1m.zip
get ofr2012_1002_bb    https://pubs.usgs.gov/of/2012/1002/GIS/raster/backscatter/BB_backscatter1m.zip
get ofr2010_1006_klein https://pubs.usgs.gov/of/2010/1006/GIS/raster/backscatter/CCB_Klein_bs_1m.zip
get ofr2009_1072_dh    https://pubs.usgs.gov/of/2009/1072/GIS/raster/backscatter/DH_USGS_backscatter1m.zip
get ofr2006_1008_bs    https://pubs.usgs.gov/of/2006/1008/gis/datazipped/bh_1mbs.zip
get ofr2016_1119_vns   https://pubs.usgs.gov/of/2016/1119/GIS_catalog/SourceData/backscatter/vns_bsmos_1m.zip
# sea-floor texture polygons (Barnhardt classes plus grain-size statistics)
get tex_ofr2012_1157 https://pubs.usgs.gov/of/2012/1157/GIS_catalog/SedimentTexture/SedCover.zip
get tex_ofr2014_1220 https://pubs.usgs.gov/of/2014/1220/GIS_catalog/SedimentTexture/BuzzardsBay_sedcover.zip
get tex_ofr2016_1119 https://pubs.usgs.gov/of/2016/1119/GIS_catalog/SedimentTexture/VineyardNantucketSound_sedcover.zip
get tex_ofr2015_1153 https://pubs.usgs.gov/of/2015/1153/GIS_catalog/SedimentTexture/SedCover.zip
find . -name "*sedcover.shp"
