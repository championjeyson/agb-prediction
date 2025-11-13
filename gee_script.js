/*******************************
 * GEE SCRIPT: Sentinel-2 + PALSAR-2 + DEM + CGLS-LC100 Landcover
 * Generate clean, consistent geospatial inputs for a given location and year.
 * Author: Jason Champion / https://github.com/championjeyson/agb-prediction
 * Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0)
 * https://creativecommons.org/licenses/by/4.0/
 *******************************/

// ========== USER PARAMETERS ==========
var lon = -123.2036;
var lat = 49.3953;
var year = 2020;

// Optional temporal control (defaults: April–September)
var startDate = ee.Date.fromYMD(year, 4, 1);
var endDate   = ee.Date.fromYMD(year, 9, 30);

// AOI parameters
var useSquare = true;          // true = square AOI, false = circular buffer
var squareSizeMeters = 15000;  // 15 km × 15 km
var bufferMeters = 5000;       // radius if useSquare = false

// Export options
var exportToDrive = true;
var exportFolder = 'GEE_exports';

// Visualization toggle
var showLayers = true;

// ========== DEFINE AOI ==========
var point = ee.Geometry.Point([lon, lat]);
var aoi;
if (useSquare) {
  var halfSide = squareSizeMeters / 2;
  var lonOffset = halfSide / (111320 * Math.cos(lat * Math.PI / 180));
  var latOffset = halfSide / 110540;
  aoi = ee.Geometry.Rectangle([
    lon - lonOffset, lat - latOffset,
    lon + lonOffset, lat + latOffset
  ]);
} else {
  aoi = point.buffer(bufferMeters).bounds();
}

Map.centerObject(point, 12);
if (showLayers) {
  Map.addLayer(point, {color: 'red'}, 'Center point');
  // Map.addLayer(aoi, {color: 'blue'}, 'AOI');
}

// ========== 1) SENTINEL-2 (Least Cloudy Composite) ==========
print('→ Fetching Sentinel-2 SR harmonized collection...');
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .sort('CLOUDY_PIXEL_PERCENTAGE', false);

// Find first image fully covering AOI
var s2_leastCloudy = ee.Image(s2.iterate(function(img, prev) {
  img = ee.Image(img);
  prev = ee.Image(prev);
  var contains = img.geometry().contains(aoi);
  return ee.Algorithms.If(contains, img, prev);
}, null));

// Fallback: median composite if no full coverage
s2_leastCloudy = ee.Image(
  ee.Algorithms.If(s2_leastCloudy, ee.Image(s2_leastCloudy), s2.median())
).clip(aoi);

// Cloud probability mask (optional)
var cloudProb = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY');
var idx = s2_leastCloudy.get('system:index');
var cloudImg = cloudProb.filter(ee.Filter.equals('system:index', idx)).first();
cloudImg = ee.Algorithms.If(cloudImg, ee.Image(cloudImg).select('probability'), ee.Image(0));
cloudImg = ee.Image(cloudImg).rename('cloud_prob');

s2_leastCloudy = ee.Image(s2_leastCloudy.addBands(cloudImg)
  .updateMask(cloudImg.lt(70).or(cloudImg.not())));

// Scale optical bands
var OPTICAL_BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'];
var sclBand = 'SCL';
var scaledOptical = s2_leastCloudy.select(OPTICAL_BANDS).multiply(0.0001);
var sclImage = s2_leastCloudy.select([sclBand]);
var leastCloudy = scaledOptical.addBands(sclImage)
  .copyProperties(s2_leastCloudy, s2_leastCloudy.propertyNames());

if (showLayers) {
  Map.addLayer(ee.Image(leastCloudy), {bands:['B4','B3','B2'], min:0, max:0.3}, 'Sentinel-2 (scaled)');
}
print('✓ Sentinel-2 processed.');

// ========== 2) PALSAR-2 ==========
print('→ Fetching PALSAR-2 mosaic...');
var sarId = 'JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH';
var sar = ee.ImageCollection(sarId)
  .filterBounds(aoi)
  .filterDate(ee.Date.fromYMD(year,1,1), ee.Date.fromYMD(year+1,1,1));

var sarImage = ee.Image(
  ee.Algorithms.If(sar.size().gt(0), sar.first(), ee.ImageCollection(sarId).sort('system:time_start').first())
).clip(aoi);

if (showLayers) Map.addLayer(sarImage.select('HH'), {min:0, max:10000}, 'PALSAR HH');
print('✓ PALSAR-2 ready.');

// ========== 3) ALOS AW3D30 DEM ==========
print('→ Loading AW3D30 DSM...');
var aw3d = ee.ImageCollection('JAXA/ALOS/AW3D30/V4_1').select('DSM');
var aw3dMosaic = aw3d.mosaic().clip(aoi).rename('DEM');
if (showLayers) Map.addLayer(aw3dMosaic, {min:0, max:1000}, 'AW3D30 DSM');
print('✓ DEM loaded.');

// ========== 4) CGLS-LC100 Landcover ==========
print('→ Loading CGLS-LC100 land cover...');

// Clamp year on the client side for dataset coverage (2015–2019)
var lcYear = Math.min(Math.max(year, 2015), 2019);
if (year < 2015 || year > 2019) {
  print('⚠️ Using LC year', lcYear, '(dataset available only 2015–2019)');
}

// Now use plain JS number in the dataset ID
var lc = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/' + lcYear)
  .select(['discrete_classification', 'discrete_classification-proba'])
  .clip(aoi);

if (showLayers) {
  Map.addLayer(lc.select('discrete_classification'), {min:0,max:220}, 'CGLS LC100');
}
print('✓ Landcover loaded.');

// ========== EXPORTS ==========
if (exportToDrive) {
  print('→ Starting exports to Google Drive...');
  var bandsToExport = OPTICAL_BANDS.concat(['SCL']);
  var renameMap = {'B1':'B01','B2':'B02','B3':'B03','B4':'B04','B5':'B05','B6':'B06','B7':'B07','B8':'B08','B8A':'B8A','B9':'B09','B11':'B11','B12':'B12','SCL':'SCL'};

  bandsToExport.forEach(function(band) {
    Export.image.toDrive({
      image: s2_leastCloudy.select(band).rename(renameMap[band]),
      description: 'S2_' + renameMap[band],
      folder: exportFolder,
      fileNamePrefix: 'S2_' + renameMap[band],
      region: aoi,
      scale: s2_leastCloudy.select(band).projection().nominalScale().getInfo(),
      maxPixels: 1e13
    });
  });

  Export.image.toDrive({
    image: sarImage.select(['HH','HV']),
    description: 'SAR_PALSAR2',
    folder: exportFolder,
    fileNamePrefix: 'SAR_PALSAR2',
    region: aoi,
    scale: 25,
    maxPixels: 1e13
  });

  Export.image.toDrive({
    image: aw3dMosaic,
    description: 'AW3D30',
    folder: exportFolder,
    fileNamePrefix: 'AW3D30',
    region: aoi,
    scale: 30,
    maxPixels: 1e13
  });

  var lcCombined = lc.rename(['map','proba']);
  Export.image.toDrive({
    image: lcCombined,
    description: 'CGLS_LC100',
    folder: exportFolder,
    fileNamePrefix: 'CGLS_LC100',
    region: aoi,
    scale: 100,
    maxPixels: 1e13
  });

  print('✅ Exports started (check Tasks tab).');
} else {
  print('⚠️ exportToDrive = false; no exports started.');
}

// ========== METADATA SUMMARY ==========
print('Sentinel-2 least cloudy image:', s2_leastCloudy);
print('PALSAR-2 image:', sarImage);
print('DEM:', aw3dMosaic);
print('Landcover:', lc);