


//////////// PREPARE DOCUMENT
function PSprepareDocument(layerSetName){
  app.preferences.rulerUnits = Units.PIXELS;
  const doc = app.activeDocument;
  var activeLayer = doc.activeLayer;
  if (activeLayer.typename === "LayerSet") return -1;
  if (activeLayer.isBackgroundLayer) return -1;
  if (activeLayer.kind !== LayerKind.NORMAL) return -1;
  if (activeLayer.bounds == 0) return -1;
  return activeLayer.id;
}



//////////// PRE-INFERENCE
function JSXPreInference(id, mimeType, tarSize, savePath){
  var layer = getArtLayerById(id);

  var bndsOrg = bndsToPxlDim(layer.bounds); // convert array of unit values to numbers
  var modInfo = modifyBounds(bndsOrg, tarSize); // modify the bounds of selected layer to a square of at least tarSize

  savePath = savePath.replace(/\\/g, "\\\\");
  app.activeDocument.suspendHistory("Runway Pre-Inference", 'preInference('+id+', "'+mimeType+'", '+modInfo.size+', '+tarSize+', "'+savePath+'")');

  // everything going back to JS land is a string, so here we use JSON
  return '{"savePath": "'+savePath+'", "bounds": ['+modInfo.bnds.join(",")+'], "size": '+modInfo.size+'}';
}

function preInference(id, mimeType, size, tarSize, savePath) {
  var layer = getArtLayerById(id);
  layer.copy();

  var docRef = app.documents.add(size, size, app.activeDocument.resolution); // create a new document
  docRef.paste(); // paste layer to new document
  if (size > tarSize) docRef.resizeImage(tarSize, tarSize, app.activeDocument.resolution); // size doc down to tarSize if needed

  // save file
  var saveFile = new File(savePath);
  if (mimeType === 'image/jpeg') { saveJPG(docRef, saveFile, 10); }
  else { savePNG(docRef, saveFile); }
  docRef.close(SaveOptions.DONOTSAVECHANGES);
}

function modifyBounds(bnds, minDim){
  var w = bndsWidth(bnds);
  var h = bndsHeight(bnds);
  var dim = (w > h ? w : h);
  dim = Math.ceil(dim*1.05);// add 5% padding to edges;
  dim = (dim % 2 == 0 ? dim : dim + 1); // ensure dim is an even number;
  dim = Math.max(dim, minDim); // ensure dim is at least as large as the minDim
  var vec = Array((dim-w)/2,(dim-h)/2);
  return {
      bnds: Array(bnds[0]-vec[0],bnds[1]-vec[1], bnds[2]+vec[0],bnds[3]+vec[1]),
      vec: vec,
      size: dim
  };
}

//////////// POST-INFERENCE


function JSXPostInference(layerSrcId, tarBounds, layerSetName, filename){
  app.activeDocument.suspendHistory("Runway Post-Inference", 'postInferenceMacro('+layerSrcId+', "'+tarBounds+'", "'+layerSetName+'", "'+filename.replace(/\\/g, "\\\\")+'")');
}

function postInferenceMacro(twinLayerId, tarBounds, layerSetName, filename) {
  tarBounds = tarBounds.split(',');
  var dstDoc = app.activeDocument;
  var activeLayerToRestore = dstDoc.activeLayer;
  var destLayerSet = getDestLayerSet(dstDoc, layerSetName); // destination layer set
  var twinLayer = getArtLayerById(twinLayerId); // layer we'll link with

  // find and remove other layers in the destination layer set that are linked with our twin
  var deletableLayers = [];
  for (var l in twinLayer.linkedLayers){
    if ((twinLayer.linkedLayers[l].name == twinLayer.name)&&(twinLayer.linkedLayers[l].parent == destLayerSet)){
      deletableLayers.push(twinLayer.linkedLayers[l]);
    }
  }
  for (var l in deletableLayers){ deletableLayers[l].remove();}

  // open inferred image and paste into this document
  var srcDoc = app.open(new File(filename));
  if ((srcDoc.width != tarBounds[2]-tarBounds[0])||(srcDoc.height != tarBounds[3]-tarBounds[1])){
    srcDoc.resizeImage(tarBounds[2]-tarBounds[0], tarBounds[3]-tarBounds[1]);
  }
  srcDoc.selection.selectAll();
  srcDoc.selection.copy();
  srcDoc.close(SaveOptions.DONOTSAVECHANGES);
  dstDoc.paste();

  // rename, resize, align, link
  dstDoc.activeLayer.name = twinLayer.name;
  var curBounds = bndsToPxlDim(dstDoc.activeLayer.bounds);
  dstDoc.activeLayer.translate(tarBounds[0] - curBounds[0], tarBounds[1] - curBounds[1]);
  dstDoc.activeLayer.link(twinLayer);
  dstDoc.activeLayer.move(destLayerSet, ElementPlacement.INSIDE);

  dstDoc.activeLayer = activeLayerToRestore;
}


//////////// UTIL

function bndsToPxlDim(bnds){ return Array( bnds[0].as("px"), bnds[1].as("px"), bnds[2].as("px"), bnds[3].as("px")) };
function bndsWidth(bnds){ return bnds[2] - bnds[0]; }
function bndsHeight(bnds){ return bnds[3] - bnds[1]; }


function getDestLayerSet(doc, lsetName) {
  var lset;
  try{
    lset = doc.layerSets.getByName(lsetName);
  } catch(error) {
    // create a new destination layer set
    lset = doc.layerSets.add();
    lset.name = lsetName;

    // make it red
    activeLayerToRestore = doc.activeLayer;
    doc.activeLayer = lset;
    var desc1 = new ActionDescriptor();
    var ref1 = new ActionReference();
    ref1.putEnumerated(app.charIDToTypeID('Lyr '), app.charIDToTypeID('Ordn'), app.charIDToTypeID('Trgt'));
    desc1.putReference(app.charIDToTypeID('null'), ref1);
    var desc2 = new ActionDescriptor();
    colorcode = 'Rd  ';
    desc2.putEnumerated(app.charIDToTypeID('Clr '), app.charIDToTypeID('Clr '), app.charIDToTypeID(colorcode));
    desc1.putObject(app.charIDToTypeID('T   '), app.charIDToTypeID('Lyr '), desc2);
    executeAction(app.charIDToTypeID('setd'), desc1, DialogModes.NO);
    doc.activeLayer = activeLayerToRestore;

    // move to the bottom of the document
    var relTo = doc.layers[doc.layers.length-1];
    if (relTo.isBackgroundLayer) relTo = doc.layers[doc.layers.length-2];
    lset.move(relTo, ElementPlacement.PLACEAFTER);
  }
  return lset;
}


function getArtLayerById(id) {
    // search the top level
    const doc = app.activeDocument;
    var layers = app.activeDocument.artLayers;
    for (var i = 0; i < layers.length; i++) {
        if (layers[i].id === id) return layers[i];
    }
    // the ArtLayer we're looking for isn't at the top level. to avoid recursion, we look just one level down.

    var sets = app.activeDocument.layerSets;
    for (var s = 0; s < sets.length; s++) {
      layers = sets[s].artLayers;
      for (var i = 0; i < layers.length; i++) {
          if (layers[i].id === id) return layers[i];
      }
    }
    alert("Error in getArtLayerById()\nNo layer found with id " + id);
}



//////////// Inherited from Runway Implementation

function saveJPG(doc, saveFile, jpegQuality) {
  var jpgSaveOptions = new JPEGSaveOptions();
  jpgSaveOptions.embedColorProfile = true;
  jpgSaveOptions.formatOptions = FormatOptions.STANDARDBASELINE;
  jpgSaveOptions.matte = MatteType.NONE;
  jpgSaveOptions.quality = jpegQuality;
  doc.saveAs(saveFile, jpgSaveOptions, true, Extension.LOWERCASE);
};

function savePNG(doc, saveFile) {
  var pngSaveOptions = new PNGSaveOptions();
  pngSaveOptions.compression = 0;
  pngSaveOptions.interlaced = false;
  doc.saveAs(saveFile, pngSaveOptions, true, Extension.LOWERCASE);
}
