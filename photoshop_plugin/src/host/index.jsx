


//////////// PREPARE DOCUMENT
function PSprepareDocument(layerSetName){
  app.preferences.rulerUnits = Units.PIXELS;
  const doc = app.activeDocument;

  try {
    var selectedLayers = getSelectedLayers(doc);
  } catch(err) {
    return '{"success":false, "message":"Cannot work with the selected layers, try unselecting the Background layer. '+err.message.replace(/"/g, '\\"').replace('\n',' ')+'"}';
  }

  var selectedIds = [];
  for( i = 0; i < selectedLayers.length; i++) {
      if (selectedLayers[i].typename === "LayerSet") return '{"success":false, "message":"One the selected layers is a LayerSet."}';
      if (selectedLayers[i].isBackgroundLayer) return '{"success":false, "message":"One of the selected layers is the Background."}';
      if (selectedLayers[i].kind !== LayerKind.NORMAL) return '{"success":false, "message":"One of the selected layers is not normal."}';
      if (selectedLayers[i].bounds == 0) return '{"success":false, "message":"One of the selected layers is empty."}';
      selectedIds.push(selectedLayers[i].id);
   }
   return '{"success":true, "layerSrcIds":['+selectedIds.join(",")+']}';

  /*
  var activeLayer = doc.activeLayer;
  if (activeLayer.typename === "LayerSet") return -1;
  if (activeLayer.isBackgroundLayer) return -1;
  if (activeLayer.kind !== LayerKind.NORMAL) return -1;
  if (activeLayer.bounds == 0) return -1;
  return activeLayer.id;
  */

}

cTID = function(s) { return app.charIDToTypeID(s); };
sTID = function(s) { return app.stringIDToTypeID(s); };

function newGroupFromLayers(doc) {
    var desc = new ActionDescriptor();
    var ref = new ActionReference();
    ref.putClass( sTID('layerSection') );
    desc.putReference( cTID('null'), ref );
    var lref = new ActionReference();
    lref.putEnumerated( cTID('Lyr '), cTID('Ordn'), cTID('Trgt') );
    desc.putReference( cTID('From'), lref);
    executeAction( cTID('Mk  '), desc, DialogModes.NO );
};

function getSelectedLayers(doc) {

  var selLayers = [];
  newGroupFromLayers(doc);

  var group = doc.activeLayer;
  var layers = group.layers;

  for (var i = 0; i < layers.length; i++) {
    selLayers.push(layers[i]);
  }

  executeAction(cTID("undo", undefined, DialogModes.NO));

  return selLayers;
};






//////////// PRE-INFERENCE
function JSXPreInference(id, mimeType, tarSize, savePath){
  var layer = getArtLayerById(id);

  var bndsOrg = bndsToPxlDim(layer.bounds); // convert array of unit values to numbers
  //var modInfo = modifyBounds(bndsOrg, tarSize); // modify the bounds of selected layer to a square of at least tarSize

  //savePath = savePath.replace(/\\/g, "\\\\");
  //app.activeDocument.suspendHistory("Runway Pre-Inference", 'preInference('+id+', "'+mimeType+'", '+modInfo.size+', '+tarSize+', "'+savePath+'")');
  //app.activeDocument.activeLayer = layer;
  selectLayerById(id);
  var base64Str = selectedLayersToBase64String(id);

  // everything going back to JS land is a string, so here we use JSON
  return '{"img64": "'+base64Str+'", "bounds": ['+bndsOrg.join(",")+']}';
}

/*
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
*/

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
  placeFile(filename);
  dstDoc.activeLayer.rasterize(RasterizeType.ENTIRELAYER);
  var w = bndsWidth(tarBounds);
  var h = bndsHeight(tarBounds);

  // scale placed image to expected dimension
  var bndsPlcd = bndsToPxlDim(dstDoc.activeLayer.bounds);
  dstDoc.activeLayer.resize( w/bndsWidth(bndsPlcd)*100,  h/bndsHeight(bndsPlcd)*100, AnchorPosition.TOPLEFT)
  // align
  var x = tarBounds[0] - bndsPlcd[0];
  var y = tarBounds[1] - bndsPlcd[1];
  dstDoc.activeLayer.translate(x,y);

  // rename, link
  dstDoc.activeLayer.name = twinLayer.name;
  dstDoc.activeLayer.link(twinLayer);
  dstDoc.activeLayer.move(destLayerSet, ElementPlacement.INSIDE);

  dstDoc.activeLayer = activeLayerToRestore;
}


//////////// UTIL

function bndsToPxlDim(bnds){ return Array( bnds[0].as("px"), bnds[1].as("px"), bnds[2].as("px"), bnds[3].as("px")) };
function bndsWidth(bnds){ return bnds[2] - bnds[0]; }
function bndsHeight(bnds){ return bnds[3] - bnds[1]; }


function selectedLayersToBase64String(layerID) {
  var pngPath, pngFile, pngData64, pngBase64Image;

  //layerID = app.activeDocument.activeLayer.id;

  pngPath = new File(Folder.temp + "/" + layerID).fsName;
  writeLayerPNGfile(pngPath);

  pngFile = new File(pngPath + ".base64");
  pngFile.open('r');
  pngFile.encoding = "UTF-8";

  pngData64 = pngFile.read();
  pngFile.close();
  pngFile.remove();

  //return pngData64;
  return "data:image/png;base64," + pngData64;
};

function writeLayerPNGfile(path) {
  var desc = new ActionDescriptor();
  desc.putBoolean(stringIDToTypeID("selectedLayers"), true); // Get data from all selected layers
  desc.putString(stringIDToTypeID("rawPixmapFilePath"), path); // Path to file
  desc.putBoolean(stringIDToTypeID("bounds"), true);
  desc.putInteger(stringIDToTypeID("width"), 10000); // Max size in pixels
  desc.putInteger(stringIDToTypeID("height"), 10000);
  desc.putInteger(stringIDToTypeID("format"), 2); // raw pixels (Don't touch this!)
  executeAction(stringIDToTypeID("sendLayerThumbnailToNetworkClient"), desc, DialogModes.NO);
}

function placeFile(path){
  var sourceFile= new File(path);
  var idPlc = charIDToTypeID( "Plc " );
  var desc3 = new ActionDescriptor();
  var idnull = charIDToTypeID( "null" );
  desc3.putPath( idnull, sourceFile);
  var idFTcs = charIDToTypeID( "FTcs" );
  var idQCSt = charIDToTypeID( "QCSt" );
  var idQcsa = charIDToTypeID( "Qcsa" );
  desc3.putEnumerated( idFTcs, idQCSt, idQcsa );
  executeAction( idPlc, desc3, DialogModes.NO );
}

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

function selectLayerById(id){
  var ref = new ActionReference();
  ref.putIdentifier(charIDToTypeID("Lyr "), id);
  var desc = new ActionDescriptor();
  desc.putReference(charIDToTypeID("null"), ref );
  //if(add) desc.putEnumerated( stringIDToTypeID( "selectionModifier" ), stringIDToTypeID( "selectionModifierType" ), stringIDToTypeID( "addToSelection" ) );
  desc.putBoolean( charIDToTypeID( "MkVs" ), false );
  executeAction(charIDToTypeID("slct"), desc, DialogModes.NO );
};




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
