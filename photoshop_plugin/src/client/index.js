const path = require('path');
const fs = require('fs');
const os = require('os');
const crypto = require("crypto");

const tmpPath = path.join(os.tmpdir(), "runway_ced_photoshop");
if (!fs.existsSync(tmpPath)){ fs.mkdirSync(tmpPath); }
console.log(`temporary image folder is at ${tmpPath}`);

const MIMETYPE='image/png' //'image/jpeg'
const IMGSIZE = 256;
const LAYERSETNAME = "generated";

/* Create an instance of CSInterface. */
var csInterface = new CSInterface();
/* Make a reference to your HTML button and add a click handler. */
var openButton = document.querySelector("#the-button");
openButton.addEventListener("click", onClick);

/* Write a helper function to pass instructions to the ExtendScript side. */
function onClick() {
  console.log("click!");
  document.body.style.background = '#aaa';
  prepareDocument(LAYERSETNAME)
    .then( async layerSrcId => {
      if (layerSrcId<0){
        console.log("!!! Active layer is not valid. Select a regular photoshop 'art' layer, not a layer group, an empty layer, the background layer, or non-raster layer.");
        document.body.style.background = '#f00';
        return false;
      }
      console.log(`... identified valid source layer with id ${layerSrcId}.`);

      const rando = crypto.randomBytes(4).toString("hex");
      const ext = (MIMETYPE === 'image/png' ? '.png' : '.jpg');
      const tmpPathSrc = path.join(tmpPath,'rw_'+rando+'_src'+ext).replace(/\\/g, "\\\\");
      const tmpPathDst = path.join(tmpPath,'rw_'+rando+'_dst'+ext).replace(/\\/g, "\\\\");
      //console.log(`saving to ${tmpPathSrc}`);

      var data = {};
      preInference(layerSrcId, tmpPathSrc, MIMETYPE, IMGSIZE)
        .then( async rslt => {
          [data.img64Src, data.bounds] = rslt;
          //data.bounds = Array(0,0,256,256);
          console.log("... pre-inference complete.");
          document.body.style.background = '#888';
          //console.log("img64Src: "+img64Src);
          doInference(data.img64Src)
            .then( async img64Dst => {
              console.log("... received inference image from Runway");
              document.body.style.background = '#666';
              postInference(layerSrcId, img64Dst, tmpPathDst, data.bounds)
                .then( async rslt => {
                  console.log("... post-inference complete.");
                  document.body.style.background = '#444';
                });
            });
        });


    });
};


async function prepareDocument() {
  const layerSrcId = await evalScriptPromise("PSprepareDocument()");
  return layerSrcId;
};

async function preInference(layerSrcId, savePath, mimeType, imgSize){
  //console.log("preInference");
  const data = await evalScriptPromise(`JSXPreInference(${layerSrcId}, "${mimeType}", ${imgSize}, "${savePath}")`, true)
    .then( saveLayerResult => {
      saveLayerResult = JSON.parse(saveLayerResult);
      //console.log(saveLayerResult.img64);
      console.log(`layer bounded to rectangle ${saveLayerResult.bounds} \t was converted to base64`);
      const bndsSve = saveLayerResult.bounds;

      var img = new Image;
      //img.onload = resizeImage;
      img.src = saveLayerResult.img64;
      return new Promise(resolve => {
        img.onload = () => {
          [img64, vec, size] = modifyHTMLImgAndConvertToBase64(img);
          var bndsGbl = Array( bndsSve[0]-vec[0], bndsSve[1]-vec[1],  bndsSve[0]-vec[0]+size, bndsSve[1]-vec[1]+size);
          console.log(`bndsGbl is ${bndsGbl}`);
          resolve([img64,bndsGbl]); // constructs data array
        };
      });

    });
  //console.log(data); // base64 image and a bounds array
  return data;
};

function modifyHTMLImgAndConvertToBase64(img) {
    console.log(`given image is ${img.naturalWidth} x  ${img.naturalHeight}`);

    // create an off-screen canvas and set to target size
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 256;

    var w = img.naturalWidth;
    var h = img.naturalHeight;
    var dim = (w > h ? w : h);
    var drawCrds = Array(128-w/2, 128-h/2, w, h);

    if (dim>256){
      canvas.width = dim;
      canvas.height = dim;
      var drawCrds = Array(dim/2-w/2, dim/2-h/2, w, h);
    }
    console.log(`drawing to a ${canvas.width} image at coords ${drawCrds}`);

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, drawCrds[0], drawCrds[1], drawCrds[2], drawCrds[3]);
    img64 = canvas.toDataURL('image/jpeg', 1.0);
    var vec = Array(drawCrds[0], drawCrds[1]);
    var size = canvas.width;
    //bndsLcl = Array(drawCrds[0], drawCrds[1], drawCrds[0]+drawCrds[2], drawCrds[1]+drawCrds[3]);
    // encode image to data-uri with base64 version of compressed image
    return [img64, vec, size]; // drawImage takes (x,y,w,h), while photoshop takes (x1,y1,x2,y2))
}


async function doInference(img64In){
  const inputs = { "image_in": img64In };

  const img64Out = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(inputs)
  })
    .then(response => response.json())
    .then(outputs => {
      const { image_out } = outputs;
      return image_out;
      // use the outputs in your project
    })

  return img64Out;
}

async function postInference(layerSrcId, img64, filename, bounds){
  console.log("postInference");
  window.cep.fs.writeFile(
    filename,
    img64.replace(/^data:image\/[a-z]+;base64,/, ""),
    window.cep.encoding.Base64
  );
  const layerName = "out";
  bounds = JSON.stringify(bounds);
  console.log(`placing to bounds ${bounds}`);
  return evalScriptPromise(`JSXPostInference(${layerSrcId}, ${bounds}, "${LAYERSETNAME}", "${filename}")`, true);
};






// CONVERT IMG TO BASE64 ENCODING

async function imgToBase64(img, mimeType){
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  c.width = img.width;
  c.height = img.height;
  ctx.drawImage(img, 0, 0);
  return c.toDataURL(mimeType);
};

function evalScriptPromise(func, verbose=false) {
  if (verbose) console.log(`[MACRO] \t ${func}`);
  return new Promise((resolve, reject) => {
    csInterface.evalScript(func, (response) => {
      if (response === 'EvalScript error.'){
        console.log(response);
        reject();
      }
      resolve(response);
    });
  });
};
