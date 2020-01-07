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
  prepareDocument(LAYERSETNAME)
    .then( async layerSrcId => {
      if (layerSrcId<0){
        console.log("!!! Active layer is not valid. Select a regular photoshop 'art' layer, not a layer group, an empty layer, the background layer, or non-raster layer.");
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
          console.log("... pre-inference complete.");
          //console.log("img64Src: "+img64Src);
          doInference(data.img64Src)
            .then( async img64Dst => {
              console.log("... received inference image from Runway");
              postInference(layerSrcId, img64Dst, tmpPathDst, data.bounds)
                .then( async rslt => {
                  console.log("... post-inference complete.");
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
      //console.log(saveLayerResult);
      saveLayerResult = JSON.parse(saveLayerResult);
      //console.log(`layer bounded to rectangle ${saveLayerResult.bounds} \t was saved to ${saveLayerResult.savePath}`);
      const bounds = saveLayerResult.bounds;
      const imgHTML = new Image();
      imgHTML.src = saveLayerResult.savePath;
      return new Promise(resolve => {
        imgHTML.onload = () => {
          // del file again. we don't need it anymore
          // window.cep.fs.deleteFile(savePath); // ks comment out for debug
          // return img
          resolve([imgHTML,bounds]);
        };
      });
    })
    .then(async rslt => {
      [imgHTML,bounds] = rslt;
      //console.log("imgHTML: "+imgHTML);
      const img64 = await imgToBase64(imgHTML, mimeType);
      return [img64,bounds]; // constructs data array
    });
  //console.log(data); // base64 image and a bounds array
  return data;
};


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
