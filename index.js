// const [width, height, tileSize] = [125826, 86165, 256];
let info, format, height, tileSize, viewer, width;
const osddiv = document.querySelector("#osd");
const pjscanvas = document.querySelector("#pjs");
const color = {
  "Region 1": "rgb(16, 128, 256)",
  "Region 2": "rgb(32, 256, 128)",
  "Region 3": "rgb(48, 256, 256)",
  "Region 4": "rgb(64, 128, 128)"
};
let reg;
let trainableSegmentationScript;
let trainSegmenterScript;
let applySegmenterScript;
let timer;

async function main() {
  let res;
  res = await fetch("./trainable_segmentation.py");
  trainableSegmentationScript = await res.text();
  res = await fetch("./train_segmenter.py");
  trainSegmenterScript = await res.text();
  res = await fetch("./apply_segmenter.py");
  applySegmenterScript = await res.text();
  
  await loadPyodide({
    indexURL : "https://cdn.jsdelivr.net/pyodide/v0.17.0/full/"
  });
  await pyodide.loadPackage(['micropip', 'numpy', 'matplotlib', 'scikit-learn', 'scikit-image']);
};

const displayResult = () => {
  const resImg = document.getElementById("py_img");
  resImg.src = pyodide.globals.get("img_str");
  resImg.style.width = 500 + "px";
  resImg.style.height = 500 * height/width + "px";
};

const getPJSData = async (width, height) => {
  const data = pjscanvas.toDataURL();
  const img = new Image();
  img.src = data;
  return await new Promise((resolve) => {
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(img, 0, 0, width/2, height/2, 0, 0, width, height);
      resolve(ctx.getImageData(0,0,width, height));
    };
  });
};

const getOSDData = () => {
  const osdcanvas = osddiv.querySelector("canvas");
  const osdctx = osdcanvas.getContext('2d');
  const osdpixdata = osdctx.getImageData(0, 0, osdcanvas.width, osdcanvas.height);

  return osdpixdata;
};

const configureImageData = async () => {
  const img = getOSDData();
  const mask = await getPJSData(img.width, img.height);

  pyodide.globals.set("img", img.data)
  pyodide.globals.set("img_width", img.width)
  pyodide.globals.set("img_height", img.height)
  
  pyodide.globals.set("mask_img", mask.data)
  pyodide.globals.set("mask_width", mask.width)
  pyodide.globals.set("mask_height", mask.height)
};

const displayJSON = (r) => {
  const path = new paper.Path();
  path.importJSON(r.annotation.path);
  path.fillColor = color[r.annotation.name];
  path.strokeColor = null;

  return path;
};

const getData = async (url) => {
  const res = await fetch(url);
  reg = await res.json();
};

const setupPJS = async (url) => {
  // setup paperjs
  pjscanvas.width = 1000;
  pjscanvas.height = 1000 * height/width;
  paper.setup(pjscanvas);
  paper.view.matrix.a = 0.5;
  paper.view.matrix.d = 0.5;

  await getData(url);
  // displayJSON(reg[18])
  for(const r of reg) {
    p = displayJSON(r);
    const area = p.getArea();
    if(Math.abs(area)<100) {
      p.remove();
      continue;
    }
  }
};

const setupOSD = async (tilesUrl, format) => {
  // setup openseadragon
  osddiv.innerHTML = "";
  osddiv.style.width = 500 + "px";
  osddiv.style.height = 500 * height/width + "px";

  viewer = OpenSeadragon({
    id: "osd",
    showNavigationControl: false,
    tileSources: {
      width,
      height,
      tileSize,
      crossOriginPolicy: 'Anonymous',
      getTileUrl: (level, x, y) => `${tilesUrl}/${level}/${x}_${y}.${format}`
    },
    prefixUrl: "//openseadragon.github.io/openseadragon/images/"
  });

  // await new Promise ((resolve) => {
  //   viewer.addHandler('open', () => {
  //     console.log("MicroDraw image loaded");
  //     resolve();
  //   });
  // });
};

const getDatasetInformation = async (jsonUrl) => {
  const res = await fetch(jsonUrl);
  const info = await res.json();

  return info;
};

const getDziUrl = async (info, sliceIndex=0) => {
  let dziUrl = info.tileSources[sliceIndex];

  if (dziUrl.slice(0,4) !== "http") {
    if(dziUrl[0] === "/") {
      dziUrl = "https://microdraw.pasteur.fr" + dziUrl;
    } else {
      dziUrl = "https://microdraw.pasteur.fr/" + dziUrl;
    }
  }

  return dziUrl;
};

const getSize = async (info, slice=0) => {
  const dziUrl = await getDziUrl(info, slice);

  const res2 = await fetch(dziUrl);
  const wh = await res2.text();
  const fields = wh.split("\n")
    .map((o)=>o.split(" ")
    .map((o)=>o.trim())).flat();
  const tileSize = fields.filter((o)=>o.match("TileSize"))[0].split("\"")[1];
  const width = fields.filter((o)=>o.match("Width"))[0].split("\"")[1];
  const height = fields.filter((o)=>o.match("Height"))[0].split("\"")[1];
  const format = fields.filter((o)=>o.match("Format"))[0].split("\"")[1];

  return [~~width, ~~height, ~~tileSize, format];
};

const displayMessage = (msg) => {
  document.querySelector("#log").innerText = msg;
};

const processMicrodraw = async (url) => {
  displayMessage("Getting dataset information...");
  const {search} = url;
  const params = new URLSearchParams(search);
  const source = params.get("source");
  const project = params.get("project");
  const slice = params.get("slice")|0;
  const jsonUrl = (new URL(source)).href;
  info = await getDatasetInformation(jsonUrl);
  console.log({info});

  displayMessage("Getting image size...");
  ([width, height, tileSize, format] = await getSize(info, slice))

  displayMessage("Getting MicroDraw image...");
  const dziUrl = await getDziUrl(info, slice);
  const tilesUrl = dziUrl.replace(".dzi", "_files");
  await setupOSD(tilesUrl, format);

  displayMessage("Getting MicroDraw annotations...");
  const dataUrl = `https://microdraw.pasteur.fr/api?source=${source}${project?"&project="+project:""}&slice=${slice}`;
  await setupPJS(dataUrl);

  displayMessage("Loading Python backend...");
  await main();

  displayMessage("Configure data...");
  await configureImageData();

  displayMessage("Loading trainable segmenter script...");
  pyodide.runPython(trainableSegmentationScript);

  displayMessage("Trainning segmenter...");
  pyodide.runPython(trainSegmenterScript);

  displayMessage("Done.");
  displayResult(width, height);
};

const getSegmentationHints = () => {
  const url = new URL(document.querySelector("#md_url").value);
  processMicrodraw(url);
};

const processOneSlice = async (sliceIndex) => {
  displayMessage("Getting image size...");
  ([width, height, tileSize, format] = await getSize(info, sliceIndex))

  displayMessage("Getting MicroDraw image...");
  const dziUrl = await getDziUrl(info, sliceIndex);
  const tilesUrl = dziUrl.replace(".dzi", "_files");
  // await setupOSD(tilesUrl, format);
  const options = {
    width,
    height,
    tileSize,
    crossOriginPolicy: 'Anonymous',
    getTileUrl: (level, x, y) => `${tilesUrl}/${level}/${x}_${y}.${format}`
  };

  viewer.open(options);
  await new Promise((resolve) => {
    setTimeout(() => {
      resolve();
    }, 2000);
  })

  displayMessage("Configuring image data...");
  const img = getOSDData();
  pyodide.globals.set("img", img.data)
  pyodide.globals.set("img_width", img.width)
  pyodide.globals.set("img_height", img.height)

  displayMessage("Segmenting one slice...");
  pyodide.runPython(applySegmenterScript);

  displayMessage("Done.");
  displayResult(width, height);
};

const processAllSlices = () => {
  const nSlices = info.tileSources.length;
  document.querySelector("#process-all").innerHTML = `Process All [${nSlices} ${(nSlices===1)?"slice":"slices"}]`
}
