window.addEventListener("DOMContentLoaded", async () => {
  const img1 = document.getElementById("img1");
  const img2 = document.getElementById("img2");
  const detectorModeSel = document.getElementById("detectorMode");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const compareBtn = document.getElementById("compareBtn");
  const resetTop = document.getElementById("resetTop");
  const resetBottom = document.getElementById("resetBottom");
  const stage = document.getElementById("stage");
  const ctx = stage.getContext("2d");
  const errorBox = document.getElementById("errorBox");
  const resultBox = document.getElementById("resultBox");
  const pairTable = document.getElementById("pairTable");
  const pairBody = document.getElementById("pairBody");
  let modelsReady = false;
  let busy = false;
  let finalImage = null;
  const colors = ["#ff4444","#ff8844","#7b4cff","#3da9fc","#00d68f","#ff6ec7","#ffd166","#06d6a0"];
  function showError(msg){
    errorBox.style.display = "block";
    errorBox.innerText = msg;
    setTimeout(()=> errorBox.style.display = "none", 6000);
  }
  function clearUI(){
    ctx.clearRect(0,0,stage.width,stage.height);
    resultBox.innerHTML = "";
    pairBody.innerHTML = "";
    pairTable.style.display = "none";
  }
  function disableControls(d){
    [img1,img2,detectorModeSel,analyzeBtn,compareBtn,resetTop,resetBottom].forEach(el => el.disabled = d);
    busy = d;
  }
  function initStage(){
    stage.width = stage.parentElement.clientWidth;
    stage.height = Math.round(stage.width * 9/16);
  }
  initStage();
  window.addEventListener("resize", () => {
    if (!busy && !finalImage) initStage();
  });
  if (/Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent)) {
    const sel = document.getElementById("detectorMode");
    if (sel) sel.value = "tiny";
  }
  function getDetectorOptions(){
    if (detectorModeSel.value === "ssd"){
      return new faceapi.SsdMobilenetv1Options({ minConfidence: 0.6 });
    }
    const w = stage.parentElement?.clientWidth || 800;
    const bucket = w <= 360 ? 256 : w <= 500 ? 320 : w <= 720 ? 416 : w <= 960 ? 512 : 608;
    return new faceapi.TinyFaceDetectorOptions({ inputSize: bucket, scoreThreshold: 0.4 });
  }
  function matchText(d){
    if (d < 0.4) return "Same person";
    if (d < 0.6) return "Possibly same person";
    return "Different person";
  }
  function showCards(dict){
    resultBox.innerHTML = "";
    for (const [k,v] of Object.entries(dict)){
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = `<div class="label">${k}</div><div class="value">${v}</div>`;
      resultBox.appendChild(card);
    }
  }
  await faceapi.tf.setBackend('webgl');
  await faceapi.tf.ready();
  Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
    faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
    faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
    faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
    faceapi.nets.ageGenderNet.loadFromUri("/models"),
    faceapi.nets.faceExpressionNet.loadFromUri("/models"),
  ]).then(async ()=>{
    const warmCanvas = document.createElement("canvas");
    warmCanvas.width = 128; warmCanvas.height = 128;
    const wctx = warmCanvas.getContext("2d");
    wctx.fillStyle = "#000"; wctx.fillRect(0,0,128,128);
    await faceapi.detectAllFaces(warmCanvas, new faceapi.TinyFaceDetectorOptions({ inputSize: 256 }));
    console.log("🔥 TFJS ready — Models warmed.");
    modelsReady = true;
  }).catch(()=>{
    showError("Failed to load models. Check /models .");
  });
  async function downscaleImage(img, maxSide = 1280){
    if (Math.max(img.width, img.height) <= maxSide) return img;
    const scale = maxSide / Math.max(img.width, img.height);
    const c = document.createElement("canvas");
    const x = c.getContext("2d");
    c.width = Math.round(img.width * scale);
    c.height = Math.round(img.height * scale);
    x.imageSmoothingEnabled = true;
    x.imageSmoothingQuality = "high";
    x.drawImage(img, 0, 0, c.width, c.height);
    const out = new Image(); out.src = c.toDataURL("image/jpeg", 0.9);
    await new Promise(r => out.onload = r);
    return out;
  }
  async function enhanceImage(img){
    const c = document.createElement("canvas");
    const x = c.getContext("2d", { willReadFrequently: true });
    c.width = img.width; c.height = img.height;
    x.drawImage(img,0,0);
    const d = x.getImageData(0,0,c.width,c.height).data;
    let sumL = 0; const step = 4 * 16;
    for (let i=0;i<d.length;i+=step){
      const r=d[i],g=d[i+1],b=d[i+2];
      sumL += 0.2126*r + 0.7152*g + 0.0722*b;
    }
    const avg = sumL / (d.length/step);
    if (avg > 120) return img;
    const c2 = document.createElement("canvas");
    const x2 = c2.getContext("2d");
    c2.width = img.width; c2.height = img.height;
    x2.drawImage(img,0,0);
    const d2 = x2.getImageData(0,0,c2.width,c2.height);
    for (let i=0;i<d2.data.length;i+=4){
      d2.data[i]   = Math.min(d2.data[i]*1.07,255);
      d2.data[i+1] = Math.min(d2.data[i+1]*1.07,255);
      d2.data[i+2] = Math.min(d2.data[i+2]*1.07,255);
    }
    x2.putImageData(d2,0,0);
    const out = new Image(); out.src = c2.toDataURL("image/jpeg", 0.95);
    await new Promise(r=> out.onload = r);
    return out;
  }
  function freezeCanvas(){
    if (finalImage) return;
    finalImage = new Image();
    finalImage.src = stage.toDataURL("image/png");
    finalImage.alt = "analysis-result";
    finalImage.style.width = "100%";
    finalImage.style.borderRadius = "14px";
    stage.replaceWith(finalImage);
  }
  function unfreezeCanvas(){
    if (!finalImage) return;
    finalImage.replaceWith(stage);
    finalImage = null;
  }
  analyzeBtn.onclick = async () => {
    if (!modelsReady || busy) return;
    if (!img1.files[0]) return showError("Please upload an image first.");
    unfreezeCanvas(); disableControls(true); clearUI();
    try{
      let image = await faceapi.bufferToImage(img1.files[0]);
      image = await downscaleImage(image);
      image = await enhanceImage(image);
      const wrapW = stage.parentElement.clientWidth;
      const scale = wrapW / image.width;
      stage.width = wrapW;
      stage.height = Math.round(image.height * scale);
      ctx.drawImage(image, 0, 0, stage.width, stage.height);
      const dets = await faceapi
        .detectAllFaces(image, getDetectorOptions())
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender()
        .withFaceDescriptors();
      if (!dets.length) { showError("No faces detected."); return; }
      const faces = [];
      dets.forEach((d,i)=>{
        const b = d.detection.box;
        const col = colors[i % colors.length];
        ctx.strokeStyle = col; ctx.lineWidth = 2;
        ctx.strokeRect(b.x*scale, b.y*scale, b.width*scale, b.height*scale);
        ctx.fillStyle = col; ctx.font = "bold 16px Segoe UI";
        ctx.fillText(`#${i+1}`, b.x*scale + 5, b.y*scale - 6);
        const emo = Object.entries(d.expressions).sort((a,b)=>b[1]-a[1])[0];
        faces.push({
          idx: i+1,
          desc: d.descriptor,
          gender: d.gender,
          gP: d.genderProbability,
          age: d.age,
          emo: emo[0],
          emoP: emo[1]
        });
      });
      if (faces.length === 1){
        const f = faces[0];
        showCards({
          "Faces Detected":"1",
          "Gender": `${f.gender} (${(f.gP*100).toFixed(1)}%)`,
          "Estimated Age": f.age.toFixed(0),
          "Dominant Emotion": `${f.emo} (${(f.emoP*100).toFixed(1)}%)`
        });
      }else{
        showCards({ "Faces Detected": String(faces.length) });
        pairBody.innerHTML = "";
        for (let i=0;i<faces.length;i++){
          for (let j=i+1;j<faces.length;j++){
            const d = faceapi.euclideanDistance(faces[i].desc, faces[j].desc);
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>Face #${faces[i].idx} ↔ Face #${faces[j].idx}</td><td>${d.toFixed(3)}</td><td>${matchText(d)}</td>`;
            pairBody.appendChild(tr);
          }
        }
        pairTable.style.display = "table";
      }
      freezeCanvas();
      (finalImage || stage).scrollIntoView({ behavior:"smooth", block:"center" });
    }catch(e){
      console.error(e);
      showError("Error analyzing image.");
    }finally{
      disableControls(false);
    }
  };
  compareBtn.onclick = async () => {
    if (!modelsReady || busy) return;
    if (!img1.files[0] || !img2.files[0]) return showError("Upload two images first.");
    unfreezeCanvas(); disableControls(true); clearUI();
    try{
      let imgA = await faceapi.bufferToImage(img1.files[0]);
      let imgB = await faceapi.bufferToImage(img2.files[0]);
      imgA = await downscaleImage(imgA);
      imgB = await downscaleImage(imgB);
      imgA = await enhanceImage(imgA);
      imgB = await enhanceImage(imgB);
      const wrapW = stage.parentElement.clientWidth;
      const halfW = Math.floor(wrapW / 2);
      const scaleA = halfW / imgA.width;
      const scaleB = halfW / imgB.width;
      const hA = Math.round(imgA.height * scaleA);
      const hB = Math.round(imgB.height * scaleB);
      stage.width = wrapW;
      stage.height = Math.max(hA, hB);
      ctx.drawImage(imgA, 0, 0, halfW, hA);
      ctx.drawImage(imgB, halfW, 0, halfW, hB);
      ctx.strokeStyle = "rgba(123,76,255,0.45)";
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(halfW, 0); ctx.lineTo(halfW, stage.height); ctx.stroke();
      const detA = await faceapi.detectSingleFace(imgA, getDetectorOptions()).withFaceLandmarks().withFaceExpressions().withAgeAndGender().withFaceDescriptor();
      const detB = await faceapi.detectSingleFace(imgB, getDetectorOptions()).withFaceLandmarks().withFaceExpressions().withAgeAndGender().withFaceDescriptor();
      if (!detA || !detB){ showError("Couldn't detect faces in both images."); return; }
      const bA = detA.detection.box, bB = detB.detection.box;
      ctx.strokeStyle = "#ff4444"; ctx.lineWidth = 2;
      ctx.strokeRect(bA.x*scaleA, bA.y*scaleA, bA.width*scaleA, bA.height*scaleA);
      ctx.strokeRect(halfW + bB.x*scaleB, bB.y*scaleB, bB.width*scaleB, bB.height*scaleB);
      const emoA = Object.entries(detA.expressions).sort((a,b)=>b[1]-a[1])[0];
      const emoB = Object.entries(detB.expressions).sort((a,b)=>b[1]-a[1])[0];
      const dist = faceapi.euclideanDistance(detA.descriptor, detB.descriptor);
      showCards({
        "Similarity Score": dist.toFixed(3),
        "Person Match": matchText(dist),
        "Image A": `Gender: ${detA.gender} (${(detA.genderProbability*100).toFixed(1)}%) — Age: ${detA.age.toFixed(0)} — Emotion: ${emoA[0]}`,
        "Image B": `Gender: ${detB.gender} (${(detB.genderProbability*100).toFixed(1)}%) — Age: ${detB.age.toFixed(0)} — Emotion: ${emoB[0]}`
      });
      freezeCanvas();
      (finalImage || stage).scrollIntoView({ behavior:"smooth", block:"center" });
    }catch(e){
      console.error(e);
      showError("Error comparing images.");
    }finally{
      disableControls(false);
    }
  };
  function doReset(){
    if (busy) return;
    resultBox.innerHTML = "";
    pairTable.style.display = "none";
    unfreezeCanvas();
    clearUI();
    initStage();
  }
  resetTop.onclick = doReset;
  resetBottom.onclick = doReset;
});
