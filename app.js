const myImage = document.getElementById("image");
const localHost = "http://127.0.0.1:5500";

Promise.all([
  // faceapi.nets.faceRecognitionNet.loadFromUri(localHost + "/models")
  faceapi.nets.faceRecognitionNet.loadFromUri(`${localHost}/models`),
  faceapi.nets.faceLandmark68Net.loadFromUri(`${localHost}/models`),
  faceapi.nets.ssdMobilenetv1.loadFromUri(`${localHost}/models`)
]).then(start);

async function start() {
  document.body.append("Tüm modeller Yüklendi..");

  myImage.addEventListener("change", async () => {
    const selectedImage = await faceapi.bufferToImage(myImage.files[0]);
    document.body.append(selectedImage);

    const detections = await faceapi
      .detectAllFaces(selectedImage)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const displaySize = {
      width: selectedImage.width,
      height: selectedImage.height
    };

    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    console.log(resizedDetections);

    const canvas = faceapi.createCanvasFromMedia(selectedImage);
    document.body.append(canvas);
    faceapi.matchDimensions(canvas, displaySize);

    // resizedDetections.forEach(d => {
    //   const box = d.detection.box;
    //   const drawBox = new faceapi.draw.DrawBox(box, { label: "Kimdir Bu?" });
    //   drawBox.draw(canvas);
    // });

    const results = resizedDetections.map(d =>
      faceMatcher.findBestMatch(d.descriptor)
    );

    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.toString()
      });
      drawBox.draw(canvas);
    });
  });

  const labelDescriptions = await loadImages();
  const faceMatcher = new faceapi.FaceMatcher(labelDescriptions, 0.6);
}

async function loadImages() {
  const labels = [
    "Cihan",
    "Gokhan",
    "Guray",
    "Ferdi",
    "Yasin",
    "Ilker",
    "Handan",
    "Defne"
  ];

  return Promise.all(
    labels.map(async label => {
      const descriptions = [];

      for (let i = 1; i <= 3; i++) {
        const image = await faceapi.fetchImage(
          `${localHost}/library/${label}/${i}.jpg`
        );

        const detections = await faceapi
          .detectSingleFace(image)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
