// handTracking.js - JavaScript for handling hand tracking

import { HandLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

const letterSelectorInput = document.getElementById("letterInput");
const startButton = document.getElementById("startButton");
const snapshotButton = document.getElementById("snapshotButton");
const exportButton = document.getElementById("exportButton");
const handData = {}; // Object to store hand tracking data
const video = document.getElementById("videoElement");

let handLandmarker;

async function initializeHandTracking() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });
    console.log("model loaded, you can start webcam")
}

async function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
        video.play();
    });
}


async function takeSnapshot() {
    if (!handLandmarker) {
        console.error("Hand tracking model is not initialized.");
        return;
    }

    const results = await handLandmarker.detectForVideo(video, performance.now());
    let detectArray = [];

    for (let hand of results.landmarks) {
        let handPoints = [];
        for (let handSingle of hand) {
            handPoints.push([handSingle.x, handSingle.y]);
        }
        detectArray.push(handPoints);
    }

    const letter = letterSelectorInput.value;
    if (!letter) {
        console.error("Please enter a letter.");
        return;
    }

    if (!handData[letter]) {
        handData[letter] = [];
    }
    handData[letter].push(detectArray);

    console.log(handData);
}

function exportHandData() {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(handData, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "hand_data.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    document.body.removeChild(downloadAnchor);
}

startButton.addEventListener("click", startWebcam);
snapshotButton.addEventListener("click", takeSnapshot);
exportButton.addEventListener("click", exportHandData);

initializeHandTracking();
