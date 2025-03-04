import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import KNear from "./knear.js";

const letterSelectorInput = document.getElementById("letterInput");
const startButton = document.getElementById("startButton");
const snapshotButton = document.getElementById("snapshotButton");
const exportButton = document.getElementById("exportButton");
const checkButton = document.getElementById("checkButton");

const handData = {}; // Object to store hand tracking data
const video = document.getElementById("videoElement");

let handLandmarker;
let recording = false;

// Initialize KNear with k = 3 (you can adjust based on needs)
const machine = new KNear(1);

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
    console.log("Model loaded. You can start the webcam.");
}

async function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
        video.play();
    });
}

async function recordHandTracking() {
    if (!handLandmarker) {
        console.error("Hand tracking model is not initialized.");
        return;
    }

    const letter = letterSelectorInput.value;
    if (!letter) {
        console.error("Please enter a letter.");
        return;
    }

    if (!handData[letter]) {
        handData[letter] = [];
    }

    recording = true;
    // time is measured in frameInterval * numFrames (In ms)
    const frameInterval = 200;
    const numFrames = 5;

    let frameCount = 0;
    let collectedData = [];

    console.log("Recording started...");

    for (let i = 0; i < numFrames; i++) {
        if (!recording) break;

        const results = await handLandmarker.detectForVideo(video, performance.now());

        let detectArray = [];
        for (let hand of results.landmarks) {
            let handPoints = [];
            for (let handSingle of hand) {
                handPoints.push([handSingle.x, handSingle.y]);
            }
            detectArray.push(handPoints);
        }

        handData[letter].push(detectArray);
        collectedData.push(detectArray);
        frameCount++;

        await new Promise(resolve => setTimeout(resolve, frameInterval)); // Wait for next frame capture
    }

    console.log("Recording stopped.");
    recording = false;

    // Convert collected data into a flattened array for KNear
    const flattenedData = collectedData.flat(Infinity); // Flatten to a single array of numbers
    console.log(flattenedData);

    if (flattenedData.length > 0) {
        machine.learn(flattenedData, letter);
        console.log(`Learned: ${letter}`);
    }

}

async function detectGesture() {
    if (!handLandmarker) {
        console.error("Hand tracking model is not initialized.");
        return;
    }

    // time is measured in frameInterval * numFrames (In ms)
    const frameInterval = 200;
    const numFrames = 5;

    let collectedData = [];

    console.log("Gesture detection started...");

    for (let i = 0; i < numFrames; i++) {
        const results = await handLandmarker.detectForVideo(video, performance.now());

        let detectArray = [];
        for (let hand of results.landmarks) {
            for (let handSingle of hand) {
                detectArray.push(handSingle.x, handSingle.y); // Store x, y directly
            }
        }

        collectedData.push(detectArray);
        await new Promise(resolve => setTimeout(resolve, frameInterval)); // Wait for next frame capture
    }

    console.log("Gesture detection finished.");


    // Fully flatten the collected data
    const flattenedData = collectedData.flat(Infinity);

    console.log("Input to classify:", flattenedData);

    if (flattenedData.length > 0) {
        const detectedLetter = machine.classify(flattenedData);
        console.log(`Detected gesture: ${detectedLetter}`);
        return detectedLetter;
    } else {
        console.error("No data collected for gesture detection.");
        return null;
    }
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
snapshotButton.addEventListener("click", recordHandTracking);
exportButton.addEventListener("click", exportHandData);
checkButton.addEventListener("click", detectGesture);

initializeHandTracking();