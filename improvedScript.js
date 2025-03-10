import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { captureHandSnapshot, drawHand } from "./functions.js"
import KNear from "./knear.js";

// empty let for the tracking
let handLandmarker;
// bool for if the webcam is running
let webcamRunning = false;

// object for the snapshot data and the current letter
let snapShotObject = {};
let currentCharacter = '';

// Define video elements
const canvas = document.getElementById("visualOutput");
const ctx = canvas.getContext("2d");
const video = document.getElementById("videoElement");

const snapShotCanvas = document.getElementById("snapShotOutput");

// Define buttons + elements + input & event listeners
const snapShotButton = document.getElementById("snapShot");
snapShotButton.addEventListener("click", takeSnapshot)

const showSnapShotButton = document.getElementById("showSnapShot");
showSnapShotButton.addEventListener("click", showSnapshot)

const characterInput = document.getElementById("characterInput");
characterInput.addEventListener('input', changeCharacter)

const letterText = document.getElementById("currentLetterText");

// Initialize hand tracking, download the model, and set settings
async function initializeHandTracking() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });

    console.log("Model loaded. You can start the webcam.");
    startWebcam();
}

async function changeCharacter(){
    const character = characterInput.value.trim(); // Get the trimmed input value
    if (character === '') {
        console.error("Please enter a character");
        return; // Stop execution if no character is entered
    } else{
        currentCharacter = character;
        console.log(`Currently selected letter is: ${currentCharacter}`);
    }
}

async function changeCharacterManual(character){
    if (character === '') {
        console.error("Please enter a character");
        return; // Stop execution if no character is entered
    } else{
        currentCharacter = character;
        console.log(`Currently selected letter is: ${currentCharacter}`);
    }
}

// Start webcam and process video frames
async function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
        video.play();
        webcamRunning = true;

        video.onloadeddata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            visualizeHands();
        };
    });
}

// Function to detect and visualize hand landmarks
async function visualizeHands() {
    // Check if model is running
    if (!handLandmarker) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas

    // Check if webcam is running
    if (!webcamRunning) return;

    // Results of the hand positions
    const results = await handLandmarker.detectForVideo(video, performance.now());

    // For all the results draw the points
    if (results.landmarks.length > 0) {
        for (let hand of results.landmarks) {
            // Function to draw the points, this function is stored in functions.js
            drawHand(hand, ctx, canvas);
        }
    }

    requestAnimationFrame(visualizeHands); // Continue visualization loop
}

async function takeSnapshot(){

    if (currentCharacter === '') {
        console.error("Please enter a character");
        return; // Stop execution if no letter is entered
    }

    const results = await handLandmarker.detectForVideo(video, performance.now());

    snapShotObject[currentCharacter] = results.landmarks;

}

async function showSnapshot(){
    console.log(snapShotObject);

    letterText.innerText = currentCharacter;

    if (snapShotObject[currentCharacter]) {
        if (snapShotObject[currentCharacter].length > 0) {
            for (let hand of snapShotObject[currentCharacter]) {
                captureHandSnapshot(hand, snapShotCanvas,canvas);
            }
        }
    } else{
        console.error("No snapshot");
    }
}

initializeHandTracking();
