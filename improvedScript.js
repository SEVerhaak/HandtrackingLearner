import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { captureHandSnapshot, drawHand } from "./functions.js"
import KNear from "./knear.js";

// vars for checking the results
let success = false;
let targetCharacter = 'a';

// empty let for the tracking
let handLandmarker;
// bool for if the webcam is running
let webcamRunning = false;

// object for the snapshot data and the current letter
let snapShotObject = {};
let currentCharacter = '';

// Define video & canvas elements
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

const importJSONButton = document.getElementById("importJSON");
importJSONButton.addEventListener("click", importJSON);

const startLoopButton = document.getElementById("startLoop");
startLoopButton.addEventListener("click", detectSignLanguageCharacterLoop);

const letterText = document.getElementById("currentLetterText");

// Initialize KNear with k = 1 (you can adjust based on needs)
const machine = new KNear(1);
// Model file locations (You can edit this so it works with your file locations, make sure it is an array with the path to the {filename}.json file
const files = ['../datasets/4set.json', '../datasets/xyz_set.json', '../datasets/wie.json'];

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
// Change current selected character (This character is used as the character that currently needs to be made
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
// Manual function that takes an string as input to set it as the current selected character
async function changeCharacterManual(character){
    if (character === '') {
        console.error("Please enter a character");
        return; // Stop execution if no character is entered
    } else{
        currentCharacter = character;
        console.log(`Currently selected letter is: ${currentCharacter}`);
    }
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

    if (!webcamRunning){
        console.error("Webcam is not active yet")
        return
    }

    const results = await handLandmarker.detectForVideo(video, performance.now());

    snapShotObject[currentCharacter] = results.landmarks;

}
// Function that displays the snapshot
async function showSnapshot(){

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

async function importJSON(){
    if (files.length > 0){
        await loadModel(files)
    } else{
        console.error("No models found.");
    }
}

async function loadModel(files) {

    for (const file of files){
        try {
            const response = await fetch(file); // Fetch the file
            if (!response.ok){
                throw new Error("Failed to load JSON");
            }

            const fileData = await response.json();

            console.log(fileData); // Use the object

            for (const [key, value] of Object.entries(fileData)) {
                console.log(`learned letter: ${key}`);
                machine.learn(value, key)
            }

        } catch (error) {
            console.error("Error loading JSON:", error);
        }
    }
}

async function detectSignLanguageCharacterLoop() {
    while (true) { // Infinite loop to keep running detection
        console.log("Starting detection in 3 seconds...");
        await new Promise(resolve => setTimeout(resolve, 3000)); // 3-second countdown before starting

        console.log("Detecting sign language character...");
        const nearestMatches = await detectSignLanguageCharachter();

        if (nearestMatches) {
            console.log(nearestMatches);
            checkResults(nearestMatches, true, targetCharacter);
            console.log("Detection complete. Waiting 3 seconds before next detection...");
            await new Promise(resolve => setTimeout(resolve, 3000)); // 3-second delay after finishing
        } else {
            console.log(nearestMatches);
            checkResults(nearestMatches, false, targetCharacter);
            console.error("Detection failed. Retrying in 3 seconds...");
            await new Promise(resolve => setTimeout(resolve, 3000)); // 3-second delay before retry
        }
    }
}

async function detectSignLanguageCharachter(){
    if (!handLandmarker) {
        console.error("Hand tracking model is not initialized.");
        return;
    }

    // Recording time is measured in frameInterval * numFrames (frameInterval in ms & numFrames in frames)
    const frameInterval = 20;
    const numFrames = 50;

    // Array containing all the data recorded
    let collectedData = [];

    for (let i = 0; i < numFrames; i++) {
        const results = await handLandmarker.detectForVideo(video, performance.now());

        let detectArray = [];

        for (let hand of results.landmarks) {
            const wrist = hand[0]; // Wrist landmark
            for (let handSingle of hand) {
                let relX = handSingle.x - wrist.x; // X relative to wrist
                let relY = handSingle.y - wrist.y; // Y relative to wrist
                let relZ = handSingle.z - wrist.z;
                detectArray.push([relX, relY, relZ]); // Store relative coordinates
            }
        }

        collectedData.push(detectArray);
        await new Promise(resolve => setTimeout(resolve, frameInterval)); // Wait for next frame capture
    }

    console.log("Gesture detection finished.");

    // Fully flatten the collected data
    const flattenedData = collectedData.flat(Infinity);

    if (flattenedData.length > 0) {
        return machine.findNearest(flattenedData, 3); // Gebruik de nieuwe functie;
    } else {
        console.error("No data collected for gesture detection.");
        return null;
    }

}

async function checkResults(nearestMatches, foundMatches, targetCharacter) {
    if (!foundMatches) {
        return console.error('Failed to detect gesture, no points');
    }

    // Total tries (in this case, it's always 3 based on your example)
    const totalTries = 3;

    // Variable to track the highest certainty
    let highestCertainty = 0;
    let predictedLetter = '';

    // Iterate through the nearestMatches to calculate the certainty
    for (let i = 0; i < nearestMatches.length; i++) {
        const [letter, matchCount] = nearestMatches[i];

        // Calculate certainty as a percentage
        const certainty = (matchCount / totalTries) * 100;

        console.log(`Letter '${letter}' has a ${certainty.toFixed(2)}% certainty.`);

        // Track the letter with the highest certainty
        if (certainty > highestCertainty) {
            highestCertainty = certainty;
            predictedLetter = letter;
        }
    }


    // Compare the highest certainty letter with the target character
    if (predictedLetter === targetCharacter) {
        success = true;
        console.log(`Success! The predicted letter '${predictedLetter}' matches the target character '${targetCharacter}'.`);
    } else {
        success = false;
        console.log(`Failed. The predicted letter '${predictedLetter}' does not match the target character '${targetCharacter}'.`);
    }

    // Return the success status
    return success;
}


initializeHandTracking();
