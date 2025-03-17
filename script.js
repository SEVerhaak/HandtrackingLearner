import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import KNear from "./knear.js";

// alle DOM elementen
const letterSelectorInput = document.getElementById("letterInput");
const startButton = document.getElementById("startButton");
const snapshotButton = document.getElementById("snapshotButton");
const exportButton = document.getElementById("exportButton");
const checkButton = document.getElementById("checkButton");
const jsonButton = document.getElementById("importJSONButton");
let dataset = {}; // Object to store data by letter

const handData = {}; // Object to store hand tracking data
const canvas = document.getElementById("output");

const ctx = canvas.getContext("2d");
const video = document.getElementById("videoElement");


let webcamRunning =  false;
let handLandmarker;
let recording = false;
let detecting = false;

// Initialize KNear with k = 3 (you can adjust based on needs)
const machine = new KNear(1);

async function initializeHandTracking() {
    // init
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
    // init
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
        video.play();
        webcamRunning = true;

        video.onloadeddata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            visualiseHands();
        };
    });
}

async function visualiseHands() {
    // voor de visualisatie van de hand positie
    if (!webcamRunning) return;

    // Fill the canvas with a black background
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const results = await handLandmarker.detectForVideo(video, performance.now());

    if (results.landmarks.length > 0) {
        for (const landmarks of results.landmarks) {
            drawHandLandmarks(landmarks);
        }
    }

    requestAnimationFrame(visualiseHands);
}

function drawHandLandmarks(landmarks) {
    ctx.fillStyle = "red";
    ctx.strokeStyle = "green";
    ctx.lineWidth = 2;


    // Define hand connections
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // Index
        [0, 9], [9, 10], [10, 11], [11, 12], // Middle
        [0, 13], [13, 14], [14, 15], [15, 16], // Ring
        [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
        [5, 9], [9, 13], [13, 17] // Palm connections
    ];

    // Draw connections
    connections.forEach(([start, end]) => {
        ctx.beginPath();
        ctx.moveTo(landmarks[start].x * canvas.width, landmarks[start].y * canvas.height);
        ctx.lineTo(landmarks[end].x * canvas.width, landmarks[end].y * canvas.height);
        ctx.stroke();
    });

    // Draw landmarks
    landmarks.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2);
        ctx.fill();
    });
}

async function importJSON() {
    const files = [
        '../datasets/allmoveset_links.json',
    ];
    for (const file of files){
        try {
            const response = await fetch(file); // Fetch the file
            if (!response.ok) throw new Error("Failed to load JSON");

            const fileData = await response.json();
            console.log(fileData); // Use the object

            for (const [key, value] of Object.entries(fileData)) {
                console.log(`learned letter: ${key}`);
                machine.learn(value, key)
            }
            // return fileData; // Return it if needed elsewhere

        } catch (error) {
            console.error("Error loading JSON:", error);
        }
    }
}

function addToDataset(letter, flattenedData) {
    dataset[letter] = flattenedData;
    console.log(dataset);
}

async function recordHandTracking() {
    // Controleer of het model is geladen
    if (!handLandmarker) {
        console.error("Hand tracking model is not initialized.");
        return;
    }

    // Haal de letter op uit de invoer
    const letter = letterSelectorInput.value;
    if (!letter) {
        console.error("Please enter a letter.");
        return;
    }

    // Maak een opslag aan voor de letter als deze nog niet bestaat
    if (!handData[letter]) {
        handData[letter] = [];
    }

    recording = true;
    let collectedData = [];
    console.log(`Recording started for letter: ${letter}...`);

    const numFrames = 10; // 🔥 Aantal frames dat we per letter opnemen
    let validFrames = 0; // Houdt bij hoeveel frames geldig zijn

    for (let i = 0; i < numFrames; i++) {
        const results = await handLandmarker.detectForVideo(video, performance.now());
        let detectArray = [];

        if (results.landmarks && results.landmarks.length > 0) { // Check of er een hand is gedetecteerd
            for (let hand of results.landmarks) {
                let handPoints = [];
                const wrist = hand[0]; // De pols als referentiepunt

                for (let handSingle of hand) {
                    let relX = handSingle.x - wrist.x;
                    let relY = handSingle.y - wrist.y;
                    let relZ = handSingle.z - wrist.z;
                    handPoints.push(relX, relY, relZ); // Relatieve coördinaten opslaan als vlakke array
                }
                detectArray.push(handPoints);
            }

            handData[letter].push(detectArray);
            collectedData.push(detectArray.flat()); // Gegevens opslaan voor het ML-model
            validFrames++; // Alleen ophogen als het een geldig frame is
        } else {
            console.warn(`Frame ${i + 1}: No hand detected, skipping...`);
        }

        await new Promise(resolve => setTimeout(resolve, 50)); // Kleine vertraging tussen frames
    }

    console.log(`Recording stopped for letter: ${letter}`);
    recording = false;

    // Controle: Voorkom NaN door alleen te middelen als er geldige frames zijn
    if (validFrames > 0) {
        let averagedData = new Array(collectedData[0].length).fill(0);

        for (let frame of collectedData) {
            frame.forEach((value, index) => {
                averagedData[index] += value;
            });
        }

        averagedData = averagedData.map(val => val / validFrames); // 🔥 Deel door het aantal geldige frames

        addToDataset(letter, averagedData);
        machine.learn(averagedData, letter);
        console.log(`Learned letter: ${letter} with ${validFrames} valid frames.`);
    } else {
        console.error("No valid data collected. Please try again.");
    }
}


function startDetection(){
    detecting = !detecting;
    detectGesture()
}

async function detectGesture() {
    return new Promise(resolve => {
        async function loop() {
            if (!handLandmarker) {
                console.error("Hand tracking model is not initialized.");
                return;
            }

            if (!detecting) {
                console.log('stopped detecting');
                return;
            }

            let collectedData = [];
            console.log("Gesture detection started...");

            const numFrames = 5; // Aantal frames om te verzamelen

            for (let i = 0; i < numFrames; i++) {
                const results = await handLandmarker.detectForVideo(video, performance.now());
                let detectArray = [];

                for (let hand of results.landmarks) {
                    const wrist = hand[0];
                    for (let handSingle of hand) {
                        let relX = handSingle.x - wrist.x;
                        let relY = handSingle.y - wrist.y;
                        let relZ = handSingle.z - wrist.z;
                        detectArray.push([relX, relY, relZ]);
                    }
                }
                collectedData.push(detectArray.flat());
                await new Promise(resolve => setTimeout(resolve, 50)); // Vertraging tussen frames
            }

            // Bereken het gemiddelde van de verzamelde frames
            let averagedData = new Array(collectedData[0].length).fill(0);
            for (let frame of collectedData) {
                frame.forEach((value, index) => {
                    averagedData[index] += value;
                });
            }
            averagedData = averagedData.map(val => val / numFrames); // Gemiddelde berekening

            if (averagedData.length > 0) {
                const nearestMatches = machine.findNearest(averagedData, 2);

                let resultText = nearestMatches.map(({ label, distance }) => `${label}: ${distance.toFixed(4)}`).join("<br>");
                document.getElementById('result-text').innerHTML = `${resultText}`;
                console.log("Nearest Matches:", nearestMatches);
            } else {
                console.error("No data collected for gesture detection.");
            }

            setTimeout(loop, 50);
        }

        loop();
    });
}


function exportHandData() {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(dataset, null, 1));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "wie.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    document.body.removeChild(downloadAnchor);
}

startButton.addEventListener("click", startWebcam);
snapshotButton.addEventListener("click", recordHandTracking);
exportButton.addEventListener("click", exportHandData);
checkButton.addEventListener("click", startDetection);
jsonButton.addEventListener("click", importJSON);

initializeHandTracking();