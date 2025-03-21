import React, {useEffect, useRef} from "react";
import {FilesetResolver, HandLandmarker} from "@mediapipe/tasks-vision";
import KNear from "./handtracker/knear.js";
import JSONData1 from "./models/allmoveset_rechts.json";
import {Color} from "excalibur";

const HandTrackingComponent = ({onDetect}) => {
    const videoRef = useRef(null);
    const handLandmarkerRef = useRef(null);
    const machine = new KNear(1);
    const files = [JSONData1];
    const animationFrameRef = useRef(null);
    const canvasRef = useRef(null);


    useEffect(() => {
        async function initializeHandTracking() {
            console.log("Initializing hand tracking...");
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
            );
            handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                    delegate: "GPU",
                },
                runningMode: "VIDEO",
                numHands: 2,
            });
            console.log("Hand tracking model loaded.");
            startWebcam();
        }

        function startWebcam() {
            console.log("Starting webcam...");
            navigator.mediaDevices.getUserMedia({video: true}).then((stream) => {
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.play();
                    videoRef.current.onloadeddata = () => {
                        console.log("Webcam started. Loading model...");
                        setTimeout(() => {
                            importJSON().then(visualizeHands);


                        }, 5000);
                        startRealTimeDetection();
                    };
                }
            }).catch((error) => {
                console.error("Error accessing webcam:", error);
            });

        }

        async function visualizeHands() {

            // DEBUG Functie kan uit kost volgens mij veel performance
            if (!handLandmarkerRef.current || !videoRef.current || !canvasRef.current) return;

            const ctx = canvasRef.current.getContext("2d");
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

            const results = await handLandmarkerRef.current.detectForVideo(videoRef.current, performance.now());

            if (results.landmarks.length > 0) {
                for (let hand of results.landmarks) {
                    // roept de visualisatie functie aan
                    drawHand(hand, ctx, canvasRef.current);
                }
            }

            requestAnimationFrame(visualizeHands);
        }

        function drawHand(landmarks, ctx, canvas) {
            ctx.lineWidth = 2;

            const connections = [
                [0, 1], [1, 2], [2, 3], [3, 4],
                [0, 5], [5, 6], [6, 7], [7, 8],
                [0, 9], [9, 10], [10, 11], [11, 12],
                [0, 13], [13, 14], [14, 15], [15, 16],
                [0, 17], [17, 18], [18, 19], [19, 20],
                [5, 9], [9, 13], [13, 17]
            ];

            // Find unique Z values for knuckles
            let zGroups = {};
            landmarks.forEach(p => {
                let roundedZ = Math.round(p.z * 10) / 10; // Round to 1 decimal for grouping
                if (!zGroups[roundedZ]) {
                    zGroups[roundedZ] = [];
                }
                zGroups[roundedZ].push(p);
            });

            // Get sorted Z depth levels (planes)
            let sortedZKeys = Object.keys(zGroups).map(Number).sort((a, b) => a - b);
            let minZ = sortedZKeys[0];
            let maxZ = sortedZKeys[sortedZKeys.length - 1];

            function getAlpha(z) {
                // Assign same alpha for all points in the same depth group
                let normalizedZ = (z - minZ) / (maxZ - minZ);
                return Math.max(0.2, 1 - normalizedZ); // At least 20% opacity
            }

            let depthAlpha = {};
            sortedZKeys.forEach(z => {
                depthAlpha[z] = getAlpha(z);
            });

            connections.forEach(([start, end]) => {
                let zStart = Math.round(landmarks[start].z * 10) / 10;
                let zEnd = Math.round(landmarks[end].z * 10) / 10;
                let alpha = (depthAlpha[zStart] + depthAlpha[zEnd]) / 2; // Average of the two depths

                ctx.strokeStyle = `rgba(245, 245, 220, ${alpha})`; // Beige lines with transparency
                ctx.lineWidth = 15;
                ctx.beginPath();
                ctx.moveTo(landmarks[start].x * canvas.width, landmarks[start].y * canvas.height);
                ctx.lineTo(landmarks[end].x * canvas.width, landmarks[end].y * canvas.height);
                ctx.stroke();
            });

            landmarks.forEach(point => {
                let zGroup = Math.round(point.z * 10) / 10;
                let alpha = depthAlpha[zGroup];

                ctx.fillStyle = `rgba(128, 128, 128, ${alpha})`; // Gray circles with transparency
                ctx.lineWidth = 10
                ctx.beginPath();
                ctx.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2);
                ctx.fill();
            });
        }


        async function importJSON() {
            if (files.length > 0) {
                console.log("Loading training data...");
                await loadModel(files);
                console.log("Training data loaded.");
            } else {
                console.error("No training files found.");
            }
        }

        async function loadModel(files) {
            for (const file of files) {
                for (const [key, value] of Object.entries(file)) {
                    console.log(`Training model with letter: ${key}`);
                    machine.learn(value, key);
                }
            }
        }

        async function processFrame() {
            if (!handLandmarkerRef.current || !videoRef.current) return;

            const results = await handLandmarkerRef.current.detectForVideo(videoRef.current, performance.now());
            // Optionally, add a check to ensure that the detection result is valid.
            if (!results || !results.landmarks || results.landmarks.length === 0) {
                animationFrameRef.current = requestAnimationFrame(processFrame);
                return;
            }
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

            if (detectArray.length > 0) {
                const flattenedData = detectArray.flat();
                const result = machine.findNearest(flattenedData, 2);
                // console.log(result);
                if (onDetect) onDetect(result);
            }

            await new Promise(resolve => {
                setTimeout(() => {
                    animationFrameRef.current = requestAnimationFrame(resolve);
                }, 200);
            });
            return processFrame();

        }

        async function startRealTimeDetection() {
            console.log("Starting real-time detection...");

            await new Promise(resolve => {
                animationFrameRef.current = requestAnimationFrame(resolve);
            });
            return processFrame();
        }

        initializeHandTracking();

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, []);

    return (
        <div style={{
            position: 'absolute',
            paddingTop: '520px',
            marginRight: '850px'


        }}>
            <video ref={videoRef} style={{
                transform: "scaleX(-1)",
                display: 'none'
            }} autoPlay>
            </video>
            <canvas
                ref={canvasRef}
                style={{
                    transform: "scaleX(-1)",
                    display: "fixed",
                    bottom: "500px",
                }}
            ></canvas>
        </div>
    );
};

export default HandTrackingComponent;
