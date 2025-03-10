export function captureHandSnapshot(landmarks, snapshotCanvas, canvas) {
    if (!landmarks || landmarks.length === 0) return;

    const snapshotCtx = snapshotCanvas.getContext("2d");
    snapshotCanvas.width = canvas.width;
    snapshotCanvas.height = canvas.height;

    // Clear previous snapshot
    snapshotCtx.clearRect(0, 0, snapshotCanvas.width, snapshotCanvas.height);

    snapshotCtx.fillStyle = "blue";
    snapshotCtx.strokeStyle = "orange";
    snapshotCtx.lineWidth = 2;

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
        snapshotCtx.beginPath();
        snapshotCtx.moveTo(landmarks[start].x * snapshotCanvas.width, landmarks[start].y * snapshotCanvas.height);
        snapshotCtx.lineTo(landmarks[end].x * snapshotCanvas.width, landmarks[end].y * snapshotCanvas.height);
        snapshotCtx.stroke();
    });

    // Draw landmarks
    landmarks.forEach(point => {
        snapshotCtx.beginPath();
        snapshotCtx.arc(point.x * snapshotCanvas.width, point.y * snapshotCanvas.height, 5, 0, Math.PI * 2);
        snapshotCtx.fill();
    });
}

// Draw red dots and green lines for hand landmarks
export function drawHand(landmarks, ctx, canvas) {
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
