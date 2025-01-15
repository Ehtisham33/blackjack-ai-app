const flaskBackendUrl = "http://localhost:5000/video_frame";  // Define your Flask backend URL here
const socket = io(flaskBackendUrl);  // Establish WebSocket connection with Flask backend

const webcamContainer = document.getElementById("webcam-container");
const webcam = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const startCaptureButton = document.getElementById("startCaptureButton");

let videoStream = null;
let capturing = false;

// Start or stop webcam capture
startCaptureButton.addEventListener("click", () => {
    if (!capturing) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                videoStream = stream;
                webcam.srcObject = stream;
                webcamContainer.style.display = 'block';
                capturing = true;
                startCaptureButton.innerText = "Stop Capture";
                // Capture frames continuously and send them to the backend
                setInterval(captureAndSendFrame, 100);  // Capture every 100ms (adjust for real-time)
            })
            .catch((error) => {
                console.error("Error accessing webcam:", error);
            });
    } else {
        // Stop the webcam capture
        videoStream.getTracks().forEach(track => track.stop());
        webcamContainer.style.display = 'none';
        capturing = false;
        startCaptureButton.innerText = "Start Capture";
    }
});

// Function to capture frame from video and send to Flask backend
function captureAndSendFrame() {
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

    // Convert canvas image to base64 (JPEG)
    const base64Image = canvas.toDataURL('image/jpeg');

    // Send the frame via WebSocket to Flask
    socket.emit('video_frame', base64Image);
}

// Listen for predictions from the backend and display them
socket.on('predictions', (data) => {
    const { frame_results, total_value } = data;
    ctx.clearRect(0, 0, canvas.width, canvas.height);  // Clear previous frame
    frame_results.forEach((result) => {
        const [x1, y1, x2, y2] = result.bbox;
        const confidence = (result.confidence * 100).toFixed(1);
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);  // Draw bounding box
        ctx.fillStyle = 'red';
        ctx.fillText(`Card: ${result.card} | Confidence: ${confidence}%`, x1, y1 - 10);  // Display label
    });
    console.log('Total Value:', total_value);  // Log total Blackjack value
});

// Toggle dark mode functionality
document.getElementById("toggleThemeButton").addEventListener("click", function () {
    // Toggle dark mode on body and header
    document.body.classList.toggle("dark-mode");
    document.querySelector("header").classList.toggle("dark-mode");
});

