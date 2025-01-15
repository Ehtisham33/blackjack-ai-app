// Connect to the backend server using Socket.IO
const socket = io.connect('http://localhost:5000/video_frame'); // Replace with your backend's URL if hosted elsewhere

const videoElement = document.getElementById('video');
const startCaptureButton = document.getElementById('startCapture');
const predictionsContainer = document.createElement('div'); // Create a container for predictions
predictionsContainer.style.marginTop = "10px";
document.body.appendChild(predictionsContainer);

let videoStream;

// Function to start video capture
async function startCapture() {
    try {
        videoStream = await navigator.mediaDevices.getDisplayMedia({ video: true });
        videoElement.srcObject = videoStream;

        // Start sending frames to the backend
        const videoTrack = videoStream.getVideoTracks()[0];
        const imageCapture = new ImageCapture(videoTrack);

        setInterval(async () => {
            try {
                const blob = await imageCapture.grabFrame();
                const canvas = document.createElement('canvas');
                canvas.width = blob.width;
                canvas.height = blob.height;
                const context = canvas.getContext('2d');
                context.drawImage(blob, 0, 0);
                const frameData = canvas.toDataURL('image/jpeg');

                // Send the frame to the backend
                socket.emit('video_frame', frameData);
            } catch (error) {
                console.error('Error capturing video frame:', error);
            }
        }, 200); // Send frames every 200ms
    } catch (err) {
        console.error('Error starting capture:', err);
    }
}

// Handle predictions from the backend
socket.on('predictions', (data) => {
    predictionsContainer.innerHTML = ''; // Clear previous predictions

    // Display total card value
    const totalValueElement = document.createElement('p');
    totalValueElement.textContent = `Total Value: ${data.total_value}`;
    predictionsContainer.appendChild(totalValueElement);

    // Display detected cards
    data.frame_results.forEach((result) => {
        const cardElement = document.createElement('p');
        cardElement.textContent = `Card: ${result.card}, Value: ${result.value}, Confidence: ${(result.confidence * 100).toFixed(2)}%`;
        predictionsContainer.appendChild(cardElement);
    });
});

// Start video capture when the button is clicked
startCaptureButton.addEventListener('click', startCapture);

// Handle connection events
socket.on('connect', () => {
    console.log('Connected to backend server');
});

socket.on('disconnect', () => {
    console.log('Disconnected from backend server');
});
