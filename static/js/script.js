const video = document.getElementById("video-feed");

// When the webcam feed image starts displaying, update the status
video.onload = function () {
  document.getElementById("status").innerText = "Ready for your sign";
};

function checkGesture() {
  fetch('/get_current_gesture')
    .then(response => response.json())
    .then(data => {
      if (data.gesture !== "No gesture detected") {
        document.getElementById("gesture-text").innerText = data.gesture;
        document.getElementById("gesture-meaning").innerText = data.meaning;
        document.getElementById("status").innerText = "Detection Complete!";
        document.getElementById("video-container").style.display = "none";
        document.getElementById("restart-btn").style.display = "inline-block";
      }
    });
}

function restartWebcam() {
  fetch('/clear_gesture')
    .then(response => response.json())
    .then(() => {
      document.getElementById("gesture-text").innerText = "Waiting for sign...";
      document.getElementById("gesture-meaning").innerText = "Meaning will appear here...";
      document.getElementById("status").innerText = "Ready for your sign";
      document.getElementById("video-container").style.display = "block";
      document.getElementById("restart-btn").style.display = "none";
    });
}

// Auto check gesture every 2 seconds
setInterval(checkGesture, 2000);
