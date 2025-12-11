// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

import { PoseExporter } from "./poseExporter.js";

export class PoseDetector {
  constructor() {
    this.poseLandmarker = undefined;
    this.runningMode = "IMAGE";
    this.webcamRunning = false;
    this.video = null;
    this.canvasElement = null;
    this.canvasCtx = null;
    this.drawingUtils = null;
    this.lastVideoTime = -1;
    this.poseExporter = new PoseExporter({
      enabled: true,
      format: 'normalized'
    });
    this.enableWebcamButton = null;
  }

  async init() {
    // Initialize MediaPipe PoseLandmarker
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
        delegate: "GPU"
      },
      runningMode: this.runningMode,
      numPoses: 2
    });

    // Get DOM elements
    this.video = document.getElementById("webcam");
    this.canvasElement = document.getElementById("output_canvas");
    if (this.canvasElement) {
      this.canvasCtx = this.canvasElement.getContext("2d");
      this.drawingUtils = new DrawingUtils(this.canvasCtx);
    }

    // Set up webcam button
    const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;
    if (hasGetUserMedia()) {
      this.enableWebcamButton = document.getElementById("webcamButton");
      if (this.enableWebcamButton) {
        this.enableWebcamButton.addEventListener("click", () => this.enableCam());
      }
    } else {
      console.warn("getUserMedia() is not supported by your browser");
    }

    console.log("PoseDetector initialized");
  }

  enableCam() {
    if (!this.poseLandmarker) {
      console.log("Wait! poseLandmarker not loaded yet.");
      return;
    }

    if (this.webcamRunning === true) {
      this.webcamRunning = false;
      if (this.enableWebcamButton) {
        this.enableWebcamButton.innerText = "ENABLE WEBCAM";
      }
      // Stop the video stream
      if (this.video && this.video.srcObject) {
        const tracks = this.video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        this.video.srcObject = null;
      }
    } else {
      this.webcamRunning = true;
      if (this.enableWebcamButton) {
        this.enableWebcamButton.innerText = "DISABLE WEBCAM";
      }

      // getUsermedia parameters.
      const constraints = {
        video: true
      };

      // Activate the webcam stream.
      navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        this.video.srcObject = stream;
        this.video.addEventListener("loadeddata", () => this.predictWebcam());
      });
    }
  }

  async predictWebcam() {
    if (!this.video || !this.canvasElement) return;

    const videoHeight = "360px";
    const videoWidth = "480px";
    this.canvasElement.style.height = videoHeight;
    this.video.style.height = videoHeight;
    this.canvasElement.style.width = videoWidth;
    this.video.style.width = videoWidth;

    // Now let's start detecting the stream.
    if (this.runningMode === "IMAGE") {
      this.runningMode = "VIDEO";
      await this.poseLandmarker.setOptions({ runningMode: "VIDEO" });
    }
    
    let startTimeMs = performance.now();
    if (this.lastVideoTime !== this.video.currentTime) {
      this.lastVideoTime = this.video.currentTime;
      const result = await this.poseLandmarker.detectForVideo(this.video, startTimeMs);
      
      // Draw landmarks on canvas if available
      if (this.canvasCtx && this.drawingUtils) {
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        for (const landmark of result.landmarks) {
          this.drawingUtils.drawLandmarks(landmark, {
            radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
          });
          this.drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
        }
        this.canvasCtx.restore();
      }
      
      // Export pose data to other applications
      this.poseExporter.exportPose(result, startTimeMs, {
        width: this.canvasElement.width,
        height: this.canvasElement.height
      });
    }

    // Call this function again to keep predicting when the browser is ready.
    if (this.webcamRunning === true) {
      window.requestAnimationFrame(() => this.predictWebcam());
    }
  }
}

