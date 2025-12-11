/**
 * PoseExporter - Broadcasts pose landmark data to other tabs/windows and applications
 * Supports multiple export methods:
 * 1. BroadcastChannel - for communication between browser tabs
 * 2. WebSocket - for sending data to external applications
 * 3. Custom Events - for same-page listeners
 */

export class PoseExporter {
  constructor(options = {}) {
    this.channel = null;
    this.websocket = null;
    this.enabled = options.enabled !== false;
    this.exportFormat = options.format || 'normalized'; // options are normalized or pixel
    
    // the BroadcastChannel we use for inter-tab communication
    if (typeof BroadcastChannel !== 'undefined') {
      this.channel = new BroadcastChannel('pose-data-channel');
      console.log('PoseExporter: BroadcastChannel initialized');
    }
    
    if (options.websocketUrl) {
      this.connectWebSocket(options.websocketUrl);
    }
    
    // statistics (visual performance monitoring)
    this.frameCount = 0;
    this.lastExportTime = 0;
  }
  
  connectWebSocket(url) {
    try {
      this.websocket = new WebSocket(url);
      
      this.websocket.onopen = () => {
        console.log('PoseExporter: WebSocket connected to', url);
      };
      
      this.websocket.onerror = (error) => {
        console.error('PoseExporter: WebSocket error', error);
      };
      
      this.websocket.onclose = () => {
        console.log('PoseExporter: WebSocket disconnected');
      };
    } catch (error) {
      console.error('PoseExporter: Failed to connect WebSocket', error);
    }
  }
  
  /**
   * Export pose landmarks
   * @param {Object} result - MediaPipe pose detection result
   * @param {number} timestamp - Current timestamp
   * @param {Object} videoInfo - Video dimensions {width, height}
   */
  exportPose(result, timestamp, videoInfo = {}) {
    if (!this.enabled || !result.landmarks || result.landmarks.length === 0) {
      return;
    }
    
    const poseData = {
      timestamp: timestamp,
      frameCount: this.frameCount++,
      poses: result.landmarks.map((landmarks, index) => ({
        personId: index,
        landmarks: landmarks.map((landmark, idx) => ({
          id: idx,
          name: this.getLandmarkName(idx),
          x: landmark.x,
          y: landmark.y,
          z: landmark.z,
          visibility: landmark.visibility || 1.0
        })),
        worldLandmarks: result.worldLandmarks?.[index]?.map((landmark, idx) => ({
          id: idx,
          name: this.getLandmarkName(idx),
          x: landmark.x,
          y: landmark.y,
          z: landmark.z,
          visibility: landmark.visibility || 1.0
        })) || []
      })),
      videoInfo: videoInfo
    };
    
    if (this.channel) {
      try {
        this.channel.postMessage(poseData);
      } catch (error) {
        console.error('PoseExporter: BroadcastChannel error', error);
      }
    }
    
    // back to sending via WebSocket
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      try {
        this.websocket.send(JSON.stringify(poseData));
      } catch (error) {
        console.error('PoseExporter: WebSocket send error', error);
      }
    }
    
    window.dispatchEvent(new CustomEvent('pose-data', {
      detail: poseData
    }));
    
    this.lastExportTime = timestamp;
  }
  
  /**
   * Get landmark name by index (MediaPipe Pose landmark indices)
   */
  getLandmarkName(index) {
    const landmarkNames = [
      'nose',                    // 0
      'left_eye_inner',          // 1
      'left_eye',                // 2
      'left_eye_outer',          // 3
      'right_eye_inner',         // 4
      'right_eye',               // 5
      'right_eye_outer',         // 6
      'left_ear',                // 7
      'right_ear',               // 8
      'mouth_left',              // 9
      'mouth_right',             // 10
      'left_shoulder',           // 11
      'right_shoulder',          // 12
      'left_elbow',              // 13
      'right_elbow',             // 14
      'left_wrist',              // 15
      'right_wrist',             // 16
      'left_pinky',              // 17
      'right_pinky',             // 18
      'left_index',              // 19
      'right_index',             // 20
      'left_thumb',              // 21
      'right_thumb',             // 22
      'left_hip',                // 23
      'right_hip',               // 24
      'left_knee',               // 25
      'right_knee',              // 26
      'left_ankle',              // 27
      'right_ankle',             // 28
      'left_heel',               // 29
      'right_heel',              // 30
      'left_foot_index',         // 31
      'right_foot_index'         // 32
    ];
    
    return landmarkNames[index] || `landmark_${index}`;
  }
  
  /**
   * Enable or disable pose export
   */
  setEnabled(enabled) {
    this.enabled = enabled;
    console.log('PoseExporter:', enabled ? 'enabled' : 'disabled');
  }
  
  /**
   * Clean up resources
   */
  destroy() {
    if (this.channel) {
      this.channel.close();
    }
    if (this.websocket) {
      this.websocket.close();
    }
  }
}

/**
 * PoseReceiver - Receives pose data from PoseExporter
 */
export class PoseReceiver {
  constructor(callback) {
    this.callback = callback;
    this.channel = null;
    this.lastReceivedTime = 0;
    
    // setting up BroadcastChannel listener
    if (typeof BroadcastChannel !== 'undefined') {
      this.channel = new BroadcastChannel('pose-data-channel');
      this.channel.onmessage = (event) => {
        this.lastReceivedTime = performance.now();
        if (this.callback) {
          this.callback(event.data);
        }
      };
      console.log('PoseReceiver: Listening for pose data');
    }
    
    // additional listen for custom events (same-page)
    window.addEventListener('pose-data', (event) => {
      this.lastReceivedTime = performance.now();
      if (this.callback) {
        this.callback(event.detail);
      }
    });
  }
  
  destroy() {
    if (this.channel) {
      this.channel.close();
    }
  }
}

