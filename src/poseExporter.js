/**
 * PoseExporter - Exports pose landmark data via custom events
 */

export class PoseExporter {
  constructor(options = {}) {
    this.enabled = options.enabled !== false;
    this.exportFormat = options.format || 'normalized'; // options are normalized or pixel
    
    // statistics (visual performance monitoring)
    this.frameCount = 0;
    this.lastExportTime = 0;
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
    // Nothing to clean up
  }
}

/**
 * PoseReceiver - Receives pose data from PoseExporter via custom events
 */
export class PoseReceiver {
  constructor(callback) {
    this.callback = callback;
    this.lastReceivedTime = 0;
    
    // Listen for custom events
    window.addEventListener('pose-data', (event) => {
      this.lastReceivedTime = performance.now();
      if (this.callback) {
        this.callback(event.detail);
      }
    });
    
    console.log('PoseReceiver: Listening for pose data');
  }
  
  destroy() {
    // Nothing to clean up
  }
}

